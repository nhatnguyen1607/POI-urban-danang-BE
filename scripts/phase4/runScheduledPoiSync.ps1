param(
  [ValidateSet('overture', 'osm', 'wikidata', 'wikimedia_commons', 'all')]
  [string]$Source = 'all',
  [string]$Repository = 'D:\UrbanAgent-work\scheduled-poi-sync',
  [string]$DataRoot = 'D:\UrbanAgent-data\stage4o',
  [string]$ArtifactRoot = 'D:\UrbanAgent-artifacts\stage4o-runs',
  [string]$TempRoot = 'D:\UrbanAgent-temp\stage4o',
  [string]$StatusPath = 'D:\UrbanAgent-artifacts\stage4o-status\status.json',
  [string]$OperationsConfig = 'config\phase4_stage4o_operations.json',
  [string]$SnapshotManifest,
  [string]$PolicyCases,
  [string]$DecisionMemory,
  [string]$RecoveredEnrichments,
  [switch]$PreparePr,
  [switch]$DryRun,
  [switch]$Resume,
  [switch]$SelfTest,
  [ValidateSet('MANUAL', 'SCHEDULED', 'CANARY')]
  [string]$TriggerType = 'MANUAL'
)

$ErrorActionPreference = 'Stop'
$resolvedRepository = (Resolve-Path -LiteralPath $Repository).Path
$resolvedOperationsConfig = if ([IO.Path]::IsPathRooted($OperationsConfig)) {
  $OperationsConfig
} else {
  Join-Path $resolvedRepository $OperationsConfig
}
$runner = Join-Path $resolvedRepository 'scripts\citypack_scheduled_runner.js'
if (-not (Test-Path -LiteralPath $runner)) { throw "Scheduled runner not found: $runner" }
New-Item -ItemType Directory -Force -Path $ArtifactRoot, $TempRoot, $DataRoot, (Split-Path $StatusPath) | Out-Null

$arguments = @(
  $runner,
  '--repository', $resolvedRepository,
  '--operations-config', $resolvedOperationsConfig,
  '--data-dir', $DataRoot,
  '--artifact-root', $ArtifactRoot,
  '--temp-dir', $TempRoot,
  '--status-path', $StatusPath
)
if ($SelfTest) {
  $arguments += '--self-test'
} else {
  if (-not $SnapshotManifest) { throw '-SnapshotManifest is required unless -SelfTest is used.' }
  $arguments += @('--snapshot-manifest', $SnapshotManifest, '--trigger-type', $TriggerType)
  if ($Source -eq 'all') { $arguments += @('--source', 'all') } else { $arguments += @('--source', $Source) }
}
if ($PreparePr -and $DryRun) { throw '-PreparePr and -DryRun are mutually exclusive.' }
if ($PreparePr) { $arguments += '--prepare-pr' }
if ($DryRun) { $arguments += '--dry-run' }
if ($Resume) { $arguments += '--resume' }
if ($PolicyCases) { $arguments += @('--policy-cases', $PolicyCases) }
if ($DecisionMemory) { $arguments += @('--decision-memory', $DecisionMemory) }
if ($RecoveredEnrichments) { $arguments += @('--recovered-enrichments', $RecoveredEnrichments) }

$wrapperLogRoot = 'D:\UrbanAgent-artifacts\stage4o-logs'
New-Item -ItemType Directory -Force -Path $wrapperLogRoot | Out-Null
$runStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$stdout = Join-Path $wrapperLogRoot "$runStamp.stdout.log"
$stderr = Join-Path $wrapperLogRoot "$runStamp.stderr.log"
$process = Start-Process -FilePath (Get-Command node.exe).Source -ArgumentList $arguments `
  -WorkingDirectory $resolvedRepository -RedirectStandardOutput $stdout `
  -RedirectStandardError $stderr -NoNewWindow -Wait -PassThru
exit $process.ExitCode
