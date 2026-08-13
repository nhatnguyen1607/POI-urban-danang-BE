param(
  [ValidateSet('overture', 'osm', 'wikidata', 'wikimedia_commons', 'all')]
  [string]$Source = 'all',
  [string]$Repository = ((Split-Path (Split-Path $PSScriptRoot))),
  [string]$DataRoot = 'D:\UrbanAgent-data\stage4n',
  [string]$ArtifactRoot = 'D:\UrbanAgent-artifacts\stage4n-sync',
  [string]$TempRoot = 'D:\UrbanAgent-temp\stage4n',
  [Parameter(Mandatory = $true)]
  [string]$SnapshotManifest,
  [string]$PolicyCases,
  [string]$DecisionMemory,
  [string]$RecoveredEnrichments,
  [switch]$PreparePr,
  [switch]$DryRun,
  [switch]$Resume
)

$ErrorActionPreference = 'Stop'
$runId = Get-Date -Format 'yyyyMMdd-HHmmss'
$artifactDir = Join-Path $ArtifactRoot $runId
$tempDir = Join-Path $TempRoot $runId
New-Item -ItemType Directory -Force -Path $artifactDir, $tempDir, $DataRoot | Out-Null

$arguments = @(
  'run', 'citypack:scheduled-sync', '--', '--no-network',
  '--artifact-dir', $artifactDir,
  '--data-dir', $DataRoot,
  '--temp-dir', $tempDir,
  '--snapshot-manifest', $SnapshotManifest
)
if ($Source -eq 'all') { $arguments += '--all-sources' } else { $arguments += @('--source', $Source) }
if ($PreparePr -and $DryRun) { throw '-PreparePr and -DryRun are mutually exclusive.' }
if ($PreparePr) { $arguments += '--prepare-pr' }
elseif ($DryRun) { $arguments += '--dry-run' }
else { $arguments += '--apply-state' }
if ($Resume) { $arguments += '--resume' }
if ($PolicyCases) { $arguments += @('--policy-cases', $PolicyCases) }
if ($DecisionMemory) { $arguments += @('--decision-memory', $DecisionMemory) }
if ($RecoveredEnrichments) { $arguments += @('--recovered-enrichments', $RecoveredEnrichments) }

$stdout = Join-Path $artifactDir 'scheduled-sync.stdout.log'
$stderr = Join-Path $artifactDir 'scheduled-sync.stderr.log'
$process = Start-Process -FilePath 'npm.cmd' -ArgumentList $arguments -WorkingDirectory $Repository `
  -RedirectStandardOutput $stdout -RedirectStandardError $stderr -NoNewWindow -Wait -PassThru
exit $process.ExitCode
