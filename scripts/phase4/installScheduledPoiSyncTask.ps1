param(
  [string]$TaskName = 'UrbanAgent POI Scheduled Sync',
  [string]$Repository = 'D:\UrbanAgent-work\scheduled-poi-sync',
  [string]$DataRoot = 'D:\UrbanAgent-data\stage4o',
  [string]$ArtifactRoot = 'D:\UrbanAgent-artifacts\stage4o-runs',
  [string]$TempRoot = 'D:\UrbanAgent-temp\stage4o',
  [string]$StatusPath = 'D:\UrbanAgent-artifacts\stage4o-status\status.json',
  [Parameter(Mandatory = $true)]
  [string]$SnapshotManifest,
  [ValidateRange(1, 24)]
  [int]$TickHours = 6,
  [switch]$Apply
)

$ErrorActionPreference = 'Stop'
$repositoryPath = (Resolve-Path -LiteralPath $Repository).Path
$wrapper = Join-Path $repositoryPath 'scripts\phase4\runScheduledPoiSync.ps1'
if (-not (Test-Path -LiteralPath $wrapper)) { throw "Wrapper not found: $wrapper" }
if (-not (Test-Path -LiteralPath $SnapshotManifest)) { throw "Snapshot manifest not found: $SnapshotManifest" }

$preflightArguments = @(
  '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $wrapper,
  '-Repository', $repositoryPath, '-DataRoot', $DataRoot, '-ArtifactRoot', $ArtifactRoot,
  '-TempRoot', $TempRoot, '-StatusPath', $StatusPath, '-SnapshotManifest', $SnapshotManifest, '-SelfTest'
)
$preflight = Start-Process -FilePath (Get-Command powershell.exe).Source -ArgumentList $preflightArguments `
  -WorkingDirectory $repositoryPath -NoNewWindow -Wait -PassThru
if ($preflight.ExitCode -ne 0) { throw 'Stage 4O preflight failed; task was not registered.' }

$powerShell = (Get-Command powershell.exe).Source
$taskArguments = @(
  '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$wrapper`"",
  '-Repository', "`"$repositoryPath`"", '-DataRoot', "`"$DataRoot`"",
  '-ArtifactRoot', "`"$ArtifactRoot`"", '-TempRoot', "`"$TempRoot`"",
  '-StatusPath', "`"$StatusPath`"", '-SnapshotManifest', "`"$SnapshotManifest`"",
  '-TriggerType', 'SCHEDULED'
) -join ' '
$action = New-ScheduledTaskAction -Execute $powerShell -Argument $taskArguments -WorkingDirectory $repositoryPath
$trigger = New-ScheduledTaskTrigger -Once -At ((Get-Date).AddMinutes(5)) `
  -RepetitionInterval (New-TimeSpan -Hours $TickHours)
$principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
  -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -StartWhenAvailable `
  -ExecutionTimeLimit (New-TimeSpan -Hours 4) -RestartCount 1 -RestartInterval (New-TimeSpan -Minutes 15)

if (-not $Apply) {
  [pscustomobject]@{ Mode = 'DRY_RUN'; TaskName = $TaskName; TickHours = $TickHours;
    Repository = $repositoryPath; Action = $powerShell; Arguments = $taskArguments;
    Principal = 'CURRENT_USER_INTERACTIVE'; ExistingTask = [bool](Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) }
  exit 0
}

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
  -Principal $principal -Settings $settings -Description 'UrbanAgent bounded incremental POI sync; no auto-merge.' -Force | Out-Null
Enable-ScheduledTask -TaskName $TaskName | Out-Null
$task = Get-ScheduledTask -TaskName $TaskName
$taskInfo = Get-ScheduledTaskInfo -TaskName $TaskName
$status = if (Test-Path -LiteralPath $StatusPath) {
  Get-Content -Raw -LiteralPath $StatusPath | ConvertFrom-Json
} else { [pscustomobject]@{ schemaVersion = 'stage4o-health-v1' } }
$status | Add-Member -NotePropertyName schedulerTaskName -NotePropertyValue $TaskName -Force
$status | Add-Member -NotePropertyName schedulerEnabled -NotePropertyValue $true -Force
$status | Add-Member -NotePropertyName schedulerCommitSha `
  -NotePropertyValue ((git -C $repositoryPath rev-parse HEAD).Trim()) -Force
$status | Add-Member -NotePropertyName nextScheduledRun `
  -NotePropertyValue $taskInfo.NextRunTime.ToString('o') -Force
New-Item -ItemType Directory -Force -Path (Split-Path $StatusPath) | Out-Null
$status | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $StatusPath -Encoding utf8
[pscustomobject]@{ Mode = 'APPLIED'; TaskName = $task.TaskName; State = $task.State;
  TickHours = $TickHours; Repository = $repositoryPath; Principal = 'CURRENT_USER_INTERACTIVE';
  NextRun = $taskInfo.NextRunTime }
