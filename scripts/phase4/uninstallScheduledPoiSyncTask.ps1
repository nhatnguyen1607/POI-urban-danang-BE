param(
  [string]$TaskName = 'UrbanAgent POI Scheduled Sync',
  [string]$StatusPath = 'D:\UrbanAgent-artifacts\stage4o-status\status.json',
  [switch]$Apply
)
$ErrorActionPreference = 'Stop'
$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if (-not $task) { Write-Output 'TASK_NOT_INSTALLED'; exit 0 }
if (-not $Apply) {
  Write-Output "DRY_RUN: would unregister only task '$TaskName'; project state and artifacts remain untouched."
  exit 0
}
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
if (Test-Path -LiteralPath $StatusPath) {
  $status = Get-Content -Raw -LiteralPath $StatusPath | ConvertFrom-Json
  $status | Add-Member -NotePropertyName schedulerEnabled -NotePropertyValue $false -Force
  $status | Add-Member -NotePropertyName nextScheduledRun -NotePropertyValue $null -Force
  $status | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $StatusPath -Encoding utf8
}
Write-Output "UNINSTALLED_TASK_ONLY: $TaskName"
