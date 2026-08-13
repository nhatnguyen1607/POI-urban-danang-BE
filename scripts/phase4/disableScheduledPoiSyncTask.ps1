param(
  [string]$TaskName = 'UrbanAgent POI Scheduled Sync',
  [string]$StatusPath = 'D:\UrbanAgent-artifacts\stage4o-status\status.json',
  [switch]$Apply
)
$ErrorActionPreference = 'Stop'
$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if (-not $task) { Write-Output 'TASK_NOT_INSTALLED'; exit 0 }
if (-not $Apply) { Write-Output "DRY_RUN: would disable $TaskName"; exit 0 }
Disable-ScheduledTask -TaskName $TaskName | Out-Null
if (Test-Path -LiteralPath $StatusPath) {
  $status = Get-Content -Raw -LiteralPath $StatusPath | ConvertFrom-Json
  $status | Add-Member -NotePropertyName schedulerEnabled -NotePropertyValue $false -Force
  $status | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $StatusPath -Encoding utf8
}
Write-Output "DISABLED: $TaskName"
