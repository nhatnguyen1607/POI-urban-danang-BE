param(
  [string]$TaskName = 'UrbanAgent-Stage4N-POI-Sync',
  [string]$WrapperPath = (Join-Path $PSScriptRoot 'runScheduledPoiSync.ps1'),
  [Parameter(Mandatory = $true)]
  [string]$SnapshotManifest,
  [string]$Cadence = 'DAILY',
  [string]$StartTime = '03:00',
  [switch]$Apply
)

$arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$WrapperPath`" -SnapshotManifest `"$SnapshotManifest`""
$command = "schtasks.exe /Create /TN `"$TaskName`" /TR `"powershell.exe $arguments`" /SC $Cadence /ST $StartTime /F"
if (-not $Apply) {
  Write-Output 'DRY_RUN_ONLY - scheduled task was not registered.'
  Write-Output $command
  exit 0
}

& schtasks.exe /Create /TN $TaskName /TR "powershell.exe $arguments" /SC $Cadence /ST $StartTime /F
exit $LASTEXITCODE
