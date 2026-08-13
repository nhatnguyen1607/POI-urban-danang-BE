param(
  [string]$TaskName = 'UrbanAgent POI Scheduled Sync',
  [string]$StatusPath = 'D:\UrbanAgent-artifacts\stage4o-status\status.json',
  [switch]$LastFailure
)
$ErrorActionPreference = 'Stop'
$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
$taskInfo = if ($task) { Get-ScheduledTaskInfo -TaskName $TaskName } else { $null }
$status = if (Test-Path -LiteralPath $StatusPath) {
  Get-Content -Raw -LiteralPath $StatusPath | ConvertFrom-Json
} else { $null }
if ($LastFailure) {
  [pscustomobject]@{ FailureClass = $status.lastFailureClass; FailureMessage = $status.lastFailureMessage;
    FailedAt = $status.lastFailure; Manifest = $status.lastManifestPath; ResumeAvailable = $status.resumeAvailable }
  exit 0
}
[pscustomobject]@{
  TaskInstalled = [bool]$task
  TaskEnabled = if ($task) { $task.State -ne 'Disabled' } else { $false }
  NextRun = $taskInfo.NextRunTime
  LastTaskRun = $taskInfo.LastRunTime
  LastTaskResult = $taskInfo.LastTaskResult
  LastUrbanAgentStatus = $status.lastRunStatus
  LastSuccessfulSync = $status.lastSuccessfulRun
  LatestPr = $status.lastPrUrl
  ExceptionCount = $status.lastHumanReviewDeltaCount
  CanonicalSha = $status.currentCanonicalSha
  StatusPath = $StatusPath
}
