# Phase 4 Stage 4O - Schedule Activation And Observability

Status: validated on 2026-08-13. Stage 4O is operational on this Windows host,
but its repository branch remains unmerged pending user review.

## Activation

- Task: `UrbanAgent POI Scheduled Sync`
- Tick: every 6 hours
- Principal: current user, interactive logon, limited run level; no stored password
- Stable repository: `D:\UrbanAgent-work\scheduled-poi-sync`
- Persistent data: `D:\UrbanAgent-data\stage4o`
- Run artifacts: `D:\UrbanAgent-artifacts\stage4o-runs`
- Health: `D:\UrbanAgent-artifacts\stage4o-status\status.json`
- Logs: `D:\UrbanAgent-artifacts\stage4o-logs`
- Temporary files: `D:\UrbanAgent-temp\stage4o`
- Installed commit at activation: `ace8e09b37d1be0c7be34e58bef4ccce3dd95e9b`

Task Scheduler uses the stable PowerShell wrapper, an explicit working
directory, `IgnoreNew`, `StartWhenAvailable`, a four-hour execution limit, and
one bounded retry after 15 minutes. It does not alter the global PowerShell
execution policy.

Source cadence is independent of the scheduler tick:

| Source | Cadence |
| --- | ---: |
| OSM | 1 day |
| Wikidata | 3 days |
| Overture | 7 days |
| Wikimedia Commons | 7 days |

Google remains absent. A tick where no source is due exits without acquisition,
normalization, resolution, branch creation, or PR creation.

## Validation

The pre-activation self-test passed Node, dependency, Git, clean-worktree,
configuration, state-schema, boundary, canonical, D-drive write, child-process,
and lock create/release checks. GitHub operation mode is
`COMPARE_URL_ONLY`; GitHub CLI availability is not a core health requirement.

The real canary was started through Windows Task Scheduler. It completed with
task result `0` and UrbanAgent status `NO_WORK`, updated the health artifact,
wrote an operational manifest, released its lock, generated zero safe
mutations, generated no human-review delta, and created no PR. All four sources
were cadence-gated as not due. The canary took approximately 1.3 seconds.

A separate no-work invocation took approximately 2.0 seconds. Process startup
dominated; heavy acquisition, normalization, and entity resolution counts were
all zero. A concurrent invocation was skipped without state mutation. The
stale-lock watchdog cleared a deliberately old lock only after proving that its
same-host PID no longer existed and retained the cleared-lock record.

Disable followed by idempotent re-install/re-enable preserved all four
persistent state files byte-for-byte. The task is left installed and enabled.
Uninstall behavior was verified non-destructively and removes only the task.

Canonical activation baseline remains 4,166 POIs with SHA-256
`39647b29308813a7ec19e4695fd2b95ffa27743db8e8cb46800c45d3a3fe6ded`.
Runtime behavior, CSV default selection, canonical data, Firebase, PostgreSQL,
frontend code, and CI scheduling were not changed.

## Operations

```powershell
npm run citypack:self-test
npm run citypack:health
npm run citypack:health -- --last-failure
.\scripts\phase4\getScheduledPoiSyncStatus.ps1
.\scripts\phase4\disableScheduledPoiSyncTask.ps1 -Apply
.\scripts\phase4\uninstallScheduledPoiSyncTask.ps1 -Apply
```

Operational failures are classified as `RUNNER_FAILURE`,
`APPLICATION_FAILURE`, `SOURCE_NETWORK_FAILURE`, `SOURCE_RATE_LIMITED`,
`SOURCE_SCHEMA_FAILURE`, `STATE_FAILURE`, `LOCK_FAILURE`, `GIT_FAILURE`,
`PR_FAILURE`, or `SAFETY_GATE_BLOCK`. Last-known-good state is preserved on
failure. Logs and disposable run manifests have bounded retention; provenance,
state, exception evidence, and open-PR rollback metadata are protected.

## Current Limitations

- The task runs only while the current user has an interactive logon; no Windows
  credential is stored.
- Source operation currently uses the approved bounded prepared-snapshot flow.
  Live city-scale acquisition and automatic `CREATE_NEW` remain disabled.
- PR automation is compare-URL-only on this host until an authenticated GitHub
  mechanism is configured.
- No intrusive human-exception notification layer is implemented.
- The earlier Stage 4N Windows child exit code 7 was transient and could not be
  reproduced in the clean workspace. Three legacy endpoint child tests and the
  direct runtime HTTP smoke passed. The scheduled path uses direct `node.exe`
  argument arrays, file-backed output, bounded startup/runtime timeouts, and
  structured error classification.

## Stage 4P Recommendation

Run a bounded operational soak and research `CREATE_NEW` policy evidence.
Automatic creation must remain disabled until separately approved with measured
quality, rollback, provenance, and human-review gates.
