# Phase 4 Stage 4N: Scheduled Incremental POI Sync

## Status And Boundary

Stage 4N composes the existing Stage 4H incremental state/checkpoints, Stage 4I
geographic triage, Stage 4J evidence semantics, Stage 4L policy/decision memory,
and Stage 4M safe executor. It does not replace those engines. It never merges a
PR, creates or deletes a canonical POI, changes a locked identity field, exposes
the enrichment sidecar at runtime, or accesses Google, production databases, or
Firebase.

The merged Stage 4M baseline is 4,166 POIs with SHA-256
`39647b29308813a7ec19e4695fd2b95ffa27743db8e8cb46800c45d3a3fe6ded`.
The expected baseline is read through the versioned canonical contract rather
than duplicated in the orchestrator.

## Operational Flow

```text
scheduled tick -> source cadence/version check -> lock -> bounded changed-source
records -> Stage 4H fingerprint delta -> affected records only -> reusable
decision check -> Stage 4L policy -> Stage 4M dry-run plan -> open-PR gate ->
optional safe feature branch/Draft PR -> atomic state promotion -> temp cleanup
```

`NO_SOURCE_CHANGE` performs no acquisition, normalization, entity resolution,
policy evaluation, branch creation, or PR creation. A new source version with
no relevant POI fingerprint delta returns `NO_RELEVANT_POI_DELTA`. `UNCHANGED`
records reuse state. A website-only change does not rerun identity resolution;
identity/location changes do. Human review output contains only newly changed
exceptions and unchanged fingerprints are deduplicated.

## Repository Command

The default CLI mode is non-mutating. `--apply-state` advances operational
source/checkpoint state without changing canonical data. `--prepare-pr` may
delegate actual safe mutations to Stage 4M after all gates pass.

```powershell
npm run citypack:scheduled-sync -- `
  --dry-run `
  --all-sources `
  --no-network `
  --snapshot-manifest <bounded-snapshot-manifest.json> `
  --data-dir D:\UrbanAgent-data\stage4n `
  --artifact-dir D:\UrbanAgent-artifacts\stage4n-sync\<run-id> `
  --temp-dir D:\UrbanAgent-temp\stage4n\<run-id> `
  --max-auto-cases 50 `
  --max-field-mutations 100
```

Use `--source overture|osm|wikidata|wikimedia_commons` instead of
`--all-sources` for a source partition. `--resume` consumes the last compatible
checkpoint. Preparing a PR additionally requires the Stage 4L/4M policy cases,
decision memory, and recovered-enrichment inputs. Credentials remain optional
environment state and are never stored in the repository.

## Cadence And Source Capabilities

Cadence lives in `config/phase4_stage4n_scheduled_sync.json`, separate from the
engine. The scheduler may tick hourly while source checks remain bounded:

| Source | Capability | Default refresh |
| --- | --- | ---: |
| Overture | snapshot | 7 days |
| OpenStreetMap | bounded polling | 1 day |
| Wikidata | bounded polling | 3 days |
| Wikimedia Commons | bounded metadata polling | 7 days |

These are polling/snapshot capabilities, not real-time feeds. Retry is bounded
with exponential backoff for 429, transient network failures, and selected 5xx
responses. The offline path requires no credentials.

## Lock, Checkpoint, And State

One city-scoped JSON lock records run ID, host/process, policy version, start
time, and snapshot references. A live lock returns `SKIPPED_ALREADY_RUNNING`;
stale locks are rotated conservatively. Checkpoints preserve completed record
partitions and produce the same final deterministic hash after resume as an
uninterrupted run.

Incremental state, media registry, source versions, decision memory references,
exception ledger, and open-PR metadata live under the configured persistent
data directory. New state is atomically promoted only after a successful stage;
failed, blocked, or incompatible-open-PR runs do not promote partial state.

## PR And Safety Rules

Stage 4M is always invoked in preview mode first. A branch or Draft PR is
created only when the preview contains actual allowed mutations and the open-PR
gate passes. An equivalent plan reuses the existing PR; an incompatible plan
returns `OPEN_PR_REQUIRES_RELEASE`. No PR is created for no-source, no-delta,
exception-only, blocked, or no-safe-change results.

Allowed automation remains a missing canonical address on a strict known match
and approved non-runtime sidecar fields. `CREATE_NEW`, `DELETE`, name,
coordinates, category, canonical/external identity changes, baseline mismatch,
policy/boundary mismatch, provenance anomaly, and mutation-budget breaches stop
at `BLOCKED_BY_SAFETY_GATE`.

## Windows Scheduling And Retention

`scripts/phase4/runScheduledPoiSync.ps1` runs the repository command in a child
process with data, artifact, temporary output, and concise logs on `D:`. Its
default operational mode promotes only safe sync state; `-DryRun` is available.
`scripts/phase4/registerScheduledPoiSync.ps1` prints the `schtasks` command by
default and requires explicit `-Apply` to alter Windows Task Scheduler. Stage 4N
validation did not activate a machine task.

Retention selection is conservative: active state, provenance evidence,
unresolved decisions, current snapshot references, open-PR rollback material,
and recent successful runs are protected. Only old disposable logs/manifests
are eligible. Successful runs clean only their supplied Stage 4N temp directory;
failed runs retain resume diagnostics.

## Validation And Limitations

Bounded real snapshots loaded 31 Overture, 33 OSM, and 12 Wikidata POIs plus 4
linked Commons media records. A synthetic new snapshot with identical relevant
records returned 76 `UNCHANGED`, zero resolution/policy work, and no PR. The
next tick returned `NO_SOURCE_CHANGE` with all heavy-work and mutation counters
at zero. Mixed-delta, lock/concurrency, interruption/resume, decision reuse,
exception deduplication, retry/backoff, open-PR, retention, safety-gate, and
circuit-breaker tests pass.

No approved live network acquisition provider is wired into Stage 4N; prepared
bounded snapshots are required. GitHub scheduled execution is not enabled
because current full-city cache/state requires persistent storage outside an
ephemeral runner. The orchestration entry point is CI-compatible once durable
state and source acquisition are approved. Runtime APIs and the non-runtime
sidecar remain unchanged.

## Proposed Stage 4O

Propose only **production-like schedule activation and observability**: run the
validated command on a persistent non-production worker, add bounded health and
run-status monitoring, and exercise Draft-PR authentication without enabling
auto-merge, `CREATE_NEW`, second-city expansion, or runtime sidecar exposure.
