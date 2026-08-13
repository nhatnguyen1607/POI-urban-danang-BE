# Phase 4 Stage 4M: Automated Safe-Enrichment PR Pipeline

## Status

Stage 4M is implemented and validated on
`phase4/stage4m-automated-safe-enrichment-pr`. It converts the merged Stage 4L
policy output into a bounded feature-branch data change. It never merges a PR,
creates a POI, deletes a POI, merges entities, calls Google, or writes to a
database or Firebase.

Stage 4L dependency:

- PR: `#25`
- merge/main SHA: `112a0110f0570a342dd13e129942c6f67464c283`
- validated head: `b5d601743dc744ed884a48ab61c94cf116c559b6`

## Repository Command

The default is a non-mutating dry run. A canonical or sidecar write requires
the explicit `--apply-safe` flag.

```powershell
npm run citypack:auto-pr -- `
  --dry-run `
  --no-network `
  --branch phase4/stage4m-automated-safe-enrichment-pr `
  --artifact-dir D:\UrbanAgent-artifacts\stage4m-auto-pr\<run-id> `
  --policy-cases <fixed-evidence-pack.jsonl> `
  --decision-memory <stage4l-decision-memory.jsonl> `
  --recovered-enrichments <stage4l-candidate-enrichments.json> `
  --max-auto-cases 50 `
  --max-field-mutations 100
```

Use `--apply-safe` only on a feature branch. A future run may use
`--create-branch --run-id <stable-run-id>` to create
`auto/poi-safe-enrichment/<stable-run-id>` only when safe changes exist.
Optional `--commit --push --create-pr` uses an authenticated `gh` CLI when
available; otherwise the command returns the exact GitHub compare URL. It
never merges the PR.

## Policy And Executor Boundary

The Stage 4L engine remains the only policy authority. Stage 4M consumes only
operations whose policy outcome is `AUTO_ACCEPT_SAFE` and verifies the policy
and boundary versions before mutation. Reusable human decisions follow the
Stage 4L identity and field fingerprints; Stage 4M does not recreate them.

The executor applies only `ENRICH_EXISTING`:

- Canonical core: `address`, only when `Address_Current` is empty and the
  incoming field is a valid `SAFE_ADDITION` with complete provenance/license.
- Non-runtime sidecar: `phone`, `website`, `openingHours`, and approved media
  metadata, with deterministic identity, ordering, serialization, and hash.
- Locked: name, coordinates, category, canonical ID, and external identity
  fields.
- Forbidden: `CREATE_NEW`, `DELETE`, and canonical entity merge.

The sidecar is `stage4l-enrichment-v1`, is marked
`NON_RUNTIME_CITYPACK_ENRICHMENT`, and has `runtimeEnabled: false`. No API v2
schema or loader exposes it.

## Safety Gates

Configured first-run limits:

| Gate | Limit | Initial result |
| --- | ---: | --- |
| AUTO cases | 50 | 38 |
| Canonical address mutations | 30 | 28 |
| Sidecar field mutations | 100 | 49 |
| Canonical target share | 1% | 0.6721% |
| CREATE_NEW | 0 | 0 |
| DELETE | 0 | 0 |
| MERGE | 0 | 0 |
| Locked-field mutations | 0 | 0 |

CLI limits can only tighten configured limits. The executor stops with
`BLOCKED_BY_SAFETY_GATE` on an unexpected baseline, policy/boundary mismatch,
Stage 4L anomaly or circuit breaker, incomplete provenance/license, invalid
field, locked/destructive operation, non-empty address overwrite, conflicting
target operations, or budget breach.

## Initial Run

Fixed Stage 4J/4L evidence, decision memory, and snapshots were reused; no
source was reacquired.

| Result | Value |
| --- | ---: |
| Canonical rows before/after | 4166 / 4166 |
| Canonical SHA before | `e1f7fd635087eecb56dac8a2f3ed810f481ff129a12d19332e2d68f08ed56f96` |
| AUTO_ACCEPT_SAFE cases | 38 |
| Raw safe operations | 87 |
| Existing canonical value skips | 10 |
| Canonical address additions | 28 |
| Sidecar additions | 49 |
| Sidecar phone / website | 38 / 11 |
| Stage 4K phone/website opportunities recovered | 16 |
| New HUMAN_REVIEW exceptions | 37 |
| Canonical SHA after | `39647b29308813a7ec19e4695fd2b95ffa27743db8e8cb46800c45d3a3fe6ded` |
| Sidecar hash | `2bb10fa42c4be338eea7a034cdab1060c1760fb1159887d8008af9c02560d7bc` |

Exactly 28 target rows changed only in `Address_Current`; all 4,138 non-target
rows were byte-preserved. Names, coordinates, categories, external IDs, row
count, and canonical IDs did not change. Field-level source, snapshot,
license, attribution, policy decision, decision-memory reference, fingerprints,
and rollback values are retained in the plan/sidecar. Google is absent.

The concise exception artifact contains only the 37 newly relevant
`HUMAN_REVIEW` cases. Such cases are never applied automatically. Creating a
new canonical POI remains a manual future decision.

## Determinism And Rollback

The same fixed input after application returned `NO_SAFE_CHANGES` with zero
canonical mutations, zero sidecar mutations, zero new exceptions, and 87
existing-value skips. No new branch, PR, media work, or source acquisition was
performed.

Rollback validation restored the exact prior canonical bytes and SHA
`e1f7fd635087eecb56dac8a2f3ed810f481ff129a12d19332e2d68f08ed56f96`
and restored/removes the sidecar to its exact prior state. The final feature
branch intentionally retains the approved safe additions.

Delta simulations passed:

- Valid website on a known match: entity decision reused; safe sidecar update.
- Large coordinate move: `HUMAN_REVIEW`; no mutation.
- NEW Tier A: `HUMAN_REVIEW`; no create.
- Outside boundary: `AUTO_REJECT`; no human queue entry.
- Already-applied unchanged enrichment: no duplicate mutation.

## Validation And Artifacts

Focused Phase 4H-4M plus Phase 0 loader/recommendation/itinerary compatibility:
`120 passed, 0 failed, 0 skipped`. Stage 4M-specific tests:
`18 passed, 0 failed`. Runtime POI count remains 4,166; recommendation and trip
preview smokes pass. Runtime implementation and API contracts did not change.

Run artifacts, plans, exception deltas, exact-diff validation, test log, and
rollback evidence are under:

`D:\UrbanAgent-artifacts\stage4m-auto-pr\20260812-230133`

## Proposed Stage 4N

Propose only: **Scheduled Incremental POI Sync And Safe PR Orchestration**.
Run checkpointed source sync, produce a delta, reuse Stage 4L policy and the
Stage 4M executor, open a PR only when safe changes exist, and emit a concise
exception artifact for human review. No uncontrolled `CREATE_NEW`; no PR on a
no-change run. Stage 4N is not implemented.
