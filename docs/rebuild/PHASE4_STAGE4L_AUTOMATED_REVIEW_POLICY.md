# Phase 4 Stage 4L - Automated Review Policy

## Status

- Verdict: `PASSED`
- Scope: deterministic review policy, decision memory, human-on-exception,
  and a non-runtime enrichment sidecar contract.
- Runtime integration: not authorized and not implemented.
- Canonical writes in Stage 4L: none.
- Source reacquisition: none. The simulation reuses fixed Stage 4I aggregates,
  Stage 4J evidence, authoritative human decisions, and Stage 4K artifacts.

## Baseline

- Stage 4K PR: `#24`
- Stage 4K merge commit: `75c319425bb6603f717db312fc86157e733c9653`
- Canonical application POIs: `4166`
- Canonical SHA-256 after the approved Stage 4K canary:
  `e1f7fd635087eecb56dac8a2f3ed810f481ff129a12d19332e2d68f08ed56f96`
- Canonical remains the CSV-default runtime.

Stage 4L updates the City Pack inspection constant to the approved post-Stage
4K baseline. It does not change the canonical CSV bytes.

## Inputs And Storage

Authoritative inputs:

- Human decisions:
  `D:\UrbanAgent-spikes\urbanagent-phase4-stage4j-human-review-20260812-112300`
- Stage 4K apply evidence:
  `D:\UrbanAgent-artifacts\stage4k-canary\20260812-134530`
- Committed Stage 4I/J fixed state and evidence under
  `data/spikes/phase4/`.

Stage 4L generated only non-repository artifacts under:

`D:\UrbanAgent-artifacts\stage4l-policy\20260812-150823`

No UrbanAgent output was written under `C:` and no provider was queried.

## Deterministic Outcomes

The versioned policy supports exactly these outcomes:

1. `SKIP_UNCHANGED`
2. `REUSE_APPROVAL`
3. `REUSE_REJECTION`
4. `AUTO_ACCEPT_SAFE`
5. `AUTO_REJECT`
6. `HUMAN_REVIEW`
7. `DEFER`

Every case is sorted by stable case ID before evaluation. Policy output,
apply-plan output, and decision memory use stable canonical hashing.

## Decision Memory

The `stage4l-decision-memory-v1` record stores:

- case, source, source ID, and canonical ID;
- decision source: `HUMAN`, `POLICY`, or `SYSTEM_SAFETY`;
- decision scope and decision reference;
- identity fingerprint and per-field fingerprints;
- evidence fingerprint;
- policy, resolver, evidence, and boundary versions;
- approved fields, provenance snapshot reference, and reviewer note.

Identity fingerprints deliberately exclude volatile retrieval timestamps.
Field fingerprints permit targeted invalidation. Boundary changes invalidate
geographic eligibility only; policy changes invalidate policy reuse; evidence
changes reopen deferred cases; identity changes always reopen identity review.

Authoritative Stage 4J import result:

| Human decision | Imported | Reuse behavior |
| --- | ---: | --- |
| APPROVE | 10 | `REUSE_APPROVAL` |
| REJECT | 5 | `REUSE_REJECTION` |
| DEFER | 10 | Suppressed until evidence/fingerprint/policy revisit |

Decision reuse rate for the 25 authoritative cases is `100%`.
The persisted simulation memory contains `53` concrete records: `25` HUMAN
records and `28` POLICY approvals. No exact evidence case produced a new
SYSTEM_SAFETY record; that decision source remains implemented and tested.

## Policy Rules

The policy is based on the approved Stage 4 polygon policy, Stage 4H resolver
and incremental state, Stage 4I review tiers, Stage 4J evidence, and Stage 4F
provenance/license rules.

`AUTO_ACCEPT_SAFE` is limited to an existing `HIGH_CONFIDENCE_MATCH` with:

- independent `STRONG` evidence;
- resolver confidence at least `0.98`;
- complete provenance and no restricted-license field;
- inside-scope geography;
- no excessive coordinate displacement;
- at least one `SAFE_ADDITION` in `address`, `phone`, `website`, or
  `openingHours`.

The generated plan may enrich only those allowed fields. It cannot change
canonical identity, name, coordinates, category, external-ID mapping, source
identity, or row identity. It cannot create or delete a POI.

`AUTO_REJECT` applies to invalid/outside/excluded records, hard provenance or
license failures, source duplicates, and invalid/excluded new-candidate tiers.
This means exclusion from the candidate workflow, never deletion from a source
or canonical dataset.

`PROBABLE_MATCH`, `AMBIGUOUS`, and Tier-A new entities require human review.
Tier-B/C work is deferred according to value and evidence. Automatic
`CREATE_NEW` is explicitly forbidden.

## Safety Budgets And Circuit Breaker

Policy configuration is versioned in `config/phase4_stage4l_policy.json`.
Initial limits are:

- at most `100` automatically accepted cases per run;
- at most `300` automatically accepted fields per run;
- auto-reject rate at most `0.95` for the evaluated batch;
- identity invalidation budget `50` for later orchestration.

Any exceeded apply budget trips the circuit breaker and suppresses the entire
apply-plan. The Stage 4L simulation accepted `28` cases and `61` safe fields;
the breaker did not trip. The plan remains non-runtime and unexecuted.

Versioned anomaly gates also stop plan generation for changed-record or
new-candidate spikes, boundary-inside distribution shifts, provenance-failure
spikes, regressed snapshot versions, unexpected canonical SHA, or any locked
field mutation attempt. The fixed-state run triggered `0` anomaly gates.

## Enrichment Layer

The `stage4l-enrichment-v1` sidecar schema contains:

- canonical ID and review case ID;
- field and candidate value;
- human decision;
- source/source ID and snapshot reference;
- full field provenance and license;
- rollback action;
- explicit `CANDIDATE_NON_RUNTIME_NOT_CANONICAL` status.

This layer supplements the stable core CSV identity contract. It is not loaded
by runtime repositories or APIs.

Stage 4L recovered all `16` Stage 4K phone/website values that were skipped
because the canonical CSV has no supported columns. Their human approvals and
provenance are preserved in `candidate_enrichments.json`. No value was written
to the canonical CSV or runtime.

## Fixed Full-City Policy Simulation

The simulation uses the fixed Stage 4I active review queue. It does not rerun
source acquisition or claim city-wide ground-truth accuracy.

- Fixed bounded-box records represented: `76,666`
- Inside-boundary records represented: `69,694`

| Active-queue outcome | Count |
| --- | ---: |
| `REUSE_APPROVAL` | 10 |
| `REUSE_REJECTION` | 5 |
| `AUTO_ACCEPT_SAFE` | 28 |
| `AUTO_REJECT` | 2,977 |
| `HUMAN_REVIEW` | 1,829 |
| `DEFER` | 26,827 |
| Total | 31,676 |

The `AUTO_REJECT` count above is the Stage 4I active source-duplicate queue.
Outside-scope records (`6,972`) and Tier-D records (`18,326`) were already
prefiltered before that active queue. Tier-C records (`22,669`) remain
prefiltered/deferred.

- Active human queue after policy: `1,829`
- Queue reduction: `29,847` (`94.23%`)
- Human review rate: `5.77%`
- Exact evidence auto-accept: `28` cases / `61` safe field operations

The reduction is an operational queue simulation, not a precision or recall
claim about the Da Nang POI population.

## No-Change And Delta Replay

An unchanged rerun over all 90 evidence cases produced:

- `SKIP_UNCHANGED`: `90`
- changed records: `0`
- expensive resolution runs: `0`
- review cases: `0`
- evidence requests: `0`
- media requests: `0`
- new policy decisions: `0`
- apply delta: `0`

Delta scenarios:

| Scenario | Invalidation | Result |
| --- | --- | --- |
| A. 100 unchanged known records | none | `SKIP_UNCHANGED` 100; no expensive work |
| B. Approved match gains website | website only | reuse entity approval; validate website only |
| C. Approved POI moves substantially | identity | `HUMAN_REVIEW` |
| D. Known rejection is unchanged | none | `REUSE_REJECTION` |
| E. One NEW Tier-A traveler POI | new identity | `HUMAN_REVIEW`; no create |
| F. New out-of-bound record | geography safety | `AUTO_REJECT`; no review/delete |

## CLI And Future Automation

The existing command now supports guarded policy planning:

```text
npm run citypack:sync -- sync --policy auto \
  --policy-cases <json> \
  --policy-memory <jsonl> \
  --policy-config config/phase4_stage4l_policy.json \
  --summary <json>
```

This mode emits a non-runtime apply-plan only. It is suitable for a later cron
or CI workflow, but Stage 4L creates no scheduler and performs no apply.

## Validation

- Focused Stage 4H-4L tests: `92/92` pass, `0` fail, `0` skip.
- Stage 4K integration with authoritative 25-case inputs: `15/15` pass.
- Stage 4L policy tests cover outcomes, fingerprints, targeted invalidation,
  human-decision reuse, safe-field filtering, new-candidate handling,
  provenance/geography safety, budgets, circuit breaker, rollback metadata,
  deterministic ordering, sidecar recovery, and guarded CLI mode.
- Independent run-1/run-2 Stage 4L hash:
  `a56bf18a18ac2def80d38b16f5849058807dfe624687200cd179410a6f8ad0f5`

## Safety Result

- Canonical count: `4166`
- Canonical SHA-256:
  `e1f7fd635087eecb56dac8a2f3ed810f481ff129a12d19332e2d68f08ed56f96`
- Canonical changed by Stage 4L: `NO`
- Runtime code changed: `NO`
- Runtime data further changed: `NO`
- POIs created/deleted: `0/0`
- Production database or Firebase touched: `NO`
- Google data included: `NO`
- Second city created: `NO`

## Proposed Stage 4M

Propose one next direction only: **Automated safe-enrichment PR pipeline**.
It should materialize policy-approved sidecar deltas as reviewable pull-request
artifacts with budget checks, rollback metadata, deterministic diffs, and a
mandatory human merge gate. It must not install sidecar data into runtime or
alter the canonical CSV without separate approval.

Stage 4M is not implemented by this stage.
