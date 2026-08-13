# Phase 4 Data Platform Closeout

## Status

- Phase 4 implementation: `CLOSED`.
- Operational soak: `IN_PROGRESS`.
- Stage 4S merge: PR #32 at
  `1b7054b85a4c0fe82d82c5ef5a3317d2695246ed`.
- Stage 4T changes do not modify canonical data or runtime API behavior.

Natural scheduler soak continues because no source has reached its next due
check. This is an operational observation, not an implementation failure.

## Official Da Nang Baseline

- Canonical POIs: 4,173 unique application POIs.
- SHA-256:
  `dcb404cc8b5c7a9b5fd70df63039ab8f828c504270e22b12a671fa4ed61583f4`.
- Previous 4,166-row SHA retained for rollback/history:
  `39647b29308813a7ec19e4695fd2b95ffa27743db8e8cb46800c45d3a3fe6ded`.
- Original 4,166 rows preserved byte-for-byte; deleted or mutated old rows: 0.

The seven Stage 4S human-approved additions are:

| Canonical ID | Name |
| --- | --- |
| `candidate:da-nang:44f30f9baa6b2340` | Khach San Golden Sea 3 |
| `candidate:da-nang:2a5e1c00c8f0bacf` | Chi House Danang |
| `candidate:da-nang:671afec933a1917a` | Kurumi |
| `candidate:da-nang:c017ebe6c5f87ec2` | Di tich danh thang Ngu Hanh Son |
| `candidate:da-nang:275459616eaeffcc` | Ocean Villas |
| `candidate:da-nang:b7de1964b4ee075e` | Banh mi & Cafe 43 - Danang |
| `candidate:da-nang:db67529b9f1809c9` | Alani Sea View Hotel |

All seven have supported categories, valid EPSG:4326 coordinates, required
core fields, and source-lineage references. The current entity-resolution
replay maps every unchanged source record to its exact canonical ID: repeated
`CREATE_NEW` proposals 0, repeated human CREATE_NEW reviews 0, unresolved
duplicates 0.

## Lifecycle And Governance

The final lifecycle is:

```text
source cadence -> source delta -> incremental fingerprints -> cached
normalization -> entity resolution -> polygon/category policy ->
provenance/license -> decision memory -> human-on-exception -> safe existing
POI enrichment -> feature branch/PR -> controlled human-approved CREATE_NEW ->
post-merge identity reuse
```

Decision memory retains seven scoped `HUMAN / CREATE_NEW_CANARY / APPLIED`
approvals, four reusable unchanged rejections, and fifteen suppressed unchanged
defers. These decisions do not authorize another candidate or automatic
creation.

The non-runtime sidecar contains 68 total records, including 19 Stage 4S
records. The 19 records have unique identities, valid canonical links,
provenance and license metadata, deterministic serialization, and no orphan.

## Source Architecture

- Overture: offline snapshot base.
- OpenStreetMap: bounded polling and enrichment under ODbL controls.
- Wikidata: bounded knowledge and identity enrichment.
- Wikimedia Commons: licensed media/provenance enrichment.
- Google: not bulk-ingested into the offline canonical dataset.

Traveler runtime requests remain provider-independent. CSV remains the default
runtime repository; PostgreSQL/PostGIS remains explicit opt-in.

## Automation Policy

- Safe enrichment of known POIs: enabled within Stage 4L/4M policy and budgets.
- `AUTO_CREATE_NEW`: false.
- Automatic delete, canonical merge, name/coordinate/category overwrite: false.
- PR auto-merge: false.
- Runtime sidecar exposure: false.
- Google bulk and second-city automation: false.

## Scheduler

`D:\UrbanAgent-work\scheduled-poi-sync` was advanced from
`c4be0c69473a0a388a3bf60962c28e71ace913b3` to merged main
`1b7054b85a4c0fe82d82c5ef5a3317d2695246ed`. Persistent state, snapshots,
decision memory, checkpoints, artifacts, and cadence were preserved.

The scheduler self-test passed Node, repository, dependency, Git, D-drive path,
state, boundary, lock, PR mode, and 4,173-row canonical checks. The Windows task
is installed, enabled, and ready. A real post-alignment scheduled invocation
returned `NO_WORK` with `LastTaskResult=0`, no mutation, branch, PR, exception,
or repeated CREATE_NEW.

Natural soak totals at closeout: 5 scheduled runs, 5 successful, 5 `NO_WORK`,
0 failed, 0 blocked, 20 source checks, 0 due checks, 0 acquisitions, 0 real
deltas, and 0 safe PR runs. Latest source health reports zero consecutive
failures for Overture, OSM, Wikidata, and Wikimedia Commons.

## Metrics

| Milestone | Result |
| --- | ---: |
| Original canonical | 4,166 |
| Post-Stage 4K canonical | 4,166 |
| Post-Stage 4M canonical | 4,166 |
| Post-Stage 4S canonical | 4,173 |
| Stage 4K safe address additions | 10 |
| Stage 4M safe address additions | 28 |
| Stage 4L active review queue | 31,676 -> 1,829 (94.23%) |
| Stage 4P raw NEW evaluated | 75,968 |
| Stage 4P strong human review | 4,217 |
| Stage 4Q canary eligible | 974 |
| Stage 4R human canary | 26 |
| Stage 4S human approvals / applied | 7 / 7 |

Focused Stage 4H-4T tests: 233 total, 224 passed, 0 failed, 9 guarded
Stage 4K skips. Phase 0/data-loader compatibility: 10 passed, 0 failed.
Runtime loader, recommendation, and trip-preview smokes pass at 4,173 POIs.

## Retained Limitations And Deferrals

Not required for Phase 4 closeout: broad `AUTO_CREATE_NEW`, another CREATE_NEW
batch, automatic deletion, a second city, Google bulk offline ingestion,
runtime sidecar exposure, production PostGIS migration, non-interactive
logged-out Windows scheduling, GitHub API PR automation, and completion of the
long-duration natural soak.

Current operational limits remain: current-user interactive task mode,
`COMPARE_URL_ONLY` PR mode, local persistent state on D:, CI scheduling
disabled, and natural soak still accumulating due-source evidence. Stage 4K-4S
human decisions, evidence, plans, hashes, rollback metadata, and source
references remain retained under `D:\UrbanAgent-artifacts`.

## Recommended Product Direction

Return to product development with one next direction: **live end-to-end
product testing and deployment readiness**. Validate the current traveler
planning, editable itinerary, saved-trip lifecycle, map synchronization,
authentication, and operational deployment as one real product flow before
adding another data-platform stage.
