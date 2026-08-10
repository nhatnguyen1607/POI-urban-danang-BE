# Phase 4 Stage 4C - Bounded Source Adapters And Entity Resolution

Updated: 2026-08-10.

Status: `SPIKE_COMPLETE`.

Implementation status: `OFFLINE_NON_RUNTIME_INFRASTRUCTURE`.

Approval basis: the user explicitly issued `APPROVED MULTI-SOURCE POI SPIKE`.

## 1. Scope

Stage 4C turns Stage 4B matching logic into a small reusable offline City Pack
preparation layer. It does not import external POIs into runtime and does not
change traveler APIs.

Implemented:

- bounded source adapter contract;
- Overture sample adapter;
- OSM sample adapter;
- Wikidata/Wikimedia sample enrichment adapter;
- normalized source-record model;
- field-level provenance/license metadata;
- conservative deterministic entity-resolution engine;
- review queue for ambiguous, new, invalid, and duplicate-source records;
- deterministic dry-run output;
- targeted Stage 4C tests.

## 2. Baseline Invariants

- Canonical runtime dataset:
  `data/canonical/urbanagent_poi_master_v1.csv`.
- Canonical runtime POIs: `4166`.
- Canonical SHA-256:
  `5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae`.
- CSV remains the default runtime.
- PostgreSQL/PostGIS remains explicit opt-in.
- No production database or Firebase data was touched.
- No frontend code changed.
- No `/api/v2` runtime code changed.
- No second City Pack was created.
- No Google bulk ingestion was added.

## 3. Adapter Contract

Module:

`src/modules/cityPackPreparation/sourceAdapterContract.js`

Contract version:

`phase4-stage4c-v1`

Each adapter exposes:

- `source`;
- `policyClass`;
- `contractVersion`;
- `normalize(rawRecords)`.

Adapters are bounded and sample-only in Stage 4C. They normalize only the fields
needed for City Pack preparation:

- source and source ID;
- name and normalized name;
- normalized category;
- latitude and longitude;
- address;
- website;
- phone;
- opening hours;
- external IDs;
- media metadata where present;
- raw source-specific data when needed;
- provenance and license metadata.

## 4. Source Adapters

| Adapter | Module | Policy class | Stage 4C role |
| --- | --- | --- | --- |
| Overture | `adapters/overtureAdapter.js` | `OPEN_PERMISSIVE_CANDIDATE` | Preferred bounded base-source records. |
| OSM | `adapters/osmAdapter.js` | `OPEN_SHAREALIKE_ISOLATED` | Complementary local tags, hours, accessibility/address context. |
| Wikidata/Wikimedia | `adapters/wikidataWikimediaAdapter.js` | `OPEN_KNOWLEDGE_AND_MEDIA_ATTRIBUTION_REQUIRED` | Landmark/entity and media-metadata enrichment. |

Google Places is not implemented in Stage 4C and remains request-time/live
enrichment only.

## 5. Provenance Model

Module:

`src/modules/cityPackPreparation/sourceRecord.js`

Each normalized field carries enough provenance to answer:

- source;
- source ID;
- field name;
- license;
- policy class;
- attribution;
- retrieval/snapshot reference.

Stage 4C does not claim final legal compatibility. It preserves the provenance
needed for later review.

## 6. Entity Resolution

Module:

`src/modules/cityPackPreparation/entityResolution.js`

Supported classifications:

- `HIGH_CONFIDENCE_MATCH`
- `PROBABLE_MATCH`
- `AMBIGUOUS`
- `NEW_CANDIDATE`
- `SOURCE_DUPLICATE`
- `INVALID`

Signals:

- normalized name similarity;
- geographic distance;
- category compatibility;
- address/district evidence;
- external identifiers where available;
- same-source duplicate detection;
- chain-branch collision detection.

Stage 4C intentionally prefers false negatives over false merges. Ambiguous
items are never auto-merged.

## 7. Review Queue

Module:

`src/modules/cityPackPreparation/reviewQueue.js`

The review queue includes:

- ambiguous candidates;
- new candidates;
- invalid-coordinate records;
- source duplicate candidates.

Suggested actions include:

- `ACCEPT_MATCH`
- `REJECT_MATCH`
- `CREATE_NEW`
- `MERGE_SOURCE_DUPLICATE`
- `KEEP_SEPARATE`
- `DEFER`

Selecting an action is not implemented in Stage 4C. No canonical data is
modified.

## 8. Dry-Run Outputs

Script:

`scripts/phase4_stage4c_entity_resolution_dry_run.js`

Outputs:

- `data/spikes/phase4/stage4c/stage4c_normalized_records.json`
- `data/spikes/phase4/stage4c/stage4c_match_results.json`
- `data/spikes/phase4/stage4c/stage4c_review_queue.json`
- `data/spikes/phase4/stage4c/stage4c_enrichment_candidates.csv`
- `data/spikes/phase4/stage4c/stage4c_summary.json`

All outputs are `NON_RUNTIME_STAGE4C_DRY_RUN` artifacts.

## 9. Dry-Run Metrics

| Metric | Count |
| --- | ---: |
| Normalized records | 17 |
| High-confidence matches | 3 |
| Probable matches | 1 |
| Ambiguous matches | 3 |
| New candidates | 9 |
| Invalid records | 1 |
| Source duplicate candidates | 1 |
| Review queue items | 14 |
| Enrichment candidates | 4 |

## 10. Validation

Targeted test:

`node --test tests/phase4/phase4Stage4cEntityResolution.test.js`

Result:

- tests: `5`
- passed: `5`
- failed: `0`

Covered:

- adapters parse bounded fixtures;
- provenance is retained;
- invalid coordinates are classified as `INVALID`;
- high-confidence, probable, ambiguous, new, and duplicate-source paths work;
- review queue is generated with suggested actions;
- dry-run output is deterministic for fixed fixtures;
- canonical row count remains `4166`;
- canonical SHA-256 remains unchanged.

## 11. Limitations

- Stage 4C still uses Stage 4B tiny fixtures, not real city-scale external
  downloads.
- Overture, OSM, and Wikidata/Wikimedia adapters are sample parsers, not
  production importers.
- Google Places is intentionally absent.
- Review action persistence is not implemented.
- No human moderation UI exists.
- No source output is written into canonical POIs.
- No second City Pack is created.

## 12. Proposed Stage 4D

Recommended next stage:

`Phase 4 Stage 4D - Reproducible Bounded City Pack Build Pipeline`

Stage 4D should:

- keep canonical runtime data unchanged;
- add reproducible bounded fetch commands for Overture, OSM, and Wikidata;
- store raw snapshots outside runtime with hashes and license metadata;
- run Stage 4C adapters against real bounded snapshots;
- produce review-approved candidate application plans;
- define a no-runtime-write build manifest;
- keep ambiguous and new candidates in review only;
- avoid Google bulk ingestion;
- avoid production DB, Firebase, frontend, second city, and runtime API changes.

Do not implement Stage 4D until separately approved.
