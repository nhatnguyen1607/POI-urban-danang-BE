# Phase 4 Stage 4B - Bounded Multi-Source Matching Dry Run

Updated: 2026-08-10.

Status: `SPIKE_COMPLETE`.

Implementation status: `NON_RUNTIME_DRY_RUN_ONLY`.

Approval basis: the user explicitly issued `APPROVED MULTI-SOURCE POI SPIKE`.

## 1. Scope

Stage 4B adds a small deterministic dry-run pipeline to test how future
external-source records could be normalized, matched, classified, and reported
against the current Da Nang canonical POI baseline.

This stage does not implement production importers and does not ingest external
sources into runtime.

## 2. Baseline Invariants

- Canonical runtime dataset:
  `data/canonical/urbanagent_poi_master_v1.csv`.
- Canonical runtime POIs: `4166`.
- Canonical SHA-256:
  `5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae`.
- CSV remains the default runtime.
- PostgreSQL/PostGIS remains explicit opt-in.
- No production database or Firebase data was touched.
- No frontend or `/api/v2` runtime behavior changed.
- No second City Pack was created.

## 3. Sample Method

The spike uses tiny source-like fixtures stored under:

`data/spikes/phase4/stage4b/source_samples/`

The fixtures are marked `NON_CANONICAL_SPIKE_ONLY`. They are intentionally
small and deterministic. They are not a city-scale Overture, OSM, Wikidata, or
Wikimedia dump.

No Google Places call was made. Google remains request-time/live enrichment
only and is not approved for canonical offline storage.

## 4. Dry-Run Pipeline

Script:

`scripts/phase4_stage4b_matching_dry_run.js`

The script:

1. reads the immutable canonical CSV;
2. verifies canonical row count and SHA-256;
3. reads the tiny Stage 4B source fixture;
4. normalizes source, source ID, name, category, coordinates, address, website,
   phone, opening hours, external IDs, and license metadata;
5. compares each source record with canonical POIs using deterministic
   explainable signals;
6. classifies records as high-confidence match, probable match, ambiguous,
   new entity candidate, invalid coordinates, or duplicate-source candidate;
7. emits machine-readable reports.

No opaque ML model is used.

## 5. Matching Signals

The deterministic matching signals are:

- normalized name similarity;
- coordinate distance in meters;
- category compatibility;
- address/district evidence;
- chain-branch collision detection for repeated names near multiple canonical
  branches;
- source duplicate detection for same-source records with highly similar names,
  close coordinates, or matching phone numbers.

Conservative Stage 4B thresholds:

- high-confidence: name similarity >= `0.90`, distance <= `50m`, category
  compatibility >= `0.70`;
- probable: name similarity >= `0.60`, distance <= `150m`, category
  compatibility >= `0.50`;
- ambiguity: multiple credible same-name/category nearby candidates or branch
  collision;
- new candidate: no credible canonical match.

## 6. Output Reports

Generated reports:

- `data/spikes/phase4/stage4b/phase4_stage4b_match_summary.json`
- `data/spikes/phase4/stage4b/phase4_stage4b_match_candidates.csv`
- `data/spikes/phase4/stage4b/phase4_stage4b_ambiguous.csv`
- `data/spikes/phase4/stage4b/phase4_stage4b_new_candidates.csv`
- `data/spikes/phase4/stage4b/phase4_stage4b_enrichment_coverage.csv`
- `data/spikes/phase4/stage4b/phase4_stage4b_threshold_sensitivity.json`
- `data/spikes/phase4/stage4b/phase4_stage4b_duplicate_source_candidates.json`

## 7. Summary Metrics

| Metric | Count |
| --- | ---: |
| Total source sample records | 17 |
| Overture sample records | 8 |
| OSM sample records | 5 |
| Wikidata records | 3 |
| Wikimedia Commons records | 1 |
| Successfully normalized | 17 |
| High-confidence matches | 3 |
| Probable matches | 1 |
| Ambiguous matches | 3 |
| New entity candidates | 9 |
| Duplicate-source candidates | 1 |
| Invalid/missing coordinates | 1 |
| Address available | 17 |
| Opening hours available | 9 |
| External IDs available | 17 |

## 8. Threshold Sensitivity

| Threshold profile | High-confidence | Probable | Ambiguous | New | Invalid coordinates |
| --- | ---: | ---: | ---: | ---: | ---: |
| Conservative | 3 | 1 | 3 | 9 | 1 |
| Balanced | 3 | 0 | 3 | 10 | 1 |
| Loose | 3 | 0 | 3 | 10 | 1 |

Stage 4B favors avoiding false merges. The ambiguous chain-branch cases show
that name and distance alone are not enough for automatic merging.

## 9. Enrichment Coverage

Among high-confidence and probable matches:

| Field | Count |
| --- | ---: |
| Matched records | 4 |
| Address available | 4 |
| Website available | 2 |
| Phone available | 2 |
| Opening hours available | 4 |
| Wikidata ID available | 0 |
| Commons media available | 0 |
| Source/license tracked | 4 |

This suggests the most useful early enrichment fields are address, website,
phone, opening hours, and namespaced source identifiers. Stage 4B does not write
any of these fields back to canonical POIs.

## 10. License And Provenance Findings

- Overture-like records must preserve source-level license, contributor, and
  notice metadata before any future canonical import.
- OSM-like records must remain license-isolated because ODbL share-alike
  obligations may affect derived databases.
- Wikidata entity data can be treated as CC0 knowledge enrichment, but QIDs and
  statement provenance should remain namespaced.
- Wikimedia Commons media requires file-level license, author, source URL,
  attribution, and derivative-use metadata.
- Google Places was not queried and remains unsuitable for canonical offline
  storage.

## 11. Quality Findings

- Exact/high-confidence records are straightforward when name, category, and
  coordinates agree tightly.
- Repeated business names create ambiguity even when coordinates are close.
- Landmarks and beaches appear as new candidates because the current canonical
  baseline is mostly restaurant/cafe/bakery/nightlife-oriented.
- Invalid or missing coordinates must be rejected from match classification and
  reported separately.
- Source duplicate detection is necessary before canonical entity matching.

## 12. Stage 4C Recommendation

Recommended next stage: `Phase 4 Stage 4C - Source Adapter And Review Queue Prototype`.

Stage 4C should:

- keep the canonical CSV unchanged;
- implement formal source adapter interfaces for bounded dry-run reads only;
- add a reproducible real bounded fetch path for Overture, OSM, and Wikidata;
- preserve license/provenance metadata at field level;
- implement a conservative entity-resolution engine with no auto-merge for
  ambiguous candidates;
- add a human-review queue model for ambiguous and probable-new candidates;
- produce the required exact, fuzzy, ambiguity, duplicate, license,
  attribution, field-completeness, stale/removed, and cost reports;
- avoid production DB, Firebase, frontend, runtime API, second-city, and Google
  bulk calls.

Do not implement Stage 4C until separately approved.
