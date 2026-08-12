# Phase 4 Stage 4I: Review And Apply Preparation

## Status

`CANDIDATE ONLY - NON-RUNTIME - NOT CANONICAL - DRY RUN ONLY`

Stage 4I asks which records may enter controlled review. It does not apply a
record, authorize a field overwrite, change runtime APIs, or modify canonical
data.

## Product-Scope Polygon

UrbanAgent documentation keeps Hoi An as a future separate City Pack. The
current OSM administrative relation now covers the post-2025 merged
municipality and therefore does not match that product scope. Stage 4I uses a
versioned pre-merger boundary instead of silently accepting the wider current
legal boundary.

- Source: OpenStreetMap relation `1891418`, `Thành phố Đà Nẵng`
- Version: `72`, timestamp `2025-03-11T13:49:11Z`, changeset `163485854`
- Reproducible Overpass snapshot time: `2025-05-31T23:59:59Z`
- Administrative level: `4`; CRS: `EPSG:4326`
- License: `ODbL-1.0`; attribution: OpenStreetMap contributors
- Artifact: `data/spikes/phase4/stage4i/boundary/osm-relation-1891418-v72.geojson`
- Artifact SHA-256:
  `c0db1b84c8fcbc77b1b5cca14ba6d280627f59713db56965bd6198b72e78a84d`

The acquisition script reconstructs the polygon from historical outer ways;
coordinates were not drawn manually. Point classification is deterministic:
`INSIDE_DANANG`, `BOUNDARY_EDGE`, `OUTSIDE_DANANG`, or
`INVALID_COORDINATE`. Polygon membership is primary. No simplistic land mask
is used, so coastal POIs, bridges, waterfront venues and peninsulas are not
discarded merely for being near water.

The fixed Stage 4G bbox had `76,666` normalized records. The product polygon
contains `69,694`, has `0` boundary-edge records, excludes `6,972`, and has `0`
invalid coordinates. The unchanged canonical dataset has `3,879` records
inside and `287` outside this pre-merger product polygon. Those 287 records are
an explicit audit limitation; Stage 4I neither deletes nor changes them.

## Validation Labels And Gates

A deterministic stratified sample contains `291` cases: high-confidence,
probable, ambiguous, new and source-duplicate cases. Resolver output is stored
only as `MACHINE_SUGGESTED`. A separate structured-evidence pass may assign
`MATCH`, `NOT_MATCH`, `DUPLICATE_SAME_PLACE`, `SEPARATE_PLACE`, or
`UNCERTAIN`; insufficient evidence always becomes `UNCERTAIN`.

Evidence-supported labels use exact stable identifiers, exact phone linkage,
or a strict combination of separately observed full name, address, category
and coordinate evidence. This is an operational sample, not scientific ground
truth. The worksheet reserves independent human-review fields and does not
convert model suggestions into labels.

| Gate | Supported | Uncertain | Precision | Result |
| --- | ---: | ---: | ---: | --- |
| `HIGH_CONFIDENCE_MATCH` | 0 | 50 | N/A | `INSUFFICIENT_EVIDENCE` |
| `PROBABLE_MATCH` | 0 | 50 | N/A | `INSUFFICIENT_EVIDENCE` |
| `SOURCE_DUPLICATE` | 54 | 16 | `1.0000` | `PASS` |

The complete sample has `58` evidence-supported labels and `233` uncertain
labels: `0 MATCH`, `4 NOT_MATCH`, `54 DUPLICATE_SAME_PLACE`, and
`0 SEPARATE_PLACE`. No entity-resolution threshold changed. The evidence does
not justify broad tuning, and false negatives remain preferable to false
merges.

## Candidate Triage And Product Value

The category map was derived from current canonical categories and the actual
normalized Stage 4G source taxonomy. It classifies only product value, not
truth. Completeness never establishes identity.

| Eligibility | Count |
| --- | ---: |
| `TRAVELER_RELEVANT` | 30,939 |
| `CONTEXTUAL` | 3,789 |
| `LOW_VALUE` | 16,296 |
| `EXCLUDED` | 17,956 |

`NEW_CANDIDATE` is never auto-created. Its `68,980` polygon-eligible records
are triaged as:

| Tier | Count | Meaning |
| --- | ---: | --- |
| `NEW_TIER_A_STRONG` | 1,168 | strong traveler-review candidate, not approved |
| `NEW_TIER_B_REVIEW` | 26,817 | traveler-relevant but needs evidence |
| `NEW_TIER_C_LOW_VALUE` | 22,669 | contextual, low-value, or duplicate-first |
| `NEW_TIER_D_INVALID_OR_EXCLUDE` | 18,326 | excluded category/name/geography |

## Prioritized Review

The Stage 4H flat queue had `79,078` items. Stage 4I has `31,676` active
prioritized items and `47,967` deferred items. High-confidence matches remain
P0 while their evidence gate is unmet.

| Priority | Count | Intent |
| --- | ---: | --- |
| P0 | 434 | potential false merge, chain collision, or unmet match gate |
| P1 | 280 | high-value probable enrichment |
| P2 | 1,168 | strong new traveler candidate |
| P3 | 2,977 | source-duplicate resolution |
| P4 | 26,817 | lower-confidence traveler review |
| DEFERRED | 47,967 | outside scope, excluded, contextual, or low value |

The canary contains `75` concise review rows balanced across P0-P4, including
`5` ambiguous/chain cases, `15` duplicate cases, and `8` excluded/low-value
category edges. It carries source identity, category, coordinates, candidate
match, evidence, provenance, license and reviewer fields without embedding raw
provider payloads.

## Apply Plan And Field Conflicts

The apply-plan schema supports `ENRICH_EXISTING`, `CREATE_NEW`,
`KEEP_EXISTING`, `KEEP_SEPARATE`, `MERGE_SOURCE_DUPLICATE`, `DEFER`, and
`REJECT`. Every operation records canonical/candidate IDs, exact field changes,
old and proposed values, field provenance, license/attribution, evidence and
reviewer status.

For name, category, coordinates, address, website, phone, opening hours,
media, and external IDs, each value is classified as `SAFE_ENRICHMENT`,
`CONFLICT_REVIEW`, `NO_CHANGE`, or `REJECTED_SOURCE_VALUE`. A populated
canonical field is never overwritten merely because a source differs. Media
and external identifiers without field license metadata are rejected.

All `75` generated canary operations remain `DEFER` with
`PENDING_HUMAN_REVIEW`. The dry-run starts with `4,166`, proposes `4,166`,
performs `0` enrichments and `0` additions, and defers `75`. Its proposed-pack
hash is deterministic:
`37b35bfd942900717b4c70edb5312898a7b6cb1cbaa1f8446039bf84824b1ba2`.
There is no canonical write command.

## Incremental Integration

Stage 4I composes with Stage 4H as:

`sync -> polygon -> normalization -> entity resolution -> category/value -> review priority -> apply-plan dry run`

`UNCHANGED` skips review and resolution. A new inside record resolves once; a
non-identity change skips resolution and invalidates only its affected stage;
a new outside record is excluded before resolution. The delta test processes
only the new and changed records and performs no full rebuild.

## Validation And Safety

- Focused Phase 4C-4I tests: `62 passed`, `0 failed`; the final Stage 4I
  canary-priority follow-up suite: `15 passed`, `0 failed`.
- Canonical rows: `4,166`.
- Canonical SHA-256:
  `5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae`.
- Canonical/runtime/frontend changed: `NO`.
- Production PostgreSQL/Firebase touched: `NO`.
- Google bulk, second city, or canonical apply: `NO`.

## Limitations And Stage 4J

Stage 4I is **not ready for controlled apply**. High-confidence and probable
match gates lack sufficient independently supported labels, and the selected
versioned product boundary is intentionally different from the current legal
post-merger municipality.

Recommend one next step only: **Stage 4J - boundary and human-evidence approval
hardening**. A human must approve the City Pack boundary policy and review the
canary/worksheet until operational match gates have adequate support. No apply
should be proposed before both blockers are closed.
