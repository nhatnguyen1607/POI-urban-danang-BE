# Phase 4 Stage 4J: Evidence and Approval Gate

## Status

Stage 4J is a non-runtime, non-canonical preparation stage. It finalizes a
versioned product boundary, supplies independent evidence for a compact review
set, and enforces a separate human approval gate. It does not authorize or
perform a canonical write.

## Boundary Policy

The Da Nang City Pack evaluation boundary is:

- source: OpenStreetMap;
- identifier: `relation/1891418`;
- source version: `72`;
- CRS: `EPSG:4326`;
- normalized UTF-8/LF artifact SHA-256:
  `c0db1b84c8fcbc77b1b5cca14ba6d280627f59713db56965bd6198b72e78a84d`;
- scope: the pre-2025-merger Da Nang product boundary used by UrbanAgent;
- current legal boundary: no.

The project context keeps Hoi An as a separate future City Pack, which is the
recorded evidence that this polygon represents the intended product scope. A
retrieval bounding box only limits acquisition; final eligibility uses the
polygon. `INSIDE_DANANG` and `BOUNDARY_EDGE` records remain eligible for
review. Edge contact alone does not reject a record. Islands, peninsulas,
bridges, waterfronts, and coastal POIs are evaluated by polygon membership and
are not removed by a simplistic land mask.

Version 72 is not permanent. Any replacement must have a new version and hash,
an explicit scope review, and a geography-only reclassification before use.
The complete machine-readable policy is in
`data/spikes/phase4/stage4j/boundary_policy.json`.

## Three Separate Decisions

1. `RESOLVER_DECISION` is a candidate-generation signal and is never ground
   truth.
2. `EVIDENCE_LABEL` is based on independently observed identifiers, concrete
   historical row linkage, phone/domain/address evidence, and compatible
   location.
3. `HUMAN_APPROVAL` is the only permission gate for a future apply plan.

Strong evidence supports evaluation but does not imply approval. Every review
decision defaults to `DEFER`; resolver output never auto-fills `APPROVE`.

## Evidence Method

The evidence pack contains 90 deterministic, human-sized cases:

| Stratum | Selected | Usable independent labels | Correct | Precision | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| HIGH_CONFIDENCE_MATCH | 40 | 40 | 40 | 1.0000 | PASS |
| PROBABLE_MATCH | 30 | 28 | 28 | 1.0000 | PASS for evaluation only |
| SOURCE_DUPLICATE | 70 rechecked | 54 | 54 | 1.0000 | PASS |
| NEW_TIER_A | 20 | canary review only | n/a | n/a | Human approval required |

`UNCERTAIN` cases are excluded from the precision denominator. Two selected
PROBABLE cases remain `UNCERTAIN`. Sixteen of 70 Stage 4I duplicate samples
remain unsupported and are excluded; only the 54 independently linked cases
are reused. These are operational canary measurements, not estimates of
city-wide statistical accuracy.

Historical `poi_urban` data was read only. A concrete
`RestaurantID = historical place_id` row link was used where it existed. The
exact 4,166-row merge lineage remains unreproducible, consistent with Stage
4H. Restricted historical Google phone/address payloads are not committed;
only the linkage outcome and evidence checks are retained.

Evidence confidence distribution is 70 `STRONG`, 17 `MODERATE`, and 3 `WEAK`.
The detailed evidence is in
`data/spikes/phase4/stage4j/evidence_pack.jsonl`.

## Human Review

The compact worksheet is
`data/spikes/phase4/stage4j/human_review.csv`. The editable decision file is
`data/spikes/phase4/stage4j/review_decisions.csv`.

Allowed decisions are `APPROVE`, `REJECT`, and `DEFER`. Optional approved
operations are `ENRICH_EXISTING`, `CREATE_NEW`, `KEEP_SEPARATE`, and
`MERGE_SOURCE_DUPLICATE`. Validation rejects unknown case IDs, duplicate
decisions, malformed operations, incompatible evidence/operation pairs, and
approved records without required provenance. No reviewer identity is
invented.

Current decisions are:

- `APPROVE`: 0;
- `REJECT`: 0;
- `DEFER`: 90.

PROBABLE and NEW records can never auto-apply. Source duplicates require human
approval unless a later policy explicitly permits an exact deterministic
identity link. P0 conflicts stay blocked until reviewed.

## Field Policy

Entity identity and field acceptance are separate. Proposed fields are marked
`SAFE_ADDITION`, `SAME_VALUE`, `CONFLICT_REVIEW`, `LOW_CONFIDENCE`,
`LICENSE_RESTRICTED`, or `REJECT`. Current counts are:

- `SAFE_ADDITION`: 275;
- `SAME_VALUE`: 117;
- `CONFLICT_REVIEW`: 155;
- `LOW_CONFIDENCE`: 19;
- `LICENSE_RESTRICTED`: 0;
- `REJECT`: 244.

Names, categories, coordinates, and external identifiers do not overwrite a
different canonical value without review. OSM-derived fields retain explicit
ODbL provenance. Commons media, when present, must retain per-file attribution.

## Apply And Rollback Plan

`data/spikes/phase4/stage4j/approved_apply_plan.json` contains only explicitly
approved cases. Each operation records its case, operation type, canonical ID
where relevant, source lineage, exact old/new field values, field provenance,
license, review decision, and rollback action.

Rollback uses previous values for enrichment, the candidate/source lineage for
new records, and all retained source identities for duplicate merges. With no
human approvals, the plan contains zero operations and the dry run remains:

- before: 4,166;
- proposed after: 4,166;
- mutation count: 0;
- deferred: 90;
- canonical writes: none.

## Incremental And Deterministic Behavior

Incremental state stores `boundaryVersion`, `boundaryHash`, and geographic
eligibility. A boundary-only change reclassifies geography but does not
invalidate normalization, entity resolution, media, vision, or provenance.

The same snapshot, boundary, decisions, and processing versions produced a
cache hit in 0.095 seconds with zero POI re-resolution, media processing,
evidence regeneration, or apply-plan changes. The deterministic Stage 4J hash
is `a8604a52ff8a6d3508de515bca9d0ffcdcb5e239dbc1f4371c51b8c17b17c628`.

## Safety And Next Gate

The canonical dataset remains 4,166 POIs with SHA-256
`5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae`.
Runtime APIs, frontend, CSV-default runtime, production databases, Firebase,
and second-city scope are unchanged. No Google bulk ingestion or canonical
apply occurred.

Stage 4K should be a controlled canary apply of only a very small explicitly
approved subset. It must not begin until the user supplies explicit human
approvals, the resulting apply plan remains deterministic, provenance/license
checks pass, rollback metadata is complete, and Stage 4K itself is separately
approved. The current exact blocker is zero explicit human `APPROVE`
decisions.
