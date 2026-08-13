# Phase 4 Stage 4Q - CREATE_NEW Evidence Hardening

## Status

`PASS_RESEARCH_ONLY`. Stage 4Q evaluates evidence only. It does not authorize
or perform canonical writes, runtime exposure, deletion, or automatic
`CREATE_NEW`.

Policy: `stage4q-independent-evidence-v1`

Evidence model: `stage4q-source-graph-v2`
Input: the 4,217 Stage 4P strong-review candidates from the cached Stage 4G
candidate pack.

## Evidence Strategy

The second-pass existence search uses indexed external identifiers and
spatial/name blocking across:

- the 4,166 canonical POIs;
- canonical aliases and source identifiers;
- available read-only historical `poi_urban` Google Maps and Foody files;
- normalized Vietnamese names and addresses;
- coordinates, category, phone, official domain, and branch evidence where
  present.

Historical lineage is explicitly partial: the available files contain useful
legacy identifiers, names, districts, categories, and coordinates, but do not
prove complete source history. Proximity to canonical or historical records
raises existing/duplicate risk; it is never positive evidence for creation.

## Source Graph And Consensus

The non-runtime in-memory graph emits explainable edges:
`EXACT_ID`, `SAME_PHONE`, `SAME_WEBSITE`, `SAME_ADDRESS`,
`SPATIAL_NAME_SUPPORT`, `HISTORICAL_LINK`, `WIKIDATA_LINK`,
`SOURCE_DUPLICATE`, and `POSSIBLE_MATCH` where evidence exists.

Support retains an explicit independence label:
`INDEPENDENT_SUPPORT`, `SHARED_UPSTREAM_SUPPORT`, or
`UNKNOWN_INDEPENDENCE`. Different provider names alone are not treated as
independence. Wikimedia media alone is not real-place or active-business proof.

Consensus remains categorical rather than one opaque score:
`EXACT_CROSS_SOURCE_ID`, `STRONG_INDEPENDENT_CONSENSUS`,
`MODERATE_CONSENSUS`, `SAME_UPSTREAM_ONLY`, `SINGLE_SOURCE_ONLY`, or
`CONFLICTING_SOURCES`.

## Canary Gate

`CREATE_CANARY_ELIGIBLE` requires all mandatory geographic, traveler-value,
validity, provenance/license, clear-absence, duplicate, freshness, and record
quality gates. It also requires evidence independent of the Stage 4P resolver
classification through one of:

1. strong independent cross-source identity support;
2. conservatively normalized phone plus official domain and supporting real
   place metadata;
3. compatible landmark/Wikidata entity linkage.

A single-source record containing only name, coordinates, and category remains
human review. ODbL/share-alike storage remains review-gated. Stale or closed
evidence cannot pass the strongest gate. `AUTO_CREATE_NEW` remains `false`.

## Full-Pool Result

| Measure | Count |
| --- | ---: |
| Strong pool evaluated | 4,217 |
| Clearly absent | 2,455 |
| Possible/likely existing | 0 |
| Uncertain existence | 1,762 |
| Cross-source supported | 283 |
| Strong independent support | 26 |
| Same-upstream only | 0 |
| Single-source only | 4,088 |
| Freshness good / unknown / stale / closed | 4,217 / 0 / 0 / 0 |
| Unresolved duplicate risk | 266 |
| `CREATE_CANARY_ELIGIBLE` | 974 |
| `STRONG_HUMAN_REVIEW` | 1,288 |
| `NORMAL_REVIEW` | 1,689 |
| `POSSIBLE_EXISTING` | 0 |
| `DUPLICATE_REVIEW` | 266 |
| `DEFER` | 0 |
| `REJECT` | 0 |

The 974 canary-eligible cases are a research pool, not approvals. Of these,
964 use the official-identity evidence path and 10 use strong independent
cross-source support. The result does not claim current opening status or
city-wide scientific truth.

## Human Review And Artifacts

The bounded worksheet contains 100 rows: 30 canary-eligible cases across 24
category groups, 35 strong-review cases, 17 normal-review cases, and 18
duplicate-review cases. Every reviewer decision is `DEFER`. This reduces the
immediate worksheet by 4,117 rows, or 97.63%, relative to the 4,217-case pool.

- Review CSV: `D:\UrbanAgent-artifacts\stage4q-create-new\stage4q-935d4629660a\create_new_evidence_review.csv`
- Evidence JSONL: `D:\UrbanAgent-artifacts\stage4q-create-new\stage4q-935d4629660a\create_new_evidence_details.jsonl`

The JSONL contains only compact backing evidence for review rows: graph edges,
second-pass candidates, independent-evidence path, provenance references, and
reason codes. No raw source dump is copied into Git.

## Incremental And Operational Result

The decision fingerprint includes source identity, coordinates, normalized
name/category/address, phone/domain evidence, source support, second-pass
existence, provenance, policy version, and evidence version. A repeated run
with the same input and versions reused Stage 4Q state and completed without
reclassifying the full pool. Changed records or a policy/evidence version
change invalidate only the relevant evidence result; raw acquisition and
unrelated caches remain valid.

The scheduler remains enabled and unchanged. Natural soak at evaluation time:
4 scheduled runs, 4 successful, 0 failed, 0 due-source checks, 0 acquisitions,
and 0 deltas. Status remains `IN_PROGRESS`; no cadence or synthetic evidence
was introduced.

## Safety And Validation

- Canonical rows: 4,166.
- Canonical SHA-256:
  `39647b29308813a7ec19e4695fd2b95ffa27743db8e8cb46800c45d3a3fe6ded`.
- Canonical modified: no.
- Runtime modified: no.
- Canonical create/delete: 0/0.
- Production DB/Firebase: untouched.
- Google bulk ingestion: absent.
- Second city: absent.
- Focused Stage 4H-4Q tests: 176 passed, 0 failed, 9 conditional skips.

## Proposed Stage 4R

Propose only: **Controlled Human-Approved CREATE_NEW Canary**. Human reviewers
should inspect at most the 30 eligible rows, explicitly approve exact cases and
fields, and retain rollback/provenance evidence. No creation may occur without
separate Stage 4R approval; automatic creation remains prohibited.
