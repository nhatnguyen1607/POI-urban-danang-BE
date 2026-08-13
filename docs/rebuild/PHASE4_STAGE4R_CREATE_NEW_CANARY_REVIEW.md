# Phase 4 Stage 4R - CREATE_NEW Canary Review Preparation

## Status

`PASS_HUMAN_REVIEW_PREPARATION_ONLY`

Stage 4R prepares a bounded review canary. It does not approve or apply a
canonical `CREATE_NEW`. `AUTO_CREATE_NEW` is false, canonical writes and delete
operations are zero, and runtime exposure is disabled.

Stage 4Q was merged through PR #30 at merge commit
`528a9b16b826937ae9486c0f2cdca0542c5d9a0f`. The Stage 4Q implementation
commit `1a47589741da5d1e1b9cd078697d39a5d011fd1c` is an ancestor of that baseline.

## Selection Contract

The selector starts from the 974 Stage 4Q `CREATE_CANARY_ELIGIBLE` cases and
inspects the 26 strong independent-support cases first. A selected case must:

- remain clearly absent after a second-pass search against the current
  canonical IDs, aliases, historical IDs, normalized identity fields,
  coordinates, contact/domain evidence, categories, and branch tokens;
- be inside the approved Da Nang product polygon and have a usable identity,
  valid coordinates, a traveler-relevant category, and stable source ID;
- pass provenance, license, freshness, traveler relevance, branch-specific,
  and semantic category/name gates;
- have evidence beyond the resolver's prior `NEW` label;
- have no unresolved source, cross-source, canonical, historical, or selected
  canary duplicate.

Evidence priority is `STRONG_INDEPENDENT`, `MODERATE_INDEPENDENT`, then
`SINGLE_SOURCE_STRONG`. Same-upstream evidence is never called independent.
Evidence quality precedes category and geographic diversity. Selection is
bounded to 26 cases, two cases per normalized identity base, and five cases per
configured geographic cell.

## Duplicate And Semantic Safety

The review preparation compares candidates with the current canonical and with
one another. Exact normalized phone, exact official domain, and nearby
compatible names are duplicate signals. Nearby alternate names for the same
cultural landmark are also blocked. Only one representative of a duplicate
group may enter the canary.

The policy rejects known non-traveler facilities and category/name conflicts,
including lodging names misclassified as attractions. Chain candidates require
branch-specific identity evidence. Address signals outside Da Nang are rejected.
These rules remain conservative and do not imply that a recommendation is a
human approval.

## Result

- Input: 974 eligible Stage 4Q cases.
- Independently supported cases inspected first: 26.
- Removed by current canonical recheck: 0.
- Removed by selected-pool duplicate recheck: 108.
- Selected: 26.
- Categories: food/cafe 6, accommodation 5, attraction/cultural 5,
  shopping/market 4, other traveler 6.
- Support: strong independent 8, moderate independent 0,
  single-source strong 18, same-upstream only 0.
- Machine recommendations: approve 26, review 0, reject 0.
- Human decisions: `DEFER` 26.
- Approved dry-run operations: 0.

The fixed-input run was executed twice. Both runs produced run ID
`stage4r-fd19a4080107`, selection hash
`be17bca5f7427a0885d612e8081ebc518a37be18e7ab83a009d94ff0d05c9d2e`,
the same ordering, proposed IDs, evidence rows, and artifact hashes.

## Human Review Contract

- `APPROVE` authorizes only that fingerprinted candidate for a future,
  controlled canary operation. It does not authorize automation, bulk import,
  another source record, unrelated fields, canonical merge, or deletion.
- `REJECT` rejects the proposal while its decision fingerprint remains valid.
- `DEFER` requests more evidence or later review.

Every generated decision starts as `DEFER`. A fingerprint covers source
identity, name, coordinates, category, address, phone/domain evidence,
cross-source support, current canonical absence, duplicate result, provenance,
and policy version. Changed evidence invalidates reuse.

## Proposed ID And Field Contract

Future IDs reuse the repository's deterministic candidate ID policy:
`candidate:<cityId>:<stable-source-identity-hash>`. Row indexes and source input
order are excluded, so the proposed ID is stable across reruns.

The minimum future canonical core is ID, name, category, coordinates, optional
verified address, and source-lineage reference. Phone, website, opening hours,
media, external IDs, and field-level provenance remain in the Stage 4L/4M
enrichment sidecar unless a separately approved canonical schema says otherwise.
Missing values stay missing.

## Dry-Run And Runtime Safety

The plan contains proposed fields, provenance, license, decision fingerprint,
and rollback metadata. Only a valid `APPROVE` could become an approved future
operation. Because all current decisions are `DEFER`, approved operations are
zero. No plan is executed or consumed by the Stage 4M scheduler.

Canonical integrity after both runs is 4,166 unique POIs with SHA-256
`39647b29308813a7ec19e4695fd2b95ffa27743db8e8cb46800c45d3a3fe6ded`.
Canonical and runtime are unchanged.

## Scheduler Soak

The stable Stage 4O scheduler workspace was read only. Its current soak is
`IN_PROGRESS`: four scheduled runs, four successful, zero failed, zero due
source checks, zero acquisitions, zero real deltas, and zero safe PR runs.
Stage 4R did not change cadence or scheduler state.

## Validation

Focused Stage 4H-4R tests: 199 total, 190 passed, 0 failed, 9 guarded Stage 4K
tests skipped. Stage 4R-specific tests: 14 passed, 0 failed.

## Proposed Stage 4S

Propose only `CONTROLLED CREATE_NEW CANARY APPLY` after genuine human
`APPROVE` decisions exist. Stage 4S must recheck the then-current canonical,
validate decision fingerprints, back up canonical data, apply only approved
rows, update the enrichment sidecar, test duplicate/integrity/runtime
compatibility and rollback, and use a feature branch and non-automatic PR.
Stage 4S is not implemented or approved by this document.
