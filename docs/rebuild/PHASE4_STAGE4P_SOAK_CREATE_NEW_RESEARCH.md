# Phase 4 Stage 4P - Soak And CREATE_NEW Research

Status: `PASS_WITH_SOAK_IN_PROGRESS`. This stage is research-only. It does not
authorize or implement automatic `CREATE_NEW`, modify canonical data, expose
research candidates at runtime, or merge a PR.

## Operational Soak

The evaluator read actual Stage 4O operational manifests and did not manufacture
runs, source timestamps, or deltas.

| Metric | Observed |
| --- | ---: |
| Scheduled invocations | 3 |
| Successful scheduled invocations | 3 |
| `NO_WORK` | 3 |
| Source checks | 12 |
| Due-source checks | 0 |
| Real acquisitions | 0 |
| Actual source deltas | 0 |
| Safe-PR runs | 0 |
| `EXCEPTIONS_ONLY` | 0 |
| Blocked / failed | 0 / 0 |
| Scheduled retries / resumes | 0 / 0 |
| Duplicate-PR suppressions | 0 |
| Average no-work duration | 1,196.67 ms |
| Stage 4O run artifacts / logs | 38,425 / 32,164 bytes |

Lock release, state integrity, failure, and duplicate-PR criteria pass. The soak
gate remains `IN_PROGRESS` because no source has naturally become due. Stage 4P
does not force source cadence or fail solely because wall-clock evidence is
still accumulating.

## Precision-First CREATE_NEW Policy

The fixed non-runtime Stage 4G candidate pack was streamed from `D:`; no source
was reacquired. All 75,968 proposed NEW candidates were evaluated against the
approved Da Nang polygon. The second existence pass uses canonical external and
alias identifiers plus spatially blocked name/category checks against all 4,166
canonical POIs. It does not use naive all-pairs matching.

CREATE_NEW has separate research states:

- `CREATE_CANARY_ELIGIBLE`
- `NEW_STRONG_CANDIDATE`
- `NEW_REVIEW`
- `NEW_LOW_CONFIDENCE`
- `NEW_REJECT`
- `NEW_POSSIBLE_EXISTING_MATCH`

Every result has an explainable scorecard covering identity absence, real-place
evidence, traveler value, source quality, freshness, cross-source support,
provenance completeness, and duplicate risk. The resolver output is never
treated as ground truth.

## Full Candidate Distribution

| Outcome | Count |
| --- | ---: |
| Input NEW candidates | 75,968 |
| Polygon eligible | 68,998 |
| Possible existing canonical | 198 |
| Non-representative source duplicates | 2,190 |
| Low-value / quality / invalid | 51,172 |
| Normal review | 18,191 |
| Strong human review | 4,217 |
| Canary eligible | 0 |
| Provenance rejected | 0 |

The strong queue is a 94.45% reduction from the raw NEW input, but 4,217 cases
remain too many for immediate human review. The stricter canary pool is empty:
no candidate simultaneously satisfied the current independent cross-source
support and permissive-storage provenance gates. OSM support remains isolated
under ODbL review and is not silently treated as unrestricted canonical
storage. This is evidence insufficiency, not permission to weaken the gates.

## Canary And Decision Memory

A deterministic 75-row validation worksheet samples strongest review cases and
difficult negatives across possible-existing, duplicate, low-confidence, and
invalid outcomes. Every `reviewer_decision` defaults to `DEFER`; no human
approval is fabricated.

`D:\UrbanAgent-artifacts\stage4p-create-new\stage4p-62732265c5aa\create_new_canary_review.csv`

CREATE_NEW decision-memory fingerprints cover source identity, name,
coordinates, category, address, external IDs, provenance, and source support.
Unchanged `APPROVE`/`REJECT` decisions may be reused only inside the research
scope; unchanged `DEFER` is suppressed. Any fingerprint change invalidates the
decision. No decision record can authorize a canonical write.

The incremental design is:

```text
NEW record -> polygon -> normalization -> existing-entity second pass ->
cross-source duplicate/support -> traveler relevance -> quality/provenance ->
CREATE_NEW research policy -> decision memory -> worthwhile HUMAN_REVIEW only
```

## Safety And Future Gate

- `AUTO_CREATE_NEW`: `FALSE`
- Canonical writes / deletes: `0 / 0`
- Canonical: 4,166 POIs, SHA-256
  `39647b29308813a7ec19e4695fd2b95ffa27743db8e8cb46800c45d3a3fe6ded`
- Runtime, frontend, production database, Firebase, Google bulk, second city:
  unchanged / absent
- Scheduler: enabled on merged main; cadence unchanged

Any future CREATE_NEW canary requires independently validated precision, low
false-existing and duplicate rates, traveler relevance, provenance/license
PASS, rollback and safety budgets, a small human-approved batch, and separate
explicit user approval.

## Stage 4Q Recommendation

Recommend one direction only: **CREATE_NEW independent-evidence and
cross-source-consensus hardening**. Continue operational soak naturally and
reduce the 4,217 strong queue to a non-empty, defensible human canary pool before
requesting controlled apply approval. Do not implement a CREATE_NEW executor.
