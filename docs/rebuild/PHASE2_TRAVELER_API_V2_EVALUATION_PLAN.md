# Phase 2 Traveler API v2 Evaluation Plan

Updated: 2026-07-26 16:35:00 +07:00.

Status: `PROPOSED FOR PHASE 2`.

This plan defines evaluation requirements for Traveler API v2. It does not
invent or claim evaluation results.

## 1. Research Questions

RQ1: Legacy/v2 behavioral compatibility.

- Hypothesis: v2 wrappers can preserve legacy traveler behavior while exposing
  clearer contracts, city metadata, and null/provenance semantics.

RQ2: Recommendation quality against simple baselines.

- Hypothesis: the current `recommendPOIs` algorithm performs better than
  category-only and rating/popularity baselines on a curated query fixture.
  Adding a v2 response wrapper alone is not treated as a recommendation-quality
  improvement.

RQ3: Robustness under missing data.

- Hypothesis: v2 responses surface missing address, rating, review count,
  freshness, opening hours, and route uncertainty without fabricating values.

RQ4: CSV/PostgreSQL parity and latency.

- Hypothesis: the same logical v2 requests return equivalent POI identities,
  counts, provenance, and null semantics in CSV-default and explicit
  PostgreSQL modes, with measured latency differences reported separately.

RQ5: Explanation fidelity and coverage.

- Hypothesis: public `reasonCodes`, reasons, and warnings reflect the actual
  public decision factors without exposing raw internal scoring weights.

## 2. Baselines

Required baselines:

- B0: current `recommendPOIs` algorithm, exposed through legacy and v2.
- B1: category-only ranking.
- B2: rating/popularity ranking.

Optional baselines may be added only if documented without changing production
behavior.

Use legacy versus v2 only for behavioral parity evaluation. Use B0 versus B1,
B2, and ablations for recommendation-quality evaluation.

## 3. Curated Query Fixture Requirement

A versioned curated query fixture is required before making recommendation
quality claims.

Fixture requirements:

- fixture version
- canonical dataset hash
- query text
- target city
- expected relevant categories or POI IDs
- optional must-not-return categories
- expected warning conditions for missing data
- annotation source and reviewer notes

No Recall@k, nDCG, MRR, or superiority claim is valid without an appropriate
ground-truth fixture.

## 4. Metrics

### Precision@k

Measures the fraction of top-k results that are relevant.

Requires:

- relevance labels for each evaluated query/result.

Inappropriate when:

- no relevance labels exist.

### Recall@k

Measures the fraction of known relevant items recovered in top-k.

Requires:

- complete or sufficiently representative relevant-set ground truth.

Inappropriate when:

- only positive examples or category labels exist without a known relevant set.

### MRR

Measures the reciprocal rank of the first relevant result.

Requires:

- at least one known relevant item per query.

Inappropriate when:

- user intent has many equally acceptable subjective answers and no ranked
  relevance labels.

### nDCG@k

Measures ranking quality with graded relevance.

Requires:

- graded relevance labels.

Inappropriate when:

- only binary or no relevance labels exist.

### Category Diversity

Measures category spread within recommendation or itinerary results.

Requires:

- verified category labels.

Inappropriate when:

- the user intent explicitly requests a narrow category.

### Geographic/District Diversity

Measures spatial or district spread.

Requires:

- verified district/admin-boundary data.

Inappropriate when:

- district data is unknown or pending spatial join.

### Catalog Coverage

Measures how much of the eligible POI catalog can appear across evaluated
queries.

Requires:

- stable eligible POI count and deterministic query set.

Inappropriate when:

- query set is too small to represent the intended product surface.

### Explanation Coverage

Measures the share of returned results with nonempty public reason and
reasonCodes.

Requires:

- reason/reasonCode presence checks.

Inappropriate when:

- evaluating raw model retrieval without public explanation layer.

### Explanation Fidelity

Measures whether public reasons match the public ranking factors and warnings.

Requires:

- an audit rule mapping result fields to allowed reasonCodes.

Inappropriate when:

- raw internal signals are the only available explanation evidence.

### Missing-Data Warning Precision

Measures whether warnings are emitted only when the corresponding data is
actually unknown or uncertain.

Requires:

- known missing-data labels from canonical quality report/serializers.

Inappropriate when:

- missing-data status is not represented explicitly.

### Itinerary Feasibility

Measures whether planned stops respect duration, transport, unknown route
semantics, and explicit constraints.

Requires:

- itinerary fixture inputs and expected structural constraints.

Inappropriate when:

- route/opening-hours data required by the assertion is unavailable.

### Route-Known Ratio

Measures `knownLegCount / (knownLegCount + unknownLegCount)`.

Requires:

- route summary fields.

Inappropriate when:

- an endpoint does not produce route legs.

### Behavioral Parity

Measures identity/count/semantic equivalence between legacy and v2 outputs, or
between CSV and explicit PostgreSQL modes.

Requires:

- fixed input requests and deterministic ordering/tie-break rules.

Inappropriate when:

- implementations intentionally differ and the decision is documented.

Legacy and v2 outputs are parity targets because both initially expose the
same current recommendation algorithm. They are not independent quality
algorithms.

### Latency Percentiles p50/p95

Measures response-time distribution.

Requires:

- repeated runs, stable environment record, and sample count.

Inappropriate when:

- single-run smoke tests are the only evidence.

### Error Rate

Measures failed requests per evaluated route.

Requires:

- repeated valid and invalid request sets.

Inappropriate when:

- failures are caused by intentionally missing optional services and not
  separated by error class.

## 5. Ablations

Evaluate recommendation behavior with:

- without rating
- without distance
- without review signal
- without preference
- without optional semantic helper

Ablation results must be reported as diagnostic evidence, not product claims,
unless the curated fixture is large enough and statistically appropriate.

## 6. Repeatability

Every evaluation run must record:

- fixed canonical hash
- fixed fixture version
- deterministic seed where randomness exists
- Node/npm versions
- operating system/environment
- repository mode, reported as internal test evidence only
- repeated run count
- timestamp
- command used

Raw result artifact format:

```json
{
  "runId": "phase2_eval_001",
  "fixtureVersion": "traveler-query-fixture-v1",
  "canonicalHash": "expected-approved-hash",
  "contractVersion": "phase2-traveler-api-v2-draft-1",
  "openApiSha256": null,
  "repositoryMode": "diagnostic-only",
  "results": []
}
```

The raw artifact must not include secrets, tokens, private credentials,
production database URLs, or sensitive payloads.

`openApiSha256` may be non-null only after Batch 1 generates an OpenAPI 3.1
artifact and the value is calculated from that artifact. It must not be
hardcoded or invented.

## 7. Statistical Reporting

When enough samples exist, report:

- paired comparisons
- confidence intervals
- effect sizes
- sample counts
- multiple-comparison caution

Do not claim statistical significance from insufficient samples.

## 8. Failure Taxonomy

Classify failures as:

- validation contract failure
- unsupported city handling failure
- null semantics failure
- provenance failure
- source identifier namespace failure
- rating contract failure
- pagination/sort instability
- recommendation relevance failure
- explanation fidelity failure
- missing-data warning failure
- route uncertainty failure
- legacy/v2 parity failure
- CSV/PostgreSQL parity failure
- auth boundary failure
- privacy/logging failure
- latency/performance regression

## 9. Qualitative Failure Analysis

For selected failures, record:

- query/request
- expected behavior
- actual behavior
- affected POI/trip IDs
- visible user impact
- likely root cause
- whether the failure is data, contract, ranking, routing, auth, or persistence
  related
- proposed follow-up

Do not include CSV contents, credentials, or sensitive user data.

## 10. Threats To Validity

Internal validity:

- Current ranking uses heuristic scoring and optional helpers; changes must be
  isolated before quality claims.
- Without deterministic tie-breaking, parity and metric runs can be unstable.
- Missing data warnings depend on serializer correctness.

External validity:

- Da Nang-only dataset limits generalization to other cities.
- Offline fixtures may not represent real traveler preferences.
- No live opening-hours, booking, or road-network routing data is available.

Construct validity:

- Category relevance may not capture traveler satisfaction.
- Rating/popularity can encode source bias.
- District diversity is invalid until district/admin data is verified.
- Explanation coverage is not the same as explanation usefulness.

User-quality limitation:

- No user-quality or production-readiness claim is valid without a user study
  or approved pilot evidence.

## 11. API Correctness Evaluation

For each v2 route:

- HTTP status code
- success/error envelope
- requestId propagation
- invalid `X-Request-Id` replacement without rejecting otherwise valid requests
- cityId requirement
- unsupported city behavior
- public metadata boundary
- no stack traces
- no secrets/tokens
- validation errors

Pass criteria:

- Contract fields match the draft.
- Unsupported explicit city never silently returns Da Nang data.
- Public responses do not expose filesystem paths, SQL/table names, internal
  repository classes, or storage-mode dependencies.

## 12. Data Semantics Regression

Required checks:

- `Global_ID` remains public canonical key.
- `RestaurantID` remains a namespaced source identifier.
- `google_maps+foody` provenance remains preserved.
- Missing values use null/status, not empty strings or zero.
- Missing route totals are null or partial, not zero.
- Unknown district is null.
- Rating source fields remain separate.
- Sample review count is not total review count.
- No `placeId` is fabricated.

## 13. City And Source Filter Evaluation

Expected current logical counts:

- Google-compatible: 3946.
- Foody-compatible: 225.
- All/canonical: 4166.

Evaluate:

- Legacy `/api/eda` keeps these counts.
- v2 POI search exposes equivalent totals with pagination.
- Batch 1 owns POI search pagination, cursor validation, deterministic
  search/list sorting, and final `Global_ID` tie-break.
- Unknown city returns `CITY_NOT_SUPPORTED`.

## 14. Recommendation Evaluation

Functional smoke:

- Query: `quan cafe yen tinh`.
- Expected: nonempty results.
- Expected: `cityId` is `da-nang`.
- Expected public fields: score, reason, reasonCodes, warnings, POI provenance.
- Raw internal signals are absent from public response.

Quality evaluation waits for the curated fixture and compares B0 against
B1/B2/ablations, not legacy versus v2 wrapper labels.

## 15. Itinerary Preview Evaluation

Functional smoke:

- Query: `quan cafe yen tinh`.
- `durationMinutes`: 180.
- Expected: nonempty stops.
- Expected missing-origin first leg:
  - `distanceKm`: null
  - `estimatedMinutes`: null
  - `distanceKnown`: false
  - `travelTimeKnown`: false
  - `calculationSource`: `missing-origin`
- Route summary:
  - `distanceFullyKnown`: false when any required leg is unknown
  - `travelTimeFullyKnown`: false when any required leg is unknown
  - `unknownLegCount` greater than zero when origin is missing
  - total distance/time null or explicitly partial

## 16. Legacy Compatibility Evaluation

Re-run legacy smokes after each implementation batch:

- `GET /api/eda?source=google_maps`
- `GET /api/eda?source=foody`
- `GET /api/pois/data-quality`
- `POST /api/agent/recommend-poi`
- legacy itinerary/preview routes applicable to the batch

Legacy endpoints must not be removed.

## 17. Repository Runtime Evaluation

Default runtime:

- CSV remains default.
- 4166 application POIs.

Explicit PostgreSQL runtime:

- Use only disposable/local approved infrastructure.
- Repository mode appears only as diagnostic evidence.
- Compare selected POI identities, provenance, rating/null fields, source
  counts, and pagination/tie-break behavior.

## 18. Performance Evaluation

Record p50 and p95 when enough repeated runs exist for:

- city status
- POI search first page
- POI search with `q`
- recommendation
- itinerary preview

Single-run smoke timing may be recorded as diagnostic only.

## 19. Security And Privacy Evaluation

Checks:

- Guest preview is stateless and non-persistent by default.
- Conditional auth routes reject missing/invalid tokens.
- Local dev fallback never works in production.
- Automated tests do not contact Firebase production.
- No secrets, tokens, filesystem paths, SQL details, internal repository class
  names, or `DATABASE_URL` in public responses/logs.

## 20. Multi-Source POI Evaluation Boundary

Phase 2 evaluation uses only the approved Da Nang canonical runtime dataset:

- `data/canonical/urbanagent_poi_master_v1.csv`
- 4166 application POIs
- CSV default runtime
- PostgreSQL opt-in only

Multi-source POI evaluation is not part of Phase 2 unless a later task is
explicitly approved with:

`APPROVED MULTI-SOURCE POI SPIKE`

Do not count external-source discovery, provider benchmarking, license review,
entity-resolution experiments, cross-provider duplicate matching, or second-city
coverage as Phase 2 Traveler API v2 validation.

Any future source spike evaluation must separately record:

- source license class and allowed uses,
- attribution obligations,
- cache/storage/deletion limits,
- field-level provenance,
- entity-resolution false merge and false split samples,
- freshness and removal behavior,
- whether restricted payloads are excluded from Git, bundles, public APIs, and
  logs.

## 21. Batch Exit Gate

Each implementation batch may close only when:

- User explicitly approved that batch.
- Code changes match approved scope.
- Default tests pass.
- New endpoint tests pass.
- Legacy compatibility checks pass.
- Null/provenance/rating/source identifier rules pass.
- CSV remains default.
- PostgreSQL remains opt-in.
- Canonical bytes remain unchanged.
- No production DB or Firebase production data was touched.
- Documentation and worklog are updated.

## 22. Final Phase 2 Exit Gate

Phase 2 can be declared complete only when:

- Core v2 endpoints are implemented and evaluated.
- Conditional endpoints are either explicitly approved and implemented or left
  documented as not implemented.
- Scientific/offline evaluation is run only with a versioned curated fixture.
- Results and limitations are documented without overclaiming.
- No Phase 3 frontend rebuild, Phase 4 City Pack automation, partner product,
  monetization, or second city work has started.

## 23. Report Template

```text
Verdict:
Batch:
Endpoints:
Contract version:
OpenAPI SHA-256:
Dataset version:
Application POI count:
Legacy compatibility:
Recommendation metrics:
Itinerary metrics:
Null/provenance/rating checks:
Pagination checks:
Repository diagnostic mode:
Latency p50/p95:
Security/privacy:
Files changed:
Risks:
Next approval needed:
```
