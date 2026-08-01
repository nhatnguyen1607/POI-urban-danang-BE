# UrbanAgent Phase 2 Batch 3 - Trip Preview Evaluation Plan

Status: IMPLEMENTED ON REVIEW BRANCH - NOT MERGED

Updated: 2026-08-01

This document defines the proposed validation plan for the later implementation of `POST /api/v2/trips/preview`.

## 1. Baseline Gates To Preserve

Batch 3 implementation must preserve the verified Phase 2 Batch 2 baseline:

- Batch 1 and Batch 2 ancestry remain present on `main`.
- Canonical runtime POIs: `4166`.
- Canonical SHA-256: `5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae`.
- CSV remains the default runtime.
- PostgreSQL remains explicit opt-in.
- Existing default tests continue to pass.
- Existing skipped disposable-DB integration behavior remains intentional when no database is configured.
- Production audit remains acceptable under the current project gate.
- No external POI source is added.
- No Firebase production access occurs.

## 2. Required Fixture Categories

Future Batch 3 tests should add focused fixtures under the existing Phase 2 fixture style.

Required fixture categories:

- `quiet-cafe-no-origin`: no `startLocation`; first leg must remain unknown.
- `quiet-cafe-with-origin`: known origin; first leg has numeric approximate travel values.
- `short-duration-trim`: duration too short for all candidates; result is trimmed deterministically.
- `unsupported-city`: non-Da Nang city returns `CITY_NOT_SUPPORTED`.
- `invalid-coordinates`: invalid latitude or longitude returns `VALIDATION_ERROR`.
- `exclusions-dedup`: excluded POIs do not appear and duplicate candidates collapse to one stop.
- `must-include-known-poi`: valid must-include POI is preserved if feasible.
- `must-include-missing-poi`: invalid must-include ID produces a warning or controlled error.
- `opening-hours-unknown`: missing approved opening-hours data produces `OPENING_HOURS_UNKNOWN`.
- `no-batch4-route`: persistence, edit, replan, and mutation routes remain absent.

Fixtures must not contain secrets, provider payloads, external scraped data, or generated canonical data.

## 3. Endpoint Smoke Gates

The later implementation should include endpoint-level smoke coverage for:

- `POST /api/v2/trips/preview` with CSV default runtime.
- `POST /api/v2/trips/preview` with explicit PostgreSQL runtime when disposable DB integration is enabled.
- `POST /api/v2/recommendations` remains unchanged and nonempty.

Trip preview smoke must verify:

- response status is successful for a valid Da Nang request,
- response envelope matches v2 conventions,
- stop count is greater than zero,
- stop IDs are unique,
- stop order is deterministic,
- reason codes are present and deterministic,
- warnings are public-safe,
- response has no persistence marker other than `persisted: false`.

## 4. Determinism Gates

For identical request bodies and runtime data:

- selected stop IDs must be identical,
- stop order must be identical,
- tie-breaking must be stable,
- warning codes must be identical,
- request IDs may differ unless caller supplies `X-Request-Id`,
- timestamps must not be injected into ranking decisions.

Tie-breaking should be validated with a fixture that creates or selects candidates with equal or near-equal recommendation scores.

## 5. Missing-Origin Gates

For a valid request without `startLocation`, the first stop must satisfy:

- `travelFromPrevious.distanceMeters` is `null`,
- `travelFromPrevious.travelDurationMinutes` is `null`,
- `travelFromPrevious.estimationMethod` is `null`,
- `travelFromPrevious.distanceKnown` is `false`,
- `travelFromPrevious.calculationSource` may be `missing-origin`.

The response must include a warning equivalent to:

- `ORIGIN_NOT_PROVIDED`
- no Da Nang center fallback was used.

Known later legs between selected POIs with valid coordinates must continue using numeric approximate distance and time values.

## 6. Known-Origin And Geometry Gates

For a request with valid origin coordinates:

- the first leg has numeric `distanceMeters`,
- the first leg has numeric `travelDurationMinutes`,
- the first leg has `estimationMethod: "local-haversine-estimate"`,
- the first leg has `distanceKnown: true`,
- `source` is `local-haversine-estimate` or an equivalent public-safe label.

The test must verify longitude and latitude are not reversed by comparing the first-leg distance against a plausible bounded result for the fixture.

No implementation may call an external routing provider in Batch 3.

## 7. Duration And Schedule Gates

Duration validation must verify:

- values below minimum are rejected,
- values above maximum are rejected,
- non-integer values are rejected,
- valid duration produces a preview,
- short duration trims stops or returns `NO_FEASIBLE_ITINERARY` deterministically.

Schedule validation must verify:

- valid `HH:mm` `startTime` is accepted,
- malformed time is rejected,
- absent `startTime` results in unknown absolute arrival/departure values or documented relative timing,
- known schedule never silently exceeds the requested duration without warning.

## 8. Opening-Hours Gates

Because the approved canonical runtime does not guarantee verified opening-hours coverage, Batch 3 must test the unknown behavior:

- missing hours do not create fabricated hours,
- missing hours produce `OPENING_HOURS_UNKNOWN`,
- missing hours do not automatically exclude a POI,
- no external live opening-hours source is queried.

If a POI already has approved runtime opening-hours data in the future, tests must separately cover open, closed, and unknown status. That extension is not required for the first Batch 3 implementation unless such data already exists.

## 9. Recommendation Integration Gates

The preview must prove integration with Batch 2 by testing:

- recommendation candidates are nonempty for the fixture query,
- preview stops are selected from eligible recommended candidates or valid must-include IDs,
- reason codes include recommendation-derived reasons where available,
- ranking remains deterministic,
- public-safe provenance remains present,
- recommendation endpoint behavior is not regressed.

## 10. Error Contract Gates

The later implementation must test:

- `VALIDATION_ERROR` for malformed body and invalid fields,
- `CITY_NOT_SUPPORTED` for unsupported city,
- `NO_FEASIBLE_ITINERARY` for a valid but impossible request,
- no stack traces in responses,
- no credential, file path, database URL, or raw provider payload in responses.

## 11. Runtime Selection Gates

CSV default gate:

- with no PostgreSQL runtime environment variable, the application uses the CSV repository.
- the preview returns from the canonical 4166 POI baseline.

PostgreSQL opt-in gate:

- with explicit PostgreSQL repository selection and a disposable DB, the preview endpoint can run against PostgreSQL.
- the implementation must not require PostgreSQL for normal tests.
- the implementation must not enable PostgreSQL by default.

## 12. Scientific-Evaluation Limitations

Batch 3 endpoint tests prove software behavior, not travel quality.

Known limitations:

- Haversine travel estimates are not road-network routes.
- Estimated minutes do not model traffic, weather, hills, bridge access, closures, parking, walking paths, or ferry constraints.
- Opening-hours handling is limited to approved runtime data.
- Missing ratings, reviews, addresses, and hours remain unknown.
- Recommendations are based on current canonical POI data and existing ranking signals.
- Fixture tests cannot prove general user satisfaction.
- Da Nang is the only supported city.
- No external source coverage improvement occurs in Batch 3.

The implementation must avoid SOTA or production-readiness claims based solely on these tests.

## 13. Proposed Validation Commands

The later implementation batch should run the repository's actual commands, including:

- dependency integrity check,
- production security audit,
- JavaScript syntax checks,
- OpenAPI parse and contract inspection after the OpenAPI artifact is explicitly updated,
- canonical SHA-256 verification,
- default test suite,
- disposable PostgreSQL integration if endpoint-switch coverage is included.

Exact commands must be recorded in `docs/rebuild/TEST_REPORT.md` and appended to `docs/rebuild/WORKLOG.md` during the implementation batch.

## 14. Acceptance Metrics And Thresholds

The Batch 3 implementation evaluation must include the following deterministic metrics.

| Metric | Numerator | Denominator | Fixture cases | Required threshold | Limitation |
| --- | --- | --- | --- | ---: | --- |
| Deterministic replay rate | identical repeated responses excluding generated request IDs | repeated valid fixture runs | all valid fixtures | 100% | Does not prove recommendation quality. |
| Exclusion violation rate | excluded POI IDs scheduled | excluded POI IDs requested | `exclusions-dedup` | 0% | Only covers explicit exclusions. |
| Duplicate-stop rate | duplicate scheduled stop IDs | scheduled stops | all success fixtures | 0% | Does not prove cross-source entity resolution. |
| Hard-constraint satisfaction rate | satisfied satisfiable hard constraints | satisfiable hard constraints | hard-constraint fixtures | 100% | Intentionally impossible cases are excluded and reported separately. |
| Must-include scheduling rate | scheduled must-include POIs | satisfiable must-include POIs | `must-include-known-poi` | 100% | Unsatisfiable must-includes are separately accounted. |
| Daily-window overflow rate | scheduled days exceeding daily window | scheduled days with daily windows | schedule fixtures | 0% | Uses local haversine estimates, not road routing. |
| Known opening-hours conflict rate | scheduled stops with known hard opening-hours conflicts | stops with approved known opening-hours constraints | known-hours fixtures | 0% | Current canonical runtime may not contain approved known hours. |
| Unscheduled explanation coverage | unscheduled items with public-safe explanations | unscheduled items | partial and infeasible fixtures | 100% | Measures presence, not human usefulness. |
| Warning-code correctness | expected warning-code assertions satisfied | deterministic warning-code assertions | warning fixtures | 100% | Fixture-bound and not global behavior proof. |
| Geographic compactness proxy | previews meeting fixture distance bound | previews evaluated for compactness | known-origin fixtures | fixture-specific documented bound | Haversine proxy; not road-network quality. |

Impossible fixtures must use expected-case accounting so they do not make hard-constraint and must-include metrics meaningless.

## 15. Warning Fixture Coverage

At minimum, implementation tests must include deterministic cases for:

- `OPENING_HOURS_UNKNOWN`
- `OPENING_HOURS_UNPARSEABLE` if approved runtime parseable/unparseable hours exist in test fixtures
- `OPENING_HOURS_CONFLICT` if approved runtime known-hour conflict fixtures exist
- `DURATION_ESTIMATED`
- `TRAVEL_TIME_ESTIMATED`
- `ORIGIN_NOT_PROVIDED`
- `COORDINATES_MISSING`
- `DAILY_WINDOW_TIGHT`
- `UNSCHEDULED_MUST_INCLUDE`
- `INSUFFICIENT_CANDIDATES`
- `PARTIAL_PREVIEW`
- `MAX_STOPS_APPLIED`
- `BUDGET_DATA_UNKNOWN`

Warnings must be emitted in the stable taxonomy order defined in `docs/rebuild/PHASE2_BATCH3_TRIP_PREVIEW_PLAN.md`.

## 16. Approval State

Implementation exists on PR #9 review branch and has not merged. Production/main runtime remains at the merged Phase 2 Batch 2 baseline until PR #9 is merged and post-merge validation passes.

Design approval has been granted through:

`APPROVED PHASE 2 BATCH 3`

Design documentation must be merged and post-merge validated before runtime implementation begins.

Approval is limited to the documented Batch 3 scope and boundaries.

## 17. Exit Criteria

Batch 3 implementation may be considered ready for review only when:

- all existing tests pass,
- all new trip-preview tests pass,
- no unapproved skipped tests are introduced,
- CSV default runtime is verified,
- PostgreSQL opt-in remains verified,
- no source, data, schema, package, frontend, or mobile expansion occurs,
- OpenAPI documentation matches the implemented endpoint,
- Phase 2 documentation records are updated,
- no production DB or Firebase was touched.

This evaluation plan does not authorize implementation.
