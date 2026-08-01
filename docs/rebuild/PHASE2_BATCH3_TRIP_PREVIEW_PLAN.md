# UrbanAgent Phase 2 Batch 3 - Trip Preview Plan

Status: APPROVED - DOCUMENTATION PENDING MERGE - NOT IMPLEMENTED

Updated: 2026-08-01

Backend baseline: `35e867a3dd8f4e9dbe27705fa9a02c7f66ea901f`

## 1. Purpose

This is the authoritative entry point and document map for the Phase 2 Batch 3 Trip Preview design package.

It records the accepted design decisions for documentation review only. It does not approve implementation and does not change runtime behavior.

## 2. Canonical References

Batch 3 design documents:

- `docs/rebuild/PHASE2_BATCH3_TRIP_PREVIEW_SCOPE.md`
- `docs/rebuild/PHASE2_BATCH3_TRIP_PREVIEW_API_CONTRACT.md`
- `docs/rebuild/PHASE2_BATCH3_TRIP_PREVIEW_EVALUATION_PLAN.md`
- `docs/rebuild/PHASE2_BATCH3_TRIP_PREVIEW_IMPLEMENTATION_BOUNDARIES.md`

Integrated Phase 2 documents:

- `docs/rebuild/PHASE2_TRAVELER_API_V2_SCOPE.md`
- `docs/rebuild/PHASE2_TRAVELER_API_V2_CONTRACT_DRAFT.md`
- `docs/rebuild/PHASE2_TRAVELER_API_V2_EVALUATION_PLAN.md`

Governance documents that remain binding:

- `docs/rebuild/MULTI_SOURCE_POI_STRATEGY.md`
- `docs/rebuild/DATA_SOURCE_LICENSE_POLICY.md`
- `docs/rebuild/DECISIONS.md`
- `docs/rebuild/CURRENT_STATE.md`

## 3. Current Verified Baseline

- Phase 2 Batch 1 is completed, merged, validated, and tagged `phase-2-batch-1`.
- Phase 2 Batch 2 is completed, merged, validated, and tagged `phase-2-batch-2`.
- Batch 2 implemented `POST /api/v2/recommendations`.
- Batch 2 recommendation behavior includes deterministic ranking, deterministic tie-breaking, and machine-readable `reasonCodes`.
- Da Nang is the only supported City Pack.
- Application POIs: `4166`.
- Canonical CSV: `data/canonical/urbanagent_poi_master_v1.csv`.
- Canonical SHA-256: `5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae`.
- CSV remains the default runtime.
- PostgreSQL/PostGIS remains explicit opt-in.
- No external POI source has been integrated.
- Phase 2 Batch 3 implementation has not started.

## 4. Approved Design Decisions For The Batch 3 Spec

The Batch 3 design package records these decisions for later implementation review:

- Endpoint: `POST /api/v2/trips/preview`.
- Endpoint type: stateless traveler trip preview.
- Persistence: not allowed in Batch 3.
- Authentication: not required for Batch 3 preview.
- City scope: Da Nang only.
- Runtime scope: current canonical dataset only.
- CSV default: preserved.
- PostgreSQL runtime: explicit opt-in only.
- Candidate source: reuse Batch 2 recommendation logic.
- Public envelope: existing v2 `ok/data/meta` envelope.
- Request ID: existing v2 request ID behavior.
- `cityId`: required and unsupported explicit city returns `CITY_NOT_SUPPORTED`.
- `query`: required nonempty string.
- request `durationMinutes`: optional; when provided, integer from 15 to 480.
- `constraints.maxStopsPerDay`: optional integer from 1 to 6.
- `recommendationOptions.limit`: optional integer from 1 to 30.
- Origin: optional.
- Missing origin: no Da Nang center fallback.
- First leg without origin: distance and time are null and unknown.
- Known later legs: may use deterministic local haversine approximation.
- Road network routing: not part of Batch 3.
- Opening hours: use only approved runtime data if present.
- Missing opening hours: do not fabricate; emit `OPENING_HOURS_UNKNOWN`.
- Unknown data: preserve null/status semantics.
- Route summary: unknown or partial route totals must not be zero.
- Public provenance: public-safe only, no runtime storage-mode dependency.
- Warnings and reason codes: deterministic and machine-readable.

## 5. Final Acceptance Policies

These policies are accepted for the design package and pending user approval for implementation.

### Duration Policy

Policy version: `phase2-batch3-duration-v1`.

Accepted request range for `durationMinutes`:

- minimum: `15`
- maximum: `480`

If a caller provides `durationMinutes`, it is treated as a requested hard scheduling budget within the accepted range.

If a stop visit duration is not explicitly requested, the preview resolves stop duration in this order:

1. approved POI-specific duration,
2. versioned category default,
3. versioned global fallback.

Every stop must expose:

- `durationMinutes`
- `durationSource`
- `durationPolicyVersion`

Allowed `durationSource` values:

- `requested`
- `poi_specific`
- `category_default`
- `fallback`

When `durationSource` is `category_default` or `fallback`, the stop must include warning code `DURATION_ESTIMATED`.

Category-duration table:

| Category family | Default minutes | Source |
| --- | ---: | --- |
| cafe, coffee, tea | 60 | `category_default` |
| restaurant, food, seafood, local_food | 75 | `category_default` |
| dessert, bakery, snack | 45 | `category_default` |
| attraction, landmark, viewpoint | 75 | `category_default` |
| museum, cultural, historic, gallery | 90 | `category_default` |
| beach, park, nature, riverside | 90 | `category_default` |
| shopping, market, mall | 75 | `category_default` |
| nightlife, bar, pub | 90 | `category_default` |
| spa, wellness | 90 | `category_default` |
| activity, entertainment | 90 | `category_default` |
| hotel, lodging | 30 | `category_default` |
| global fallback | 60 | `fallback` |

These are planning defaults, not scientifically exact visit durations.

### Trip And Daily Limits

All limits are deterministic and testable.

| Field | Rule |
| --- | --- |
| `dayCount` | integer, minimum `1`, maximum `7` |
| `dailyWindow.start` | required when `dailyWindow` is supplied; local `HH:mm` |
| `dailyWindow.end` | required when `dailyWindow` is supplied; local `HH:mm`, strictly after start on the same day |
| `dailyWindow` span | minimum `15` minutes, maximum `480` minutes |
| `maxStopsPerDay` | integer, minimum `1`, maximum `6` |
| `mustIncludePoiIds` | optional unique array, maximum `20` canonical IDs |
| `excludePoiIds` | optional unique array, maximum `100` canonical IDs |

The proposed `dayCount` maximum is `7` because Batch 3 is a bounded traveler preview, not full trip management. A one-week maximum is enough for MVP planning while keeping validation, scheduling, and deterministic fixture coverage tractable.

Default `maxStopsPerDay` by pace:

| Pace | Default max stops per day |
| --- | ---: |
| `relaxed` | 3 |
| `balanced` | 4 |
| `packed` | 6 |

### Travel-Time Policy

Policy version: `phase2-batch3-travel-time-v1`.

Every calculated leg must contain:

- `distanceMeters`
- `travelDurationMinutes`
- `travelMode`
- `estimationMethod`
- `estimationPolicyVersion`
- warning code `TRAVEL_TIME_ESTIMATED`

Every calculated leg uses `estimationMethod: "local-haversine-estimate"`.

Travel estimates are not road-network routing, live traffic, routing-provider output, or production ETA.

| Travel mode | Assumed speed km/h | Fixed transfer overhead minutes | Minimum rounded minutes |
| --- | ---: | ---: | ---: |
| `walk` | 4.5 | 0 | 2 |
| `motorbike` | 22 | 3 | 4 |
| `car` | 18 | 5 | 5 |
| `taxi` | 18 | 7 | 7 |

Rounding:

- distance is calculated by haversine and rounded to the nearest `10` meters,
- time is calculated from rounded distance and assumed speed, then overhead is added,
- final time is rounded up to the next whole minute,
- minimum rounded minutes are applied after overhead.

Excessive-distance behavior:

- a known leg above `30000` meters is not invalid by itself,
- optional stops may be unscheduled if the leg cannot fit the daily window,
- hard must-include stops may produce `PARTIAL`, `INFEASIBLE`, or narrow `NO_FEASIBLE_ITINERARY` depending on hard-constraint rules,
- no external routing provider may be queried to repair the estimate.

### Missing-Origin Semantics

When origin is not supplied, the first leg must expose:

- `distanceMeters: null`
- `travelDurationMinutes: null`
- `estimationMethod: null`
- `calculationSource: "missing-origin"` may be included as an explanatory field.

`calculationSource` must not replace `estimationMethod`.

Travel from the unknown origin to the first stop is not included in schedule feasibility.

No Da Nang center fallback, hotel fallback, or synthetic coordinate may be used.

### Opening-Hours Policy

Opening hours may be used only when approved runtime data exists.

When approved runtime opening-hours data is absent, the preview must not infer hours from category, rating, review text, popularity, or external services. It should continue construction and emit `OPENING_HOURS_UNKNOWN`.

When approved runtime opening-hours data exists but cannot be parsed deterministically, emit `OPENING_HOURS_UNPARSEABLE` and continue unless a hard constraint requires verified open status.

When approved runtime opening-hours data proves a hard schedule conflict, emit `OPENING_HOURS_CONFLICT` and unschedule or fail according to hard-constraint rules.

### Feasibility Statuses

Feasibility status exists at preview level and day level.

| Status | Criteria |
| --- | --- |
| `FEASIBLE` | At least one stop is scheduled, all hard constraints are satisfied, no hard warning exists, and the plan fits known schedule limits. |
| `FEASIBLE_WITH_WARNINGS` | At least one stop is scheduled, all hard constraints are satisfied, and only soft warnings or unknown-data warnings are present. |
| `PARTIAL` | At least one stop is scheduled, optional or soft-constrained items were unscheduled, or one or more must-include POIs are unscheduled for explainable non-contradictory reasons. |
| `INFEASIBLE` | Zero stops are scheduled for a structurally valid request, but no deterministic hard-constraint contradiction requires HTTP 422. The response remains HTTP 200 with explanations. |

Feasibility status does not replace request validation.

### NO_FEASIBLE_ITINERARY Semantics

`NO_FEASIBLE_ITINERARY` is allowed only as HTTP 422 when all are true:

- the request is structurally valid,
- explicit user hard constraints must all be satisfied,
- no itinerary containing any valid compliant schedule can be constructed,
- silently relaxing hard constraints is forbidden.

Examples:

- mutually impossible must-include POIs and time windows,
- every required POI conflicts with a known hard scheduling constraint,
- the requested mandatory set cannot fit within any requested day.

Do not return HTTP 422 merely because:

- recommendation candidates are limited,
- opening hours are unknown,
- some optional stops cannot be scheduled,
- the preview is incomplete,
- coordinates are missing for some optional POIs.

Those conditions normally return HTTP 200 with `PARTIAL` or `INFEASIBLE`, warnings, and unscheduled explanations.

### Warning Taxonomy

Public warning severity enum:

- `info`
- `warning`
- `error`

Do not use public warning severities `hard`, `hard/warning`, `soft`, `fatal`, or `conditional`.

Warning ordering is stable by the order below, then by stable scope position, then by canonical `Global_ID` where applicable.

| Code | Trigger | Scope | Severity | Continue? | Traveler-safe meaning | Required fixture |
| --- | --- | --- | --- | --- | --- | --- |
| `OPENING_HOURS_CONFLICT` | Approved runtime hours prove a stop conflicts with a hard schedule. | stop, day, preview | warning | conditional | A known hours conflict was found; optional conflicts must be replaced or unscheduled. | `opening-hours-conflict` |
| `UNSCHEDULED_MUST_INCLUDE` | A requested must-include POI cannot be scheduled in an HTTP 200 `PARTIAL` preview. | stop, day, preview | warning | conditional | A requested stop was not placed. | `unscheduled-must-include` |
| `COORDINATES_MISSING` | Optional POI lacks valid coordinates for route ordering. | stop, preview | warning | yes | Some route math cannot be calculated; coordinates are not fabricated. | `coordinates-missing` |
| `OPENING_HOURS_UNPARSEABLE` | Approved runtime hours exist but cannot be parsed. | stop, preview | warning | yes | Hours exist but cannot be safely interpreted. | `opening-hours-unparseable` |
| `DAILY_WINDOW_TIGHT` | Known schedule fits only after trimming optional stops or leaves no buffer. | day, preview | warning | yes | The day may feel tight. | `daily-window-tight` |
| `INSUFFICIENT_CANDIDATES` | Candidate pool has fewer eligible POIs than requested soft stop count. | preview | warning | yes | Fewer matching stops were found. | `insufficient-candidates` |
| `PARTIAL_PREVIEW` | Preview schedules at least one stop but omits optional or explainable requested items. | preview | warning | yes | The preview is incomplete. | `partial-preview` |
| `OPENING_HOURS_UNKNOWN` | Approved runtime opening-hours data is absent. | stop, preview | info | yes | Hours are not verified. | `opening-hours-unknown` |
| `DURATION_ESTIMATED` | Duration source is `category_default` or `fallback`. | stop | info | yes | Visit time is estimated. | `duration-estimated` |
| `TRAVEL_TIME_ESTIMATED` | A known leg uses local haversine estimate. | leg | info | yes | Travel time is approximate. | `travel-time-estimated` |
| `ORIGIN_NOT_PROVIDED` | `startLocation` is absent for first leg. | first leg, preview | info | yes | Travel from start to first stop is unknown. | `quiet-cafe-no-origin` |
| `MAX_STOPS_APPLIED` | Candidate list was trimmed by `maxStopsPerDay`. | day, preview | info | yes | Stop count limit was applied. | `max-stops-applied` |
| `BUDGET_DATA_UNKNOWN` | Caller supplies budget preference but approved POI cost data is unavailable. | preview | info | yes | Budget fit cannot be verified. | `budget-data-unknown` |

### Evaluation Metrics

| Metric | Numerator | Denominator | Fixture cases | Required threshold | Limitation |
| --- | --- | --- | --- | ---: | --- |
| Deterministic replay rate | identical repeated responses | repeated valid fixture runs | all valid fixtures | 100% | request IDs may differ unless supplied |
| Exclusion violation rate | excluded IDs returned | excluded IDs requested | `exclusions-dedup` | 0% | only tests explicit exclusions |
| Duplicate-stop rate | duplicate stop IDs | scheduled stops | all success fixtures | 0% | does not prove entity-resolution quality |
| Hard-constraint satisfaction rate | satisfied hard constraints | satisfiable hard constraints | valid hard-constraint fixtures | 100% | impossible fixtures accounted separately |
| Must-include scheduling rate | scheduled must-include IDs | satisfiable must-include IDs | `must-include-known-poi` | 100% | impossible must-includes excluded from denominator |
| Daily-window overflow rate | scheduled days exceeding window | scheduled days with window | schedule fixtures | 0% | uses approximate travel time |
| Known opening-hours conflict rate | scheduled known-closed stops | stops with known hard hours | known-hours fixtures | 0% | current canonical data may have no known hours |
| Unscheduled explanation coverage | unscheduled items with explanation | unscheduled items | partial/infeasible fixtures | 100% | explanation quality is structural only |
| Warning-code correctness | expected warning matches | deterministic warning assertions | warning fixtures | 100% | does not measure usefulness |
| Geographic compactness proxy | routes within fixture bound | routes evaluated for compactness | known-origin fixtures | documented threshold per fixture | haversine proxy, not road quality |

Impossible fixtures must be reported separately so they do not make hard-constraint or must-include metrics meaningless.

## 6. Resolved Draft Contradictions

The initial draft documents were audited and reconciled as follows:

- Response envelope now follows the existing v2 `ok/data/meta` convention, not a separate `success/error` convention.
- Trip preview response uses `data.trip.stops`, aligned with the existing `TripPlan` model, rather than a separate top-level `itinerary` array.
- Travel legs use `estimationMethod` for calculated legs. Missing-origin legs use `distanceMeters: null`, `travelDurationMinutes: null`, `estimationMethod: null`, and may also expose `calculationSource: "missing-origin"`.
- Route summary fields align with the existing v2 contract:
  - `totalDistanceKm`
  - `totalTravelMinutes`
  - `totalStayMinutes`
  - `totalPlanMinutes`
  - `distanceFullyKnown`
  - `travelTimeFullyKnown`
  - `knownLegCount`
  - `unknownLegCount`
  - `calculationSource`
  - `status`
- Public responses must not expose repository mode as a client contract. CSV/PostgreSQL mode remains diagnostic evidence only.
- `NO_FEASIBLE_ITINERARY` is documented as a narrow controlled error for structurally valid requests with impossible hard constraints.

## 7. Explicit Non-Approvals

This design package does not approve:

- runtime implementation,
- test implementation,
- OpenAPI JSON changes,
- trip persistence,
- itinerary mutation,
- feedback persistence,
- frontend work,
- mobile work,
- production PostgreSQL cutover,
- database migrations,
- second-city support,
- external source integration,
- multi-source POI spike.

The following approval phrases are present only as future gates and have not been granted:

- `APPROVED MULTI-SOURCE POI SPIKE`
- `APPROVED DATA SOURCE LICENSE POLICY`

## 8. Approval State

Implementation has not started.

Design approval has been granted through:

`APPROVED PHASE 2 BATCH 3`

Design documentation must be merged and post-merge validated before runtime implementation begins.

Approval is limited to the documented Batch 3 scope and boundaries.

## 9. Exit Criteria Before Implementation

Before Batch 3 implementation begins, the user must explicitly approve implementation after reviewing:

- this authoritative plan,
- the scope document,
- the API contract,
- the evaluation plan,
- the implementation boundaries.

The implementation approval must not be inferred from this document.
