# Current State

Updated: 2026-07-31 19:55:42 +07:00.

## Phase

`PHASE_2_BATCH_2_IMPLEMENTED_PENDING_REVIEW`

Phase 2 Batch 2 has been explicitly approved and implemented after the merged
Batch 1 baseline. This batch is limited to Traveler API v2 recommendation
endpoint coverage, public recommendation serialization, `reasonCodes`,
deterministic recommendation ordering, a recommendation smoke/evaluation
fixture foundation, OpenAPI update, and focused backend tests.

No itinerary preview v2 endpoint, trip persistence, trip edit/replan, feedback
persistence, PostgreSQL default-runtime switch, frontend source change,
production database access, Firebase production access, Phase 2 Batch 3 work,
or later-phase work has been started.

## Repository State

Backend repository:

- Path: `D:\POI-urban-danang-BE`
- Branch: `main`
- Local `main`: `ec06a905e2f61f9f4c90dc7857bf31cf2bb78fb6`
- `origin/main`: `ec06a905e2f61f9f4c90dc7857bf31cf2bb78fb6`
- Phase 1 Batch 3 commit
  `2c34471747f6fd33d73130db2ab47df054d4f35c` is in `origin/main`.
- Annotated tag `phase-1-batch-3` is present on current `main`.
- Working tree contained Phase 2 planning documentation changes before Batch 1
  implementation began.
- Phase 2 Batch 1 implementation changes are currently uncommitted and pending
  user review.

Frontend repository:

- Path: `D:\POI-urban-danang-FE`
- Read for context only.
- No frontend files were changed in this Phase 2 planning batch.

## Canonical Dataset Status

- Path: `data/canonical/urbanagent_poi_master_v1.csv`
- Rows: `4166`
- Unique `Global_ID`: `4166`
- SHA-256:
  `5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae`
- Runtime semantics from Phase 0 remain binding:
  - no coordinate fallback to Da Nang center
  - nullable rating/review/address/freshness values
  - `RestaurantID` remains source identifier
  - `Global_ID` remains canonical key
  - `google_maps+foody` provenance preserved
  - urban-void rows excluded from traveler runtime counts

## Phase 1 Status

Phase 1 data-platform foundation is complete through Batch 3 and merged to
`main`.

Validated Phase 1 records show:

- Default backend tests: 15 passed, 0 failed, 1 disposable-DB integration test
  skipped when DB is absent.
- Full disposable PostGIS integration: 16 passed, 0 failed, 0 skipped.
- Migration apply, rollback, reapply, canonical import, second import,
  idempotency, geometry, constraints, Postgres repository integration, and
  CSV/Postgres parity passed.
- Canonical counts:
  - POI entities: `4166`
  - source records: `4166`
  - external IDs: `8337`
  - aliases: `985`
  - images: `16246`
  - review summaries: `4166`
- `npm audit --omit=dev`: 0 production vulnerabilities after targeted
  remediation.
- CSV remains the default runtime.
- PostgreSQL remains explicit opt-in through
  `URBANAGENT_POI_REPOSITORY=postgres`.
- No production database or Firebase production data was touched.

## Current API Implementation

`CURRENT IMPLEMENTATION`:

- Primary Express server: `src/server.js`.
- Legacy compatibility server: `server.js`.
- Current traveler routes:
  - `POST /api/agent/recommend-poi`
  - `POST /api/agent/create-itinerary`
  - `POST /api/agent/guest-itinerary-preview`
  - `POST /api/agent/update-itinerary`
  - `GET /api/agent/itineraries`
  - `POST /api/agent/itineraries`
  - `POST /api/agent/feedback`
- Current data/status routes:
  - `GET /api/eda`
  - `GET /api/pois/data-quality`
  - `GET /api/weather/forecast`
  - `POST /api/route/matrix`
- Partner/seller/admin routes exist but are outside Phase 2 Traveler API v2
  scope.

New Phase 2 Batch 1 Traveler API v2 routes are mounted under `src/server.js` at
`/api/v2`:

- `GET /api/v2/cities`
- `GET /api/v2/cities/:cityId/status`
- `GET /api/v2/pois/search`
- `GET /api/v2/pois/:poiId`
- `POST /api/v2/recommendations`

Batch 1 routes expose a common v2 envelope with `ok`, `data` or `error`, and
`meta`. Every v2 response includes `apiVersion` and `requestId`; city-scoped
responses also include `cityId`.

Implemented Batch 1 behavior:

- missing or invalid `X-Request-Id` generates a new server request ID and does
  not reject an otherwise valid request.
- unknown `cityId` returns `CITY_NOT_SUPPORTED`.
- missing required `cityId` returns `VALIDATION_ERROR`.
- POI search pagination uses default limit `20`, maximum limit `100`, opaque
  cursors, cursor validation, deterministic sorting, and canonical `Global_ID`
  tie-breaks.
- POI search keeps legacy EDA source compatibility counts:
  - Google-compatible: `3946`
  - Foody-compatible: `225`
  - All/canonical: `4166`
- POI responses expose traveler-safe provenance through typed source
  identifiers and do not expose ambiguous legacy `placeId` or raw `sourceIds`.
- Recommendation v2 validates body-scoped `cityId`, rejects unsupported cities
  with `CITY_NOT_SUPPORTED`, returns nonempty canonical Da Nang results for the
  smoke query `quan cafe yen tinh`, exposes public `score`, `reason`,
  `reasonCodes`, `warnings`, and POI provenance, and omits raw scoring signals.
- Recommendation v2 uses deterministic tie-breaking by score descending,
  normalized name ascending, and canonical `Global_ID` ascending.

## New Phase 2 Planning Artifacts

Current Phase 2 planning artifacts revised in this batch:

- `docs/rebuild/PHASE2_TRAVELER_API_V2_SCOPE.md`
- `docs/rebuild/PHASE2_TRAVELER_API_V2_CONTRACT_DRAFT.md`
- `docs/rebuild/PHASE2_TRAVELER_API_V2_EVALUATION_PLAN.md`
- `docs/rebuild/PHASE2_TRAVELER_API_V2_OPENAPI_DRAFT.json`

Revision corrections applied:

- public API examples no longer expose dataset filesystem paths, database
  internals, repository class names, SQL details, or storage-mode dependencies.
- public metadata uses `contractVersion` instead of a false contract hash.
- common response metadata is minimal: every v2 response requires `apiVersion`
  and `requestId`; city-scoped responses also require `cityId`.
- `datasetVersion`, `applicationPoiCount`, `qualitySummary`, and
  `capabilityStatus` are used only where relevant, primarily city status.
- OpenAPI 3.1 is approved and required in Batch 1; `openApiSha256` may be
  recorded only after an artifact exists and is hashed.
- unknown values are modeled as `null` and/or explicit status fields, not empty
  strings or zero.
- route summaries distinguish partial and unknown legs with known/unknown leg
  counts and `distanceFullyKnown` / `travelTimeFullyKnown`.
- normalized ratings and source ratings are separated.
- source identifiers are namespaced objects; `RestaurantID` is not a Google
  Place ID.
- capability states use `unavailable`, `planned`, `experimental`, or
  `available`; no capability is true before implementation and validation.
- Phase 2 endpoints are split into core approved scope and conditional
  persistence scope.
- raw internal recommendation scoring signals are not public API.
- request ID behavior is final: invalid/missing `X-Request-Id` generates a new
  server requestId and does not reject otherwise valid requests.
- authentication contracts are documented.
- Batch 1 owns POI search pagination, cursor validation, deterministic
  search/list sorting, and OpenAPI; Batch 2 owns recommendation ranking and
  evaluation fixture foundation.
- recommendation evaluation treats legacy/v2 as behavioral parity and compares
  current `recommendPOIs` against category-only/rating-popularity baselines and
  ablations.
- scientific evaluation plan now includes research questions, hypotheses,
  baselines, fixture requirements, metrics, ablations, repeatability,
  statistical reporting, failure taxonomy, and validity threats.
- OpenAPI draft artifact SHA-256:
  `371e5de7db74b3fdeaf52999e2f417db0078309edb9ff5fe399dfec210c60da9`

Updated in this batch:

- `docs/rebuild/CURRENT_STATE.md`
- `docs/rebuild/DECISIONS.md`
- `docs/rebuild/TEST_REPORT.md`
- `docs/rebuild/WORKLOG.md` appended only
- `docs/rebuild/PHASE2_TRAVELER_API_V2_SCOPE.md`
- `docs/rebuild/PHASE2_TRAVELER_API_V2_CONTRACT_DRAFT.md`

## Build/Test Status

Phase 2 Batch 2 validation:

- Syntax checks for changed/new Traveler API v2 modules and Phase 2 tests:
  PASS.
- OpenAPI JSON parse: PASS.
- OpenAPI SHA-256:
  `371e5de7db74b3fdeaf52999e2f417db0078309edb9ff5fe399dfec210c60da9`.
- `npm.cmd test`: PASS, 28 tests total, 27 passed, 0 failed, 1 skipped.
- The skipped test is the existing guarded Phase 1 disposable PostGIS
  integration test when DB integration env vars are absent.
- Phase 2 Batch 1 endpoint smoke: PASS.
- Phase 2 Batch 2 recommendation endpoint smoke: PASS.
- Runtime source counts observed through v2 search:
  - Google-compatible: `3946`
  - Foody-compatible: `225`
  - All/canonical: `4166`
- Runtime recommendation smoke:
  - `POST /api/v2/recommendations`: PASS, nonempty canonical Da Nang results.
  - deterministic repeated response IDs: PASS.
  - public raw scoring fields absent: PASS.
- CSV remains default runtime.
- PostgreSQL remains explicit opt-in.

## Risks

- Batch 2 intentionally implements only standalone recommendation v2.
  Itinerary preview v2 remains unimplemented until Phase 2 Batch 3 approval.
- Existing traveler responses include some mojibake display strings; Phase 2
  contract should test structure and semantics first, then handle copy/encoding
  deliberately.
- Current route estimates are local haversine approximations, not road-network
  routing.
- Canonical dataset lacks verified opening hours, address, admin boundary,
  phone, website, and freshness data; v2 must surface uncertainty.
- Current server entrypoint listens immediately, so Phase 2 Batch 1 endpoint
  tests use child-process smoke tests.
- Firestore persistence couples traveler, seller, admin, and POI persistence;
  Phase 2 should not deepen partner/traveler coupling.

## Next Step

User review of Phase 2 Batch 1 implementation and validation.

Do not start Phase 2 Batch 2 until the user explicitly approves it.
