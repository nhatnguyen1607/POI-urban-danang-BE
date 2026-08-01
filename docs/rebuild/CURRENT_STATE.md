# Current State

Updated: 2026-08-01 15:48:48 +07:00.

## Phase

`PHASE_2_BATCH_3_DESIGN_APPROVED_DOCUMENTATION_PENDING_MERGE`

Phase 2 Batch 1 and Batch 2 are completed, merged, validated, and tagged.
Phase 2 Batch 3 Trip Preview design is approved through
`APPROVED PHASE 2 BATCH 3`. Documentation is not yet merged at the start of
this officialization task.

Completed Batch 2 scope is limited to Traveler API v2 recommendation endpoint
coverage, public recommendation serialization, `reasonCodes`, deterministic
recommendation ordering, a recommendation smoke/evaluation fixture foundation,
OpenAPI update, and focused backend tests.

`POST /api/v2/trips/preview` is not implemented. Runtime remains Batch 2.
No trip persistence, trip edit/replan, feedback persistence, PostgreSQL
default-runtime switch, frontend source change, mobile work, production
database access, Firebase production access, external POI integration, or
later-phase work has been started.

## Repository State

Backend repository state:

- Path: `D:\POI-urban-danang-BE`
- Original working copy remains untouched during documentation
  synchronization.
- Documentation clean clone:
  `C:\tmp\urbanagent-docs-be-20260731-clean`
- Documentation branch: `docs/multi-source-poi-governance`
- Previous documentation branch commit:
  `a9bf00d2de0a35a3b5dacdf570b0e1e8d14d71cd`
- `origin/main`: `707cce556cf37986d9bd78fdf25902d76850242c`
- Phase 2 Batch 2 implementation commit
  `7718cd5c9e4d4d07a083f1d10aa9ad539035e14b` is in `origin/main`.
- Annotated tag `phase-2-batch-2` is present.
- Phase 1 Batch 3 commit
  `2c34471747f6fd33d73130db2ab47df054d4f35c` is in `origin/main`.
- Annotated tag `phase-1-batch-3` is present.

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

Backend runtime remains Da Nang only. The approved application POI count
remains `4166`, and the canonical SHA-256 remains
`5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae`.
CSV remains the default runtime. PostgreSQL/PostGIS remains explicit opt-in.

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
- `POST /api/v2/trips/preview`: not implemented.

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

## Phase 2 Batch 3 Design Package

Status:

`APPROVED - DOCUMENTATION PENDING MERGE`

Authoritative documents:

- `docs/rebuild/PHASE2_BATCH3_TRIP_PREVIEW_PLAN.md`
- `docs/rebuild/PHASE2_BATCH3_TRIP_PREVIEW_SCOPE.md`
- `docs/rebuild/PHASE2_BATCH3_TRIP_PREVIEW_API_CONTRACT.md`
- `docs/rebuild/PHASE2_BATCH3_TRIP_PREVIEW_EVALUATION_PLAN.md`
- `docs/rebuild/PHASE2_BATCH3_TRIP_PREVIEW_IMPLEMENTATION_BOUNDARIES.md`

Integrated canonical documents:

- `docs/rebuild/PHASE2_TRAVELER_API_V2_SCOPE.md`
- `docs/rebuild/PHASE2_TRAVELER_API_V2_CONTRACT_DRAFT.md`
- `docs/rebuild/PHASE2_TRAVELER_API_V2_EVALUATION_PLAN.md`
- `docs/rebuild/MASTER_PLAN.md`
- `docs/rebuild/CURRENT_STATE.md`
- `docs/rebuild/DECISIONS.md`
- `docs/rebuild/WORKLOG.md`

Design approval has been granted through `APPROVED PHASE 2 BATCH 3`.
Documentation must be merged and post-merge validated before runtime
implementation begins. Runtime implementation has not started.

No frontend or mobile Batch 3 work has started.

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

User review of the synchronized backend multi-source documentation Draft PR #5.

Do not start Phase 2 Batch 3 until the user explicitly approves it.

## Future Multi-Source POI Direction

A future multi-source POI and City Pack strategy has been documented.

No multi-source implementation has started.

Current runtime remains:

- Da Nang only,
- 4166 canonical application POIs,
- canonical CSV unchanged,
- CSV default,
- PostgreSQL opt-in.

Phase 2 continues on the fixed canonical baseline.

No Overture, OpenStreetMap, Wikidata, Google Places, Foursquare, Tripadvisor,
booking-provider, or competitor data has been added to runtime.

New planning documents:

- `docs/rebuild/MULTI_SOURCE_POI_STRATEGY.md`
- `docs/rebuild/DATA_SOURCE_LICENSE_POLICY.md`

The backend and frontend `URBANAGENT_CODEX_CONTEXT.md` files now contain the
same shared multi-source governance section.

The next source-expansion action requires explicit approval after Phase 2 work
is safely paused or completed.

Mobile product work has not started.
