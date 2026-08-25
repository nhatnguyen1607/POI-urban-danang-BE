# Current State

Updated: 2026-08-10 14:55:00 +07:00.

## Current Phase 4 State

`PHASE_4_STAGE_4E_REAL_BOUNDED_VALIDATION_REVIEW_BRANCH`

Branch: `phase4/stage4e-real-source-validation`.

Stage 4D is merged into backend main at
`30bd6d48f3e490f3be7ab43ce2b3724a0003dca1`. Stage 4E validates small real
bounded Overture, OSM, Wikidata, and Wikimedia Commons metadata snapshots
through the existing Stage 4C adapters and Stage 4D candidate build pipeline.

Current Stage 4E result:

- 76 normalized records, 0 invalid.
- 6 high-confidence matches, 3 probable matches, 1 ambiguous, 66 new
  candidates, and 3 source-duplicate pairs.
- 70 review-queue items; no ambiguous/new/duplicate auto-merge.
- Candidate build: 9 matched enrichments, 0 approved new POIs.
- Fixed-snapshot build determinism: PASS.
- Focused Stage 4C/4D/4E tests: 22 passed, 0 failed.
- Canonical count remains 4166 and canonical SHA-256 remains
  `5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae`.
- Runtime, CSV default repository, Firestore, and production databases are
  unchanged. Google Places is absent.

Recommended next direction, not approved or started: Stage 4F option C,
provenance/license hardening for per-file Commons metadata and ODbL artifact
boundaries.

## Previous Integrated Demo Phase

`INTEGRATED_DEMO_RELEASE_BRANCH`

Phase 2 Batch 3 is merged, validated, tagged, and backed up. The current
video-ready demo work is being prepared on release branch:

`release/integrated-demo-2026-08`

Clean integration clones:

- Backend: `C:\tmp\urbanagent-integrated-backend-20260808-175300`
- Frontend: `C:\tmp\urbanagent-integrated-frontend-20260808-175300`

This branch integrates the approved Traveler API v2 recommendations and trip
preview flow into the existing main web application route `/urban-agent`,
while preserving existing login, layout, navigation, dashboard, Urban Agent
page, Leaflet map modal, CSV default runtime, and legacy endpoints.

Current validation:

- Backend `npm.cmd test`: PASS, `40` total, `39` passed, `0` failed,
  `1` guarded optional PostGIS skip.
- Backend `npm.cmd audit --omit=dev`: PASS, `0` vulnerabilities.
- Backend JavaScript syntax checks: PASS.
- Frontend production build: PASS.
- Frontend `npm.cmd audit --omit=dev`: PASS, `0` vulnerabilities.
- Frontend scoped lint: FAIL on pre-existing lint debt in existing large
  files; TypeScript build remains PASS.
- Canonical CSV SHA-256 unchanged:
  `5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae`.

No production database, Firebase production, external POI provider, external
routing provider, PostgreSQL default switch, second city, multi-source
implementation, mobile work, trip persistence, replan, stop mutation,
feedback persistence, or Batch 4 work has been started.

## Previous Demo Sprint Backend State

`DEMO_SPRINT_BACKEND_PER_DAY_WINDOWS_BRANCH`

Phase 2 Batch 3 is merged, validated, tagged, and backed up. The demo sprint
backend branch is:

`demo/2026-08-07-per-day-windows`

Branch base:

`e5f2f975b009d561dfef037932068836bde24918`

This branch adds only a backward-compatible demo extension to
`POST /api/v2/trips/preview`:

- preserve existing `trip.dailyWindow` behavior,
- accept `trip.dailyWindow.startTime` / `endTime` aliases in addition to
  `start` / `end`,
- accept optional `trip.dayWindows[]` per-day overrides for day counts 1-7,
- keep the response grouped by day with each day exposing its resolved
  `dailyWindow`.

Validation on this branch:

- `npm.cmd test`: PASS, `40` total, `39` passed, `0` failed, `1` guarded
  optional PostGIS skip.
- `npm.cmd audit --omit=dev`: PASS, `0` vulnerabilities.
- Per-day-window HTTP smoke: PASS, `200`, `5` stops, day windows resolved as
  `09:00-17:00`, `11:00-14:00`, and `08:30-12:30`.
- OpenAPI SHA-256:
  `5dc88bb27797626e4564c6b324e19ed286ae1239bd03eda8d9c736a3aa892988`.

No persistence, authentication, trip ownership/history, replan mutation, stop
mutation, feedback persistence, external routing, external POI ingestion,
second city, mobile work, production database access, Firebase production
access, multi-source implementation, or Batch 4 work was started.

## Previous Phase 2 Batch 3 State

`PHASE_2_BATCH_3_IMPLEMENTED_REVIEW_BRANCH_NOT_MERGED`

Phase 2 Batch 1 and Batch 2 are completed, merged, validated, and tagged.
Phase 2 Batch 3 Trip Preview design is approved through
`APPROVED PHASE 2 BATCH 3`. The approved design documentation was merged and
post-merge validated before runtime implementation began.

Phase 2 Batch 3 runtime implementation now exists on review branch
`phase2/batch3-traveler-api-v2-trip-preview` in clean implementation clone
`C:\tmp\urbanagent-phase2-batch3-implementation-20260801-164730`.

The implementation adds exactly one new Traveler API v2 runtime endpoint:

- `POST /api/v2/trips/preview`

Batch 3 remains unmerged at the time of this record. Production runtime is
unchanged until the review branch is approved and merged.

Completed Batch 2 scope is limited to Traveler API v2 recommendation endpoint
coverage, public recommendation serialization, `reasonCodes`, deterministic
recommendation ordering, a recommendation smoke/evaluation fixture foundation,
OpenAPI update, and focused backend tests.

No trip persistence, trip edit/replan, feedback persistence, PostgreSQL
default-runtime switch, frontend source change, mobile work, production
database access, Firebase production access, external POI integration,
second-city work, Batch 4, Phase 3, or later-phase work has been started.

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
- `POST /api/v2/trips/preview`: implemented on
  `phase2/batch3-traveler-api-v2-trip-preview` review branch; not merged to
  production/main until PR #9 is approved, merged, and post-merge validated.

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

`IMPLEMENTED ON REVIEW BRANCH - NOT MERGED`

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
Runtime implementation exists on PR #9 review branch and has not merged.
Production/main runtime remains at the merged Phase 2 Batch 2 baseline until
PR #9 is merged and post-merge validation passes.

No frontend or mobile Batch 3 work has started.

## Build/Test Status

Phase 2 Batch 3 implementation branch validation observed during the
implementation task:

- Baseline main SHA: `03bcc2ba90be7aa2618b5381f763ac1933469deb`.
- Branch: `phase2/batch3-traveler-api-v2-trip-preview`.
- `npm.cmd ci`: PASS.
- `npm.cmd ls --omit=dev --all`: PASS, no ELSPROBLEMS.
- `npm.cmd audit --omit=dev`: PASS, 0 vulnerabilities.
- `npm.cmd audit --omit=dev --json`: PASS, total vulnerabilities `0`.
- Batch 3 focused test file:
  `tests/phase2/phase2TravelerApiV2Batch3.test.js`.
- `node --test tests/phase2/phase2TravelerApiV2Batch3.test.js`: PASS,
  10 tests, 10 passed, 0 failed.
- `npm.cmd test`: PASS, 38 tests total, 37 passed, 0 failed, 1 skipped.
- The skipped test remains the existing guarded Phase 1 disposable PostGIS
  integration test when DB integration env vars are absent.
- OpenAPI JSON parse after Batch 3 update: PASS.
- OpenAPI SHA-256:
  `0cf59e434e270cee80154ac20cf4c32b14c4147b6f8d174364ea93caae326034`.
- Canonical CSV SHA-256 remains
  `5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae`.
- Canonical application POIs remain `4166`.
- CSV remains the default runtime.
- PostgreSQL/PostGIS remains explicit opt-in.
- Endpoint smoke coverage for `POST /api/v2/trips/preview`: PASS in local
  CSV-default child-process tests.

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

- Batch 3 trip preview exists only on PR #9 review branch at the time of this
  record; production/main remains at the merged Batch 2 runtime until review,
  merge, and post-merge validation complete.
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

Strict re-audit of PR #9 after targeted Batch 3 corrections.

Do not merge PR #9 until strict re-audit passes and the user approves merge.

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

## Phase 2 Batch 5 State

`PHASE_2_BATCH_5_IMPLEMENTED_REVIEW_BRANCH_NOT_MERGED`

Approved scope:

- authenticated trip persistence,
- saved-trip list/open/update/delete,
- owner-only access,
- `/urban-agent` saved-trip integration.

Implementation branches:

- Backend: `phase2/batch5-trip-persistence`
- Frontend: `phase2/batch5-trip-persistence`

Persistence:

- Firebase Auth plus Firestore for production saved user trips.
- Guarded nonproduction memory adapter only for tests/local smoke.
- POI catalog remains canonical CSV default; PostgreSQL/PostGIS remains
  explicit opt-in.

Validation:

- Backend `npm.cmd test`: PASS, 40 passed, 0 failed, 1 guarded PostGIS skip.
- Frontend `npm.cmd run build`: PASS.
- Frontend scoped lint on `src/utils/apiClient.ts`: PASS.
- Integrated local HTTP smoke: PASS.

No production database, Firebase production, external source, second city,
multi-source implementation, mobile work, booking/payment, or Batch 6 work was
started.

## Phase 2 Final Gate State

`PHASE_2_CLOSED`

Date: 2026-08-09 14:49:50 +07:00.

Current backend final-gate branch:

- `phase2/final-gates`

Baseline:

- Backend main at final-gate start:
  `1d75ec7ae32eae17697895dd94523017ac094d32`
- Frontend main at final-gate start:
  `e0c75bed038befbc53dd35a9ba26b447eafd6649`

Validated in the clean final-gate backend clone:

- Canonical runtime POIs: `4166`
- Canonical SHA-256:
  `5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae`
- CSV remains default runtime.
- PostgreSQL remains explicit opt-in.
- Legacy/v2 source counts: Google `3946`, Foody `225`, All `4166`.
- Recommendation smoke: PASS, nonempty.
- Trip preview smoke: PASS, nonempty.
- Missing-origin null semantics: PASS.
- Saved-trip replan preserves same tripId: PASS.
- Feedback persistence: DEFERRED and not implemented in Phase 2.
- CSV/PostgreSQL parity: PASS through the existing documented disposable
  PostGIS integration test, `1` total, `1` passed, `0` failed, `0` skipped.
- Backend tests before parity close: PASS, `42` total, `41` passed, `0`
  failed, `1` guarded optional PostGIS skip.
- Backend tests after final-gate documentation update: PASS, `42` total, `41`
  passed, `0` failed, `1` guarded optional PostGIS skip.
- Production audit: PASS, `0 vulnerabilities`.

Cleanup:

- Disposable PostGIS container and volume were removed after parity validation.

Phase 2 is closed.

## Phase 5 Stage 5A State

`PASSED_WITH_DEPLOYMENT_BLOCKER`

Date: 2026-08-14 (+07:00).

- Backend branch: `phase5/stage5a-product-e2e`.
- Frontend branch: `phase5/stage5a-product-e2e`.
- Runtime: `4173` unique POIs, `0` invalid; all seven Stage 4S POIs load.
- Canonical SHA-256:
  `dcb404cc8b5c7a9b5fd70df63039ab8f828c504270e22b12a671fa4ed61583f4`.
- Five traveler scenarios, itinerary, replan, saved-trip lifecycle, Leaflet
  synchronization, desktop and mobile browser checks: PASS.
- Backend tests: `48` total, `47` passed, `0` failed, `1` guarded PostGIS skip.
- Frontend production build and scoped lint: PASS.
- CSV remains default; PostgreSQL remains opt-in; production DB and Firebase
  were not accessed.
- Deployment blocker: a clean clone cannot download the canonical CSV because
  the Git LFS budget is exceeded.
- Next proposed work: `FIX CONCRETE DEPLOYMENT BLOCKER` only.
## Phase 5 Stage 5B - 2026-08-14

- Status: `PASSED` on branch `phase5/stage5b-runtime-data-packaging`.
- Stage 5A merged through backend PR #34 (`5da2ee1172e75f1011aa9aa70f8b8bed93bf8ffb`)
  and frontend PR #9 (`44ba7a323a966cba034df47ede1a57cfcf620c1a`).
- Canonical runtime data is a normal Git blob, not an LFS pointer.
- Canonical contract remains 4173 POIs, 7220189 bytes, SHA-256
  `dcb404cc8b5c7a9b5fd70df63039ab8f828c504270e22b12a671fa4ed61583f4`.
- `npm run data:verify` and startup fail closed on missing, pointer, size, hash,
  schema, duplicate-ID, or invalid-record failures.
- True clean clone with LFS smudge disabled passed install, verifier, startup,
  loader, recommendation, trip-preview, replan, and saved-trip checks.
- CSV remains the default runtime; PostgreSQL remains explicit opt-in.
- Production CORS now requires exact origins from
  `URBANAGENT_CORS_ALLOWED_ORIGINS`; frontend API URL was already configurable.
- Frontend changes: none.
- Next proposed step: Stage 5C staging deployment. Stage 5C has not started.
## Phase 5 Stage 5C - 2026-08-14

- Status: `STAGING_PREPARED_BLOCKED_BY_PLATFORM_ACCESS`.
- Stage 5B merged through backend PR #35 at
  `bcbb4c0ab1cac07040393dd57679a702b63bca2c`.
- Backend staging container preparation uses lockfile install, build-time data
  verification, startup verification, and `/api/v2/cities` health checks.
- Backend platform discovery: Docker/Hugging Face architecture exists, but no
  Space, deployment record, environment, platform secret, or authenticated
  Hugging Face session is available.
- Frontend platform discovery: Vercel integration exists, but the authenticated
  local account cannot access the UrbanAgent project or configure Preview env.
- No backend or frontend staging URL was created. Deployed E2E was not run.
- Local pre-deploy validation passed for 4173 POIs, canonical integrity,
  recommendation/preview/replan/saved-trip tests, production CORS, backend
  startup, and frontend production build.
- Firebase staging credentials are unavailable; saved trips are blocked for
  staging rather than bypassed.
- Production was not deployed or modified. Stage 5D has not started.
## Phase 5 Stage 5C safe auto-deploy continuation - 2026-08-14

- Latest status: `SAFE_AUTO_DEPLOY_PREPARED_FOR_REVIEW`; this supersedes the
  earlier backend platform-access blocker above.
- Existing target verified: Hugging Face Docker Space `nhttngy/back-end`, port
  7860, rollback revision
  `240c962c170c1b6ddef333863f9692c825808e2c`.
- Added a deterministic tracked-file payload and GitHub Action. Pushes to main
  validate automatically; upload remains gated until a first manual dispatch
  succeeds or `HF_AUTO_DEPLOY_ENABLED=true` is explicitly configured.
- Runtime legacy LFS assets are excluded; no LFS-managed file is required by
  the default traveler runtime.
- Local payload validation: 13 tests passed, payload install/data/startup/CORS
  passed, 4173 POIs and approved SHA preserved.
- Remote deployment: not executed. Space variables/secrets were not changed.
- Next action: branch review, merge, validate main workflow, then first manual
  dispatch. Stage 5D remains not started.

## Phase 5 Stage 5D.1 destination geocoding support - 2026-08-20

- Branch: `phase5/stage5d1-geocoding`.
- Added a request-time destination-search contract for the traveler map.
- No geocoder is enabled by default. `URBANAGENT_GEOCODER_URL` is required;
  otherwise the endpoint fails closed with `GEOCODER_NOT_CONFIGURED`.
- Photon-shaped responses are normalized without creating UrbanAgent POI IDs.
- Results remain request-time only and include OpenStreetMap attribution.
- Canonical data, CSV-default runtime, database, Firebase, and deployment are
  unchanged.
- Runtime provider approval/configuration and live address verification remain
  pending; no external query was executed in this batch.

## Phase 5 Stage 5D.1 geocoder release gate - 2026-08-20

- Verdict remains `PARTIAL` at the staging-configuration gate.
- Full PR #37 review found no committed credential, browser-exposed provider
  secret, user-controlled outbound URL, fabricated result, or canonical write.
- Hardened the disabled-by-default proxy with an HTTPS hostname allowlist,
  bounded global staging rate limit, controlled upstream errors, and explicit
  Da Nang-only city validation. No response cache was added.
- Required staging variables are `URBANAGENT_GEOCODER_URL`,
  `URBANAGENT_GEOCODER_ALLOWED_HOSTS`, and optional
  `URBANAGENT_GEOCODER_RATE_LIMIT_PER_MINUTE`.
- No approved provider values are available in the current environment.
  Provider activation, live address queries, authenticated route E2E, and GPS
  acceptance were not performed or represented as passing.
- Frontend PR #12 now redacts number-bearing address queries before optional
  search analytics persistence; precise GPS coordinates are not written by the
  touched analytics payloads.

## Product hotfix - 2026-08-21

- Status: `IMPLEMENTED_ON_REVIEW_BRANCH_NOT_MERGED`.
- Backend branch: `fix/external-place-resolution-trip-integration`.
- Frontend branch: `fix/full-route-external-place-search-ui-cleanup`.
- The traveler UI now provides an all-trip/day road-route view built from the
  existing authenticated route endpoint, without an artificial overnight leg.
- Generic Da Nang address results can enter preview/replan as bounded,
  request-time temporary places. They remain `canonical=false` and are not
  written to the canonical dataset, Firestore, or PostgreSQL.
- The Map & Data page is now a focused place/address search and navigation
  surface; the traveler-facing source selector, metrics, overview map, and
  featured list were removed.
- Local browser and HTTP smoke passed. Canonical runtime remains 4173 POIs with
  SHA-256 `dcb404cc8b5c7a9b5fd70df63039ab8f828c504270e22b12a671fa4ed61583f4`.
- No merge or deployment has occurred. Existing automatic canonical creation
  remains disabled.

## Product hotfix final hardening - 2026-08-21

- Status: `HARDENED_ON_REVIEW_BRANCH_NOT_MERGED`.
- The full-route UI now reports routed-segment coverage. Complete routes retain
  normal total labels; partial routes label distance/time as calculated values
  and state how many segments remain unresolved.
- Temporary-place persistence is covered through create, reload, and same-trip
  replan with outbound resolver calls blocked. The immutable request snapshot
  retains temporary ID, name, address, coordinates, source, attribution, and
  `canonical=false`.
- External result cards show useful address context, result type/category, and
  GPS distance when available. Manual-pin fallback and canonical search remain
  operational when external geocoding fails.
- Focused backend tests, frontend production build, scoped lint, and browser
  acceptance at desktop and 390x844 passed. No blocker or high-severity issue
  remains in this review scope.
- Canonical runtime remains 4173 POIs at SHA-256
  `dcb404cc8b5c7a9b5fd70df63039ab8f828c504270e22b12a671fa4ed61583f4`.
- No merge, deployment, canonical write, database/Firebase write, or automatic
  POI creation occurred.

## Google Maps Platform search upgrade - 2026-08-24

- Status: `GOOGLE_MAPS_PLATFORM_CONFIGURATION_REQUIRED` on the existing
  product-hotfix review branches; no merge or deployment occurred.
- Google Places (New) Text Search, Nearby Search, Autocomplete, Place Details,
  and Google Geocoding are implemented behind separate browser/server key
  boundaries. Photon/OpenStreetMap remains the bounded fallback.
- Query intent, search-origin priority, 3/5/10 km category expansion, 20 km
  normal-search cap, deterministic distance-aware ranking, exact rooftop
  validation, approximate-address pin confirmation, and provider identity
  preservation are covered by mock tests.
- The discovery map uses Google Maps JavaScript API only when configured to
  display Google content. Existing OSRM route/navigation behavior is unchanged.
- Full backend suite: 69 total, 68 passed, 0 failed, 1 guarded PostGIS skip.
  Frontend production build and scoped lint passed; local desktop/mobile
  fallback smoke passed.
- Canonical runtime remains 4173 POIs at SHA-256
  `dcb404cc8b5c7a9b5fd70df63039ab8f828c504270e22b12a671fa4ed61583f4`.
- Next action: configure restricted staging keys, enable Maps JavaScript API,
  Places API (New), and Geocoding API, then run the required live Google test
  matrix before merge or deployment.

## Phase 5 Admin authentication foundation - 2026-08-25

- Status: `IMPLEMENTED_ON_REVIEW_BRANCH_NOT_MERGED`.
- Backend branch: `phase5/admin-backend-foundation`, based on the latest remote
  `fix/external-place-resolution-trip-integration` at
  `772a40c243c85bebee3be4492731f15a7ff8949c`.
- Frontend branch remains `phase5/base44-ui-integration`, starting from accepted
  commit `86b741c50138be1f747081de957a9704218118b3`.
- `/api/admin/*` now requires a verified Firebase ID token and the trusted
  custom claim `admin: true`; client roles, local storage, URL state, and local
  development tokens cannot authorize Admin.
- The Admin namespace is read-only and provides identity, capabilities,
  paginated Firebase Auth users, canonical POI summary, and safe health state.
- The frontend Admin guard verifies `/api/admin/me` before mounting any Admin
  content. The old hardcoded/local Admin login mechanism has been removed.
- Backend suite: 79 total, 78 passed, 0 failed, 1 guarded PostGIS skip. Frontend
  production build and scoped lint passed. Browser negative-authorization and
  traveler smoke passed; no live Firebase Admin account was used.
- Google provider implementation is preserved and remains
  `GOOGLE_LIVE_CONFIGURATION_PENDING`.
- No merge, deployment, production Firebase access, canonical data change, or
  Admin write capability occurred.
