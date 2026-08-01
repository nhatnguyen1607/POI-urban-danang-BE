# Phase 2 Traveler API v2 Scope

Updated: 2026-07-26 22:04:35 +07:00.

Status: `PHASE 2 BATCH 2 IMPLEMENTED FOR APPROVED RECOMMENDATION FOUNDATION`.

This document separates accepted decisions, current implementation, proposed
Phase 2 core scope, conditional scope, and open questions. Phase 2 Batch 1
implemented the approved city/status and POI read/search foundation. Phase 2
Batch 2 implements the approved standalone recommendation v2 foundation.
Itinerary preview v2, persistence/edit/replan/feedback, and later batches have
not started.

## 1. Baseline Verification

- Backend repository: UrbanAgent backend.
- Current branch at planning time: `main`.
- Phase 1 Batch 3 is merged into `main` and tagged `phase-1-batch-3`.
- Working tree was clean before Phase 2 planning documentation began.

## 2. Current Implementation

`CURRENT IMPLEMENTATION`:

- Primary Express runtime: `src/server.js`.
- Legacy compatibility runtime: root `server.js`.
- Current traveler capabilities are served by legacy `/api/agent/*` endpoints.
- Data access goes through a repository abstraction.
- CSV remains default runtime.
- PostgreSQL remains explicit opt-in.
- `/api/v2` traveler Batch 1 endpoints now exist in `src/server.js` through
  `src/modules/travelerApiV2/router.js`.
- Implemented Batch 1 endpoints:
  - `GET /api/v2/cities`
  - `GET /api/v2/cities/:cityId/status`
  - `GET /api/v2/pois/search`
  - `GET /api/v2/pois/:poiId`
- Implemented Batch 2 endpoints:
  - `POST /api/v2/recommendations`
- No production database or Firebase production access is part of this planning
  or implementation batch.

## 3. Existing Data Decisions To Preserve

- Da Nang is the only supported city in Phase 2.
- Canonical runtime POI count is 4166.
- `Global_ID` remains the canonical public POI key.
- `RestaurantID` is a source identifier, never a verified Google Place ID.
- Source IDs must be namespaced objects, not ambiguous strings.
- `Alias_Global_IDs` preserve merged source-row aliases and do not create extra
  product POIs.
- `google_maps+foody` provenance remains preserved.
- Missing coordinates are not replaced by Da Nang center.
- Missing address, admin boundary, rating, review count, opening hours, phone,
  website, and freshness remain unknown/null unless verified.
- Urban-void rows remain excluded from traveler POIs, recommendations,
  itineraries, maps, and product counts.

## 4. Public Metadata Boundary

`PROPOSED FOR PHASE 2`:

Traveler API v2 must not expose local filesystem paths, database table names,
SQL details, `DATABASE_URL`, internal repository class names, or storage-mode
dependencies.

Public metadata should expose:

- `datasetVersion`
- `contractVersion`
- `applicationPoiCount`
- `qualitySummary`
- `capabilityStatus`

Repository mode may remain available only as diagnostic metadata or
non-production test evidence. Traveler clients must not branch behavior on CSV
versus PostgreSQL.

`contractVersion` is a version identifier, not a cryptographic hash. Use
`phase2-traveler-api-v2-draft-1`. Batch 1 must generate an OpenAPI 3.1
artifact. Only after that artifact exists may validation record an actual
`openApiSha256`, calculated from the generated artifact.

## 5. Null Semantics

`PROPOSED FOR PHASE 2` normative rule:

- Unknown is not an empty string.
- Unknown is not zero.
- Unknown is not `false` unless the field is truly Boolean.
- Unknown is represented by `null` and/or an explicit status field.

Required examples:

- Unknown district: `null`.
- Missing rating/review: `null` with status `unknown`.
- Missing route totals: `null` or explicitly partial, not zero.
- Missing freshness: `null` with status `unknown`.
- Missing origin: unknown leg distance/time and `missing-origin` source/status.

## 6. Phase 2 Core Approved Scope

The following endpoints are approved for Phase 2 specification. Batch 1
implements only the first four read-only foundation endpoints.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/v2/cities` | List supported cities. |
| GET | `/api/v2/cities/:cityId/status` | City dataset contract, quality summary, and capabilities. |
| GET | `/api/v2/pois/search` | Paginated traveler-safe POI search/list. |
| GET | `/api/v2/pois/:poiId` | POI detail by canonical `Global_ID`. |
| POST | `/api/v2/recommendations` | Standalone recommendation endpoint. |
| POST | `/api/v2/trips/preview` | Stateless, guest-safe itinerary preview. |

Approved contract decisions:

- API base path: `/api/v2`.
- Standalone recommendation endpoint is accepted.
- Guest preview is stateless and non-persistent by default.
- Raw internal scoring signals are not a public contract.
- Public recommendations expose score, reason, reasonCodes, warnings, and POI
  provenance.
- `cityId` is required for city-scoped endpoints.
- Only `GET /api/v2/cities` is not city-scoped.
- Unknown city returns `CITY_NOT_SUPPORTED`.
- No silent Da Nang fallback for explicit invalid `cityId`.
- OpenAPI 3.1 is `APPROVED AND REQUIRED` in Phase 2 Batch 1 for approved Batch
  1 endpoints and common schemas.

## 7. Conditional Scope

The following remain contract drafts only.

Status: `CONDITIONAL - NOT APPROVED FOR IMPLEMENTATION`.

- `POST /api/v2/trips`
- `GET /api/v2/trips/:tripId`
- `PATCH /api/v2/trips/:tripId`
- `POST /api/v2/trips/:tripId/replan`
- `POST /api/v2/trips/:tripId/stops`
- `DELETE /api/v2/trips/:tripId/stops/:stopId`
- `POST /api/v2/trips/:tripId/feedback`

These require an explicit persistence decision before implementation. Until
that decision exists:

- `tripSave` is unavailable.
- `tripEdit` is unavailable.
- `tripReplan` is unavailable.
- feedback persistence is unavailable.

## 8. Capability Truthfulness

Capabilities must not be `true` before implementation and validation.

Allowed states:

- `unavailable`
- `planned`
- `experimental`
- `available`

Phase 2 should use `PHASE2_DRAFT` or `EXPERIMENTAL`, not `READY_FOR_BETA`, when
no production readiness claim is intended.

After a Batch 1 endpoint is implemented and validated, its capability may move
from `planned` to `experimental` for local/test status. It becomes `available`
only when approved environment criteria pass. Implemented endpoints must not
remain permanently marked `planned`.

## 9. Existing Endpoint Inventory

Primary `src/server.js` currently registers:

- Data/status: `/api/eda`, `/api/pois/data-quality`,
  `/api/weather/forecast`, `/api/route/matrix`.
- Traveler/agent: `/api/agent/recommend-poi`,
  `/api/agent/create-itinerary`, `/api/agent/guest-itinerary-preview`,
  `/api/agent/update-itinerary`, `/api/agent/itineraries`,
  `/api/agent/feedback`.
- Account/customer: `/api/auth/*`, `/api/customer/profile`,
  `/api/agent/memory`, `/api/user-preferences/rebuild`.
- Partner/seller/admin/research/legacy model routes.

Root `server.js` remains a smaller legacy compatibility server for EDA,
figures, training metrics, inference, and route demo paths.

Partner/seller/admin routes are out of Traveler API v2 scope.

## 10. Backward Compatibility

Legacy endpoints remain unchanged while v2 endpoints are added later.

Compatibility mapping:

| Legacy behavior | Phase 2 core v2 target |
| --- | --- |
| EDA source counts | POI search totals and city status quality summary |
| POI quality report | City status |
| POI recommendation | `POST /api/v2/recommendations` |
| Guest itinerary preview | `POST /api/v2/trips/preview` |

Conditional mapping:

| Legacy behavior | Conditional v2 target |
| --- | --- |
| Authenticated itinerary create/save | `POST /api/v2/trips` |
| Itinerary list/detail/update | trip detail/edit endpoints |
| Feedback persistence | trip feedback endpoint |

## 11. Pagination Contract

`PROPOSED FOR PHASE 2`:

- Default limit: 20.
- Maximum limit: 100.
- Cursor: opaque.
- Cursor validation is required.
- Response page metadata: `total`, `limit`, `nextCursor`.
- If `q` is present: relevance descending, normalized name ascending,
  canonical `Global_ID` ascending.
- If `q` is absent: normalized name ascending, canonical `Global_ID` ascending.
- Logical pagination must behave consistently for CSV and PostgreSQL.
- Batch 1 owns all POI search pagination behavior, including limit validation,
  default limit, maximum limit, opaque cursor, cursor validation,
  deterministic search/list sorting, final `Global_ID` tie-break, `total`,
  `nextCursor`, CSV-default endpoint behavior, and OpenAPI contract coverage.

## 12. Recommendation Score Contract

`PROPOSED FOR PHASE 2`:

Public fields:

- `score`: integer or number range 0 to 100.
- `reason`: display explanation.
- `reasonCodes`: machine-readable explanation categories.
- `warnings`: public uncertainty and quality warnings.
- POI provenance.

Raw internal signal weights are not stable public API. They may exist only in
internal diagnostics or explicit non-production debug mode.

Deterministic tie-breaking:

1. score descending
2. normalized name ascending
3. canonical `Global_ID` ascending

## 13. Route Summary Semantics

`PROPOSED FOR PHASE 2`:

Route summary must prevent partial or unknown routes from implying zero
distance or zero travel time.

Required fields:

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

When one or more required legs are unknown, total distance and total travel time
must be `null` or explicitly marked partial.

## 14. Rating Contract

`PROPOSED FOR PHASE 2`:

Rating must separate:

- normalized rating on scale 5
- Google source rating/count on scale 5 where available
- Foody source rating on scale 10 where available
- nullable review count
- explicit unknown review-count status
- sample review count, which is not total review count

Do not collapse source ratings and normalized ratings into one ambiguous value.

## 15. Source Identifier Contract

`PROPOSED FOR PHASE 2`:

Use typed source identifiers:

```json
{
  "namespace": "restaurant_id",
  "value": "123456",
  "source": "foody"
}
```

`RestaurantID` is a source identifier and is never assumed to be a Google Place
ID.

## 16. Request ID And Logging

`PROPOSED FOR PHASE 2`:

- Server generates requestId when absent.
- Optional `X-Request-Id` is accepted only after validation.
- Accepted `X-Request-Id` length is 1 to 128 characters.
- Accepted characters are ASCII letters, digits, dot, underscore, colon, and
  hyphen.
- If the header is missing or invalid, the server generates a new requestId,
  continues the request, and does not echo the invalid supplied value.
- Response returns effective requestId.
- Logs use the same requestId.
- Tokens and sensitive payloads are not logged.
- An otherwise valid request must not be rejected solely because the tracing
  header is invalid.

## 17. Authentication

`PROPOSED FOR PHASE 2`:

Conditional authenticated endpoints use Firebase Bearer tokens. Missing token
returns `AUTH_REQUIRED`; invalid token returns `INVALID_TOKEN`.

Local development fallback must never work in production. Automated tests must
not contact Firebase production. Phase 2 does not redesign authentication.

## 18. Revised Batch Plan

Each batch requires explicit user approval before implementation.

Batch 1:

- v2 router
- envelope/error/requestId
- city metadata/status
- POI search/detail
- POI search pagination: limit validation, default limit 20, max limit 100,
  opaque cursor, cursor validation, deterministic search/list sorting,
  canonical `Global_ID` final tie-break, `total`, and `nextCursor`
- CSV-default endpoint behavior
- OpenAPI 3.1 artifact covering only approved Batch 1 endpoints and common
  schemas

Batch 2:

- recommendation v2
- reasonCodes
- recommendation deterministic ranking
- curated evaluation fixture foundation
- quality evaluation preparation

Batch 2 implemented:

- `POST /api/v2/recommendations`
- request validation for body `cityId`, nonempty `query`, valid
  `context.location`, and bounded `limit`
- default recommendation limit `5`, maximum `20`
- public response fields `poi`, `score`, `reason`, `reasonCodes`, `warnings`,
  and `provenance`
- deterministic public tie-breaking: score descending, normalized name
  ascending, canonical `Global_ID` ascending
- no public raw scoring signals, `scoreRaw`, ambiguous `sourceIds`, or
  `placeId`
- smoke/evaluation fixture foundation without quality superiority claims

Batch 3:

- stateless itinerary preview v2
- unknown-route semantics
- explanation/provenance

Conditional Batch 4:

- authenticated persistence/edit/replan/feedback
- only after explicit persistence approval

Final Batch:

- CSV/Postgres parity
- performance
- scientific/offline evaluation
- documentation and compatibility gate

## 19. Validation And Testing

Phase 2 tests should cover:

- envelope and requestId behavior
- city required/unsupported behavior
- pagination limit/cursor/sort contract
- source-filter counts: Google 3946, Foody 225, All 4166
- recommendation nonempty smoke
- no public raw scoring signals
- stateless itinerary preview
- unknown-origin route summary null/partial semantics
- rating/source identifier/null semantics
- legacy endpoint compatibility
- CSV default runtime
- optional PostgreSQL parity only in explicit disposable mode

## 20. Risks

- Opening hours, address, admin boundary, freshness, phone, and website remain
  unknown in the approved dataset.
- Current route estimates are local haversine approximations, not road-network
  routes.
- Current server startup shape may require child-process endpoint tests unless
  a later approved batch refactors server exports.
- Conditional persistence may require Firestore/emulator decisions before safe
  tests can be implemented.
- No user-quality claim can be made without a user study.

## 21. Open Questions

- Does conditional trip persistence belong in Phase 2?
- If persistence is approved, does Firestore remain the persistence mechanism?
- Is a Firebase emulator required for conditional persistence tests?
- What process and reviewers will create the curated query fixture?

## 22. Multi-Source POI Boundary

Phase 2 Traveler API v2 remains scoped to the approved Da Nang canonical
baseline. Multi-source POI strategy, license policy, and source governance are
documented for future planning only.

Phase 2 must not:

- ingest, scrape, sample, download, persist, cache, or merge an external POI
  source;
- replace or regenerate `data/canonical/urbanagent_poi_master_v1.csv`;
- change the approved 4166 application-POI runtime baseline;
- add a second city or global source pipeline;
- expose provider-restricted fields through public traveler contracts.

Future source work requires a separate source onboarding decision and the exact
approval phrase:

`APPROVED MULTI-SOURCE POI SPIKE`

The governing references are:

- `docs/rebuild/MULTI_SOURCE_POI_STRATEGY.md`
- `docs/rebuild/DATA_SOURCE_LICENSE_POLICY.md`

## 23. Definition Of Done

Phase 2 can be closed only when:

- Core v2 traveler endpoints are implemented and tested after approval.
- Conditional endpoints are implemented only if separately approved.
- Legacy compatibility remains tested.
- CSV remains default runtime.
- PostgreSQL remains opt-in.
- Canonical dataset bytes remain unchanged.
- No production database or Firebase production data is touched.
- No frontend, partner, second-city, monetization, or Phase 3+ work enters
  Phase 2.
- Documentation and evaluation reports are updated.
