# UrbanAgent Rebuild Master Plan

Updated: 2026-07-26 15:35:00 +07:00.

This plan is synchronized with the approved Phase 0 canonical dataset and the
completed Phase 1 data-platform foundation. Phase 1 is merged into `main` and
tagged `phase-1-batch-3`. Phase 2 is now in planning/specification only after
explicit user approval.

Phase 2 planning artifacts:

- `docs/rebuild/PHASE2_TRAVELER_API_V2_SCOPE.md`
- `docs/rebuild/PHASE2_TRAVELER_API_V2_CONTRACT_DRAFT.md`
- `docs/rebuild/PHASE2_TRAVELER_API_V2_EVALUATION_PLAN.md`

Phase 2 Batch 3 Trip Preview design package:

- `docs/rebuild/PHASE2_BATCH3_TRIP_PREVIEW_PLAN.md`
- `docs/rebuild/PHASE2_BATCH3_TRIP_PREVIEW_SCOPE.md`
- `docs/rebuild/PHASE2_BATCH3_TRIP_PREVIEW_API_CONTRACT.md`
- `docs/rebuild/PHASE2_BATCH3_TRIP_PREVIEW_EVALUATION_PLAN.md`
- `docs/rebuild/PHASE2_BATCH3_TRIP_PREVIEW_IMPLEMENTATION_BOUNDARIES.md`

All Phase 2 API v2 architecture in this master plan remains `PROPOSED FOR
PHASE 2` until implemented and tested in a later approved batch.

Phase 2 Batch 3 status:

`APPROVED - DOCUMENTATION PENDING MERGE`

Phase 2 Batch 1 is completed. Phase 2 Batch 2 is completed. Phase 2 Batch 3
design has been approved through `APPROVED PHASE 2 BATCH 3`, but documentation
is pending merge and post-merge validation. Runtime implementation has not
started. Multi-source implementation has not started. Frontend and mobile
Batch 3 work have not started.

## 1. Current Architecture

Backend:

- Node/Express runtime is `src/server.js`; root `server.js` remains a legacy compatibility server.
- Current Phase 0 POI data access goes through `src/services/poiDataService.js`, `src/services/poiRepository.js`, and `src/services/canonicalCsvPoiRepository.js`.
- Canonical runtime POI CSV is `data/canonical/urbanagent_poi_master_v1.csv`.
- Retrieval is in `src/services/poiRetrievalService.js`.
- Itinerary generation is in `src/services/itineraryPlannerService.js`.
- Firestore persistence is in `src/services/firestorePersistenceService.js`.
- Route/expert-system code exists under both `src/expert_system/` and legacy root `ES-system/`.
- Python inference remains in `src/inference.py` and top-level `inference.py`.

Frontend:

- React/Vite/TypeScript runtime is in `D:\POI-urban-danang-FE\src`.
- Router is centralized in `src/App.tsx`.
- Auth state is in `src/auth/AuthContext.tsx`.
- API client is `src/utils/apiClient.ts`.
- Main traveler surface is `src/pages/urban-agent/UrbanAgentPage.tsx`.
- POI live/review/map layer is `src/pages/urban-agent/PoiExperienceLayer.tsx`.

## 2. Actual User Request To Itinerary Flow

1. FE gathers query, traveler context, optional location, duration, transport, and semantic config.
2. Signed-in itinerary creation calls `POST /api/agent/create-itinerary`; recommendations use `POST /api/agent/recommend-poi`.
3. Backend `createItinerary()` calls `recommendPOIs()`.
4. `recommendPOIs()` loads canonical POIs for `cityId`, scores keyword/intent/preference/rating/distance/review signals, optionally calls semantic retrieval, reranks, and returns compatible recommendation objects.
5. `createItinerary()` selects intent-matching POIs, fills remaining stops, orders by origin if valid, then by previous stop.
6. If the start location is missing, Phase 0 does not substitute Da Nang center; the first travel segment is marked `missing-origin`.
7. The response includes `cityId`, itinerary stops, timings, warnings, semantic metadata, and legacy-compatible actions.

## 3. Existing Endpoint Inventory

Primary `src/server.js` endpoints remain compatible and include:

- `GET /api/health/firebase`
- `GET /api/figures/:version/:filename`
- `GET /api/eda`
- `GET /api/metrics/training-loss`
- `POST /api/auth/ensure-user`
- `POST /api/auth/role`
- `GET /api/customer/profile`
- `POST /api/customer/profile`
- `GET /api/agent/memory`
- `POST /api/agent/memory`
- `GET /api/pois`
- `POST /api/pois`
- `POST /api/pois/sync-local`
- `GET /api/pois/data-quality`
- `POST /api/agent/recommend-poi`
- `POST /api/recommend-pois`
- `POST /api/user-preferences/rebuild`
- `POST /api/agent/create-itinerary`
- `POST /api/agent/guest-itinerary-preview` behind `FEATURE_GUEST_ITINERARY_PREVIEW`
- `POST /api/agent/update-itinerary`
- `GET /api/agent/itineraries`
- `POST /api/agent/itineraries`
- `POST /api/agent/business-location`
- `POST /api/agent/business-insight`
- seller, admin, feedback, training-status, inference, route, matrix, and weather endpoints from the earlier inventory.

## 4. Demo Features Existing In Code

- Traveler POI recommendation.
- Signed-in itinerary creation/update/list/save.
- Weather forecast.
- Local route/matrix estimates.
- Version 4 text/image inference compatibility.
- User memory/preference feedback.
- Seller/admin/customer role flows.
- Firestore-backed POI persistence and review utilities.
- Map/timeline behavior in the current Urban Agent page.

## 5. Documented But Not Implemented

- PostgreSQL/PostGIS canonical database.
- Multi-city City Pack automation beyond `da-nang`.
- External Overture/OSM/Wikidata/Foursquare/Google Places adapters.
- `/api/v2/*` traveler API.
- Full guest itinerary preview product flow; Phase 0 only has a disabled feature-flag endpoint contract.
- Opening-hours feasibility optimization.
- Partner product under `/partners`.
- Monetization/referral/sponsored-result features.

## 6. Hard-Coded Da Nang Positions

Resolved for Phase 0 runtime POI normalization/retrieval:

- Canonical loader no longer falls back to Da Nang center.
- Retrieval and itinerary no longer substitute Da Nang center for missing user/POI coordinates.

Still present by design or out of Phase 0 scope:

- `DEFAULT_CITY_ID = "da-nang"` in repository/service compatibility.
- Da Nang bbox in canonical quality report.
- `businessLocationScorer.js` accessibility scoring uses Da Nang center.
- Frontend map and copy defaults still target Da Nang.
- Research/demo scripts still contain Da Nang examples.

## 7. CSV Dependencies

Current runtime traveler POI source:

- `data/canonical/urbanagent_poi_master_v1.csv`

Raw/legacy inputs:

- must remain source/backup inputs.
- are not used as Phase 0 traveler runtime recommendation data.

Still CSV-dependent:

- canonical CSV repository.
- expert-system rule CSV files.
- research/training/report scripts and artifacts.

## 8. POI Data Risk Points

- `Address_Current` is empty for all canonical rows; UI/API must not invent address from district.
- Missing rating/review counts exist and must stay null.
- `RestaurantID` is not guaranteed to be Google `place_id`.
- Cross-source merge is intentionally conservative; do not auto-merge more duplicates without evidence.
- No freshness timestamps or license metadata per field yet.
- Admin normalization is still `pending_spatial_join`.

## 9. CSV Schema Versus Mapping

Canonical schema has 34 columns listed in `DATA_AUDIT.md`.

Current backend mapping:

- `Global_ID` -> `id`, `globalId`, `legacyId`.
- `City_ID` -> `cityId`.
- `RestaurantID` -> `sourceId`.
- `Source_IDs` -> `sourceIds`.
- `Alias_Global_IDs` -> `aliasGlobalIds`.
- `Address_Current`/`Address_Raw` -> address fields only when present.
- `Overall Rating`, `Rating_Count`, `Review_Sample_Count`, `Google_*`, `Foody_*` are preserved separately.
- `Image_URL` -> `imageUrls` as a unique ordered URL array and `imageUrl` as a legacy first-image compatibility field.
- UTF-8 BOM in the first header is stripped in `csv-parser` `mapHeaders`.

## 10. Large Or Mixed Frontend Files

- `src/pages/urban-agent/UrbanAgentPage.tsx`: large, mixes query, itinerary, route, weather, feedback, role UI, and map state.
- `src/pages/urban-agent/PoiExperienceLayer.tsx`: large, mixes map, GPS, route, reviews, analytics, and UI state.
- `src/pages/role/RolePages.tsx`: large, mixes customer, seller, and admin flows.

No frontend application refactor was done in this blocker fix.

## 11. Target Modular Monolith

Future backend modules:

- cities
- pois
- ingestion
- search
- itineraries
- routing
- weather
- preferences
- feedback
- partners
- admin

Express remains one deployable modular monolith for MVP.

## 12. City Pack Design

Da Nang remains the only approved Phase 0 city:

- `cityId = "da-nang"`
- city bbox retained for quality validation
- timezone/currency/category/dwell/source policy stay future configuration work

No second City Pack was implemented.

## 13. PostgreSQL/PostGIS And Migration

Phase 1 added and validated the PostgreSQL/PostGIS data-platform foundation
against disposable local infrastructure only. CSV remains the default runtime;
PostgreSQL remains explicit opt-in and no production migration/import has been
approved or run.

## 14. Proposed API V2

Status: `PROPOSED FOR PHASE 2`.

- `GET /api/v2/cities`
- `GET /api/v2/pois`
- `POST /api/v2/trips/preview`
- `POST /api/v2/trips`
- trip detail/edit/replan/feedback routes
- route matrix and weather v2 routes

## 15. Legacy API Compatibility

- No old endpoint was removed.
- New fields are additive: `cityId`, `globalId`, `sourceId`, provenance fields, coordinate status.
- Guest preview remains feature-flagged and disabled by default.
- Old APIs now read canonical POIs through the repository abstraction.

## 16. Traveler Versus Partner Split

Traveler remains `/urban-agent`; partner/seller surfaces remain separate in current role pages. Full `/partners` split is Phase 3+ or later, not Phase 0.

## 17. Frontend Timeline And Map Plan

Future frontend plan remains:

- timeline and map synchronized.
- numbered markers.
- mobile itinerary/map view switch.
- guest preview save/share boundaries.
- partner panels outside traveler flow.

No redesign was done in this fix.

## 18. Testing Plan

Phase 0 backend tests now exist in `tests/phase0CanonicalData.test.js`.

Still needed later:

- endpoint-level Express tests.
- frontend tests.
- route feasibility tests with real address/admin data.
- Firestore emulator tests for persistence flows.

## 19. Rollback Plan

- Revert this fix batch by restoring the listed changed files from the working tree before commit.
- Do not modify or regenerate the canonical CSV unless a new dataset decision is approved.
- Old endpoints were preserved, so API rollback is service-level.
- Keep Phase 1 migration rollback design for the future; none exists yet because no migration was added.

## 20. Phase Criteria

Phase 0 blocker criteria now verified:

- canonical CSV loads 4,166 POIs.
- BOM header bug fixed without editing CSV.
- no POI coordinate fallback to Da Nang center in canonical runtime loader.
- `cityId` filter works.
- duplicate/global ID quality checks exist.
- no automatic new duplicate merge.
- `RestaurantID` source semantics preserved.
- rating/review null semantics tested.
- legacy root expert-system CSV dependency removed.
- backend tests pass from `tests/phase0/`.

Phase 1 is complete through Batch 3. Phase 2 Batch 1 and Batch 2 are complete.
Phase 2 Batch 3 trip-preview design is approved for implementation planning
through `APPROVED PHASE 2 BATCH 3`, but documentation is pending merge and
post-merge validation. Batch 3 runtime implementation has not started.

## Future Data Platform Expansion - Multi-Source City Packs

Status: `PLANNED, NOT STARTED`.

This roadmap item does not alter Phase 2.

Phase 2 continues to use the approved 4166-POI Da Nang canonical baseline.

Future City Pack and POI expansion should proceed through:

### Stage 4A - Source Discovery Spike

- Evaluate approved open-source candidates.
- Use a bounded Da Nang development sample.
- Produce coverage, quality, license, and cost reports.
- Do not modify canonical runtime data.

### Stage 4B - License Registry And Source Governance

- Create a machine-readable source registry.
- Record storage, caching, display, API, attribution, ML, and deletion rights.
- Reject sources with unclear or incompatible rights.

### Stage 4C - Entity Resolution Prototype

- Compare new candidates against the current canonical entities.
- Produce exact, fuzzy, ambiguous, and probable-new-entity reports.
- Keep all merges reversible.
- Do not auto-merge ambiguous records.

### Stage 4D - Local Verification Workflow

- Support verified business profiles.
- Support local operator review.
- Record evidence, timestamps, and freshness.
- Keep submitted and verified values distinct.

### Stage 4E - City Pack Automation

- Convert approved source records and local enrichment into a reproducible
  City Pack.
- Preserve field-level provenance and license records.
- Keep provider-specific restrictions enforceable.

### Stage 4F - Second-City Readiness

- Select the second city through a separate product and data decision.
- Do not assume Da Nang matching rules or categories generalize automatically.

Required gate before Stage 4A:

`APPROVED MULTI-SOURCE POI SPIKE`

Mobile execution-companion design remains a later product track and is not
part of this data-platform roadmap update.
