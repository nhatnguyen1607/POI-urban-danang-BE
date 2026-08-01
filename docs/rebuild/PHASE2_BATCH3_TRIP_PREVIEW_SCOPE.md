# UrbanAgent Phase 2 Batch 3 - Trip Preview Scope

Status: APPROVED - DOCUMENTATION PENDING MERGE - NOT IMPLEMENTED

Updated: 2026-08-01

Backend baseline: `35e867a3dd8f4e9dbe27705fa9a02c7f66ea901f`

Phase 2 Batch 2 merge commit: `707cce556cf37986d9bd78fdf25902d76850242c`

Phase 2 Batch 2 implementation commit: `7718cd5c9e4d4d07a083f1d10aa9ad539035e14b`

Canonical dataset:

- path: `data/canonical/urbanagent_poi_master_v1.csv`
- application POIs: `4166`
- SHA-256: `5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae`

## 1. Purpose

Phase 2 Batch 3 designs the first Traveler API v2 itinerary preview endpoint:

`POST /api/v2/trips/preview`

The endpoint is a stateless, non-persistent trip preview. It converts a user intent and trip constraints into an ordered list of candidate stops using the existing Da Nang canonical POI runtime and the Batch 2 recommendation foundation.

The preview must be deterministic, public-safe, and honest about missing data. It must not create saved trips, mutate user data, fabricate coordinates, or claim road-network accuracy.

## 2. Current Baseline

The verified baseline before Batch 3 implementation is:

- Da Nang is the only approved City Pack.
- CSV remains the default POI runtime.
- PostgreSQL/PostGIS remains explicit opt-in.
- `POST /api/v2/recommendations` exists from Batch 2.
- Recommendation ranking is deterministic.
- Recommendation tie-breaking is deterministic.
- Recommendation responses expose machine-readable `reasonCodes`.
- Recommendation responses expose public-safe provenance and warnings.
- No external POI source has been integrated.
- Multi-source governance is documentation only.
- Neither multi-source approval gate has been granted:
  - `APPROVED MULTI-SOURCE POI SPIKE`
  - `APPROVED DATA SOURCE LICENSE POLICY`

## 3. Batch 3 In Scope

Batch 3 may implement only the stateless trip preview surface described in this design package:

- Add `POST /api/v2/trips/preview`.
- Use Da Nang canonical application POIs only.
- Use the existing repository selection rules:
  - CSV is default.
  - PostgreSQL is used only when explicitly selected by environment.
- Reuse Batch 2 recommendation logic as the candidate generator.
- Build a deterministic ordered stop list from recommended candidates.
- Support optional known origin coordinates.
- Preserve unknown-origin semantics:
  - no Da Nang center fallback,
  - first-leg `distanceMeters` is `null`,
  - first-leg `travelDurationMinutes` is `null`,
  - first-leg `estimationMethod` is `null`,
  - first-leg `distanceKnown` is `false`,
  - first-leg explanatory `calculationSource` may be `missing-origin`.
- Estimate known travel legs with transparent local approximation only.
- Validate trip duration and schedule feasibility.
- Use opening-hours information only if already present in approved runtime data.
- Report opening-hours uncertainty when approved data is absent.
- Preserve missing values as null or unknown.
- Return deterministic warnings and reason codes.
- Return public-safe provenance without exposing secrets, credentials, internal file paths, stack traces, or raw provider payloads.
- Add focused Batch 3 tests and fixtures during the later implementation batch.
- Update Phase 2 docs and OpenAPI contract during the later implementation batch only after explicit implementation approval.

## 4. Batch 3 Out Of Scope

Batch 3 must not implement:

- Trip persistence.
- Saved trips.
- Trip editing.
- Trip replan mutation.
- Stop add, remove, reorder, or replace mutation.
- Feedback persistence.
- Authentication requirements.
- Authorization rules.
- Payments.
- Booking.
- Collaboration.
- Frontend UI.
- Mobile UI or mobile APIs beyond this backend contract.
- Production PostgreSQL cutover.
- New database schema or migration.
- New external POI provider.
- Overture, OSM, Wikidata, Google Places, Foursquare, Tripadvisor, Yelp, Wanderlog, or competitor data ingestion.
- Multi-source entity resolution.
- Second city support.
- Batch 4 or Phase 3 work.

Batch 3 must not modify:

- canonical CSV bytes,
- canonical manifest,
- approved 4166 POI baseline,
- source governance approval gates,
- Firebase production data,
- production or shared databases,
- frontend repository.

## 5. Product Boundary

The endpoint is a traveler product preview, not a partner product feature.

It may help a traveler inspect a possible route for a short trip in Da Nang. It must not expose partner analytics, monetization metadata, private ranking weights, source payloads, moderation status, or business-owner tooling.

## 6. Compatibility Boundary

Batch 3 must preserve:

- all legacy endpoints,
- all existing v2 endpoints,
- Batch 2 recommendation behavior except for explicitly tested shared helper reuse,
- CSV default runtime,
- PostgreSQL opt-in runtime.

If a legacy guest-itinerary endpoint already exists, Batch 3 must not delete it or alter its response contract except for separately approved bug fixes.

## 7. Approval State

Implementation has not started.

Design approval has been granted through:

`APPROVED PHASE 2 BATCH 3`

Design documentation must be merged and post-merge validated before runtime implementation begins.

Approval is limited to the documented Batch 3 scope and boundaries.

## 8. Design Deliverables

This planning package contains:

- `PHASE2_BATCH3_TRIP_PREVIEW_PLAN.md`
- `PHASE2_BATCH3_TRIP_PREVIEW_SCOPE.md`
- `PHASE2_BATCH3_TRIP_PREVIEW_API_CONTRACT.md`
- `PHASE2_BATCH3_TRIP_PREVIEW_EVALUATION_PLAN.md`
- `PHASE2_BATCH3_TRIP_PREVIEW_IMPLEMENTATION_BOUNDARIES.md`

The authoritative index and contradiction-resolution record is
`PHASE2_BATCH3_TRIP_PREVIEW_PLAN.md`.

These files are design artifacts only. Their presence does not authorize implementation.

## 9. Exit Criteria For Implementation Approval

Before implementation can begin, the user must explicitly approve Phase 2 Batch 3 implementation after reviewing this package.

The implementation approval should confirm:

- the request contract,
- the response contract,
- the accepted travel-time approximation,
- the accepted opening-hours limitation,
- the test gates,
- the out-of-scope boundaries.

Until then, no runtime code, tests, OpenAPI JSON, migrations, package files, canonical data, frontend code, mobile code, or production systems may be changed for Batch 3.
