# UrbanAgent Phase 2 Batch 3 - Implementation Boundaries

Status: IMPLEMENTED ON REVIEW BRANCH - NOT MERGED

Updated: 2026-08-01

This document defines the boundaries for a future implementation of `POST /api/v2/trips/preview`. It is not implementation approval.

## 1. Allowed Future Batch 3 Work

After explicit user approval, Batch 3 may make narrow backend changes for:

- request validation for `POST /api/v2/trips/preview`,
- stateless trip-preview service logic,
- v2 router mounting,
- serialization helpers for trip preview responses,
- focused Phase 2 Batch 3 tests,
- focused fixtures for trip preview,
- OpenAPI contract update for the new endpoint,
- Phase 2 documentation and validation records.

Any changed-file list must be declared before implementation starts.

## 2. Expected Future Code Areas

The expected implementation surface is limited to the existing Traveler API v2 module and related tests/docs.

Possible future files:

- `src/modules/travelerApiV2/tripPreview.js`
- `src/modules/travelerApiV2/router.js`
- `src/modules/travelerApiV2/serializers.js`
- `src/modules/travelerApiV2/constants.js`
- `tests/phase2/phase2TravelerApiV2Batch3.test.js`
- `tests/fixtures/phase2/tripPreviewQueries.json`
- `docs/rebuild/PHASE2_TRAVELER_API_V2_CONTRACT_DRAFT.md`
- `docs/rebuild/PHASE2_TRAVELER_API_V2_EVALUATION_PLAN.md`
- `docs/rebuild/PHASE2_BATCH3_TRIP_PREVIEW_PLAN.md`
- `docs/rebuild/CURRENT_STATE.md`
- `docs/rebuild/DECISIONS.md`
- `docs/rebuild/TEST_REPORT.md`
- `docs/rebuild/WORKLOG.md`

This list is not automatic permission to edit those files. It is the expected boundary for a later approved implementation.

## 3. Forbidden Future Batch 3 Work

Batch 3 must not implement or modify:

- trip persistence,
- saved-trip storage,
- trip edit,
- replan mutation,
- stop add, remove, reorder, or replace mutation,
- feedback persistence,
- authentication,
- authorization,
- payments,
- booking,
- collaboration,
- frontend UI,
- mobile UI,
- mobile-specific backend routes,
- production PostgreSQL cutover,
- new migrations,
- schema changes,
- package dependency upgrades,
- new package dependencies,
- external POI sources,
- multi-source entity resolution,
- second City Pack,
- generated model artifacts,
- research model weights.

## 4. Data Safety Boundary

Batch 3 must not change:

- `data/canonical/urbanagent_poi_master_v1.csv`,
- canonical manifest contents,
- canonical SHA-256,
- canonical Global_ID values,
- canonical application POI count,
- raw or legacy CSV inputs,
- Firebase production data,
- production or shared databases.

The approved canonical runtime baseline remains:

- application POIs: `4166`,
- SHA-256: `5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae`.

## 5. Runtime Boundary

CSV must remain the default runtime.

PostgreSQL/PostGIS may be tested only through explicit opt-in and disposable local infrastructure. Batch 3 must not require PostgreSQL for the default endpoint behavior.

No production or shared database may be contacted.

## 6. External Source Boundary

Batch 3 must not:

- query external POI providers,
- scrape websites,
- download Overture, OSM, Wikidata, Google Places, Foursquare, Tripadvisor, Yelp, Wanderlog, or competitor data,
- add provider adapters,
- cache external provider payloads,
- train on external provider data,
- expose external provider payloads.

The documentary approval phrases do not grant source-expansion approval.

The following phrases remain future gates, not granted approvals:

- `APPROVED MULTI-SOURCE POI SPIKE`
- `APPROVED DATA SOURCE LICENSE POLICY`

## 7. Compatibility Boundary

Batch 3 must preserve:

- `POST /api/v2/recommendations`,
- existing v2 city and POI endpoints,
- legacy endpoints,
- source compatibility rules from Phase 0,
- provenance preservation such as `google_maps+foody`,
- missing-origin null semantics,
- missing rating and review null semantics,
- image URL semantics.

No legacy endpoint may be deleted.

## 8. Implementation Sequence For Later Approval

When the user explicitly approves Batch 3 implementation, the recommended sequence is:

1. Re-read `AGENTS.md`, `URBANAGENT_CODEX_CONTEXT.md`, and Phase 2 docs.
2. Record Git state.
3. Declare the exact changed-file list.
4. Add request validation and constants.
5. Add pure trip-preview construction logic.
6. Mount `POST /api/v2/trips/preview`.
7. Add serializers and public-safe response helpers if needed.
8. Add Batch 3 fixtures.
9. Add endpoint and service tests.
10. Update OpenAPI JSON and contract docs.
11. Run default tests.
12. Run optional disposable PostgreSQL endpoint-switch tests if configured.
13. Update `CURRENT_STATE.md`, `TEST_REPORT.md`, `DECISIONS.md`, and append `WORKLOG.md`.
14. Report changed files, tests, risks, and whether the batch is ready for review.
15. Stop without starting Batch 4 or Phase 3.

## 9. Rollback Plan

Because Batch 3 should be stateless and isolated, rollback should be simple:

- disable or remove the new route,
- remove the trip-preview service module,
- remove Batch 3 tests and fixtures,
- restore OpenAPI and docs to the previous version,
- keep canonical data untouched.

No database rollback should be necessary because Batch 3 must not add migrations or persistent writes.

## 10. Approval State

Implementation exists on PR #9 review branch and has not merged. Production/main runtime remains at the merged Phase 2 Batch 2 baseline until PR #9 is merged and post-merge validation passes.

Design approval has been granted through:

`APPROVED PHASE 2 BATCH 3`

Design documentation must be merged and post-merge validated before runtime implementation begins.

Approval is limited to the documented Batch 3 scope and boundaries.

## 11. Review Checklist

Before a future Batch 3 PR can be considered ready:

- no application data was changed,
- no production DB or Firebase was touched,
- no external source was used,
- no new dependency was added without approval,
- CSV default runtime remains verified,
- PostgreSQL remains opt-in,
- all existing tests pass,
- all new trip-preview tests pass,
- missing-origin first leg remains null/unknown,
- route estimates are documented as approximate,
- warnings are deterministic and public-safe,
- no Batch 4 route exists.

This boundary document does not authorize implementation.

## 12. Implementation Review Record

Status: `IMPLEMENTED_ON_REVIEW_BRANCH_NOT_MERGED`.

Date: 2026-08-01.

The implementation task was explicitly approved by the user through:

`APPROVED PHASE 2 BATCH 3`

Implemented branch:

`phase2/batch3-traveler-api-v2-trip-preview`

Boundary audit:

- Allowed Traveler API v2 route implementation: used.
- Allowed stateless trip-preview service logic: used.
- Allowed v2 router mounting: used.
- Allowed focused Phase 2 Batch 3 tests and fixtures: used.
- Allowed OpenAPI contract update: used.
- Allowed truthful Phase 2 documentation records: used.
- Forbidden persistence, saved-trip, edit, replan, stop mutation, feedback,
  authentication, frontend, mobile, new migrations, package dependencies,
  second-city, external provider, production database, Firebase production,
  generated model artifact, and research model weight changes: not used.

The implementation is not merged and must pass final review before any
post-merge validation or later batch begins.
