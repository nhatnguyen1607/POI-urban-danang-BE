# Decisions

Updated: 2026-07-26 16:35:00 +07:00.

## Accepted

- BE `URBANAGENT_CODEX_CONTEXT.md` remains the canonical context.
- FE `URBANAGENT_CODEX_CONTEXT.md` is a mirror and is byte-identical after this sync.
- The approved Phase 0 runtime dataset is `data/canonical/urbanagent_poi_master_v1.csv`.
- The canonical CSV and manifest were not modified.
- The dataset decision file was not modified.
- `Global_ID` is the canonical legacy key exposed by compatibility APIs.
- `RestaurantID` is a source identifier, not a verified Google `place_id`.
- `Alias_Global_IDs` preserves merged row IDs and must not create extra product POIs.
- Urban-void rows are excluded from traveler POIs, recommendations, itineraries, maps, and POI counts.
- Missing coordinates, address, admin-boundary, rating, review count, and freshness values must stay unknown/null unless verified.
- Phase 0 may add tests and safety foundation only.
- Phase 1 was explicitly approved, implemented through Batch 3, merged to
  `main`, and tagged `phase-1-batch-3`.
- Phase 2 was explicitly approved on 2026-07-26 for traveler API v2 planning
  and specification.

## New Decisions In This Fix Batch

- Fix BOM at the CSV parser header layer instead of editing the canonical CSV.
- Use Node built-in `node:test` for Phase 0 backend tests, avoiding new dependencies.
- Keep Phase 0 tests under `tests/phase0/`.
- Expose `imageUrls` as the canonical normalized image collection and keep `imageUrl` as a compatibility first-image field.
- Keep frontend app code unchanged in this blocker fix; only sync frontend rules/context.
- Run frontend production build to `C:\tmp\urbanagent-fe-build-phase0-fix` to avoid modifying repo `dist`.
- Leave broad frontend lint debt unresolved and reported, per user instruction.

## Phase 1 Decisions

- Start Phase 1 only after explicit user approval on 2026-07-26.
- Use a modular monolith data foundation instead of introducing microservices.
- Keep CSV as the default runtime repository until a Postgres migration/import is
  explicitly run and verified.
- Add PostgreSQL/PostGIS migration SQL, but do not run migrations in this batch.
- Add `pg` as the backend Postgres driver dependency.
- Preserve legacy API-compatible POI shape through a `PostgresPoiRepository`
  mapper so old endpoints can switch repositories later without response-field
  churn.
- Treat the Phase 0 canonical CSV as a legacy-canonical import source that
  produces Bronze source records, Gold POI entities, external IDs, aliases,
  image records, and review summaries.
- Keep missing freshness, address, rating, review, and admin-boundary values as
  null/unknown in the import plan.
- Do not auto-merge duplicate candidates in this batch.
- Use Docker Compose with `postgis/postgis:16-3.5-alpine` and port `55432` for
  disposable Postgres/PostGIS verification only.
- Require `URBANAGENT_ALLOW_PHASE1_DB_WRITE=true` for migration, rollback, and
  importer write mode.
- Restrict guarded Phase 1 DB writes to disposable localhost `55432` database
  names containing `phase1`, `test`, or `disposable`.
- Keep importer dry-run as the default; real write mode requires `--write`.
- Do not drop PostGIS or pgcrypto in rollback because extensions may be shared
  in non-disposable databases.
- Use `(poi_id, provider, external_id)` as the external ID primary key so
  identical ID values in different namespaces and duplicate row prevention are
  explicit without implying RestaurantID is a Google Place ID.
- Add unique image association indexes to keep second imports idempotent.
- Treat repeated import runs as new `ingestion_runs`; core entity/source/image/
  alias/review counts must remain unchanged.
- Close the Phase 1 Batch 2 security gate by documenting npm CLI audit failure
  and classifying production advisories through the official npm Bulk Advisory
  POST endpoint with gzip fallback decoding.
- Do not change dependencies during audit closure; direct `pg@8.22.0` remains
  accepted for Phase 1 because no production advisory affects the `pg` path.
- After explicit user approval for Phase 1 security remediation, apply targeted
  production dependency remediation instead of `npm audit fix`, broad updates,
  or dependency dedupe.
- Update direct production dependencies `firebase-admin` to `^14.2.0` and
  `multer` to `^2.2.0`.
- Use explicit npm `overrides` for production transitive advisories:
  `body-parser@2.3.0`, `brace-expansion@5.0.8`, `form-data@2.5.6`,
  `protobufjs@7.6.5`, `qs@6.15.3`, and `uuid@11.1.1`.
- Add three narrow integrity overrides, `call-bind-apply-helpers@1.0.2`,
  `get-intrinsic@1.3.0`, and `hasown@2.0.4`, only to make npm's full
  production dependency tree validation reproducible and free of
  `ELSPROBLEMS`.
- Keep `pg@8.22.0` unchanged because no production advisory affects the new
  PostgreSQL path.
- Treat the dependency integrity gate as passed only after
  `npm.cmd ci --ignore-scripts`, `npm.cmd ls --omit=dev --all`,
  `npm.cmd audit --omit=dev`, backend tests, syntax checks, and Firebase
  Admin/Multer smoke all pass.
- Start Phase 1 Batch 3 only after explicit user approval on 2026-07-26.
- Limit Batch 3 to endpoint-level PostgreSQL runtime-switch smoke coverage and
  a local disposable PostGIS runbook.
- Keep `src/server.js` runtime behavior intact for Batch 3; use a child-process
  smoke test instead of refactoring the server export shape.
- Run Batch 3 endpoint smoke only inside the guarded disposable DB integration
  path with `URBANAGENT_PHASE1_INTEGRATION=true`.
- Continue to require `URBANAGENT_POI_REPOSITORY=postgres` for PostgreSQL API
  runtime tests; CSV remains default when the env var is absent.
- Cover existing endpoint compatibility before adding API v2 behavior:
  `/api/eda`, `/api/pois/data-quality`, `/api/agent/recommend-poi`, and
  `/api/agent/create-itinerary`.

## Phase 2 Planning Decisions

- Limit this batch to planning/specification documentation only.
- Phase 2 scope is backend Traveler API v2.
- Do not modify application code, tests, migrations, package files, canonical
  CSV, manifest, context files, AGENTS files, or frontend source in this
  planning batch.
- Treat all new `/api/v2` endpoint definitions as `PROPOSED FOR PHASE 2` until
  a later implementation batch adds and tests them.
- Preserve all legacy endpoints until equivalent v2 endpoints pass tests.
- Keep CSV as default runtime and PostgreSQL as explicit opt-in during Phase 2.
- Do not perform production database migration/import in Phase 2 planning.
- Keep partner/seller/admin product behavior outside Traveler API v2 scope.
- Defer frontend traveler rebuild to Phase 3.
- Defer City Pack automation/second city to Phase 4+.
- Defer monetization and partner product work to later approved phases.
- Use the new Phase 2 planning artifacts as the review basis:
  - `docs/rebuild/PHASE2_TRAVELER_API_V2_SCOPE.md`
  - `docs/rebuild/PHASE2_TRAVELER_API_V2_CONTRACT_DRAFT.md`
  - `docs/rebuild/PHASE2_TRAVELER_API_V2_EVALUATION_PLAN.md`
- API base path is `/api/v2`.
- Trip resource name is `/trips`.
- Standalone `POST /api/v2/recommendations` endpoint is accepted.
- Guest preview is stateless and non-persistent by default.
- Raw internal scoring signals are not public API contract.
- Public recommendations expose `score`, `reason`, `reasonCodes`,
  `warnings`, and POI provenance.
- `cityId` is required for city-scoped endpoints.
- Only `GET /api/v2/cities` is not city-scoped.
- Da Nang is the only supported Phase 2 city.
- Explicit unknown city returns `CITY_NOT_SUPPORTED`; no silent Da Nang
  fallback is allowed for invalid `cityId`.
- Public v2 metadata must not expose local dataset paths, database table names,
  SQL details, `DATABASE_URL`, repository class names, or storage-mode
  dependencies.
- Public contract metadata uses `contractVersion`, not a fake hash. The Phase
  2 draft contract version is `phase2-traveler-api-v2-draft-1`.
- OpenAPI 3.1 is `APPROVED AND REQUIRED` in Phase 2 Batch 1.
- Batch 1 must create a draft OpenAPI artifact covering only approved Batch 1
  endpoints and common schemas.
- `openApiSha256` may be recorded only after the OpenAPI artifact exists and
  the SHA-256 is calculated from that artifact.
- Common v2 response metadata is minimal: every response requires `apiVersion`
  and `requestId`; city-scoped responses also require `cityId`.
- `datasetVersion` appears only where lineage context is relevant.
- `applicationPoiCount`, `qualitySummary`, and `capabilityStatus` primarily
  belong in `GET /api/v2/cities/:cityId/status`.
- Unknown values are represented by `null` and/or explicit status fields, not
  empty strings or zero.
- Route summaries must mark partial/unknown routes with known/unknown leg
  counts and must not imply zero distance/time.
- Rating contract separates normalized scale-5 rating, Google scale-5 source
  rating/count, Foody scale-10 source rating, nullable review count, and sample
  review data.
- Source identifiers are typed objects with namespace/value/source;
  `RestaurantID` is not a Google Place ID.
- Capabilities use states such as `unavailable`, `planned`, `experimental`, or
  `available`; no capability is public `true` before implementation and
  validation.
- Phase 2 core approved scope is limited to cities, city status, POI
  search/detail, recommendations, and stateless trip preview.
- Trip persistence/edit/replan/stop edit/feedback persistence routes are
  `CONDITIONAL - NOT APPROVED FOR IMPLEMENTATION`.
- Request IDs are generated server-side when absent; optional `X-Request-Id`
  requires validation and must be returned in responses/logs without logging
  tokens or sensitive payloads.
- Accepted `X-Request-Id` length is 1 to 128 characters and may contain ASCII
  letters, digits, dot, underscore, colon, and hyphen.
- Missing or invalid `X-Request-Id` generates a new server requestId, does not
  echo the invalid value, continues the request, and must not reject an
  otherwise valid API request.
- Conditional authenticated routes use Firebase Bearer tokens; local dev
  fallback must never work in production and tests must not contact Firebase
  production.
- Batch 1 owns all POI search pagination behavior: limit validation, default
  limit 20, maximum limit 100, opaque cursor, cursor validation, deterministic
  search/list sorting, canonical `Global_ID` final tie-break, `total`,
  `nextCursor`, CSV-default endpoint behavior, and OpenAPI contract.
- Batch 2 owns recommendation v2, recommendation deterministic ranking,
  reasonCodes, recommendation fixture foundation, and quality evaluation
  preparation.
- Legacy versus v2 is used for behavioral parity evaluation, not as separate
  ranking algorithms.
- Recommendation-quality evaluation compares B0 current `recommendPOIs`
  against B1 category-only, B2 rating/popularity, and approved ablations.
- Capability transition rule: planning examples may show `planned`; after an
  endpoint is implemented and validated it may become `experimental` for
  local/test status, and becomes `available` only when approved environment
  criteria pass.

## Phase 2 Batch 1 Implementation Decisions

- Implement only the approved read-only Traveler API v2 foundation endpoints in
  this batch:
  - `GET /api/v2/cities`
  - `GET /api/v2/cities/:cityId/status`
  - `GET /api/v2/pois/search`
  - `GET /api/v2/pois/:poiId`
- Mount the v2 router only in the primary backend runtime `src/server.js`.
- Keep legacy routes unchanged and keep CSV as the default POI runtime.
- Keep PostgreSQL explicit opt-in; do not switch default runtime behavior.
- Use shared Traveler API v2 helpers for response envelopes, request ID
  resolution, pagination, POI search, and POI serialization.
- Accept valid `X-Request-Id` values with ASCII letters, digits, dot,
  underscore, colon, and hyphen, length 1 to 128 characters.
- Replace missing or invalid request IDs with generated server request IDs;
  never echo invalid caller values and do not reject otherwise valid requests
  for tracing-header issues.
- Use `CITY_NOT_SUPPORTED` for unsupported explicit `cityId`; do not silently
  fallback to Da Nang.
- Require `cityId` for all city-scoped v2 endpoints except `GET /api/v2/cities`.
- Implement POI search pagination with default limit `20`, maximum limit `100`,
  opaque base64url cursors, cursor validation, deterministic sorting, and
  canonical `Global_ID` tie-break behavior.
- Expose POI provenance through typed `sourceIdentifiers` and do not expose
  ambiguous legacy `placeId` or raw `sourceIds` in Traveler API v2 POI
  responses.
- Mark implemented Batch 1 capabilities as `experimental`, planned future
  Traveler API v2 capabilities as `planned`, and unavailable external/live or
  persistence capabilities as `unavailable`.
- Create the OpenAPI 3.1 draft artifact at
  `docs/rebuild/PHASE2_TRAVELER_API_V2_OPENAPI_DRAFT.json`.
- Record the real OpenAPI SHA-256 only after generating the artifact:
  `58599da0dd29023c5d25eee1fc74da7f52339d3e131d6d5542344974b6577a9b`.
- Do not implement recommendation v2, itinerary preview v2, persistence,
  edit/replan, feedback persistence, frontend changes, production DB work, or
  Phase 2 Batch 2 in this batch.

## Phase 2 Proposals Pending User Acceptance

- Add versioned backend traveler endpoints under `/api/v2` in separately
  approved implementation batches.
- Add core city/status, POI search/detail, recommendation, and stateless
  itinerary preview endpoints first.
- Keep trip persistence/edit/replan/stop edit/feedback persistence as
  conditional contract drafts only until persistence approval.
- Use a stable v2 response envelope with `ok`, `data` or `error`, and `meta`.
- Return `CITY_NOT_SUPPORTED` for unknown city IDs instead of silently falling
  back to Da Nang.
- Add machine-readable reason/warning codes while preserving display reasons.
- Implement Phase 2 in small batches with legacy compatibility tests after each
  batch.
- Create the required OpenAPI 3.1 specification in Batch 1.

## Still Open

- Decide whether restored root CSV Git LFS pointer files should remain tracked for legacy compatibility in the eventual Phase 0 commit.
- Decide whether `.gitignore` changes from the earlier Phase 0 patch should be kept.
- Decide when to add a real frontend test runner.
- Decide whether Phase 1 should restore road-name density through verified address/admin data or leave route density proximity-only until PostGIS.
- Decide whether conditional trip persistence belongs in Phase 2.
- If persistence is approved, decide whether Firestore remains the persistence
  mechanism.
- Decide whether a Firebase emulator is required for conditional persistence
  tests.
- Decide what process and reviewers will create the curated query fixture.

## Phase 2 Batch 2 Implementation Decisions

- Implement only the approved standalone Traveler API v2 recommendation
  endpoint:
  - `POST /api/v2/recommendations`
- Keep the existing `recommendPOIs` algorithm as B0/current behavior. Batch 2
  wraps it in a public v2 contract and does not claim recommendation-quality
  superiority.
- Require `cityId` in the recommendation request body. Missing `cityId`
  returns `VALIDATION_ERROR`; unsupported explicit `cityId` returns
  `CITY_NOT_SUPPORTED`.
- Keep Da Nang as the only supported Phase 2 city and keep CSV as the default
  runtime.
- Use recommendation limit default `5` and maximum `20` for the v2 endpoint.
- Public recommendation items expose only `poi`, `score`, `reason`,
  `reasonCodes`, `warnings`, and `provenance`.
- Do not expose raw internal scoring signals, `scoreRaw`, ambiguous
  `sourceIds`, or `placeId` in Traveler API v2 recommendation responses.
- Apply deterministic public ordering by score descending, normalized name
  ascending, and canonical `Global_ID` ascending.
- Add a curated smoke/evaluation fixture foundation for `quan cafe yen tinh`.
  The fixture is not a scored relevance benchmark and must not be used for
  quality superiority claims.
- Mark `recommendations` capability as `experimental` after implementation and
  validation. Keep `tripPreview` as `planned`; persistence/edit/replan/feedback
  remain `unavailable`.
- Do not implement itinerary preview v2, trip persistence, frontend changes,
  PostgreSQL default runtime, production database work, Firebase production
  access, Phase 2 Batch 3, or later-phase work in Batch 2.

## Decision - Future Multi-Source POI Architecture

Date: 2026-07-31.

Status: `APPROVED AS DOCUMENTED FUTURE DIRECTION`.

UrbanAgent will retain the current 4166-POI Da Nang canonical dataset as the
fixed Phase 2 runtime and evaluation baseline.

Future data expansion will use a layered architecture:

1. approved persistent open sources,
2. licensed or request-time commercial enrichment,
3. UrbanAgent-owned local enrichment,
4. evidence-backed derived intelligence.

The existing application `Global_ID` remains the legacy canonical POI key.

External identifiers remain namespaced source identifiers and do not create
independent product POIs without entity resolution.

No external source may be ingested without license, storage, display,
redistribution, attribution, deletion, refresh, cost, security, and quality
review.

Competitor products, including Wanderlog, Mindtrip, and Layla, are approved
for benchmarking only and are prohibited as scraped POI sources.

Phase 2 must not:

- change the canonical dataset,
- change the 4166 application POI baseline,
- add a second city,
- introduce a source ingestion pipeline,
- claim global coverage.

Detailed governance:

- `docs/rebuild/MULTI_SOURCE_POI_STRATEGY.md`
- `docs/rebuild/DATA_SOURCE_LICENSE_POLICY.md`

Implementation requires a separate approval:

`APPROVED MULTI-SOURCE POI SPIKE`

Mobile product design remains deferred and is not part of this decision.
