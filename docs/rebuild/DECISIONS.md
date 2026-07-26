# Decisions

Updated: 2026-07-26 01:49:31 +07:00.

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
- Phase 1 remains blocked until explicit approval.

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
- Keep `pg@8.22.0` unchanged because no production advisory affects the new
  PostgreSQL path.
- Accept `npm.cmd ls --omit=dev --all` `ELSPROBLEMS` override-tree warnings as
  a remaining review item because `npm audit --omit=dev`, direct dependency
  listing, syntax checks, and tests pass.

## Still Open

- Decide whether restored root CSV Git LFS pointer files should remain tracked for legacy compatibility in the eventual Phase 0 commit.
- Decide whether `.gitignore` changes from the earlier Phase 0 patch should be kept.
- Decide when to add a real frontend test runner.
- Decide whether Phase 1 should restore road-name density through verified address/admin data or leave route density proximity-only until PostGIS.
- Decide whether the next Phase 1 batch should run a local Postgres/PostGIS
  verification environment, add rollback SQL first, or add endpoint-level tests
  for repository switching first.
- Track npm override-tree validation warnings under `get-intrinsic` /
  `call-bind-apply-helpers` as a residual package-manager review item before
  production release.
