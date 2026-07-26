# Current State

Updated: 2026-07-26 12:31:18 +07:00.

## Phase

`PHASE_1_DATA_PLATFORM_FOUNDATION_BATCH_2_PASSED`

Phase 1 Batch 2 validated the PostgreSQL/PostGIS schema and importer against a
disposable Docker Compose PostGIS database. Migration, rollback, reapply, real
write import, idempotency, diagnostics, Postgres repository integration, and
CSV/Postgres parity passed. No production database or Firebase data was touched.

The Batch 2 security audit gate is now closed. `npm audit --omit=dev` still
fails in this environment because npm 11.6.2 receives a gzip/binary advisory
response and tries to parse it as plain JSON, but the advisory payload was
successfully classified through the official npm Bulk Advisory POST endpoint
with explicit gzip fallback decoding. No `npm audit fix` was run.

## Repository Visibility

- Backend repo visible: `D:\POI-urban-danang-BE`
- Frontend repo visible: `D:\POI-urban-danang-FE`

## Branches And Working Tree

Backend branch: `phase1/data-platform-foundation`.

Backend status after implementation:

- modified: `docs/rebuild/CURRENT_STATE.md`
- modified: `docs/rebuild/DECISIONS.md`
- modified: `docs/rebuild/TEST_REPORT.md`
- modified: `docs/rebuild/WORKLOG.md`
- modified: `package-lock.json`
- modified: `package.json`
- modified: `src/services/poiRepository.js`
- untracked: `docker-compose.phase1.yml`
- untracked: `migrations/`
- untracked: `scripts/phase1_db_diagnostics.js`
- untracked: `scripts/phase1_db_migrate.js`
- untracked: `scripts/phase1_db_rollback.js`
- untracked: `scripts/phase1_import_canonical_pois.js`
- untracked: `src/infrastructure/`
- untracked: `src/modules/`
- untracked: `tests/phase1/`

Frontend branch: `main`.

Frontend status:

- clean: `## main...origin/main`
- No frontend application source file was changed.

## Canonical Dataset Status

- Path: `data/canonical/urbanagent_poi_master_v1.csv`
- Rows: `4166`
- Unique `Global_ID`: `4166`
- SHA-256: `5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae`
- Runtime repository load count: `4166`
- Header contract: pass
- City filter `da-nang`: `4166`

## Phase 1 Data Platform Status

- Migration file exists: `migrations/phase1/001_core_postgis_schema.sql`.
- Rollback file exists: `migrations/phase1/001_core_postgis_schema.down.sql`.
- Disposable Compose file exists: `docker-compose.phase1.yml`.
- Migration declares PostGIS and pgcrypto extensions.
- Core tables defined: `cities`, `ingestion_runs`, `poi_entities`,
  `poi_source_records`, `poi_external_ids`, `poi_aliases`, `poi_images`,
  `poi_reviews_summary`, `poi_merge_candidates`, `data_quality_issues`.
- Da Nang City Pack config exists in `src/modules/cities/cityConfig.js`.
- Canonical import dry-run produces:
  - application POIs: `4166`
  - source records: `4166`
  - external IDs: `8337`
  - aliases: `985`
  - images: `16246`
  - review summaries: `4166`
- Existing runtime still defaults to CSV through the repository adapter.
- Postgres runtime can be selected later with `URBANAGENT_POI_REPOSITORY=postgres`
  after migration/import setup.
- Disposable database method used: Docker Compose with
  `postgis/postgis:16-3.5-alpine`, host port `55432`.
- Migration apply: PASS.
- First real write import: PASS.
- Second real write import: PASS; core counts unchanged.
- Rollback: PASS; Phase 1 tables removed.
- Reapply and final import: PASS.
- Final diagnostics:
  - cities: `1`
  - POI entities: `4166`
  - source records: `4166`
  - external IDs: `8337`
  - aliases: `985`
  - images: `16246`
  - review summaries: `4166`
  - duplicate checks: `0`
  - orphan checks: `0`
  - invalid longitude/latitude: `0`
  - null geometry: `0`
  - wrong SRID: `0`
  - coordinate mismatch: `0`
  - outside Da Nang envelope: `0`
  - merged provenance rows: `5`
  - GiST index: `poi_entities_location_gix`
- Disposable container and volume cleanup: PASS; no Phase 1 container or volume remains.

## Build/Test Status

Backend:

- `npm.cmd test`: PASS, 15 tests passed, 0 failed.
- `node --check` for Phase 1 files and server entrypoints: PASS.
- `npm.cmd run phase1:import:canonical`: PASS in dry-run mode.
- `npm.cmd test` with `URBANAGENT_PHASE1_INTEGRATION=true`: PASS, 16 tests passed, 0 failed, 0 skipped.
- `npm.cmd run phase1:db:diagnostics`: PASS.
- `npm.cmd audit --omit=dev`: FAIL due npm registry invalid JSON/gzip response.
- Audit closure classifier: PASS through official npm Bulk Advisory POST.

Frontend:

- `npm.cmd run build`: PASS.
- Vite warning remains: one JS chunk is `1,368.16 kB`, above the 500 kB warning threshold.
- Frontend lint and missing frontend test script remain accepted pre-existing debt from Phase 0.

## Remaining Risks

- `npm audit --omit=dev` still fails in this environment; use the documented
  official Bulk Advisory POST diagnostic as the current production-only
  classification until npm CLI/registry decompression behavior is fixed.
- Production-only advisory classification: 9 advisory records, 1 low, 4
  moderate, 4 high; 2 direct and 7 transitive; 0 affect the new `pg` path.
- The disposable write importer was verified, but production migration/import has not been approved or run.
- The optional Postgres repository adapter is still not enabled by default.
- Address/admin-boundary spatial joins remain pending because no boundary dataset has been approved.
- Dependency install reported `12 vulnerabilities`; no `npm audit fix` was run because that would be a broader dependency change.
- Phase 1 is not complete; this is only the first foundation batch.

## Next Step

Review Batch 2 results, then decide whether Batch 3 should add endpoint-level
repository-switch tests, rollback packaging, or a local-only migration runbook.
Do not cut over runtime to PostgreSQL without a separate explicit approval.
