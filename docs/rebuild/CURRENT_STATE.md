# Current State

Updated: 2026-07-26 13:28:46 +07:00.

## Phase

`PHASE_1_SECURITY_REMEDIATION_PASSED`

Phase 1 Batch 2 validated the PostgreSQL/PostGIS schema and importer against a
disposable Docker Compose PostGIS database. Migration, rollback, reapply, real
write import, idempotency, diagnostics, Postgres repository integration, and
CSV/Postgres parity passed. No production database or Firebase data was touched.

The Batch 2 security remediation gate is now closed. Targeted production
dependency remediation was applied without `npm audit fix`, broad dependency
updates, application behavior changes, production database access, Firebase
access, or Phase 1 Batch 3 work.

Security remediation result:

- `npm.cmd audit --omit=dev`: PASS, `0 vulnerabilities`.
- `npm.cmd audit --omit=dev --json`: PASS, total vulnerabilities `0`.
- `pg@8.22.0`: retained; no production advisory found.
- Direct dependency updates:
  - `firebase-admin`: `^14.0.0` -> `^14.2.0`.
  - `multer`: `^2.1.1` -> `^2.2.0`.
- Production overrides:
  - `body-parser@2.3.0`
  - `brace-expansion@5.0.8`
  - `call-bind-apply-helpers@1.0.2`
  - `form-data@2.5.6`
  - `get-intrinsic@1.3.0`
  - `hasown@2.0.4`
  - `protobufjs@7.6.5`
  - `qs@6.15.3`
  - `uuid@11.1.1`
- Dependency integrity: `npm.cmd ci --ignore-scripts` PASS.
- Clean-clone `npm.cmd ci` without flags: PASS.
- `npm.cmd ls --omit=dev --all`: PASS, no `ELSPROBLEMS`.
- Firebase Admin and Multer compatibility smoke: PASS without connecting to
  production Firebase.
- Final disposable PostGIS integration rerun: PASS, 16 tests passed, 0 failed,
  0 skipped.
- Disposable container and volume cleanup after final gate: PASS.

## Repository Visibility

- Backend repo visible: `D:\POI-urban-danang-BE`
- Frontend repo visible: `D:\POI-urban-danang-FE`

## Branches And Working Tree

Backend branch: `phase1/data-platform-foundation`.

Backend status after security remediation:

- modified: `package.json`
- modified: `docs/rebuild/CURRENT_STATE.md`
- modified: `docs/rebuild/DECISIONS.md`
- modified: `docs/rebuild/TEST_REPORT.md`
- modified: `docs/rebuild/WORKLOG.md`

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

- `npm.cmd test`: PASS, 16 tests total, 15 passed, 0 failed, 1 skipped
  when the disposable Postgres database is absent.
- `node --check` for Phase 1 files and server entrypoints: PASS.
- `npm.cmd run phase1:import:canonical`: PASS in dry-run mode.
- `npm.cmd test` with `URBANAGENT_PHASE1_INTEGRATION=true`: PASS, 16 tests passed, 0 failed, 0 skipped.
- `npm.cmd run phase1:db:diagnostics`: PASS.
- `npm.cmd audit --omit=dev`: PASS, `0 vulnerabilities`.
- `npm.cmd audit --omit=dev --json`: PASS, total vulnerabilities `0`.
- Security remediation: PASS; previous 9 production advisory records are no
  longer present in npm audit output.
- `npm.cmd ci --ignore-scripts`: PASS; lockfile is reproducible from a clean
  install.
- Clean clone at `C:\tmp\urbanagent-phase1-readiness-clone`:
  - `git lfs pull`: PASS.
  - `npm.cmd ci` without `--ignore-scripts`, `--force`, or
    `--legacy-peer-deps`: PASS.
  - `git status --short --branch`: clean before and after checks.
  - `npm.cmd ls --omit=dev --all`: PASS.
  - `npm.cmd audit --omit=dev`: PASS, `0 vulnerabilities`.
  - `npm.cmd audit --omit=dev --json`: PASS, total vulnerabilities `0`.
  - `npm.cmd test`: PASS, 16 tests total, 15 passed, 0 failed, 1 skipped
    because the disposable DB is absent.
- `npm.cmd ls --omit=dev --all`: PASS; no `ELSPROBLEMS`.
- Firebase Admin and Multer smoke: PASS; package entrypoints used by
  UrbanAgent load without initializing production Firebase.
- Full disposable PostGIS integration: PASS, 16 passed, 0 failed, 0 skipped.
- Full integration diagnostics: PASS, POI entities `4166`, source records
  `4166`, external IDs `8337`, aliases `985`, images `16246`, review summaries
  `4166`, duplicate checks `0`, orphan checks `0`, geometry checks `0`, GiST
  index `poi_entities_location_gix`.
- Canonical SHA-256 recheck: PASS,
  `5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae`.
- CSV default runtime recheck: PASS, `4166` POIs with no
  `URBANAGENT_POI_REPOSITORY` override.
- Disposable cleanup: PASS; no `urbanagent-phase1-postgis` container or
  `urbanagent_phase1_postgis_data` volume remains.
- Branch synchronization: PASS, local `HEAD` equals
  `origin/phase1/data-platform-foundation`.
- PR unmerged by Git state: PASS, `HEAD` is not an ancestor of `origin/main`.

Frontend:

- `npm.cmd run build`: PASS.
- Vite warning remains: one JS chunk is `1,368.16 kB`, above the 500 kB warning threshold.
- Frontend lint and missing frontend test script remain accepted pre-existing debt from Phase 0.

## Remaining Risks

- The disposable write importer was verified, but production migration/import has not been approved or run.
- The optional Postgres repository adapter is still not enabled by default.
- Address/admin-boundary spatial joins remain pending because no boundary dataset has been approved.
- No `npm audit fix`, `npm update`, `npm upgrade`, or `npm dedupe` was run.
- Phase 1 is not complete; this gate closes Batch 1-2 dependency integrity and
  draft PR readiness validation only.

## Next Step

Review the Phase 1 Batch 1-2 pull request with the security remediation
included. Do not cut over runtime to PostgreSQL and do not start Batch 3
without separate explicit approval.
