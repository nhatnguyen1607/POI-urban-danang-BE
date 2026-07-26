# Worklog

This file is append-only. New entries must be appended below previous entries.

## 2026-07-25 Audit

Mode: planning-only audit. No application source code was changed.

Files read:

- `D:\POI-urban-danang-BE\AGENTS.md`
- `D:\POI-urban-danang-FE\AGENTS.md`
- `D:\POI-urban-danang-BE\URBANAGENT_CODEX_CONTEXT.md`
- `D:\POI-urban-danang-FE\URBANAGENT_CODEX_CONTEXT.md`
- `D:\POI-urban-danang-BE\PLANNING.md`
- `D:\POI-urban-danang-FE\PLANNING.md`
- `D:\POI-urban-danang-BE\README.md`
- `D:\POI-urban-danang-FE\README.md`
- `D:\POI-urban-danang-BE\package.json`
- `D:\POI-urban-danang-FE\package.json`
- `D:\POI-urban-danang-BE\src\server.js`
- `D:\POI-urban-danang-BE\src\services\poiDataService.js`
- `D:\POI-urban-danang-BE\src\services\poiRetrievalService.js`
- `D:\POI-urban-danang-BE\src\services\itineraryPlannerService.js`
- `D:\POI-urban-danang-BE\src\services\routeMatrixService.js`
- `D:\POI-urban-danang-BE\src\services\weatherService.js`
- `D:\POI-urban-danang-BE\src\services\intentService.js`
- `D:\POI-urban-danang-BE\src\services\businessLocationScorer.js`
- `D:\POI-urban-danang-BE\src\services\firestorePersistenceService.js`
- `D:\POI-urban-danang-BE\src\services\contextualPoiRecommenderService.js`
- `D:\POI-urban-danang-BE\src\expert_system\poi_density_engine.js`
- `D:\POI-urban-danang-BE\src\inference.py`
- `D:\POI-urban-danang-FE\src\App.tsx`
- `D:\POI-urban-danang-FE\src\utils\apiClient.ts`
- `D:\POI-urban-danang-FE\src\services\poiExperienceService.ts`
- selected read-only snippets from `D:\POI-urban-danang-FE\src\pages\urban-agent\UrbanAgentPage.tsx`
- selected read-only snippets from `D:\POI-urban-danang-FE\src\pages\urban-agent\PoiExperienceLayer.tsx`

Read-only commands run:

```text
Get-ChildItem -Force -LiteralPath D:\POI-urban-danang-BE
Get-ChildItem -Force -LiteralPath D:\POI-urban-danang-FE
Get-Content -LiteralPath ...\AGENTS.md
Get-FileHash -Algorithm SHA256 -LiteralPath ...\URBANAGENT_CODEX_CONTEXT.md
Get-Content -LiteralPath ...\URBANAGENT_CODEX_CONTEXT.md
Get-Content -LiteralPath ...\PLANNING.md
Get-Content -LiteralPath ...\README.md
Get-Content -LiteralPath ...\package.json
rg --files
rg -n "app\.(get|post|put|patch|delete)\("
rg -n "16\.0544|108\.2022|Da Nang|Danang|Đà|DANANG|da nang|poi_data_|csv|CSV"
Get-ChildItem -Recurse -File -LiteralPath ...\data
git status --short --branch
Test-Path -LiteralPath ...\data\raw\legacy\poi_data_ggmap.csv
Test-Path -LiteralPath ...\data\raw\legacy\poi_data_foody.csv
python -c <read-only CSV schema/stat audit>
npm test
npm.cmd test
npm run lint
npm.cmd run lint
```

Write commands run:

```text
New-Item -ItemType Directory -Force -Path D:\POI-urban-danang-BE\docs\rebuild
apply_patch to add docs/rebuild/*.md
```

Key conclusions:

- Both repositories are visible.
- BE and FE context files are identical by SHA-256.
- BE context is canonical per both `AGENTS.md` files.
- Exact requested raw legacy paths `data/raw/legacy/poi_data_*.csv` do not exist.
- Runtime backend reads `data/poi_data_ggmap.csv` and `data/poi_data_foody.csv`.
- Existing raw legacy files are named `master_nodes_google_maps_clean.csv` and `master_nodes_foody_clean.csv`.
- Multiple backend and frontend paths hard-code Da Nang center.
- Multiple backend paths fabricate Da Nang-center coordinates when lat/lon are missing.
- Runtime Google CSV has duplicate IDs and lacks address/review-count columns.
- FE lint currently fails with 119 errors and 3 warnings.

## 2026-07-26 Phase 0 Canonical Loader Fix Batch

Mode: Phase 0 blocker fix only. No Phase 1 work, migration, dependency install, commit, push, Firebase production data change, canonical CSV edit, manifest edit, or dataset decision edit was performed.

Files read:

- `D:\POI-urban-danang-BE\AGENTS.md`
- `D:\POI-urban-danang-FE\AGENTS.md`
- `D:\POI-urban-danang-BE\URBANAGENT_CODEX_CONTEXT.md`
- `D:\POI-urban-danang-FE\URBANAGENT_CODEX_CONTEXT.md`
- `D:\POI-urban-danang-BE\docs\rebuild\URBANAGENT_DATASET_DECISION.md`
- `D:\POI-urban-danang-BE\docs\rebuild\MASTER_PLAN.md`
- `D:\POI-urban-danang-BE\docs\rebuild\DATA_AUDIT.md`
- `D:\POI-urban-danang-BE\docs\rebuild\CURRENT_STATE.md`
- `D:\POI-urban-danang-BE\docs\rebuild\DECISIONS.md`
- `D:\POI-urban-danang-BE\docs\rebuild\WORKLOG.md`
- `D:\POI-urban-danang-BE\docs\rebuild\TEST_REPORT.md`
- `D:\POI-urban-danang-BE\docs\rebuild\URBANAGENT_PHASE0_CONSOLIDATED_REVIEW.md`
- backend and frontend `package.json`
- current Phase 0 diffs and relevant backend services.

Commands run:

```text
Get-Content -LiteralPath ...
git status --short --branch --untracked-files=all
git diff --name-status
git diff --stat
rg -n ...
Get-FileHash -Algorithm SHA256 ...
node -e <canonical loader and quality diagnostic>
npm.cmd test
node --check src\server.js
node --check server.js
node --check src\services\canonicalCsvPoiRepository.js
node --check src\services\poiRepository.js
node --check src\services\poiDataService.js
node --check src\services\poiRetrievalService.js
node --check src\services\itineraryPlannerService.js
node --check src\services\firestorePersistenceService.js
node --check src\services\contextualPoiRecommenderService.js
node --check ES-system\poi_density_engine.js
node --check tests\phase0CanonicalData.test.js
npm.cmd run lint
D:\POI-urban-danang-FE\node_modules\.bin\tsc.cmd --noEmit --pretty false --project tsconfig.app.json
D:\POI-urban-danang-FE\node_modules\.bin\vite.cmd build --outDir C:\tmp\urbanagent-fe-build-phase0-fix --emptyOutDir
npm.cmd test
```

Write actions:

- Updated `src/services/canonicalCsvPoiRepository.js` to normalize CSV headers and strip UTF-8 BOM from the first header.
- Updated `ES-system/poi_density_engine.js` to load canonical POIs instead of deleted legacy root CSV files.
- Updated POI rating/review normalization in Firestore/contextual services to preserve unknown values as null.
- Added `tests/phase0CanonicalData.test.js`.
- Changed backend `package.json` test script to run Node built-in tests.
- Synced BE/FE `AGENTS.md` and `URBANAGENT_CODEX_CONTEXT.md` with the approved canonical dataset semantics.
- Updated rebuild docs: `MASTER_PLAN.md`, `CURRENT_STATE.md`, `DATA_AUDIT.md`, `DECISIONS.md`, and `TEST_REPORT.md`.

Key conclusions:

- Canonical CSV hash remains `5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae`.
- Context files are byte-identical after sync with SHA-256 `2EC5ACC2E8AF8B1553E94EB17A332C8C6D03675EF74D214DF4AD0D010D28BF0F`.
- `loadPOIs()` now returns 4,166 POIs.
- Data-quality report now has `headerMatchesExpected: true`, `applicationPois: 4166`, and `invalidRows: 0`.
- Backend tests pass 7/7.
- Frontend lint still fails with the existing `122 problems (119 errors, 3 warnings)`.
- Frontend type-check and temp build pass.
- Frontend test script is missing.

## 2026-07-26 Final Phase 0 Small Fixes

Mode: final small Phase 0 fixes and revalidation. No Phase 1 work, migration, dependency install, commit, push, Firebase production data change, canonical CSV edit, manifest edit, dataset decision edit, frontend redesign, or model artifact edit was performed.

Commands run:

```text
Get-Content -LiteralPath ...
git show HEAD:data/poi_data_ggmap.csv
git show HEAD:data/poi_data_foody.csv
git status --short -- data\poi_data_ggmap.csv data\poi_data_foody.csv
git diff -- data\poi_data_ggmap.csv data\poi_data_foody.csv
npm.cmd test
node --check src\services\canonicalCsvPoiRepository.js
node --check tests\phase0\phase0CanonicalData.test.js
node -e <canonical loader/imageUrls diagnostic>
node -e <recommendation smoke>
node -e <itinerary smoke>
rg -n ...
```

Write actions:

- Added `parseImageUrls()` in `src/services/canonicalCsvPoiRepository.js`.
- Exposed normalized `imageUrls` and compatibility `imageUrl`.
- Moved Phase 0 backend tests into `tests/phase0/phase0CanonicalData.test.js`.
- Updated backend `package.json` test script to `node --test tests/phase0/*.test.js`.
- Added `Image_URL` semantics tests.
- Added raw input immutability wording to frontend `AGENTS.md`.
- Restored root legacy CSV Git LFS pointer files so they are no longer marked deleted.

Key conclusions:

- Backend tests pass 8/8.
- Canonical loader still returns 4,166 POIs.
- Quality report still has `rows: 4166`, `applicationPois: 4166`, `invalidRows: 0`, and `headerMatchesExpected: true`.
- Recommendation smoke returns 3 results.
- Itinerary smoke returns 3 stops.
- Root legacy CSV paths are no longer marked deleted; they show modified with no textual diff because they were restored as LFS pointer files.

## 2026-07-26 Final Phase 0 Itinerary Unknown-Origin Correction

Mode: targeted Phase 0 semantic correction only. No Phase 1 work, commit, push, migration, canonical CSV edit, manifest edit, dataset decision edit, production Firebase data change, frontend redesign, or model artifact change was performed.

Files changed:

- `src/services/itineraryPlannerService.js`
- `tests/phase0/phase0CanonicalData.test.js`
- `docs/rebuild/TEST_REPORT.md`
- `docs/rebuild/WORKLOG.md`

Commands run:

```text
Get-Content -LiteralPath URBANAGENT_CODEX_CONTEXT.md
Get-Content -LiteralPath PLANNING.md
Get-Content -LiteralPath README.md
Get-Content -LiteralPath package.json
Get-Content -LiteralPath D:\POI-urban-danang-FE\PLANNING.md
Get-Content -LiteralPath D:\POI-urban-danang-FE\README.md
Get-Content -LiteralPath D:\POI-urban-danang-FE\package.json
npm.cmd test
node --check src\services\itineraryPlannerService.js
node --check tests\phase0\phase0CanonicalData.test.js
node -e <itinerary unknown-origin smoke>
```

Key conclusions:

- Backend tests pass 8/8.
- When itinerary start location is missing, the first stop now reports `distanceKm: null`, `estimatedMinutes: null`, `distanceKnown: false`, and `source: missing-origin`.
- Later itinerary legs still report numeric distance/time using local haversine estimates.
- The no-Da-Nang-center-fallback warning remains.

## 2026-07-26 Final Phase 0 Provenance And EDA Compatibility Fix

Mode: targeted Phase 0 semantic and compatibility fix only. No Phase 1 work, commit, push, migration, canonical CSV edit, manifest edit, dataset decision edit, production Firebase data change, frontend redesign, or model artifact change was performed.

Files changed:

- `src/services/firestorePersistenceService.js`
- `src/services/poiDataService.js`
- `src/server.js`
- `server.js`
- `tests/phase0/phase0CanonicalData.test.js`
- `docs/rebuild/TEST_REPORT.md`
- `docs/rebuild/WORKLOG.md`

Commands run:

```text
rg -n "api/eda|source.*foody|source.*google|normalizePoi|source: \[|google_maps\+foody|loadPOIs|getPoiDataQualityReport" src\server.js server.js src\services tests\phase0\phase0CanonicalData.test.js
Get-Content -LiteralPath src\services\firestorePersistenceService.js
Get-Content -LiteralPath src\server.js
Get-Content -LiteralPath server.js
Get-Content -LiteralPath src\services\poiDataService.js
npm.cmd test
node --check src\services\poiDataService.js
node --check src\services\firestorePersistenceService.js
node --check src\server.js
node --check server.js
node --check tests\phase0\phase0CanonicalData.test.js
node -e <provenance diagnostic>
node -e <EDA source-filter diagnostic>
```

Test output:

```text
tests 10
pass 10
fail 0
```

Diagnostics:

```text
provenance: { source: "google_maps+foody", preserved: true }
EDA undefined/missing/ggmap/google/google_maps/unknown: 3946
EDA foody: 225
EDA all/canonical: 4166
```

Key conclusions:

- `normalizePoi()` now preserves `source: "google_maps+foody"` instead of normalizing it to `manual`.
- EDA source filtering is shared by `server.js` and `src/server.js` through `poiDataService`.
- Google-compatible EDA views include `google_maps+foody`; Foody EDA views include `google_maps+foody`; all/canonical returns every canonical POI.

## 2026-07-26 Final Phase 0 Data Artifact Tracking Fix

Mode: data artifact tracking fix only. No application code, tests, canonical CSV contents, manifest contents, context files, AGENTS files, package files, dependencies, Firebase data, Git history, commit, push, or Phase 1 work was performed.

Files changed:

- `.gitignore`
- `.gitattributes`
- `docs/rebuild/URBANAGENT_PHASE0_CONSOLIDATED_REVIEW.md`
- `docs/rebuild/WORKLOG.md`

Commands run:

```text
rg -n "data/canonical/urbanagent_poi_master_v1\.csv|data/raw/legacy|canonical" .gitignore .gitattributes
git lfs env
Get-Content -LiteralPath .gitattributes
git check-attr filter -- data/canonical/urbanagent_poi_master_v1.csv
git check-ignore -v data/canonical/urbanagent_poi_master_v1.csv
Get-FileHash -Algorithm SHA256 -LiteralPath data\canonical\urbanagent_poi_master_v1.csv
git lfs track "data/canonical/urbanagent_poi_master_v1.csv"
npm.cmd test
```

Verification:

```text
git lfs env: git-lfs/3.5.1
git check-attr filter -- data/canonical/urbanagent_poi_master_v1.csv: filter: lfs
git check-ignore -v data/canonical/urbanagent_poi_master_v1.csv: no output
canonical CSV SHA-256: 5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae
git status --short -- data/canonical/urbanagent_poi_master_v1.csv .gitignore .gitattributes:
 M .gitattributes
 M .gitignore
?? data/canonical/urbanagent_poi_master_v1.csv
npm.cmd test: 10 pass, 0 fail
```

Key conclusions:

- Removed only the `.gitignore` line that hid `data/canonical/urbanagent_poi_master_v1.csv`.
- Raw/legacy input ignore rules remain unchanged.
- The canonical runtime CSV is now visible to Git and tracked through Git LFS attributes.
- The canonical CSV bytes were not changed; the approved SHA-256 still matches.
- A clean clone will receive the runtime dataset once Phase 0 is committed and pushed with LFS objects.

## 2026-07-26 Phase 1 Data Platform Foundation Batch 1

Mode: Phase 1 started after explicit user approval. Implemented backend data
foundation only. No Phase 2 work, frontend redesign, production Firebase change,
database migration execution, canonical CSV edit, or external data-source
integration was performed.

Changed files:

- `package.json`
- `package-lock.json`
- `migrations/phase1/001_core_postgis_schema.sql`
- `src/infrastructure/db/postgresClient.js`
- `src/modules/cities/cityConfig.js`
- `src/modules/pois/canonicalPoiImportPlan.js`
- `src/modules/pois/postgresPoiRepository.js`
- `src/services/poiRepository.js`
- `scripts/phase1_import_canonical_pois.js`
- `tests/phase1/phase1DataPlatform.test.js`
- `docs/rebuild/CURRENT_STATE.md`
- `docs/rebuild/DECISIONS.md`
- `docs/rebuild/TEST_REPORT.md`
- `docs/rebuild/WORKLOG.md`

Commands run:

```text
Get-Content -LiteralPath AGENTS.md
Get-Content -LiteralPath D:\POI-urban-danang-FE\AGENTS.md
Get-Content -LiteralPath URBANAGENT_CODEX_CONTEXT.md
Get-Content -LiteralPath docs\rebuild\MASTER_PLAN.md
Get-Content -LiteralPath PLANNING.md
Get-Content -LiteralPath README.md
Get-Content -LiteralPath package.json
Get-Content -LiteralPath docs\rebuild\CURRENT_STATE.md
Get-Content -LiteralPath D:\POI-urban-danang-FE\PLANNING.md
Get-Content -LiteralPath D:\POI-urban-danang-FE\package.json
rg --files src tests scripts config db migrations docs
git --no-pager status --short --branch
git switch -c phase1/data-platform-foundation
npm.cmd install
npm.cmd test
node --check src\infrastructure\db\postgresClient.js
node --check src\modules\cities\cityConfig.js
node --check src\modules\pois\canonicalPoiImportPlan.js
node --check src\modules\pois\postgresPoiRepository.js
node --check scripts\phase1_import_canonical_pois.js
node --check tests\phase1\phase1DataPlatform.test.js
node --check src\services\poiRepository.js
node --check src\server.js
node --check server.js
npm.cmd run phase1:import:canonical
cd D:\POI-urban-danang-FE
npm.cmd run build
```

Test output:

```text
tests 15
pass 15
fail 0
```

Import dry-run output:

```text
applicationPois: 4166
sourceRecords: 4166
externalIds: 8337
aliases: 985
images: 16246
reviewSummaries: 4166
```

Key conclusions:

- Added a self-contained Phase 1 schema migration with PostGIS, pgcrypto,
  cities, ingestion runs, canonical POIs, source records, external IDs, aliases,
  images, review summaries, merge candidates, and data quality issues.
- Added a Da Nang City Pack config while keeping a single approved city.
- Added a canonical legacy import plan that preserves approved CSV hash and null
  semantics without modifying the CSV.
- Added an optional Postgres POI repository adapter while keeping CSV as the
  default runtime repository.
- Added tests for migration shape, City Pack config, import planning, null
  semantics/provenance, and Postgres-to-legacy POI mapping.
- No migration or write import was run.

## 2026-07-26 Phase 1 Batch 2 Disposable Postgres/PostGIS Integration

Mode: Phase 1 Batch 2. Disposable database verification only. No production
database, Firebase production, frontend application code, canonical CSV bytes,
runtime cutover, commit, push, or Batch 3 work was performed.

Changed files:

- `docker-compose.phase1.yml`
- `migrations/phase1/001_core_postgis_schema.sql`
- `migrations/phase1/001_core_postgis_schema.down.sql`
- `package.json`
- `src/infrastructure/db/phase1MigrationRunner.js`
- `src/modules/pois/postgresDiagnostics.js`
- `src/modules/pois/postgresPoiRepository.js`
- `scripts/phase1_db_diagnostics.js`
- `scripts/phase1_db_migrate.js`
- `scripts/phase1_db_rollback.js`
- `scripts/phase1_import_canonical_pois.js`
- `tests/phase1/phase1PostgresIntegration.test.js`
- `docs/rebuild/CURRENT_STATE.md`
- `docs/rebuild/DECISIONS.md`
- `docs/rebuild/TEST_REPORT.md`
- `docs/rebuild/WORKLOG.md`

Commands run:

```text
git status --short --branch
git --no-pager diff --name-status
git --no-pager diff --stat
git ls-files --others --exclude-standard
git --no-pager log --oneline --decorate -5
docker --version
wsl.exe docker --version
wsl.exe docker compose -f docker-compose.phase1.yml version
node --check src\infrastructure\db\phase1MigrationRunner.js
node --check src\modules\pois\postgresDiagnostics.js
node --check scripts\phase1_db_migrate.js
node --check scripts\phase1_db_rollback.js
node --check scripts\phase1_db_diagnostics.js
node --check scripts\phase1_import_canonical_pois.js
node --check tests\phase1\phase1PostgresIntegration.test.js
npm.cmd test
wsl.exe docker compose -f docker-compose.phase1.yml up -d
wsl.exe docker inspect -f "{{.State.Health.Status}}" urbanagent-phase1-postgis
npm.cmd run phase1:db:migrate
npm.cmd run phase1:import:canonical -- --write
npm.cmd run phase1:db:diagnostics
npm.cmd run phase1:import:canonical -- --write
npm.cmd run phase1:db:diagnostics
npm.cmd run phase1:db:rollback
node -e <list Phase 1 tables after rollback>
npm.cmd run phase1:db:migrate
npm.cmd run phase1:import:canonical -- --write
npm.cmd run phase1:db:diagnostics
npm.cmd test with URBANAGENT_PHASE1_INTEGRATION=true
Get-FileHash -Algorithm SHA256 -LiteralPath data\canonical\urbanagent_poi_master_v1.csv
npm.cmd run phase1:import:canonical -- --dry-run
npm.cmd run phase1:db:diagnostics
node -e <CSV default runtime check>
node --check <Phase 1 scripts, services, and tests>
npm.cmd audit --omit=dev
npm.cmd audit --omit=dev --json
npm.cmd audit --omit=dev --registry=http://registry.npmjs.org
npm.cmd ls pg --omit=dev
npm.cmd ls --omit=dev --depth=0
wsl.exe docker compose -f docker-compose.phase1.yml down -v
wsl.exe docker ps -a --filter name=urbanagent-phase1-postgis
wsl.exe docker volume ls --filter name=urbanagent_phase1_postgis_data
```

Verification:

```text
Docker Compose image: postgis/postgis:16-3.5-alpine
Container health: healthy
Migration apply: PASS
First write import: PASS
Second write import: PASS
Rollback: PASS
Tables after rollback: []
Reapply migration: PASS
Final import: PASS
Full tests with integration: 16 pass, 0 fail, 0 skip
Canonical SHA-256: 5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae
CSV default runtime: CanonicalCsvPoiRepository, 4166 POIs
Container cleanup: PASS
Volume cleanup: PASS
```

Final diagnostics:

```text
poi_entities: 4166
source_records: 4166
external_ids: 8337
aliases: 985
images: 16246
review_summaries: 4166
duplicates: 0
orphans: 0
invalid longitude/latitude: 0
wrong SRID: 0
coordinate mismatch: 0
outside Da Nang envelope: 0
merged provenance rows: 5
GiST index: poi_entities_location_gix
```

NPM audit:

```text
npm.cmd audit --omit=dev: FAIL
npm.cmd audit --omit=dev --json: FAIL
npm.cmd audit --omit=dev --registry=http://registry.npmjs.org: FAIL, 426 Upgrade Required
```

Key conclusions:

- Disposable Postgres/PostGIS integration works end-to-end.
- Importer write mode is guarded and dry-run remains default.
- Repeated imports are idempotent for core tables; `ingestion_runs` increments
  as import history.
- Rollback removes Phase 1 tables in reverse dependency order without dropping
  extensions.
- Postgres repository returns API-compatible POIs and matches selected CSV
  records for Google-only, Foody-only, merged provenance, null rating, multiple
  images, and aliases.
- Application runtime remains CSV by default.
- Vulnerability classification is still unresolved because npm audit did not
  return a valid advisory payload.

## 2026-07-26 Phase 1 Batch 2 Security Audit Closure

Mode: security audit closure only. No source code, migration SQL, tests,
Docker Compose design, frontend files, canonical CSV, manifest, package.json,
package-lock.json, dependency install/update/dedupe, database workflow rerun,
Firebase access, commit, push, or Batch 3 work was performed.

Changed files:

- `docs/rebuild/CURRENT_STATE.md`
- `docs/rebuild/DECISIONS.md`
- `docs/rebuild/TEST_REPORT.md`
- `docs/rebuild/WORKLOG.md`

Commands run:

```text
node --version
npm.cmd --version
npm.cmd config get registry
npm.cmd config get userconfig
npm.cmd config get globalconfig
npm.cmd config get proxy
npm.cmd config get https-proxy
npm.cmd ping --registry=https://registry.npmjs.org/
npm.cmd audit --omit=dev --registry=https://registry.npmjs.org/
npm.cmd audit --omit=dev --json --registry=https://registry.npmjs.org/
npm.cmd audit --omit=dev --json --registry=https://registry.npmjs.org/ --userconfig=<empty temp npmrc>
npm.cmd audit --omit=dev --json --registry=https://registry.npmjs.org/ --cache=<temp cache> --prefer-online
wsl.exe sh -lc "npm audit --omit=dev --json --registry=https://registry.npmjs.org/"
npm.cmd config ls -l | rg -n "audit|compress|gzip|encoding|fetch|registry|proxy"
node <temp official npm bulk advisory classifier>
Remove-Item <temp audit artifacts>
```

Environment:

```text
Node: v24.11.1
npm: 11.6.2
registry: https://registry.npmjs.org/
proxy: null
https-proxy: null
npm ping: PASS
```

NPM CLI audit result:

```text
All npm CLI audit attempts failed with invalid JSON/gzip response parsing.
HTTP fallback failed with 426 Upgrade Required and was not used as audit result.
```

Official advisory classification:

```text
Source: npm Bulk Advisory POST endpoint
Response: HTTP 200, gzip decoded locally
Production packages checked: 252
Advisory records: 9
Severity: 1 low, 4 moderate, 4 high
Direct: 2
Transitive: 7
Affects new pg path: 0
```

Affected production modules:

```text
body-parser@2.2.2: low, transitive via express, pg path no
brace-expansion@2.1.1: high, transitive via firebase-admin/google stack, pg path no
brace-expansion@2.1.1: high, transitive via firebase-admin/google stack, pg path no
form-data@2.5.5: high, transitive via firebase-admin/google stack, pg path no
multer@2.1.1: high, direct, pg path no
multer@2.1.1: moderate, direct, pg path no
protobufjs@7.6.3: moderate, transitive via firebase-admin/google stack, pg path no
qs@6.15.1: moderate, transitive via express/body-parser, pg path no
uuid@9.0.1: moderate, transitive via firebase-admin/google stack, pg path no
```

Key conclusions:

- `npm audit --omit=dev` remains broken in this environment due gzip response
  parsing, despite registry ping success and isolated npm CLI attempts.
- A usable production-only advisory classification was obtained from the
  official npm Bulk Advisory POST endpoint.
- No advisory affects the newly added Phase 1 `pg@8.22.0` path.
- Existing advisories affect Express/Multer/Firebase dependency surfaces and
  remain unresolved because dependency changes were explicitly out of scope.
- Temporary audit script/cache files were removed after classification.

## 2026-07-26 Phase 1 Security Remediation

Time: 2026-07-26 12:52:16 +07:00.

Changed files:

```text
package.json
package-lock.json
docs/rebuild/CURRENT_STATE.md
docs/rebuild/DECISIONS.md
docs/rebuild/TEST_REPORT.md
docs/rebuild/WORKLOG.md
```

Commands run:

```text
git status --short --branch
git --no-pager diff --name-status
git --no-pager diff --stat
git --no-pager diff -- package.json package-lock.json
npm.cmd view multer version versions --json
npm.cmd view body-parser version versions --json
npm.cmd view qs version versions --json
npm.cmd view protobufjs version versions --json
npm.cmd view form-data version versions --json
npm.cmd view uuid version versions --json
npm.cmd view brace-expansion version versions --json
npm.cmd view firebase-admin version dependencies --json
npm.cmd install --ignore-scripts
npm.cmd ls body-parser brace-expansion form-data multer protobufjs qs uuid pg firebase-admin --omit=dev --all
npm.cmd audit --omit=dev
npm.cmd audit --omit=dev --json
npm.cmd test
node --check src\server.js
node --check server.js
node --check src\infrastructure\db\phase1MigrationRunner.js
node --check src\modules\pois\postgresPoiRepository.js
node --check scripts\phase1_import_canonical_pois.js
node --check tests\phase1\phase1PostgresIntegration.test.js
npm.cmd ls --omit=dev --depth=0
```

Key conclusions:

- Targeted production dependency remediation was applied after explicit user
  approval.
- Direct dependencies updated: `firebase-admin` to `^14.2.0`, `multer` to
  `^2.2.0`.
- Production overrides added for `body-parser`, `brace-expansion`, `form-data`,
  `protobufjs`, `qs`, and `uuid`.
- `npm.cmd audit --omit=dev` now passes with `0 vulnerabilities`.
- `npm.cmd audit --omit=dev --json` now reports total vulnerabilities `0`.
- Backend default test run passes: `16` tests total, `15` passed, `0` failed,
  `1` skipped because the disposable Postgres database is absent.
- Syntax checks pass for server entrypoints, Phase 1 migration runner,
  Postgres repository, importer, and Phase 1 integration test.
- `npm.cmd ls --omit=dev --all` still reports npm override-tree
  `ELSPROBLEMS` under `get-intrinsic` / `call-bind-apply-helpers`; this is
  documented as a residual review item because audit and tests pass.
- No `npm audit fix`, broad update, dependency dedupe, application code change,
  canonical CSV change, production database access, Firebase access, commit,
  push, or Phase 1 Batch 3 work occurred during this remediation step.

## 2026-07-26 Phase 1 Final Dependency Integrity Gate

Time: 2026-07-26 13:08:43 +07:00.

Changed files:

```text
package.json
docs/rebuild/CURRENT_STATE.md
docs/rebuild/DECISIONS.md
docs/rebuild/TEST_REPORT.md
docs/rebuild/WORKLOG.md
```

Commands run:

```text
npm.cmd ls --omit=dev --all
npm.cmd explain get-intrinsic
npm.cmd explain call-bind-apply-helpers
npm.cmd explain body-parser
npm.cmd explain qs
npm.cmd explain firebase-admin
npm.cmd explain multer
npm.cmd explain hasown
npm.cmd install --ignore-scripts
npm.cmd ci --ignore-scripts
npm.cmd audit --omit=dev
npm.cmd audit --omit=dev --json
npm.cmd test
node --check src\server.js
node --check server.js
node --check src\config\firebaseAdmin.js
node --check src\infrastructure\db\phase1MigrationRunner.js
node --check src\modules\pois\postgresPoiRepository.js
node --check scripts\phase1_import_canonical_pois.js
node --check tests\phase1\phase1PostgresIntegration.test.js
node -e <firebase-admin and multer compatibility smoke>
git status --short --branch
git --no-pager diff --stat
```

Key conclusions:

- The initial full production dependency tree check failed with
  `ELSPROBLEMS`.
- `get-intrinsic@1.3.0` and `call-bind-apply-helpers@1.0.2` were first
  reported invalid even though installed versions satisfied their dependency
  ranges; adding narrow overrides moved npm to the remaining invalid package.
- `hasown@2.0.4` was then reported invalid under the `form-data@2.5.6` /
  `es-set-tostringtag@2.1.0` path; adding a narrow override resolved the final
  tree validation error.
- Final `npm.cmd ls --omit=dev --all` passes with exit code `0`.
- `npm.cmd ci --ignore-scripts` passes, proving the lockfile is reproducible
  from a clean install.
- `npm.cmd audit --omit=dev` and `npm.cmd audit --omit=dev --json` both pass
  with total vulnerabilities `0`.
- Backend default tests pass: `16` tests total, `15` passed, `0` failed, `1`
  skipped because the disposable Postgres database is absent.
- Syntax checks pass for server entrypoints, Firebase Admin config, Phase 1
  migration runner, Postgres repository, importer, and integration test.
- Firebase Admin and Multer compatibility smoke passes without initializing
  Firebase or connecting to production.
- No app source, migration SQL, tests, canonical CSV, manifest, frontend,
  production database, Firebase data, `npm audit fix`, broad update, dedupe,
  merge, or Batch 3 work occurred in this gate.
