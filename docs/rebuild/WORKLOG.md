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

## 2026-07-26 Phase 1 Draft PR Final Readiness Gate

Time: 2026-07-26 13:28:46 +07:00.

Mode: final readiness checks for Draft PR #2. No application source, migration
SQL, tests, canonical CSV, manifest, frontend code, production database, or
Firebase production data was modified.

Changed files:

```text
docs/rebuild/CURRENT_STATE.md
docs/rebuild/TEST_REPORT.md
docs/rebuild/WORKLOG.md
```

Commands run:

```text
node -e <package dependencies/devDependencies/overrides inspection>
rg -n <package override and lockfile path inspection>
git --no-pager diff --name-status main...HEAD
git --no-pager diff --stat main...HEAD
git --no-pager log --oneline --decorate main..HEAD
git remote get-url origin
git lfs version
git clone --branch phase1/data-platform-foundation <origin> C:\tmp\urbanagent-phase1-readiness-clone
git lfs pull
git lfs status
npm.cmd ci
git status --short --branch
git diff -- package.json package-lock.json
npm.cmd ls --omit=dev --all
npm.cmd audit --omit=dev
npm.cmd audit --omit=dev --json
npm.cmd test
wsl.exe docker --version
wsl.exe docker compose -f docker-compose.phase1.yml version
wsl.exe docker ps -a --filter name=urbanagent-phase1-postgis
wsl.exe docker volume ls --filter name=urbanagent_phase1_postgis_data
wsl.exe docker compose -f docker-compose.phase1.yml up -d
docker healthcheck wait for urbanagent-phase1-postgis
URBANAGENT_PHASE1_INTEGRATION=true npm.cmd test
npm.cmd run phase1:db:diagnostics
wsl.exe docker compose -f docker-compose.phase1.yml down -v
wsl.exe docker ps -a --filter name=urbanagent-phase1-postgis
wsl.exe docker volume ls --filter name=urbanagent_phase1_postgis_data
Get-FileHash -Algorithm SHA256 data\canonical\urbanagent_poi_master_v1.csv
node -e <canonical repository count>
node -e <CSV default runtime count>
git fetch origin
git merge-base --is-ancestor HEAD origin/main
git --no-pager diff --name-only main...HEAD | rg -n <forbidden file pattern>
```

Key conclusions:

- `call-bind-apply-helpers@1.0.2`, `get-intrinsic@1.3.0`, and
  `hasown@2.0.4` are under `overrides`, not top-level production
  dependencies.
- Clean clone at `C:\tmp\urbanagent-phase1-readiness-clone` checked out
  `c160f61edc45cbde5da772f1c9f564336b4de41c`.
- `git lfs pull` in the clean clone passed.
- Clean-clone `npm.cmd ci` passed without `--ignore-scripts`, `--force`, or
  `--legacy-peer-deps`; package files remained unchanged and Git status stayed
  clean.
- Clean-clone `npm.cmd ls --omit=dev --all` passed with no `ELSPROBLEMS`.
- Clean-clone `npm.cmd audit --omit=dev` and `npm.cmd audit --omit=dev --json`
  passed with total vulnerabilities `0`.
- Clean-clone default `npm.cmd test` passed with `15` passed, `0` failed, `1`
  skipped because disposable DB integration was off.
- Full disposable PostGIS integration with
  `postgis/postgis:16-3.5-alpine` passed: `16` passed, `0` failed, `0`
  skipped.
- Full integration diagnostics passed with POI entities `4166`, source records
  `4166`, external IDs `8337`, aliases `985`, images `16246`, review summaries
  `4166`, duplicate/orphan/geometry checks `0`, and GiST index
  `poi_entities_location_gix`.
- Disposable PostGIS container and volume were removed after validation.
- Canonical SHA-256 matched
  `5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae`.
- Canonical runtime and CSV default runtime both returned `4166` POIs.
- PR diff hygiene check found no `.env`, Firebase credential JSON, database
  dump, diagnostic bundle, raw audit output, `node_modules`, `dist`, temporary
  workspace file, accidental `status` file, or full patch artifact.
- Local `HEAD` equals `origin/phase1/data-platform-foundation`; Git state
  confirms Phase 1 `HEAD` is not in `origin/main`, so PR #2 was not merged in
  this gate.
- No production/shared database or Firebase production system was touched.
- Batch 3 was not started.

## 2026-07-26 14:26:14 +07:00 - Phase 1 Batch 3 endpoint runtime-switch coverage

Scope:

```text
Approved Phase 1 Batch 3 only.
No Phase 2 work.
No production/shared database.
No Firebase production access.
No canonical CSV byte changes.
CSV remains default runtime.
PostgreSQL remains explicit opt-in.
```

Changed files:

```text
tests/phase1/phase1PostgresIntegration.test.js
docs/rebuild/CURRENT_STATE.md
docs/rebuild/DECISIONS.md
docs/rebuild/PHASE1_LOCAL_POSTGIS_RUNBOOK.md
docs/rebuild/TEST_REPORT.md
docs/rebuild/WORKLOG.md
```

Commands run:

```text
git status --short --branch
Get-Content -LiteralPath docs\rebuild\CURRENT_STATE.md
Get-Content -LiteralPath docs\rebuild\TEST_REPORT.md
Get-Content -LiteralPath docs\rebuild\DECISIONS.md
Get-Content -LiteralPath docs\rebuild\WORKLOG.md -Tail 80
git --no-pager diff -- tests/phase1/phase1PostgresIntegration.test.js
git --no-pager diff --stat
node --check tests\phase1\phase1PostgresIntegration.test.js
npm.cmd test
wsl.exe docker compose -f docker-compose.phase1.yml up -d
docker healthcheck wait for urbanagent-phase1-postgis
URBANAGENT_PHASE1_INTEGRATION=true npm.cmd test
npm.cmd run phase1:db:diagnostics
wsl.exe docker compose -f docker-compose.phase1.yml down -v
wsl.exe docker ps -a --filter name=urbanagent-phase1-postgis
wsl.exe docker volume ls --filter name=urbanagent_phase1_postgis_data
Get-FileHash -Algorithm SHA256 data\canonical\urbanagent_poi_master_v1.csv
node -e <canonical repository count>
node -e <CSV default runtime check>
```

Results:

```text
Phase 1 integration test syntax: PASS.
Default npm.cmd test: PASS, 16 total, 15 passed, 0 failed, 1 skipped.
Disposable PostGIS healthcheck: PASS.
Full disposable DB npm.cmd test: PASS, 16 total, 16 passed, 0 failed, 0 skipped.
Endpoint runtime-switch smoke: PASS under URBANAGENT_POI_REPOSITORY=postgres.
Diagnostics: PASS.
Canonical SHA-256: 5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae.
CSV default runtime: PASS, csv-default, 4166 POIs.
Disposable container and volume cleanup: PASS.
```

Endpoint smoke evidence:

```text
GET /api/eda?source=google_maps: 3946 POIs.
GET /api/eda?source=foody: 225 POIs.
GET /api/pois/data-quality: 4166 application POIs, expected headers.
POST /api/agent/recommend-poi: nonempty da-nang results.
POST /api/agent/create-itinerary: nonempty da-nang itinerary.
Missing-origin first leg: distanceKm null, distanceKnown false.
```

Key conclusions:

- Existing endpoint compatibility now has explicit PostgreSQL runtime-switch
  smoke coverage for EDA, data quality, recommendation, and itinerary paths.
- The endpoint smoke test starts a local child process and does not require a
  server export refactor.
- The test path is guarded by the disposable DB integration flag and does not
  run against production/shared databases.
- Local runbook documents the exact disposable PostGIS workflow and cleanup.
- CSV remains the default runtime when `URBANAGENT_POI_REPOSITORY` is absent.
- No production database, shared database, Firebase production data, frontend
  code, or canonical CSV bytes were touched.

## 2026-07-26 15:35:00 +07:00 - Phase 2 Traveler API v2 planning and specification

Scope:

```text
Planning/specification only after APPROVED PHASE 2.
No application source changes.
No test changes.
No migration changes.
No package/dependency changes.
No canonical CSV, manifest, context, AGENTS, or frontend changes.
No Docker/PostGIS workflow.
No production database or Firebase access.
No commit or push.
```

Files read:

```text
AGENTS.md
URBANAGENT_CODEX_CONTEXT.md
PLANNING.md
README.md
package.json
docs/rebuild/MASTER_PLAN.md
docs/rebuild/CURRENT_STATE.md
docs/rebuild/DECISIONS.md
docs/rebuild/DATA_AUDIT.md
docs/rebuild/TEST_REPORT.md
docs/rebuild/WORKLOG.md
docs/rebuild/URBANAGENT_DATASET_DECISION.md
docs/rebuild/PHASE1_LOCAL_POSTGIS_RUNBOOK.md
..\POI-urban-danang-FE\AGENTS.md
..\POI-urban-danang-FE\PLANNING.md
..\POI-urban-danang-FE\package.json
server.js
src/server.js
src/services/poiRepository.js
src/services/poiDataService.js
src/services/canonicalCsvPoiRepository.js
src/services/poiRetrievalService.js
src/services/itineraryPlannerService.js
src/services/routeMatrixService.js
src/services/weatherService.js
src/services/firestorePersistenceService.js
src/middleware/firebaseAuth.js
src/modules/cities/cityConfig.js
src/modules/pois/postgresPoiRepository.js
tests/phase0/phase0CanonicalData.test.js
tests/phase1/phase1DataPlatform.test.js
tests/phase1/phase1PostgresIntegration.test.js
```

Commands run:

```text
git fetch origin --tags --prune
git status --short --branch
git branch --show-current
git rev-parse HEAD
git rev-parse origin/main
git merge-base --is-ancestor 2c34471 origin/main
git --no-pager log --oneline --decorate -10
rg -n "app\.(get|post|put|patch|delete)\(" server.js src\server.js
rg -n "Traveler API v2|API v2|Phase 2|Phase 3|Phase 4|Phase 5|Phase 6|Every recommendation|Source_IDs|City_ID|RestaurantID|Alias_Global_IDs|urban-void|missing|null|guest|PostgreSQL|CSV remains|default runtime" URBANAGENT_CODEX_CONTEXT.md PLANNING.md docs\rebuild\MASTER_PLAN.md docs\rebuild\CURRENT_STATE.md docs\rebuild\DECISIONS.md docs\rebuild\TEST_REPORT.md docs\rebuild\DATA_AUDIT.md
rg -n "function |const .* =|module\.exports|exports\." src\services src\middleware src\config tests
rg --files | rg "cityConfig|cities|phase1|postgres|import"
Get-Content -LiteralPath <required docs and source files>
Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"
```

Files created:

```text
docs/rebuild/PHASE2_TRAVELER_API_V2_SCOPE.md
docs/rebuild/PHASE2_TRAVELER_API_V2_CONTRACT_DRAFT.md
docs/rebuild/PHASE2_TRAVELER_API_V2_EVALUATION_PLAN.md
```

Files updated:

```text
docs/rebuild/MASTER_PLAN.md
docs/rebuild/CURRENT_STATE.md
docs/rebuild/DECISIONS.md
docs/rebuild/WORKLOG.md
```

Key conclusions:

- Phase 1 Batch 3 is merged into `main` and tagged `phase-1-batch-3`.
- Current API v2 traveler endpoints do not exist yet.
- Current traveler functionality is available through legacy `/api/agent/*`
  endpoints and repository-backed POI/data-quality routes.
- Primary server has 40 Express endpoint registrations; legacy root server has
  5 compatibility registrations.
- Phase 2 should add backend Traveler API v2 only and keep all legacy endpoints.
- CSV remains default runtime; PostgreSQL remains explicit opt-in.
- The only approved city in Phase 2 is `da-nang`.
- Unknown city behavior must be explicit in v2 and must not silently return Da
  Nang data.
- Partner/seller/admin product behavior is out of Phase 2 traveler scope.
- No build/test was run because this was documentation-only planning and no
  application/test code changed.

## 2026-07-26 16:17:26 +07:00 - Phase 2 Traveler API v2 specification revision

Scope:

```text
Documentation revision only.
No Phase 2 implementation.
No application source changes.
No test changes.
No migration changes.
No package/dependency changes.
No canonical data changes.
No manifest/context/AGENTS changes.
No frontend changes.
No runtime configuration changes.
No commit or push.
```

Files read:

```text
C:\Users\ADMIN\.codex\attachments\c9b876cc-899b-4479-ad66-d22e998c8893\pasted-text.txt
docs/rebuild/PHASE2_TRAVELER_API_V2_SCOPE.md
docs/rebuild/PHASE2_TRAVELER_API_V2_CONTRACT_DRAFT.md
docs/rebuild/PHASE2_TRAVELER_API_V2_EVALUATION_PLAN.md
docs/rebuild/CURRENT_STATE.md
docs/rebuild/DECISIONS.md
```

Commands run:

```text
Get-Content -LiteralPath <attached request>
rg -n <Phase 2 public metadata/null/capability/signal checks>
Get-Content -LiteralPath docs\rebuild\PHASE2_TRAVELER_API_V2_CONTRACT_DRAFT.md
Get-Content -LiteralPath docs\rebuild\PHASE2_TRAVELER_API_V2_SCOPE.md
Get-Content -LiteralPath docs\rebuild\PHASE2_TRAVELER_API_V2_EVALUATION_PLAN.md
Get-Content -LiteralPath docs\rebuild\CURRENT_STATE.md
Get-Content -LiteralPath docs\rebuild\DECISIONS.md
Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"
```

Files revised:

```text
docs/rebuild/PHASE2_TRAVELER_API_V2_SCOPE.md
docs/rebuild/PHASE2_TRAVELER_API_V2_CONTRACT_DRAFT.md
docs/rebuild/PHASE2_TRAVELER_API_V2_EVALUATION_PLAN.md
docs/rebuild/CURRENT_STATE.md
docs/rebuild/DECISIONS.md
docs/rebuild/WORKLOG.md
```

Key conclusions:

- Public API examples were revised to avoid local dataset paths, SQL/table
  details, repository class names, `DATABASE_URL`, and storage-mode
  dependencies.
- Public metadata now uses `datasetVersion`, `contractHash`,
  `applicationPoiCount`, `qualitySummary`, and `capabilityStatus`.
- Unknown values are specified as `null` and/or explicit status fields, not
  empty strings or zero.
- Route summaries now model partial/unknown routes with known/unknown leg
  counts and full-known flags.
- Rating contract now separates normalized scale-5 rating, Google source
  rating/count, Foody scale-10 source rating, review count, and sample review
  data.
- Source identifiers are typed objects; `RestaurantID` is explicitly not a
  Google Place ID.
- Capabilities now use state strings and are not public `true` before
  implementation and validation.
- Phase 2 is split into core approved endpoints and conditional persistence
  endpoints that are not approved for implementation.
- Scientific evaluation plan now includes RQs, hypotheses, baselines, fixture
  requirements, metrics, ablations, repeatability, statistical reporting,
  failure taxonomy, qualitative analysis, and validity threats.
- Phase 2 implementation remains not started.

## 2026-07-26 21:50:51 +07:00 - Phase 2 final specification correction before Batch 1

Scope:

```text
Final documentation-only correction.
No Phase 2 code implementation.
No application source changes.
No test changes.
No migration changes.
No package/dependency changes.
No canonical data changes.
No runtime configuration changes.
No frontend changes.
No git add, commit, or push.
```

Files read:

```text
C:\Users\ADMIN\.codex\attachments\2fbeb369-e5e7-42c1-a2e9-aef2f5237396\pasted-text.txt
docs/rebuild/PHASE2_TRAVELER_API_V2_SCOPE.md
docs/rebuild/PHASE2_TRAVELER_API_V2_CONTRACT_DRAFT.md
docs/rebuild/PHASE2_TRAVELER_API_V2_EVALUATION_PLAN.md
docs/rebuild/CURRENT_STATE.md
docs/rebuild/DECISIONS.md
```

Commands run:

```text
Get-Content -LiteralPath <attached request>
rg -n <Phase 2 contractVersion/OpenAPI/requestId/pagination/baseline checks>
Get-Content -LiteralPath docs\rebuild\PHASE2_TRAVELER_API_V2_CONTRACT_DRAFT.md
Get-Content -LiteralPath docs\rebuild\PHASE2_TRAVELER_API_V2_SCOPE.md
Get-Content -LiteralPath docs\rebuild\PHASE2_TRAVELER_API_V2_EVALUATION_PLAN.md
Get-Content -LiteralPath docs\rebuild\DECISIONS.md
Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"
```

Files revised:

```text
docs/rebuild/PHASE2_TRAVELER_API_V2_SCOPE.md
docs/rebuild/PHASE2_TRAVELER_API_V2_CONTRACT_DRAFT.md
docs/rebuild/PHASE2_TRAVELER_API_V2_EVALUATION_PLAN.md
docs/rebuild/CURRENT_STATE.md
docs/rebuild/DECISIONS.md
docs/rebuild/WORKLOG.md
```

Key conclusions:

- Replaced false `contractHash` semantics with `contractVersion`.
- Phase 2 draft contract version is
  `phase2-traveler-api-v2-draft-1`.
- OpenAPI 3.1 is approved and required in Batch 1.
- `openApiSha256` may be recorded only after generating and hashing the
  OpenAPI artifact.
- Batch 1 owns POI search pagination, cursor validation, deterministic
  search/list sorting, CSV-default endpoint behavior, and OpenAPI contract.
- Batch 2 owns recommendation v2 ranking, reasonCodes, fixture foundation, and
  quality evaluation preparation.
- Request ID behavior is final: invalid/missing `X-Request-Id` generates a new
  server requestId, does not echo invalid values, and does not reject otherwise
  valid requests.
- Common metadata is minimal: `apiVersion` and `requestId`, plus `cityId` for
  city-scoped responses.
- Legacy/v2 comparison is behavioral parity, not a quality algorithm
  comparison.
- Remaining open questions are limited to conditional persistence, Firestore,
  Firebase emulator for conditional persistence tests, and curated query
  fixture process/reviewers.
- Phase 2 implementation remains not started.

## 2026-07-26 22:04:35 +07:00 - Phase 2 Batch 1 Traveler API v2 Foundation

User authorization:

```text
APPROVED PHASE 2 BATCH 1
```

Files read/inspected:

```text
git status --short --branch
git --no-pager diff --name-status
src/server.js
package.json
src/modules/travelerApiV2/constants.js
src/modules/travelerApiV2/requestContext.js
src/modules/travelerApiV2/pagination.js
src/modules/travelerApiV2/serializers.js
src/modules/travelerApiV2/poiSearch.js
src/modules/travelerApiV2/router.js
tests/phase0/phase0CanonicalData.test.js
tests/phase1/phase1DataPlatform.test.js
tests/phase1/phase1PostgresIntegration.test.js
docs/rebuild/CURRENT_STATE.md
docs/rebuild/DECISIONS.md
docs/rebuild/TEST_REPORT.md
docs/rebuild/PHASE2_TRAVELER_API_V2_SCOPE.md
docs/rebuild/PHASE2_TRAVELER_API_V2_CONTRACT_DRAFT.md
```

Changed/added files:

```text
src/server.js
src/modules/travelerApiV2/constants.js
src/modules/travelerApiV2/requestContext.js
src/modules/travelerApiV2/pagination.js
src/modules/travelerApiV2/serializers.js
src/modules/travelerApiV2/poiSearch.js
src/modules/travelerApiV2/router.js
tests/phase2/phase2TravelerApiV2Batch1.test.js
package.json
docs/rebuild/PHASE2_TRAVELER_API_V2_OPENAPI_DRAFT.json
docs/rebuild/CURRENT_STATE.md
docs/rebuild/DECISIONS.md
docs/rebuild/TEST_REPORT.md
docs/rebuild/PHASE2_TRAVELER_API_V2_SCOPE.md
docs/rebuild/PHASE2_TRAVELER_API_V2_CONTRACT_DRAFT.md
docs/rebuild/WORKLOG.md
```

Implementation summary:

- Mounted a new read-only Traveler API v2 router at `/api/v2` in the primary
  backend runtime.
- Added common v2 response envelope and error envelope helpers.
- Added request ID validation with server-generated replacement for missing or
  invalid `X-Request-Id`.
- Added city metadata/status endpoints for the single approved Phase 2 city.
- Added POI search/detail endpoints backed by the existing repository flow.
- Added POI search pagination with default limit `20`, maximum `100`, opaque
  cursor validation, deterministic sort, and canonical `Global_ID` tie-breaks.
- Added traveler-safe POI serialization that preserves null/unknown semantics
  and exposes typed source identifiers instead of ambiguous legacy fields.
- Added OpenAPI 3.1 Batch 1 artifact covering only the approved Batch 1
  endpoints and common schemas.
- Added focused Phase 2 Batch 1 tests.
- Updated the backend test script to include `tests/phase2/*.test.js`.

Commands run:

```text
node --check src\modules\travelerApiV2\constants.js
node --check src\modules\travelerApiV2\requestContext.js
node --check src\modules\travelerApiV2\pagination.js
node --check src\modules\travelerApiV2\serializers.js
node --check src\modules\travelerApiV2\poiSearch.js
node --check src\modules\travelerApiV2\router.js
node --check tests\phase2\phase2TravelerApiV2Batch1.test.js
node -e <OpenAPI JSON parse>
node -e <OpenAPI SHA-256>
npm test
npm.cmd test
```

Command results:

- `npm test`: failed before running tests because PowerShell blocked
  `D:\Apps\npm.ps1` under the local execution policy.
- `npm.cmd test`: PASS.
- Test totals: 21 tests, 20 passed, 0 failed, 1 skipped.
- Skipped test: existing guarded Phase 1 disposable PostGIS integration test
  when DB integration env vars are absent.
- OpenAPI SHA-256:
  `58599da0dd29023c5d25eee1fc74da7f52339d3e131d6d5542344974b6577a9b`.

Endpoint smoke conclusions:

- `GET /api/v2/cities`: PASS.
- `GET /api/v2/cities/:cityId/status`: PASS, application POIs `4166`.
- Unknown city: PASS, `CITY_NOT_SUPPORTED`.
- Missing `cityId`: PASS, `VALIDATION_ERROR`.
- Invalid pagination limit: PASS, `VALIDATION_ERROR`.
- Google-compatible POI search count: `3946`.
- Foody-compatible POI search count: `225`.
- All/canonical POI search count: `4166`.
- POI detail: PASS.
- Unknown POI detail: PASS, `NOT_FOUND`.

Safety conclusions:

- CSV remains the default runtime.
- PostgreSQL remains explicit opt-in.
- No production/shared database was used.
- No Firebase production data was touched.
- Canonical CSV bytes were not modified.
- No frontend files were changed.
- Recommendation v2, itinerary preview v2, persistence, edit/replan, feedback
  persistence, and Phase 2 Batch 2 were not started.

## 2026-07-26 22:04:35 +07:00 - Phase 2 Batch 1 Final Review Test Tightening

Scope:

- Focused review before commit and Draft PR.
- Added narrow Batch 1 tests only.
- No Phase 2 Batch 2 implementation.
- No recommendation v2, itinerary preview v2, persistence/edit/replan,
  feedback persistence, frontend, canonical CSV, production database, or
  Firebase production changes.

Additional test coverage added:

- Missing request ID generates a server request ID.
- Error responses include `meta.apiVersion` and `meta.requestId`.
- Non-integer limit is rejected.
- Malformed cursor is rejected with structured `VALIDATION_ERROR`.
- `source=canonical` returns all `4166` POIs.
- Repeated identical paginated requests return identical order/cursor behavior.
- Consecutive cursor pages have no duplicate `Global_ID` and match the
  equivalent wider first page.
- Public endpoint response bodies do not expose local paths, storage internals,
  SQL details, repository class names, stack traces, bearer strings, or token
  text.
- Serializer semantics preserve canonical public IDs, `google_maps+foody`
  provenance, typed source identifiers, alias IDs, nullable rating/review
  values, scale-5 normalized/Google ratings, scale-10 Foody ratings,
  sample-review semantics, image URL array compatibility, and null district
  behavior.

Commands run:

```text
node -e <canonical fixture availability diagnostic>
npm.cmd test
```

Results:

- `npm.cmd test`: PASS.
- Test totals: 22 tests, 21 passed, 0 failed, 1 skipped.
- Skipped test remains the existing guarded Phase 1 disposable PostGIS
  integration test when DB integration env vars are absent.

## 2026-07-31 19:55:42 +07:00 - Phase 2 Batch 2 Traveler API v2 Recommendations

Scope:

- User approved Phase 2 Batch 2.
- Implemented only standalone Traveler API v2 recommendations.
- Did not implement itinerary preview v2, trip persistence/edit/replan,
  feedback persistence, frontend changes, PostgreSQL default runtime, production
  database work, Firebase production access, Batch 3, or later-phase work.
- Used the clean validation clone:
  `C:\tmp\urbanagent-phase2-batch1-validation-clone-20260726-225244`.
- Did not modify the locked original working copy at
  `D:\POI-urban-danang-BE`.

Files changed:

```text
docs/rebuild/CURRENT_STATE.md
docs/rebuild/DECISIONS.md
docs/rebuild/PHASE2_TRAVELER_API_V2_CONTRACT_DRAFT.md
docs/rebuild/PHASE2_TRAVELER_API_V2_OPENAPI_DRAFT.json
docs/rebuild/PHASE2_TRAVELER_API_V2_SCOPE.md
docs/rebuild/TEST_REPORT.md
docs/rebuild/WORKLOG.md
src/modules/travelerApiV2/constants.js
src/modules/travelerApiV2/recommendations.js
src/modules/travelerApiV2/router.js
src/modules/travelerApiV2/serializers.js
tests/fixtures/phase2/recommendationQueries.json
tests/phase2/phase2TravelerApiV2Batch1.test.js
tests/phase2/phase2TravelerApiV2Batch2.test.js
```

Commands run:

```text
git status --short --branch
git branch --show-current
git rev-parse HEAD
Get-Content AGENTS.md
Get-Content URBANAGENT_CODEX_CONTEXT.md
Get-Content PLANNING.md
Get-Content README.md
Get-Content package.json
Get-Content D:\POI-urban-danang-FE\AGENTS.md
Get-Content D:\POI-urban-danang-FE\URBANAGENT_CODEX_CONTEXT.md
Get-Content docs\rebuild\PHASE2_TRAVELER_API_V2_SCOPE.md
Get-Content docs\rebuild\PHASE2_TRAVELER_API_V2_CONTRACT_DRAFT.md
Get-Content docs\rebuild\PHASE2_TRAVELER_API_V2_EVALUATION_PLAN.md
Get-Content docs\rebuild\CURRENT_STATE.md
Get-Content docs\rebuild\DECISIONS.md
Get-Content docs\rebuild\TEST_REPORT.md
Get-Content docs\rebuild\WORKLOG.md
rg --files src\modules\travelerApiV2 src\services tests\phase2 docs\rebuild
rg -n <traveler API v2 and recommendation references>
node --check src\modules\travelerApiV2\constants.js
node --check src\modules\travelerApiV2\recommendations.js
node --check src\modules\travelerApiV2\router.js
node --check tests\phase2\phase2TravelerApiV2Batch1.test.js
node --check tests\phase2\phase2TravelerApiV2Batch2.test.js
node -e <OpenAPI JSON parse and path list>
npm.cmd test
Get-FileHash -Algorithm SHA256 docs\rebuild\PHASE2_TRAVELER_API_V2_OPENAPI_DRAFT.json
Get-FileHash -Algorithm SHA256 data\canonical\urbanagent_poi_master_v1.csv
node -e <CSV default runtime and count diagnostic>
git status --short
git diff --name-status
```

Results:

- Syntax checks: PASS.
- OpenAPI JSON parse: PASS.
- OpenAPI implemented core paths:
  - `/api/v2/cities`
  - `/api/v2/cities/{cityId}/status`
  - `/api/v2/pois/search`
  - `/api/v2/pois/{poiId}`
  - `/api/v2/recommendations`
- OpenAPI SHA-256:
  `371e5de7db74b3fdeaf52999e2f417db0078309edb9ff5fe399dfec210c60da9`.
- Canonical CSV SHA-256:
  `5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae`.
- CSV default runtime: `csv-default`, count `4166`.
- `npm.cmd test`: PASS, 28 tests total, 27 passed, 0 failed, 1 skipped.
- Skipped test remains the existing guarded Phase 1 disposable PostGIS
  integration test when DB integration env vars are absent.

Endpoint and semantic conclusions:

- `POST /api/v2/recommendations`: PASS.
- Query `quan cafe yen tinh`: nonempty canonical Da Nang recommendations.
- Missing `cityId`: `400 VALIDATION_ERROR`.
- Unsupported `cityId=hue`: `422 CITY_NOT_SUPPORTED`.
- Missing/blank `query`: `400 VALIDATION_ERROR`.
- Invalid recommendation `limit`: `400 VALIDATION_ERROR`.
- Public recommendations expose `poi`, `score`, `reason`, `reasonCodes`,
  `warnings`, and `provenance`.
- Raw fields are absent from v2 recommendation responses: `signals`,
  `scoreRaw`, `sourceIds`, and `placeId`.
- Repeated v2 recommendation requests return deterministic POI IDs.
- `recommendations` capability is now `experimental`; `tripPreview` remains
  `planned`.
- The recommendation fixture is a smoke/evaluation foundation only and does not
  support quality superiority claims.

Safety:

- CSV remains the default runtime.
- PostgreSQL remains explicit opt-in.
- No production or shared database was used.
- No Firebase production data was touched.
- Canonical CSV bytes were not modified.
- No frontend source was changed.
- No commit or push occurred.
- Phase 2 Batch 3 was not started.

## 2026-07-31 - Multi-Source POI Strategy Documentation Completion

Type: documentation-only planning repair in clean clones.

Context:

- Phase 2 continues to use the approved 4166-POI Da Nang canonical baseline.
- No source expansion was added to an active implementation batch.
- Mobile product design remains deferred.
- The original backend and frontend working copies were read only.
- Corrected documentation work was performed in clean clones under `C:\tmp`.

Created in the backend documentation clone:

- `docs/rebuild/MULTI_SOURCE_POI_STRATEGY.md`
- `docs/rebuild/DATA_SOURCE_LICENSE_POLICY.md`

Updated in the backend documentation clone:

- `AGENTS.md`
- `URBANAGENT_CODEX_CONTEXT.md`
- `docs/rebuild/MASTER_PLAN.md`
- `docs/rebuild/CURRENT_STATE.md`
- `docs/rebuild/DECISIONS.md`
- `docs/rebuild/WORKLOG.md`

Updated in the frontend documentation clone:

- `AGENTS.md`
- `URBANAGENT_CODEX_CONTEXT.md`

Corrections:

- Completed `DATA_SOURCE_LICENSE_POLICY.md`; the registry code fence is closed
  and the policy now includes approval status, provider guardrails,
  attribution, storage/cache/deletion, ML/derived-use, security/privacy,
  dry-run, Phase 2 freeze, and approval-gate sections.
- Added the same shared multi-source governance section to both backend and
  frontend `URBANAGENT_CODEX_CONTEXT.md`.
- Backend and frontend context files now remain equivalent after the shared
  governance addition.
- Recorded that future source expansion requires:
  `APPROVED MULTI-SOURCE POI SPIKE`.

No application source, tests, migrations, package files, OpenAPI runtime
artifacts, canonical data, runtime configuration, frontend implementation code,
or mobile code was changed.

No external POI source was downloaded, queried, sampled, scraped, or ingested.

No production database or Firebase production resource was accessed.

No multi-source implementation, Phase 2 Batch 3, commit, or push occurred.

## 2026-08-01 13:17:32 +07:00 - Synchronize Backend Multi-Source Docs After Phase 2 Batch 2 Merge

Type: documentation-only Draft PR synchronization in clean backend clone.

Context:

- Existing Draft PR #5 remains the backend multi-source governance
  documentation PR.
- Phase 2 Batch 2 has been merged into `origin/main` at
  `707cce556cf37986d9bd78fdf25902d76850242c`.
- Phase 2 Batch 2 implementation commit
  `7718cd5c9e4d4d07a083f1d10aa9ad539035e14b` is an ancestor of `origin/main`.
- Documentation branch before synchronization:
  `a9bf00d2de0a35a3b5dacdf570b0e1e8d14d71cd`.
- The original backend working copy at `D:\POI-urban-danang-BE` was not
  modified.

Files changed in the documentation branch diff versus `origin/main`:

```text
AGENTS.md
URBANAGENT_CODEX_CONTEXT.md
docs/rebuild/CURRENT_STATE.md
docs/rebuild/DATA_SOURCE_LICENSE_POLICY.md
docs/rebuild/DECISIONS.md
docs/rebuild/MASTER_PLAN.md
docs/rebuild/MULTI_SOURCE_POI_STRATEGY.md
docs/rebuild/PHASE2_TRAVELER_API_V2_EVALUATION_PLAN.md
docs/rebuild/PHASE2_TRAVELER_API_V2_SCOPE.md
docs/rebuild/WORKLOG.md
```

Commands run:

```text
git status --short --branch
git branch --show-current
git rev-parse HEAD
git remote -v
git fetch origin --tags --prune
Invoke-RestMethod https://api.github.com/repos/nhatnguyen1607/POI-urban-danang-BE/pulls/5
git rev-parse origin/main
git merge-base --is-ancestor 7718cd5c9e4d4d07a083f1d10aa9ad539035e14b origin/main
git diff --name-status origin/main...HEAD
git merge --no-ff origin/main
rg -n "<<<<<<<|=======|>>>>>>>" docs/rebuild
git diff --name-status origin/main
git diff --stat origin/main
```

Conflict resolution:

- `docs/rebuild/DECISIONS.md`: preserved both the completed Phase 2 Batch 2
  implementation decisions and the future multi-source governance decision.
- `docs/rebuild/WORKLOG.md`: preserved both the Phase 2 Batch 2 implementation
  entry and the multi-source documentation completion entry.
- `docs/rebuild/CURRENT_STATE.md`: updated stale Batch 1/Batch 2 next-step
  language so the document reflects the merged Batch 2 baseline and waits for
  Draft PR #5 review.

Safety:

- No application source, tests, package files, migration SQL, OpenAPI JSON,
  runtime configuration, canonical data, environment files, production
  database, Firebase production resource, frontend code, mobile code, or
  external POI source was modified.
- PR #5 remains Draft and unmerged.
- Phase 2 Batch 3 was not started.

## 2026-08-01 15:48:48 +07:00 - Phase 2 Batch 3 Trip Preview Design Acceptance Package

Type: documentation-only design acceptance completion in clean planning clone.

Planning clone:

```text
C:\tmp\urbanagent-phase2-batch3-planning-20260801-142927
```

Files finalized:

```text
docs/rebuild/PHASE2_BATCH3_TRIP_PREVIEW_PLAN.md
docs/rebuild/PHASE2_BATCH3_TRIP_PREVIEW_SCOPE.md
docs/rebuild/PHASE2_BATCH3_TRIP_PREVIEW_API_CONTRACT.md
docs/rebuild/PHASE2_BATCH3_TRIP_PREVIEW_EVALUATION_PLAN.md
docs/rebuild/PHASE2_BATCH3_TRIP_PREVIEW_IMPLEMENTATION_BOUNDARIES.md
```

Canonical documents integrated:

```text
docs/rebuild/PHASE2_TRAVELER_API_V2_SCOPE.md
docs/rebuild/PHASE2_TRAVELER_API_V2_CONTRACT_DRAFT.md
docs/rebuild/PHASE2_TRAVELER_API_V2_EVALUATION_PLAN.md
docs/rebuild/MASTER_PLAN.md
docs/rebuild/CURRENT_STATE.md
docs/rebuild/DECISIONS.md
docs/rebuild/WORKLOG.md
```

Work completed:

- Renamed the prior authoritative entry-point draft to
  `PHASE2_BATCH3_TRIP_PREVIEW_PLAN.md`.
- Finalized the documentation-only Batch 3 design package.
- Integrated the design package into seven canonical documentation records.
- Documented final request limits, duration policy, trip/day limits,
  travel-time policy, missing-origin semantics, opening-hours behavior,
  warning taxonomy, feasibility statuses, `NO_FEASIBLE_ITINERARY` semantics,
  and evaluation metrics/thresholds.

Safety:

- No source, test, package, OpenAPI JSON, canonical CSV, migration, database
  configuration, runtime configuration, environment, credential, frontend, or
  mobile file was modified.
- No external POI source was queried, downloaded, sampled, scraped, ingested,
  cached, persisted, or merged.
- No production database or Firebase production resource was accessed.
- No implementation work was started.
- No commit, push, or Pull Request was created.
- User approval has not been granted.

## 2026-08-01 16:05:12 +07:00 - Phase 2 Batch 3 Design Approval Officialization

Type: documentation-only approval officialization in clean planning clone.

User approval:

```text
APPROVED PHASE 2 BATCH 3
```

Work completed:

- Applied final public warning-severity normalization.
- Limited public warning severity values to `info`, `warning`, and `error`.
- Recorded that Phase 2 Batch 3 design approval has been granted.
- Recorded that documentation must be committed, reviewed, merged, and
  post-merge validated before runtime implementation begins.
- Preserved the approved request limits, duration policy, travel-time policy,
  missing-origin semantics, opening-hours semantics, feasibility statuses, and
  evaluation gates.

Safety:

- Documentation officialization started.
- Runtime implementation did not start.
- No source, test, package, OpenAPI JSON, canonical CSV, migration, database
  configuration, runtime configuration, environment, credential, frontend, or
  mobile file was modified.
- No production database or Firebase production resource was accessed.
- No external POI source was queried, downloaded, sampled, scraped, ingested,
  cached, persisted, or merged.
- Multi-source approvals remain not granted.

## 2026-08-01 17:11:45 +07:00 - Phase 2 Batch 3 Trip Preview Runtime Implementation

Type: approved backend runtime implementation in fresh clean clone.

Implementation clone:

```text
C:\tmp\urbanagent-phase2-batch3-implementation-20260801-164730
```

Branch:

```text
phase2/batch3-traveler-api-v2-trip-preview
```

User approval:

```text
APPROVED PHASE 2 BATCH 3
```

Baseline verification:

- `origin/main`: `03bcc2ba90be7aa2618b5381f763ac1933469deb`.
- Approved Batch 3 design commit
  `b6fc14f02d84ca17d1a1eb1dc5bad17ab0971632` is an ancestor of
  `origin/main`.
- Batch 2 implementation commit
  `7718cd5c9e4d4d07a083f1d10aa9ad539035e14b` is an ancestor of
  `origin/main`.
- Tag `phase-2-batch-2` resolves to
  `707cce556cf37986d9bd78fdf25902d76850242c`.
- Canonical CSV SHA-256 verified:
  `5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae`.
- Canonical row count verified: `4166`.
- Five authoritative Batch 3 design-document Git blob hashes matched the
  approved expected values.

Files changed or added:

```text
docs/rebuild/CURRENT_STATE.md
docs/rebuild/DECISIONS.md
docs/rebuild/PHASE2_BATCH3_TRIP_PREVIEW_IMPLEMENTATION_BOUNDARIES.md
docs/rebuild/PHASE2_BATCH3_TRIP_PREVIEW_PLAN.md
docs/rebuild/PHASE2_TRAVELER_API_V2_OPENAPI_DRAFT.json
docs/rebuild/TEST_REPORT.md
docs/rebuild/WORKLOG.md
src/modules/travelerApiV2/constants.js
src/modules/travelerApiV2/recommendations.js
src/modules/travelerApiV2/router.js
src/modules/travelerApiV2/serializers.js
src/modules/travelerApiV2/tripPreview.js
src/modules/travelerApiV2/tripPreviewDurationPolicy.js
src/modules/travelerApiV2/tripPreviewTravelPolicy.js
src/modules/travelerApiV2/tripPreviewValidation.js
src/modules/travelerApiV2/tripPreviewWarnings.js
tests/fixtures/phase2/tripPreviewQueries.json
tests/phase2/phase2TravelerApiV2Batch1.test.js
tests/phase2/phase2TravelerApiV2Batch3.test.js
```

Commands run:

```text
git clone --branch main --single-branch <backend-origin-url> <implementation-clone>
git -C <implementation-clone> fetch origin --tags --prune
git -C <implementation-clone> lfs pull
git -C <implementation-clone> merge-base --is-ancestor b6fc14f02d84ca17d1a1eb1dc5bad17ab0971632 origin/main
git -C <implementation-clone> merge-base --is-ancestor 7718cd5c9e4d4d07a083f1d10aa9ad539035e14b origin/main
git -C <implementation-clone> rev-parse "phase-2-batch-2^{}"
Get-FileHash -Algorithm SHA256 data\canonical\urbanagent_poi_master_v1.csv
node -e <five Batch 3 design Git-blob SHA-256 calculation>
git -C <implementation-clone> branch --list phase2/batch3-traveler-api-v2-trip-preview
git -C <implementation-clone> ls-remote --heads origin phase2/batch3-traveler-api-v2-trip-preview
git -C <implementation-clone> switch -c phase2/batch3-traveler-api-v2-trip-preview
Import-Csv data\canonical\urbanagent_poi_master_v1.csv <opening-hours/category inspection>
npm.cmd ci
node --check src\modules\travelerApiV2\tripPreview.js
node --check src\modules\travelerApiV2\tripPreviewValidation.js
node --check src\modules\travelerApiV2\router.js
node --test tests\phase2\phase2TravelerApiV2Batch3.test.js
node -e <OpenAPI JSON parse and path list>
npm.cmd test
git -C <implementation-clone> status --short --branch
git -C <implementation-clone> diff --name-status
```

Validation output recorded:

```text
node --test tests/phase2/phase2TravelerApiV2Batch3.test.js
tests 8
pass 8
fail 0
skipped 0

npm.cmd test
tests 36
pass 35
fail 0
skipped 1
```

Implementation notes:

- Added deterministic validation, duration policy, travel policy, warning
  taxonomy, and trip-preview orchestration modules.
- Mounted only `POST /api/v2/trips/preview`.
- Reused Batch 2 recommendation semantics through a shared candidate helper.
- Updated OpenAPI 3.1 draft to include the Batch 3 preview route and no
  persistence routes.
- Updated the existing OpenAPI regression test to allow only the new preview
  route and continue forbidding persistence/replan/feedback routes.
- Added 18-case fixture `phase2-trip-preview-smoke-v1`.
- Added focused Batch 3 tests covering request contract, determinism,
  constraints, duration, travel, missing-origin, warning taxonomy, feasibility,
  `NO_FEASIBLE_ITINERARY`, endpoint smokes, and no persistence routes.

Safety:

- Original repository `D:\POI-urban-danang-BE` was not modified.
- Canonical CSV bytes were not modified.
- No package dependency was added.
- No migration, schema, production database, shared database, Firebase
  production, external POI source, external routing provider, live
  opening-hours provider, frontend, mobile, second-city, persistence, saved
  trip, replan, stop mutation, feedback persistence, Batch 4, Phase 3, or
  later-phase work was started.
- Multi-source approvals remain not granted.

## 2026-08-01 17:24:00 +07:00 - Phase 2 Batch 3 Validation Continuation

Type: non-destructive validation in implementation clone.

Implementation clone:

```text
C:\tmp\urbanagent-phase2-batch3-implementation-20260801-164730
```

Commands run:

```text
npm.cmd ls --omit=dev --all
npm.cmd audit --omit=dev
npm.cmd audit --omit=dev --json
node --check src/server.js and all JavaScript files under src/modules/travelerApiV2 plus tests/phase2
npm.cmd test
node -e <18-scenario endpoint smoke script>
Get-FileHash -Algorithm SHA256 docs\rebuild\PHASE2_TRAVELER_API_V2_OPENAPI_DRAFT.json
Get-FileHash -Algorithm SHA256 data\canonical\urbanagent_poi_master_v1.csv
node -e <CSV-default runtime repository and application POI count diagnostic>
git -C <implementation-clone> diff --name-status
git -C <implementation-clone> diff --stat
```

Results:

```text
npm.cmd ls --omit=dev --all: PASS, no ELSPROBLEMS
npm.cmd audit --omit=dev: PASS, found 0 vulnerabilities
npm.cmd audit --omit=dev --json: PASS, total vulnerabilities 0
syntax checks: PASS, 16 files
npm.cmd test: PASS, 36 total, 35 passed, 0 failed, 1 skipped
endpoint smoke script: PASS, 18/18 scenarios
OpenAPI SHA-256: a6add259e5da16beaba5fc5be4d3e34a542a6077e4bf4a4b9f2d6b8c78788d31
canonical CSV SHA-256: 5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae
application POIs: 4166
CSV-default repository: CanonicalCsvPoiRepository
```

Endpoint smoke scenarios:

```text
valid one-day preview: PASS
valid multi-day preview: PASS
repeated identical preview: PASS
missing origin: PASS
explicit origin: PASS
explicit duration: PASS
category-default duration: PASS
excluded POI: PASS
must-include POI: PASS
partial preview: PASS
infeasible preview: PASS as deterministic PARTIAL with explanations
impossible hard constraints: PASS
invalid city: PASS
invalid time window: PASS
invalid X-Request-Id: PASS
valid X-Request-Id: PASS
no persistent trip route: PASS
no replan route: PASS
```

Safety:

- Original repository `D:\POI-urban-danang-BE` remained untouched.
- No production database, shared database, Firebase production, external POI
  source, external routing provider, live opening-hours provider, frontend,
  mobile, second-city, persistence, saved-trip, replan, mutation, feedback,
  Batch 4, Phase 3, tag, or main push was used.

## 2026-08-01 22:23:15 +07:00 - Phase 2 Batch 3 PR #9 Targeted Fixes

Type: targeted correction and validation in implementation clone.

Implementation clone:

```text
C:\tmp\urbanagent-phase2-batch3-implementation-20260801-164730
```

Branch:

```text
phase2/batch3-traveler-api-v2-trip-preview
```

Previous branch HEAD:

```text
f0d02306512e87e28ed05f6c4c7c95d3844a5efa
```

Files changed:

- `src/modules/travelerApiV2/tripPreview.js`
- `tests/fixtures/phase2/tripPreviewQueries.json`
- `tests/phase2/phase2TravelerApiV2Batch3.test.js`
- `docs/rebuild/PHASE2_TRAVELER_API_V2_OPENAPI_DRAFT.json`
- `docs/rebuild/PHASE2_BATCH3_TRIP_PREVIEW_API_CONTRACT.md`
- `docs/rebuild/PHASE2_BATCH3_TRIP_PREVIEW_PLAN.md`
- `docs/rebuild/PHASE2_BATCH3_TRIP_PREVIEW_SCOPE.md`
- `docs/rebuild/PHASE2_BATCH3_TRIP_PREVIEW_EVALUATION_PLAN.md`
- `docs/rebuild/PHASE2_BATCH3_TRIP_PREVIEW_IMPLEMENTATION_BOUNDARIES.md`
- `docs/rebuild/PHASE2_TRAVELER_API_V2_SCOPE.md`
- `docs/rebuild/PHASE2_TRAVELER_API_V2_CONTRACT_DRAFT.md`
- `docs/rebuild/PHASE2_TRAVELER_API_V2_EVALUATION_PLAN.md`
- `docs/rebuild/CURRENT_STATE.md`
- `docs/rebuild/DECISIONS.md`
- `docs/rebuild/MASTER_PLAN.md`
- `docs/rebuild/TEST_REPORT.md`
- `docs/rebuild/WORKLOG.md`

Commands run:

```text
Get-Content <strict audit report>
git status --short --branch
git branch --show-current
git rev-parse HEAD
git fetch origin --tags --prune
git rev-parse origin/main
git rev-parse origin/phase2/batch3-traveler-api-v2-trip-preview
GitHub API PR #9 metadata check
rg <stale status and contract terms>
npm.cmd ci
npm.cmd ls --omit=dev --all
npm.cmd audit --omit=dev
npm.cmd audit --omit=dev --json
node --check src/server.js and all Traveler API v2 / Phase 2 test JavaScript files
node --test tests/phase2/phase2TravelerApiV2Batch3.test.js
npm.cmd test
node -e <CSV-default endpoint and engine-unit smoke script>
```

Results:

```text
npm.cmd ci: PASS
npm.cmd ls --omit=dev --all: PASS, no ELSPROBLEMS
npm.cmd audit --omit=dev: PASS, 0 vulnerabilities
npm.cmd audit --omit=dev --json: PASS, total vulnerabilities 0
syntax checks: PASS, 16 files
Batch 3 focused tests: PASS, 10 total, 10 passed, 0 failed, 0 skipped
npm.cmd test: PASS, 38 total, 37 passed, 0 failed, 1 skipped
endpoint smoke: PASS, 16/16 endpoint fixture cases
engine-unit smoke: PASS, 2/2 synthetic fixture cases
fixture execution: PASS, 18/18 meaningful assertions
OpenAPI SHA-256: 0cf59e434e270cee80154ac20cf4c32b14c4147b6f8d174364ea93caae326034
canonical CSV SHA-256: 5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae
application POIs: 4166
CSV-default repository: CanonicalCsvPoiRepository
```

Evaluation metrics:

```text
deterministic replay rate: 15/15
exclusion violation rate: 0/1 violations
duplicate-stop rate: 0/15 duplicates
hard-constraint satisfaction rate on satisfiable cases: 15/15
must-include scheduling rate on satisfiable cases: 4/4
daily-window overflow rate: 0/4 overflows
known opening-hours conflict rate: 3/3 detected
unscheduled explanation coverage: 14/14
warning-code correctness: 16/16
geographic compactness proxy: 15/15 within <= 80 km or unknown route
```

Targeted corrections:

- Reconciled public runtime and OpenAPI fields.
- Removed public internal fields:
  `trip.recommendationSummary`, `stop.durationPolicyCategory`,
  `stop.recommendation`, and `travelFromPrevious.legOrder`.
- Retained and documented traveler-safe fields:
  `trip.explanation`, `trip.dataFreshness`, and
  `travelFromPrevious.warnings`.
- Active `durationSource` enum is exactly `requested`,
  `category_default`, and `fallback`.
- Marked all 18 fixture cases with explicit execution mode.
- Executed the two synthetic cases through production scheduling modules.
- Corrected Batch 3 status wording to `IMPLEMENTED ON REVIEW BRANCH - NOT
  MERGED`.

Safety:

- Original repository `D:\POI-urban-danang-BE` remained untouched.
- Canonical CSV, package files, migrations, environment files, frontend,
  mobile, database schemas, and credentials were not modified.
- No production/shared database or Firebase production was touched.
- No external POI source, routing provider, persistence, saved-trip, replan,
  mutation, feedback, second-city, Batch 4, or later-phase work occurred.

## 2026-08-02 08:42:21 +07:00 - Phase 2 Batch 3 OpenAPI LF Reproducibility Fix

Type: repository-normalization correction for PR #9.

Implementation clone:

```text
C:\tmp\urbanagent-phase2-batch3-implementation-20260801-164730
```

Previous branch HEAD:

```text
a670f9e4287b2e5309abc73d79d139bedad3c67d
```

Correction:

- Added a narrowly scoped `.gitattributes` rule for the OpenAPI artifact:
  `docs/rebuild/PHASE2_TRAVELER_API_V2_OPENAPI_DRAFT.json text eol=lf`.
- Preserved OpenAPI semantic JSON content.
- Clarified in `TEST_REPORT.md` that the expected OpenAPI SHA-256 is the
  repository-controlled LF checkout byte hash.

Pre-commit validation plan:

- Verify raw OpenAPI bytes hash to
  `0cf59e434e270cee80154ac20cf4c32b14c4147b6f8d174364ea93caae326034`.
- Verify fresh clones with `core.autocrlf=true` and `core.autocrlf=false`
  both check out the OpenAPI artifact with the same LF byte hash.
- Run dependency, audit, test, OpenAPI parse, canonical hash, and POI count
  checks before push.

Safety:

- No runtime JavaScript, tests, fixtures, package files, canonical CSV,
  migrations, environment files, frontend, mobile, database, Firebase,
  external-source, second-city, Batch 4, main push, tag, or PR merge occurred.
## 2026-08-02 10:28 +07:00 - Demo Sprint Part B Backend Per-Day Windows

Repository clone:

`C:\tmp\urbanagent-demo-backend-20260802-101316`

Branch:

`demo/2026-08-07-per-day-windows`

Read-only original repositories preserved:

- `D:\POI-urban-danang-BE`
- `D:\POI-urban-danang-FE`

Commands run:

```text
git status --short --branch
git branch --show-current
git rev-parse HEAD
git rev-parse origin/main
Get-Content AGENTS.md
Get-Content URBANAGENT_CODEX_CONTEXT.md
Get-Content D:\POI-urban-danang-FE\AGENTS.md
Get-Content D:\POI-urban-danang-FE\PLANNING.md
Get-Content D:\POI-urban-danang-FE\URBANAGENT_CODEX_CONTEXT.md
Get-Content D:\POI-urban-danang-FE\package.json
git switch -c demo/2026-08-07-per-day-windows
rg -n "dailyWindow|startTime|dayCount|days|scheduleCandidates|validateTripPreviewRequest|TripPreviewRequest|TripPreviewDay" src\modules\travelerApiV2 tests\phase2 docs\rebuild\PHASE2_TRAVELER_API_V2_OPENAPI_DRAFT.json
node --check src\modules\travelerApiV2\tripPreviewValidation.js
node --check src\modules\travelerApiV2\tripPreview.js
node --check tests\phase2\phase2TravelerApiV2Batch3.test.js
node -e <OpenAPI JSON parse>
node --test tests\phase2\phase2TravelerApiV2Batch3.test.js
npm.cmd test
npm.cmd audit --omit=dev
npm.cmd ls --omit=dev --all
node -e <per-day-window HTTP smoke>
Get-FileHash -Algorithm SHA256 docs\rebuild\PHASE2_TRAVELER_API_V2_OPENAPI_DRAFT.json
git diff --name-status
git diff --stat
```

Files changed:

```text
docs/rebuild/CURRENT_STATE.md
docs/rebuild/DECISIONS.md
docs/rebuild/PHASE2_BATCH3_TRIP_PREVIEW_API_CONTRACT.md
docs/rebuild/PHASE2_TRAVELER_API_V2_OPENAPI_DRAFT.json
docs/rebuild/TEST_REPORT.md
docs/rebuild/WORKLOG.md
src/modules/travelerApiV2/tripPreview.js
src/modules/travelerApiV2/tripPreviewValidation.js
tests/phase2/phase2TravelerApiV2Batch3.test.js
```

Implementation summary:

- Preserved existing `trip.dailyWindow.start` / `end` parsing.
- Added support for `trip.dailyWindow.startTime` / `endTime` aliases.
- Added optional `trip.dayWindows[]` per-day overrides with unique
  `dayNumber` from 1 to 7 and `dayNumber` within `trip.dayCount`.
- Scheduler now resolves a per-day override first, then falls back to
  `trip.dailyWindow`, then `trip.startTime`.
- Response remains stateless; each `trip.days[]` item exposes its resolved
  `dailyWindow`.
- No trip persistence, mutation, authentication, external routing, external
  POI source, production DB, Firebase production, second city, mobile, or
  Batch 4 work was started.

Validation results:

```text
Batch 3 focused tests: 12 total, 12 passed, 0 failed
Full backend tests: 40 total, 39 passed, 0 failed, 1 guarded PostGIS skip
npm audit --omit=dev: 0 vulnerabilities
npm ls --omit=dev --all: PASS, no ELSPROBLEMS
Per-day-window HTTP smoke: PASS, 200, 5 stops
OpenAPI SHA-256: 270fedc98292fab7bb5661609b6b2baf5c72f2139538bd1a9d3045711fb1fd05
Canonical SHA-256 unchanged: 5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae
Canonical application POIs: 4166
```

Next recommended action:

Continue the video-ready demo sprint with frontend integration in a fresh
frontend clone, using the real backend contract and deterministic local demo
presets. Do not modify the original `D:\` repositories.

## 2026-08-02 10:45 +07:00 - Demo Sprint Part B Backend Window Span Correction

Reason:

The required two-day video demo uses Day 1 `09:00-20:00`, which is an
11-hour local day window. The first Part B implementation still used the
earlier 480-minute maximum window span, so the mandatory smoke request was
correctly rejected as `span_minutes_between_15_and_480`.

Correction:

- Keep `trip.durationMinutes` maximum at `480`.
- Increase local day-window span validation to `15-960` minutes.
- Preserve same-day-only windows.
- Keep per-day overrides stateless and deterministic.

Validation:

```text
Focused Batch 3 tests after correction: 12 total, 12 passed, 0 failed
Mandatory backend HTTP smoke:
  default dailyWindow 09:00-20:00: PASS
  two-day plan Day 1 09:00-20:00 and Day 2 08:00-16:00: PASS
  Day 2 latest departure: 10:19
  Day 2 ends by 16:00: true
  repeated request deterministic: true
New OpenAPI SHA-256: 5dc88bb27797626e4564c6b324e19ed286ae1239bd03eda8d9c736a3aa892988
```
