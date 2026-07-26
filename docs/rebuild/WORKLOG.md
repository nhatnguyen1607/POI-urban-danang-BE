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
