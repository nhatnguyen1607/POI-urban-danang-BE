# Test Report

Updated: 2026-07-26 08:42:18 +07:00.

## Backend

Commands run:

```text
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
node --check tests\phase0\phase0CanonicalData.test.js
node -e <provenance diagnostic>
node -e <EDA source-filter diagnostic>
```

Results:

- `npm.cmd test`: PASS.
- Node test runner summary: 10 tests, 10 pass, 0 fail.
- Syntax checks: PASS.
- Provenance diagnostic: `google_maps+foody` preserved.
- EDA source-filter diagnostic: Google `3946`, Foody `225`, All `4166`.

Backend tests now cover:

- approved canonical CSV path, row count, unique `Global_ID`, and SHA-256.
- BOM/header normalization for `Global_ID`.
- full canonical repository load of 4,166 application POIs.
- `cityId` filtering.
- no urban-void rows in runtime POIs.
- no duplicate `Global_ID`.
- no missing/invalid coordinates and no Da Nang-center coordinate fabrication.
- `RestaurantID` preserved as `sourceId`, not exposed as `placeId`.
- alias IDs preserved without creating extra POIs.
- `Image_URL` split into unique ordered `imageUrls` with `imageUrl` as the first valid URL or null.
- missing rating/review count remains null.
- `Foody_Review_Sample_Count` is not converted into total review count.
- invalid rows are rejected instead of filled.
- Firestore POI normalization preserves unknown rating/review semantics.
- Firestore POI normalization preserves merged-source `google_maps+foody` provenance.
- EDA source compatibility includes merged-source POIs in Google and Foody views.
- itinerary first leg keeps unknown origin distance/time as `null` with `distanceKnown: false`, while later legs remain numeric.
- recommendation and itinerary service smokes return canonical POIs.
- legacy root `ES-system/poi_density_engine.js` no longer reads old `data/poi_data_*.csv` files.

## Frontend

Commands run:

```text
npm.cmd run lint
D:\POI-urban-danang-FE\node_modules\.bin\tsc.cmd --noEmit --pretty false --project tsconfig.app.json
D:\POI-urban-danang-FE\node_modules\.bin\vite.cmd build --outDir C:\tmp\urbanagent-fe-build-phase0-fix --emptyOutDir
npm.cmd test
```

Results:

- `npm.cmd run lint`: FAIL, `122 problems (119 errors, 3 warnings)`.
- `tsc --noEmit`: PASS.
- Vite temp build: PASS after sandbox escalation for `C:\tmp` output.
- Vite warning: one JS chunk is `1,368.16 kB`, above the 500 kB warning threshold.
- `npm.cmd test`: FAIL because no frontend `test` script exists.

Frontend lint failure remains intentionally unresolved because this task explicitly said not to fix all existing frontend lint errors.
