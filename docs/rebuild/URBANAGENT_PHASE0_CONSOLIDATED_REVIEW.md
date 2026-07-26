# UrbanAgent Phase 0 Consolidated Review

Review time: 2026-07-26 09:01:19 +07:00.

Scope: final Phase 0 report artifact refresh only. Application code, tests, CSV files, context files, AGENTS files, package files, dependencies, Firebase data, Git history, and Phase 1 implementation were not modified in this report-refresh pass.

## Final Verdict

`PHASE 0 PASSED — READY FOR USER REVIEW AND COMMIT`

Phase 1 has not started.

## Final Data Artifact Tracking Fix

The canonical runtime CSV is included in repository delivery and is no longer hidden by `.gitignore`.

Canonical file:

```text
data/canonical/urbanagent_poi_master_v1.csv
```

Tracking mode: Git LFS.

Evidence:

```text
git lfs env: git-lfs/3.5.1
.gitattributes: data/canonical/urbanagent_poi_master_v1.csv filter=lfs diff=lfs merge=lfs -text
git check-attr filter -- data/canonical/urbanagent_poi_master_v1.csv: filter: lfs
git check-ignore -v data/canonical/urbanagent_poi_master_v1.csv: no output
git status --short -- data/canonical/urbanagent_poi_master_v1.csv .gitignore .gitattributes:
 M .gitattributes
 M .gitignore
?? data/canonical/urbanagent_poi_master_v1.csv
```

SHA-256 after tracking fix:

```text
5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae
```

Raw/legacy input ignore rules were kept unchanged.

Clean clone delivery: once this Phase 0 work is committed and pushed with Git LFS objects, a clean clone will contain the runtime dataset required by the backend tests and runtime loader.

## Final Validation Results

Backend tests:

```text
npm.cmd test
tests: 10
pass: 10
fail: 0
```

Canonical runtime:

```text
runtime POIs: 4166
quality.rows: 4166
quality.applicationPois: 4166
quality.invalidRows: 0
quality.headerMatchesExpected: true
```

Recommendation smoke:

```text
count: 3
```

Itinerary smoke:

```text
stops: 3
```

Missing-origin itinerary diagnostic:

```json
{
  "distanceKm": null,
  "estimatedMinutes": null,
  "transport": "motorbike",
  "source": "missing-origin",
  "distanceKnown": false
}
```

Merged-source provenance diagnostic:

```json
{
  "source": "google_maps+foody",
  "preserved": true
}
```

EDA source compatibility diagnostic:

```text
google_maps: 3946
foody: 225
all: 4166
unknown: 3946
```

Frontend:

```text
TypeScript no-emit: PASS
production build: PASS
```

Production build command:

```text
D:\POI-urban-danang-FE\node_modules\.bin\vite.cmd build --outDir C:\tmp\urbanagent-fe-build-phase0-final-artifacts --emptyOutDir
```

Build note: Vite still warns that one JS chunk is larger than 500 kB. This is accepted pre-existing technical debt for Phase 0.

## Corrections Documented

1. BOM/header handling:
   - `csv-parser` uses `mapHeaders`.
   - UTF-8 BOM is stripped from the first header.
   - surrounding header whitespace is trimmed.

2. Canonical 4166 POI runtime:
   - canonical `loadPOIs({ cityId: "da-nang" })` returns 4166 POIs.
   - quality report passes expected row/application/invalid/header checks.

3. `imageUrls` semantics:
   - `Image_URL` is split by comma.
   - whitespace is trimmed.
   - empty and invalid non-http/non-https values are removed.
   - duplicate URLs are removed while preserving source order.
   - normalized POIs expose `imageUrls`.
   - backward-compatible `imageUrl` equals the first valid URL or `null`.

4. Nullable missing rating/review values:
   - missing ratings and review counts remain `null`.
   - Foody review sample counts are not treated as total review counts.

5. Nullable unknown-origin itinerary distance/time:
   - missing start location keeps `distanceKm: null`.
   - missing start location keeps `estimatedMinutes: null`.
   - `distanceKnown` remains `false`.
   - `source` remains `missing-origin`.
   - no Da Nang-center origin fallback is used.
   - later known itinerary legs continue using numeric distance/time.

6. `google_maps+foody` provenance preservation:
   - Firestore POI normalization preserves `source: "google_maps+foody"`.
   - it no longer normalizes merged-source POIs to `manual`.

7. EDA source compatibility:
   - missing/ggmap/google/google_maps/unknown source requests map to Google-compatible POIs.
   - Google-compatible count includes `google_maps+foody`: 3946.
   - Foody-compatible count includes `google_maps+foody`: 225.
   - all/canonical returns all POIs: 4166.
   - shared helper is used by both `server.js` and `src/server.js`.

8. Backend tests:
   - 10/10 pass.
   - tests live at `tests/phase0/phase0CanonicalData.test.js`.

9. Frontend TypeScript/build:
   - TypeScript no-emit passes.
   - production build passes to `C:\tmp`.

10. Accepted frontend technical debt:
   - frontend lint remains pre-existing debt from earlier verification.
   - frontend has no test script.
   - both are accepted as outside Phase 0 correction scope.

11. Legacy CSV restoration:
   - `data/poi_data_ggmap.csv` and `data/poi_data_foody.csv` are not deleted or modified.
   - `git status --short -- data\poi_data_ggmap.csv data\poi_data_foody.csv` returned no output.

12. No Phase 1 work:
   - no PostgreSQL/PostGIS work was started.
   - no migration was created or run.
   - no external data adapter was added.
   - no frontend redesign was performed.

## Context Hash Equality

Backend context:

```text
D:\POI-urban-danang-BE\URBANAGENT_CODEX_CONTEXT.md
SHA-256: 2EC5ACC2E8AF8B1553E94EB17A332C8C6D03675EF74D214DF4AD0D010D28BF0F
```

Frontend context:

```text
D:\POI-urban-danang-FE\URBANAGENT_CODEX_CONTEXT.md
SHA-256: 2EC5ACC2E8AF8B1553E94EB17A332C8C6D03675EF74D214DF4AD0D010D28BF0F
```

Status: PASS.

## Exact Git Status

Backend:

```text
## main...origin/main
 M .gitattributes
 M .gitignore
 M ES-system/poi_density_engine.js
 M inference.py
 M package.json
 M server.js
 M src/expert_system/poi_density_engine.js
 M src/inference.py
 M src/server.js
 M src/services/contextualPoiRecommenderService.js
 M src/services/firestorePersistenceService.js
 M src/services/itineraryPlannerService.js
 M src/services/poiDataService.js
 M src/services/poiRetrievalService.js
?? AGENTS.md
?? URBANAGENT_CODEX_CONTEXT.md
?? data/canonical/urbanagent_poi_master_v1.csv
?? data/canonical/urbanagent_poi_master_v1_manifest.json
?? docs/rebuild/CURRENT_STATE.md
?? docs/rebuild/DATA_AUDIT.md
?? docs/rebuild/DECISIONS.md
?? docs/rebuild/MASTER_PLAN.md
?? docs/rebuild/TEST_REPORT.md
?? docs/rebuild/URBANAGENT_DATASET_DECISION.md
?? docs/rebuild/URBANAGENT_PHASE0_CONSOLIDATED_REVIEW.md
?? docs/rebuild/URBANAGENT_PHASE0_FULL.patch
?? docs/rebuild/WORKLOG.md
?? src/services/canonicalCsvPoiRepository.js
?? src/services/legacyCsvPoiRepository.js
?? src/services/poiRepository.js
?? tests/phase0/phase0CanonicalData.test.js
```

Frontend:

```text
## main...origin/main
?? AGENTS.md
?? URBANAGENT_CODEX_CONTEXT.md
?? poi_urban_web.code-workspace
```

Legacy root CSV status:

```text
git status --short -- data\poi_data_ggmap.csv data\poi_data_foody.csv
```

Result: no output.

## Commit Recommendation

Phase 0 may be committed after user review.

Recommended commit hygiene:

- stage only intended Phase 0 source, tests, docs, context/rule files, and approved data artifacts.
- do not stage `.env`, Firebase credentials, `node_modules`, `dist`, temp build output, secrets, or unrelated local files.
- decide whether `poi_urban_web.code-workspace` should remain untracked.

## Phase 1 Readiness

Phase 1 may not begin yet.

Phase 1 requires explicit user approval:

```text
APPROVED PHASE 1
```
