# Phase 5 Stage 5A - Real Product E2E

Status: `PASSED_WITH_DEPLOYMENT_BLOCKER`

Date: 2026-08-14 (+07:00)

## Runtime baseline

- Canonical traveler POIs: `4173` unique, `0` invalid.
- Stage 4S entities loadable: `7/7`.
- Canonical SHA-256:
  `dcb404cc8b5c7a9b5fd70df63039ab8f828c504270e22b12a671fa4ed61583f4`.
- CSV remains the default runtime; PostgreSQL remains explicit opt-in.
- Canonical bytes, scheduler policy, Firebase and production databases were not changed.

## Traveler scenarios

Five bounded Vietnamese scenarios passed through the real recommendation and
trip-preview endpoints: cafe plus seafood; one-day landmarks, local food and
beach; relaxed family travel; named Ngũ Hành Sơn plus nearby food/cafe; and
sunset plus dinner/cafe. Every response was nonempty, unique, coordinate-valid
and inside the selected daily end time. The named Ngũ Hành Sơn entity was
present in its scenario.

Browser validation passed login, `/urban-agent`, recommendations, two-day trip
preview, marker-to-stop focus, remove/add/reorder, dirty state, replan, save and
list, day switching, mobile layout and logout. Replan kept the removed POI
excluded. Saved-trip create/list/get/replan/delete and owner isolation passed
against the guarded memory adapter.

## Defects fixed

- BLOCKER: none found.
- HIGH: anonymous requests inherited global demo memory; substring intent
  matching confused `biển` with words such as `Điện Biên`; multi-intent and
  named-place retrieval were truncated; reranker boosts saturated scores;
  frontend sent `walking` instead of API v2 `walk`; overlapping Leaflet
  transitions raised a runtime exception; desktop timeline was compressed;
  malformed JSON exposed Express HTML; duplicate warning keys produced React
  errors. All were fixed.
- MEDIUM: backend-unavailable state exposed `Failed to fetch`; it now shows a
  Vietnamese retryable message.
- Test and opt-in Postgres diagnostics were synchronized to the approved 4173
  baseline (`4173` entities/source/reviews and `8344` external IDs).

## Validation and performance

- Backend syntax checks: PASS.
- Backend tests: `48` total, `47` passed, `0` failed, `1` guarded PostGIS skip.
- Frontend production build: PASS; scoped lint: PASS.
- Browser smoke: PASS with no new console/runtime error after the final fixes.
- Recommendation: approximately `1.6-2.3 s` across five bounded scenarios.
- Trip preview: approximately `1.5-1.8 s`; replan approximately `1.6 s`.
- Vite development startup: approximately `3.5 s`.
- Production build still reports the accepted existing large-chunk warning
  (`~1.41 MB`, `~416 KB` gzip).

## Deployment readiness

Frontend clean install/build passed. The backend start command and configurable
API base URL work locally, and no committed secret or machine-specific runtime
path was introduced. Production must set the API base URL and review CORS and
the frontend dependency advisories before launch.

`DEPLOYMENT_BLOCKED_BY_LFS`: in a clean backend clone, `git lfs pull` cannot
download `data/canonical/urbanagent_poi_master_v1.csv` because the repository
LFS budget is exceeded. The pointer remains unresolved without hidden local
state, so a clean deployment cannot obtain the required runtime dataset. The
local product validation used separately verified bytes only and does not prove
deployment reproducibility.

## Recommended next step

`FIX CONCRETE DEPLOYMENT BLOCKER`: restore a repository-native, reproducible way
for a clean environment to obtain the approved canonical runtime dataset before
staging deployment.
