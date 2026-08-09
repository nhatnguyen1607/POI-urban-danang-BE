# Phase 2 Final Exit Report

Updated: 2026-08-09 14:49:50 +07:00.

Verdict: `PHASE 2 FINAL GATE PENDING`.

Phase 2 cannot be declared closed yet because the mandatory CSV/PostgreSQL
parity gate could not execute: no local Docker daemon was available for the
approved disposable PostGIS service.

## Completed Phase 2 Scope

Closed before this final gate:

- Batch 1: Traveler API v2 foundation.
- Batch 2: `POST /api/v2/recommendations`.
- Batch 3: `POST /api/v2/trips/preview`.
- Batch 4: editable itinerary and frontend replanning workflow.
- Batch 5: authenticated saved-trip persistence.
- Batch 6: saved-trip lifecycle routes and UI.

Feedback persistence is intentionally not implemented in Phase 2.

`POST /api/v2/trips/:tripId/feedback` remains:

- status: `DEFERRED`
- implementation: `NOT_IMPLEMENTED_IN_PHASE_2`
- approval: requires future explicit user approval

## Canonical Integrity

Result: `PASS`.

- Canonical dataset:
  `data/canonical/urbanagent_poi_master_v1.csv`
- SHA-256:
  `5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae`
- Application POIs: `4166`
- Invalid rows: `0`
- Header matches expected schema: `true`
- Canonical CSV bytes were not modified in this final batch.

## Runtime Mode

Result: `PASS`.

- Default repository: `CanonicalCsvPoiRepository`
- CSV remains default runtime.
- PostgreSQL remains explicit opt-in through
  `URBANAGENT_POI_REPOSITORY=postgres`.
- No production database or Firebase production data was touched.
- No external POI source was queried, downloaded, scraped, cached, merged, or
  ingested.

## Legacy Compatibility

Result: `PASS` for CSV-default local final-gate smoke.

Observed counts:

- Legacy `/api/eda?source=google_maps`: `3946`
- Legacy `/api/eda?source=foody`: `225`
- Legacy `/api/eda?source=all`: `4166`
- v2 `/api/v2/pois/search?source=google_maps`: `3946`
- v2 `/api/v2/pois/search?source=foody`: `225`
- v2 `/api/v2/pois/search?source=all`: `4166`

Traveler runtime smoke:

- `POST /api/v2/recommendations`: `PASS`, `5` results.
- `POST /api/v2/trips/preview`: `PASS`, `3` stops.
- Missing-origin first leg:
  - `distanceMeters`: `null`
  - `travelDurationMinutes`: `null`
  - `distanceKnown`: `false`
  - `calculationSource`: `missing-origin`
- Saved-trip replan preserved the same `tripId`: `PASS`.
- `POST /api/v2/trips/:tripId/feedback`: `404`, expected because feedback is
  deferred and not implemented in Phase 2.

## CSV/PostgreSQL Parity

Result: `PENDING`.

The existing guarded parity/integration test was not executed in PostgreSQL
mode because the approved disposable Docker/PostGIS prerequisite was not
available.

Observed Docker blocker:

- `docker compose -f docker-compose.phase1.yml up -d`
- Result: `FAIL`
- Reason: Docker CLI could not connect to the Docker Desktop Linux engine at
  `npipe:////./pipe/dockerDesktopLinuxEngine`.
- The default Docker context also had no reachable daemon at
  `npipe:////./pipe/docker_engine`.

No PostgreSQL PASS is claimed in this report.

## Performance

Result: `PASS_WITH_LIMITATION`.

The documented evaluation plan requires p50/p95 recording when enough repeated
runs exist. It does not define numeric latency thresholds.

Local CSV-default repeated run count per endpoint: `10`.

| Endpoint | p50 ms | p95 ms |
| --- | ---: | ---: |
| city status | 0.43 | 0.87 |
| POI search first page | 56.39 | 60.19 |
| POI search q | 53.05 | 75.28 |
| recommendation | 362.44 | 493.26 |
| itinerary preview | 357.44 | 431.95 |

Threshold status: `NO_DOCUMENTED_NUMERIC_THRESHOLD`.

## Scientific/Offline Evaluation

Result: `PASS_WITH_LIMITATION`.

Versioned fixtures present:

- Recommendation fixture: `phase2-recommendation-smoke-v1`
- Trip preview fixture: `phase2-trip-preview-smoke-v1`
- Recommendation cases: `1`
- Trip preview cases: `18`

These fixtures are structural smoke/evaluation-foundation fixtures only. They
are not a scored relevance benchmark. No Recall@k, nDCG, MRR, statistical
significance, production-readiness, or traveler-quality superiority claim is
made.

## Backend Tests And Audit

Result: `PASS`.

- `node --check scripts/phase2_final_gate_evaluation.js`: `PASS`
- `node scripts/phase2_final_gate_evaluation.js`: `PASS`
- `npm.cmd test`: `PASS`, `42` total, `41` passed, `0` failed, `1` skipped
- Skipped test: guarded optional disposable PostGIS integration
- `npm.cmd audit --omit=dev`: `PASS`, `0 vulnerabilities`

## Remaining Gate

To close Phase 2, rerun the final gate with a working local disposable
PostGIS environment and execute the CSV/PostgreSQL parity test in explicit
PostgreSQL mode.

Required prerequisite:

- Docker daemon available for `postgis/postgis:16-3.5-alpine`

When that gate passes and documentation is refreshed with the actual parity
result, Phase 2 may be declared closed.
