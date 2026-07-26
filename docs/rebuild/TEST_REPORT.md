# Test Report

Updated: 2026-07-26 13:28:46 +07:00.

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

## Phase 1 Batch 1

Updated: 2026-07-26 10:08:23 +07:00.

Commands run:

```text
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

Results:

- `npm.cmd install`: PASS; added `pg` dependency. NPM audit summary reported `12 vulnerabilities (1 low, 8 moderate, 3 high)`. No audit fix was run.
- `npm.cmd test`: PASS.
- Node test runner summary: 15 tests, 15 pass, 0 fail.
- Syntax checks: PASS.
- Import dry-run: PASS.
- Frontend production build: PASS.
- Vite warning remains: one JS chunk is `1,368.16 kB`.

Phase 1 test coverage added:

- migration defines PostGIS/pgcrypto and required foundation tables.
- migration does not contain `DROP TABLE` or `TRUNCATE`.
- Da Nang City Pack config is explicit and converts to a PostGIS bbox polygon.
- canonical import plan preserves approved dataset hash and plans 4,166 POIs.
- import plan produces source records, external IDs, aliases, images, and review summaries.
- POI entity records preserve null semantics and field provenance.
- Postgres repository mapper preserves the legacy API-compatible POI shape and does not expose `placeId`.

Import dry-run output:

```text
mode: dry-run
cityId: da-nang
source provider: legacy_canonical_csv
source policyClass: legacy-canonical
source path: data/canonical/urbanagent_poi_master_v1.csv
source sha256: 5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae
source rows: 4166
applicationPois: 4166
invalidRows: 0
headerMatchesExpected: true
sourceRecords: 4166
externalIds: 8337
aliases: 985
images: 16246
reviewSummaries: 4166
```

Not run:

- No database migration was run.
- No write import was run.
- No production Firebase operation was run.

## Phase 1 Batch 2

Updated: 2026-07-26 12:20:17 +07:00.

Disposable database:

```text
Docker Compose file: docker-compose.phase1.yml
Image: postgis/postgis:16-3.5-alpine
Host port: 55432
Database: disposable local Phase 1 test database
```

Commands run:

```text
docker compose -f docker-compose.phase1.yml up -d
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
npm.cmd test
npm.cmd run phase1:import:canonical -- --dry-run
Get-FileHash -Algorithm SHA256 data\canonical\urbanagent_poi_master_v1.csv
node -e <CSV default runtime check>
node --check <Phase 1 scripts, services, and tests>
npm.cmd audit --omit=dev
npm.cmd audit --omit=dev --json
npm.cmd audit --omit=dev --registry=http://registry.npmjs.org
npm.cmd ls pg --omit=dev
npm.cmd ls --omit=dev --depth=0
docker compose -f docker-compose.phase1.yml down -v
```

Results:

- Migration apply: PASS.
- First real write import: PASS.
- First import counts:
  - POI entities: `4166`
  - source records: `4166`
  - external IDs: `8337`
  - aliases: `985`
  - images: `16246`
  - review summaries: `4166`
- Second real write import: PASS.
- Second import core counts: unchanged.
- Idempotency: PASS for core tables; `ingestion_runs` increments by design.
- Rollback: PASS; Phase 1 tables removed.
- Reapply migration and final import: PASS.
- Final diagnostics: PASS.
- PostgresPoiRepository integration: PASS.
- CSV/Postgres selected parity: PASS.
- CSV default runtime: PASS, `CanonicalCsvPoiRepository`, count `4166`.
- Canonical SHA-256: `5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae`.
- Container/volume cleanup: PASS.

Final database diagnostics:

```text
cities: 1
poi_entities: 4166
source_records: 4166
external_ids: 8337
aliases: 985
images: 16246
review_summaries: 4166
duplicate source records: 0
duplicate external IDs: 0
duplicate aliases: 0
duplicate image associations: 0
orphan source records/external IDs/aliases/images/reviews: 0
invalid longitude: 0
invalid latitude: 0
null geometry: 0
wrong SRID: 0
coordinate mismatch: 0
outside Da Nang envelope: 0
merged provenance rows: 5
invalid rating ranges: 0
image ordering anomalies: 0
GiST index: poi_entities_location_gix
EXPLAIN observed the GiST index in the representative bbox query.
```

Tests:

```text
Unit/default suite without DB integration:
tests 16
pass 15
fail 0
skipped 1

Full suite with URBANAGENT_PHASE1_INTEGRATION=true:
tests 16
pass 16
fail 0
skipped 0
```

`npm audit --omit=dev`:

- Result: FAIL due npm audit endpoint response parsing failure.
- Error observed: invalid JSON response body from
  `https://registry.npmjs.org/-/npm/v1/security/advisories/bulk`; response
  began with gzip/binary bytes and npm could not parse it as JSON.
- HTTP fallback returned `426 Upgrade Required`.
- No `npm audit fix` or `npm audit fix --force` was run.
- `npm install` previously reported `12 vulnerabilities (1 low, 8 moderate, 3 high)`.
- Direct production dependencies are `cors`, `csv-parser`, `express`,
  `firebase-admin`, `multer`, and `pg`.
- New `pg` direct dependency is installed as `pg@8.22.0`.
- Official npm Bulk Advisory POST with gzip fallback decoding returned a usable
  production-only classification.

## Phase 1 Batch 2 Security Audit Closure

Updated: 2026-07-26 12:31:18 +07:00.

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

NPM CLI audit behavior:

```text
npm audit --omit=dev: FAIL
npm audit --omit=dev --json: FAIL
empty userconfig audit: FAIL
isolated cache audit: FAIL
WSL npm audit: FAIL
HTTP registry fallback: FAIL, 426 Upgrade Required
```

Observed failure:

```text
invalid json response body at https://registry.npmjs.org/-/npm/v1/security/advisories/bulk
Unexpected token '\u001f', gzip/binary response was parsed as JSON
```

Production advisory classification source:

```text
Official npm Bulk Advisory POST endpoint
Accept-Encoding: identity
Response was still gzip; decoded locally before JSON parsing
Production dependency tree source: npm ls --omit=dev --all --json
Package/version source: package-lock.json production packages only
```

Production-only summary:

```text
production packages checked: 252
advisory records affecting installed production versions: 9
severity: 1 low, 4 moderate, 4 high
direct advisories: 2
transitive advisories: 7
advisories affecting new pg path: 0
pg version: 8.22.0
```

Production advisories:

```text
body-parser@2.2.2
- severity: low
- range: >=2.0.0 <2.3.0
- path: express@5.2.1 > body-parser@2.2.2
- direct: no
- fix availability from bulk payload: none listed
- remediation character: likely transitive Express/body-parser update when available
- affects pg path: no

brace-expansion@2.1.1
- severity: high
- range: >=2.0.0 <2.1.2
- path: firebase-admin > @google-cloud/firestore > google-gax > rimraf > glob > minimatch > brace-expansion
- direct: no
- fix availability from bulk payload: none listed
- remediation character: transitive Firebase/Google dependency update or override only with approval
- affects pg path: no

brace-expansion@2.1.1
- severity: high
- range: <=5.0.7
- path: same Firebase/Google transitive path
- direct: no
- fix availability from bulk payload: none listed
- remediation character: transitive Firebase/Google dependency update or override only with approval
- affects pg path: no

form-data@2.5.5
- severity: high
- range: <2.5.6
- path: firebase-admin > @google-cloud/storage > retry-request > @types/request > form-data
- direct: no
- fix availability from bulk payload: none listed
- remediation character: transitive Firebase/Google dependency update or override only with approval
- affects pg path: no

multer@2.1.1
- severity: high
- range: >=1.0.0 <2.2.0
- path: multer@2.1.1
- direct: yes
- fix availability from bulk payload: none listed
- remediation character: direct minor/patch update likely needed, but not changed in this task
- affects pg path: no

multer@2.1.1
- severity: moderate
- range: >=2.0.0-alpha.1 <2.2.0
- path: multer@2.1.1
- direct: yes
- fix availability from bulk payload: none listed
- remediation character: direct minor/patch update likely needed, but not changed in this task
- affects pg path: no

protobufjs@7.6.3
- severity: moderate
- range: >=7.5.0 <=7.6.4
- paths: Firebase/Google Firestore/gax/protobufjs transitive paths
- direct: no
- fix availability from bulk payload: none listed
- remediation character: transitive Firebase/Google dependency update or override only with approval
- affects pg path: no

qs@6.15.1
- severity: moderate
- range: >=6.11.1 <=6.15.1
- paths: express@5.2.1 > qs and express > body-parser > qs
- direct: no
- fix availability from bulk payload: none listed
- remediation character: transitive Express/qs update or override only with approval
- affects pg path: no

uuid@9.0.1
- severity: moderate
- range: <11.1.1
- paths: firebase-admin > @google-cloud/storage > gaxios/teeny-request > uuid
- direct: no
- fix availability from bulk payload: none listed
- remediation character: transitive Firebase/Google dependency update or override only with approval
- affects pg path: no
```

Conclusion:

- The new Phase 1 `pg@8.22.0` dependency and its dependency path have no
  production advisories in this classification.
- Existing production advisories are in Express/Multer/Firebase transitive
  surfaces and remain unresolved because this task prohibited dependency changes.
- No package file, lockfile, application code, migration SQL, tests, frontend
  file, production database, or Firebase data was modified during audit closure.

## Phase 1 Security Remediation

Updated: 2026-07-26 12:52:16 +07:00.

Commands run:

```text
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

Remediation applied:

```text
firebase-admin: ^14.0.0 -> ^14.2.0
multer: ^2.1.1 -> ^2.2.0
override body-parser -> 2.3.0
override brace-expansion -> 5.0.8
override form-data -> 2.5.6
override protobufjs -> 7.6.5
override qs -> 6.15.3
override uuid -> 11.1.1
pg remains 8.22.0
```

Results:

- `npm.cmd audit --omit=dev`: PASS, `found 0 vulnerabilities`.
- `npm.cmd audit --omit=dev --json`: PASS.
  - info: `0`
  - low: `0`
  - moderate: `0`
  - high: `0`
  - critical: `0`
  - total: `0`
  - production dependencies counted by audit metadata: `156`
  - total dependencies counted by audit metadata: `298`
- `npm.cmd test`: PASS.
  - tests: `16`
  - pass: `15`
  - fail: `0`
  - skipped: `1` because the disposable Postgres integration test skips when
    database environment variables are absent.
- Syntax checks: PASS for server entrypoints, Phase 1 migration runner,
  Postgres repository, importer, and Phase 1 integration test.
- `npm.cmd ls --omit=dev --depth=0`: PASS.
  - `cors@2.8.6`
  - `csv-parser@3.2.0`
  - `express@5.2.1`
  - `firebase-admin@14.2.0`
  - `multer@2.2.0`
  - `pg@8.22.0`
- `npm.cmd ls --omit=dev --all`: residual `ELSPROBLEMS` warning from npm's
  override-tree validation under `get-intrinsic` /
  `call-bind-apply-helpers`; no audit advisory remains.

Safety confirmations:

- No `npm audit fix`, `npm audit fix --force`, `npm update`, `npm upgrade`, or
  `npm dedupe` was run.
- No application source code, migration SQL, tests, canonical CSV, manifest,
  frontend file, production database, or Firebase data was modified.
- PostgreSQL remains opt-in and CSV remains the default runtime.
- Phase 1 Batch 3 was not started.

## Phase 1 Final Dependency Integrity Gate

Updated: 2026-07-26 13:08:43 +07:00.

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
```

Integrity correction:

```text
override call-bind-apply-helpers -> 1.0.2
override get-intrinsic -> 1.3.0
override hasown -> 2.0.4
```

Results:

- Initial `npm.cmd ls --omit=dev --all`: FAIL, exit code `1`.
  - First invalid paths: `get-intrinsic@1.3.0` and
    `call-bind-apply-helpers@1.0.2` under the `qs` / `side-channel` tree.
  - After adding those two overrides, the remaining invalid path was
    `hasown@2.0.4` under `form-data@2.5.6` / `es-set-tostringtag@2.1.0`.
- Final `npm.cmd ls --omit=dev --all`: PASS, exit code `0`; no
  `ELSPROBLEMS`.
- `npm.cmd ci --ignore-scripts`: PASS; clean install from `package-lock.json`
  is reproducible.
- `npm.cmd audit --omit=dev`: PASS, `found 0 vulnerabilities`.
- `npm.cmd audit --omit=dev --json`: PASS.
  - info: `0`
  - low: `0`
  - moderate: `0`
  - high: `0`
  - critical: `0`
  - total: `0`
  - production dependencies counted by audit metadata: `156`
  - total dependencies counted by audit metadata: `298`
- `npm.cmd test`: PASS.
  - tests: `16`
  - pass: `15`
  - fail: `0`
  - skipped: `1` because the disposable Postgres integration test skips when
    database environment variables are absent.
- Syntax checks: PASS for server entrypoints, Firebase Admin config, Phase 1
  migration runner, Postgres repository, importer, and Phase 1 integration
  test.
- Firebase Admin compatibility smoke: PASS for `require('firebase-admin')`,
  `require('firebase-admin/auth')`, and `require('firebase-admin/firestore')`
  entrypoints used by UrbanAgent. The smoke did not initialize Firebase and did
  not connect to production.
- Multer compatibility smoke: PASS for `multer(...).single('image')`, matching
  UrbanAgent upload middleware usage.

Safety confirmations:

- No `npm audit fix`, `npm audit fix --force`, `npm update`, `npm upgrade`, or
  `npm dedupe` was run.
- No application source code, migration SQL, tests, canonical CSV, manifest,
  frontend file, production database, or Firebase data was modified.
- PostgreSQL remains opt-in and CSV remains the default runtime.
- Draft PR #2 remains unmerged and Batch 3 was not started.

## Phase 1 Draft PR Final Readiness Gate

Updated: 2026-07-26 13:28:46 +07:00.

Package placement:

- `call-bind-apply-helpers@1.0.2`, `get-intrinsic@1.3.0`, and
  `hasown@2.0.4` are under `overrides`.
- They are not top-level production dependencies.
- Top-level production dependencies remain `cors`, `csv-parser`, `express`,
  `firebase-admin`, `multer`, and `pg`.
- `devDependencies` is empty.
- Package-lock paths resolve the three integrity overrides under
  `node_modules/call-bind-apply-helpers`, `node_modules/get-intrinsic`, and
  `node_modules/hasown`.

Clean clone:

```text
path: C:\tmp\urbanagent-phase1-readiness-clone
branch: phase1/data-platform-foundation
HEAD: c160f61edc45cbde5da772f1c9f564336b4de41c
git lfs pull: PASS
npm.cmd ci: PASS
git status after npm ci and checks: clean
package.json/package-lock diff after npm ci: none
```

Clean-clone commands and results:

```text
npm.cmd ci
  PASS, added 298 packages and audited 299 packages.
  No lifecycle script failed.
  No credentials were required.

npm.cmd ls --omit=dev --all
  PASS, exit code 0, no ELSPROBLEMS.

npm.cmd audit --omit=dev
  PASS, found 0 vulnerabilities.

npm.cmd audit --omit=dev --json
  PASS, total vulnerabilities 0.

npm.cmd test
  PASS, tests 16, pass 15, fail 0, skipped 1.
  The single skipped test is the disposable Postgres integration test because
  the clean clone did not enable the DB integration environment.
```

Full disposable PostGIS integration:

```text
image: postgis/postgis:16-3.5-alpine
compose file: docker-compose.phase1.yml
host port: 55432
database: urbanagent_phase1_test
container: urbanagent-phase1-postgis

wsl.exe docker compose -f docker-compose.phase1.yml up -d
  PASS

healthcheck
  PASS, healthy

URBANAGENT_PHASE1_INTEGRATION=true npm.cmd test
  PASS, tests 16, pass 16, fail 0, skipped 0.

npm.cmd run phase1:db:diagnostics
  PASS

wsl.exe docker compose -f docker-compose.phase1.yml down -v
  PASS

container cleanup check
  PASS, no urbanagent-phase1-postgis container remains.

volume cleanup check
  PASS, no urbanagent_phase1_postgis_data volume remains.
```

Final DB diagnostics:

```text
poi_entities: 4166
source_records: 4166
external_ids: 8337
aliases: 985
images: 16246
review_summaries: 4166
duplicate checks: 0
orphan checks: 0
invalid longitude: 0
invalid latitude: 0
null geometry: 0
wrong SRID: 0
coordinate mismatch: 0
outside Da Nang envelope: 0
GiST index: poi_entities_location_gix
```

Canonical data:

```text
SHA-256: 5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae
runtime count: 4166
CSV default runtime count without URBANAGENT_POI_REPOSITORY: 4166
```

PR file hygiene:

- `git --no-pager diff --name-status main...HEAD`: reviewed.
- `git --no-pager diff --stat main...HEAD`: reviewed.
- `git --no-pager log --oneline --decorate main..HEAD`: reviewed.
- No `.env`, Firebase credential JSON, database dump, diagnostic bundle, raw
  audit output, `node_modules`, `dist`, temporary workspace file, accidental
  `status` file, or full patch artifact was found in the PR diff.
- `package.json` and `package-lock.json` changes are limited to approved
  Phase 1 `pg` dependency, approved security remediation, and final dependency
  integrity overrides.

Safety confirmations:

- No application source, migration SQL, tests, canonical CSV, manifest,
  frontend code, production/shared database, or Firebase production data was
  modified during this final gate.
- No `npm audit fix`, `npm audit fix --force`, `npm update`, `npm upgrade`, or
  `npm dedupe` was run.
- PostgreSQL remains explicit opt-in and CSV remains the default runtime.
- Branch is synchronized with `origin/phase1/data-platform-foundation`.
- Git state confirms Phase 1 `HEAD` is not in `origin/main`; PR #2 was not
  merged in this gate.
- Batch 3 was not started.

Final readiness verdict:

`READY FOR REVIEW`
