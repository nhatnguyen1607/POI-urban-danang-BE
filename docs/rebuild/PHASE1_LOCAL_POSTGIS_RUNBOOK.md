# Phase 1 Local PostGIS Runbook

Updated: 2026-07-26 14:26:14 +07:00.

## Scope

This runbook is for disposable local Phase 1 PostgreSQL/PostGIS validation only.
It must not be used with a production, shared, or Firebase-backed data store.

CSV remains the default runtime. PostgreSQL runtime is explicit opt-in only with:

```text
URBANAGENT_POI_REPOSITORY=postgres
```

## Disposable Database

```text
Compose file: docker-compose.phase1.yml
Image: postgis/postgis:16-3.5-alpine
Container: urbanagent-phase1-postgis
Host port: 55432
Database: urbanagent_phase1_test
User: urbanagent_test
```

Connection string:

```text
postgres://urbanagent_test:urbanagent_test_password@localhost:55432/urbanagent_phase1_test
```

## Start

```powershell
wsl.exe docker compose -f docker-compose.phase1.yml up -d
```

Wait until the container is healthy:

```powershell
wsl.exe docker inspect --format "{{.State.Health.Status}}" urbanagent-phase1-postgis
```

Expected value:

```text
healthy
```

## Apply Migration And Import

Write operations require the guarded Phase 1 flag:

```powershell
$env:DATABASE_URL="postgres://urbanagent_test:urbanagent_test_password@localhost:55432/urbanagent_phase1_test"
$env:URBANAGENT_ALLOW_PHASE1_DB_WRITE="true"
```

Apply the schema:

```powershell
npm.cmd run phase1:db:migrate
```

Importer dry-run remains the default:

```powershell
npm.cmd run phase1:import:canonical
```

Real write mode requires explicit `--write` plus the guard above:

```powershell
npm.cmd run phase1:import:canonical -- --write
```

## Integration Tests

Enable the disposable DB integration suite:

```powershell
$env:DATABASE_URL="postgres://urbanagent_test:urbanagent_test_password@localhost:55432/urbanagent_phase1_test"
$env:URBANAGENT_ALLOW_PHASE1_DB_WRITE="true"
$env:URBANAGENT_PHASE1_INTEGRATION="true"
npm.cmd test
```

Expected result:

```text
tests 16
pass 16
fail 0
skipped 0
```

The integration suite also starts a local backend child process with
`URBANAGENT_POI_REPOSITORY=postgres` and smokes these endpoints:

```text
GET /api/eda?source=google_maps
GET /api/eda?source=foody
GET /api/pois/data-quality
POST /api/agent/recommend-poi
POST /api/agent/create-itinerary
```

Expected endpoint evidence:

```text
EDA google_maps: 3946
EDA foody: 225
Quality applicationPois: 4166
Recommendation: nonempty, cityId da-nang
Itinerary: nonempty, cityId da-nang
Missing-origin first leg: distanceKm null, distanceKnown false
```

## Diagnostics

```powershell
npm.cmd run phase1:db:diagnostics
```

Expected canonical counts:

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

## Rollback

Rollback is also guarded:

```powershell
$env:DATABASE_URL="postgres://urbanagent_test:urbanagent_test_password@localhost:55432/urbanagent_phase1_test"
$env:URBANAGENT_ALLOW_PHASE1_DB_WRITE="true"
npm.cmd run phase1:db:rollback
```

Reapply and reimport with the same migration/import commands above.

## Cleanup

Always remove the disposable container and volume unless retaining them is
needed to diagnose a failure:

```powershell
wsl.exe docker compose -f docker-compose.phase1.yml down -v
```

Verify cleanup:

```powershell
wsl.exe docker ps -a --filter name=urbanagent-phase1-postgis
wsl.exe docker volume ls --filter name=urbanagent_phase1_postgis_data
```

Expected result: no matching container or volume remains.

## Safety Notes

- Do not use production or shared PostgreSQL databases.
- Do not touch Firebase production.
- Do not modify the canonical CSV bytes.
- Do not set PostgreSQL as the default runtime.
- Do not run Phase 2 work from this runbook.
