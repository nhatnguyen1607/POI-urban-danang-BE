# Phase 4 Stage 4D - Reproducible Bounded City Pack Build Pipeline

Updated: 2026-08-10.

Status: `SPIKE_COMPLETE`.

Implementation status: `OFFLINE_NON_RUNTIME_CANDIDATE_BUILD`.

Approval basis: the user explicitly issued `APPROVED MULTI-SOURCE POI SPIKE`.

## 1. Scope

Stage 4D creates a reproducible offline pipeline that turns bounded source
snapshots, Stage 4C adapters, entity-resolution output, and file-based review
decisions into a candidate City Pack.

The output is explicitly:

`CANDIDATE_NON_RUNTIME_NOT_CANONICAL`

No output is installed into runtime.

## 2. Baseline Invariants

- Runtime canonical dataset remains
  `data/canonical/urbanagent_poi_master_v1.csv`.
- Canonical runtime POIs remain `4166`.
- Canonical SHA-256 remains
  `5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae`.
- CSV remains default runtime.
- PostgreSQL/PostGIS remains explicit opt-in.
- No `/api/v2` runtime code changed.
- No production DB, Firebase, API provider, or City Pack install occurred.
- Google Places is not part of the offline candidate pack.
- No second city was created.

## 3. Build Command

Script:

`scripts/phase4_stage4d_citypack_build.js`

Default build:

```powershell
node scripts\phase4_stage4d_citypack_build.js
```

Optional deterministic build ID:

```powershell
$env:URBANAGENT_STAGE4D_BUILD_ID = "stage4d-fixture-build"
node scripts\phase4_stage4d_citypack_build.js
```

## 4. Build Flow

The pipeline:

1. verifies canonical count and SHA-256;
2. loads bounded Stage 4B fixture snapshots;
3. normalizes records through Stage 4C adapters;
4. runs conservative Stage 4C entity resolution;
5. builds the Stage 4C review queue;
6. loads file-based review decisions;
7. validates decisions;
8. separates matched enrichment, approved new candidates, rejected records,
   deferred/unresolved items, and source duplicate groups;
9. validates candidate records;
10. emits candidate City Pack artifacts;
11. emits license/attribution manifest;
12. emits deterministic artifact hashes.

## 5. Review Decision Format

Fixture:

`data/spikes/phase4/stage4d/review_decisions/stage4d_review_decisions.fixture.json`

Supported actions:

- `ACCEPT_MATCH`
- `REJECT_MATCH`
- `CREATE_NEW`
- `MERGE_SOURCE_DUPLICATE`
- `KEEP_SEPARATE`
- `DEFER`

Validator rejects:

- unknown queue IDs;
- unsupported actions;
- duplicate or conflicting decisions;
- invalid canonical IDs;
- unsafe ambiguous acceptance outside listed canonical candidates.

No interactive admin UI exists in Stage 4D.

## 6. Candidate Pack Structure

Output directory:

`data/citypacks/candidates/danang/stage4d-fixture-build/`

Artifacts:

- `candidate_city_pack.json`
- `candidate_city_pack_index.csv`
- `license_attribution_manifest.json`
- `build_summary.json`

Candidate pack sections:

- canonical baseline metadata;
- source snapshot identifiers;
- matched enrichment records;
- approved new candidates;
- source duplicate groups;
- rejected records;
- deferred or unresolved review items.

Candidate records include stable candidate IDs, source provenance, field-level
provenance, license metadata, and attribution metadata where available.

## 7. Build Metrics

Default Stage 4D fixture build:

| Metric | Count |
| --- | ---: |
| Canonical baseline count | 4166 |
| Matched/enriched | 5 |
| Approved new candidates | 1 |
| Rejected | 1 |
| Deferred | 3 |
| Unresolved | 7 |
| Source duplicate groups | 1 |
| Candidate total | 6 |

Source contribution counts:

| Source adapter | Records |
| --- | ---: |
| Overture | 8 |
| OSM | 5 |
| Wikidata/Wikimedia | 4 |

## 8. Provenance And License Manifest

Manifest:

`license_attribution_manifest.json`

Rules preserved:

- Overture-like rows keep source/license/attribution metadata.
- OSM-like rows remain identifiable as `OPEN_SHAREALIKE_ISOLATED`.
- Wikidata-like rows remain identifiable as CC0-style knowledge enrichment.
- Wikimedia Commons-like rows retain media attribution/license requirements.
- Google Places is explicitly absent from offline candidate output.

Stage 4D does not claim final legal compatibility beyond Stage 4A conclusions.

## 9. Reproducibility

Given identical:

- canonical CSV;
- Stage 4B fixture snapshots;
- Stage 4C adapter logic;
- review decision file;
- build ID;

the build produces identical:

- candidate pack ordering;
- build summary;
- license manifest;
- artifact hashes.

Volatile timestamps are excluded from deterministic content.

## 10. Validation

Targeted command:

```powershell
node --test tests\phase4\phase4Stage4cEntityResolution.test.js tests\phase4\phase4Stage4dCityPackBuild.test.js
```

Validated:

- valid build;
- invalid review decision;
- conflicting duplicate decision;
- unresolved ambiguous records excluded;
- approved match/enrichment;
- approved `CREATE_NEW`;
- source duplicate handling;
- provenance/license manifest;
- deterministic repeated build;
- canonical integrity.

Result:

- tests: `16`
- passed: `16`
- failed: `0`

## 11. Fixture vs Real Bounded Sample

Stage 4D validation is `FIXTURE VALIDATION`.

It uses the deterministic Stage 4B/4C fixtures already committed under
`data/spikes/phase4/`.

It does not fetch or process real Overture, OSM, Wikidata, Wikimedia, or Google
payloads.

## 12. Limitations

- No production importer.
- No runtime City Pack install.
- No canonical CSV mutation.
- No admin review UI.
- No real bounded external snapshot fetch.
- No second-city readiness claim.
- No Google bulk ingestion.

## 13. Proposed Stage 4E

Recommended next stage:

`Phase 4 Stage 4E - Controlled Real Bounded Snapshot Validation`

Stage 4E should:

- keep canonical runtime data unchanged;
- fetch a tiny real Da Nang bounded Overture sample;
- fetch a tiny real OSM bounded sample;
- fetch a tiny Wikidata landmark sample;
- record snapshot hashes and license/provenance metadata;
- run Stage 4C adapters and Stage 4D candidate build on those snapshots;
- compare fixture behavior with real bounded sample behavior;
- keep all candidate applications in review output only;
- avoid production DB, Firebase, frontend, Google bulk ingestion, and second
  city creation.

Do not implement Stage 4E until separately approved.
