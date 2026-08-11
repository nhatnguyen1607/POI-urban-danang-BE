# Phase 4 Stage 4G: Full Da Nang Candidate Generation

## Status

`CANDIDATE ONLY - NON-RUNTIME - NOT CANONICAL`

Stage 4G ran the existing Stage 4C-F preparation pipeline over the full
supported Da Nang boundary. No candidate was applied to canonical or exposed
through a runtime API.

## Boundary

- Source: `src/services/canonicalCsvPoiRepository.js:DA_NANG_BBOX`
- Version: `pending_spatial_join`
- CRS: `EPSG:4326`
- Bbox: west `107.8`, south `15.8`, east `108.5`, north `16.3`
- Limitation: the repository has no approved Da Nang administrative polygon.
  Explicit Hoi An and Quang Nam address/admin signals were rejected, but a
  polygon remains necessary before canonical application.

## Fixed Source Snapshots

| Source | Access/version | Raw | Normalized | Snapshot SHA-256 |
| --- | --- | ---: | ---: | --- |
| Overture Places | Overture client/STAC, release `2026-07-22.0` | 78,625 | 70,030 | `9794aea3203d34c88cf0302857e8ed1659ef9b676a67885a9bd1b1ab2d1be06c` |
| OpenStreetMap | Overpass, named traveler-relevant POIs | 6,021 | 5,811 | `3bb4f55a06abac23f90c5e05ee6fe410bfc5927a9d1e7487884f8757e834065c` |
| Wikidata | WDQS `wikibase:box` | 994 | 825 | `f5c01bfc8e77bac7f896bebef65173df04635184403e60d6510a12810eeedf32` |
| Wikimedia Commons | linked media metadata only; no binaries | 81 requested | 72 retained media relationships | `7bd8138e1585da0b57c7d95c4aa23e0b3f64d4570fce7271f88c2ab1b7f9726c` |

Snapshot retrieval completed at `2026-08-10T09:03:58Z`. Raw payloads and the
full generated candidate artifacts are not committed to Git.

## Resolution And Candidate Metrics

| Outcome | Count |
| --- | ---: |
| Normalized source records | 76,666 |
| `HIGH_CONFIDENCE_MATCH` | 393 |
| `PROBABLE_MATCH` | 253 |
| `AMBIGUOUS` | 52 |
| `NEW_CANDIDATE` | 75,968 |
| `SOURCE_DUPLICATE` evidence edges | 3,112 |
| Source duplicate groups | 2,640 |
| Invalid | 0 |
| Clearly out-of-city/rejected | 8,974 |
| Proposed matched/enrichment records | 646 |
| Proposed new records | 75,968 |
| Candidate pack total | 76,614 |
| Review queue | 79,174 |

The review queue contains every ambiguous/new/duplicate outcome plus 42
conflicting-enrichment or provenance/license items. No unresolved action was
applied. Duplicate groups retain source records and provenance rather than
collapsing records.

## Field Coverage

- Overture: name/coordinates/address `100%`, category `95.18%`, phone `88.48%`,
  website `39.67%`, opening hours `0%`.
- OSM: name/category/coordinates `100%`, address `35.57%`, phone `12.89%`,
  opening hours `11.84%`, website `8.12%`.
- Wikidata: name/category/coordinates `100%`, media `8.73%`, website `2.67%`,
  address/phone/opening hours `0%`.

Missing values remain missing. The pipeline did not fabricate fields.

## Quality Spot Check

- High-confidence samples had matching names/categories and sub-3-meter
  coordinate differences.
- Probable samples had strong names/categories but about 50-meter coordinate
  differences; they remain traceable and reviewable under the existing policy.
- Ambiguous samples correctly exposed same-name chain branches, including
  `Dong Tien Bakery`, rather than auto-merging them.
- Several new-candidate samples had exact-name canonical records just beyond
  the 150-meter probable threshold. This threshold-edge behavior is a reason
  to harden entity resolution before any canonical apply.
- Duplicate samples after hardening were plausible same-name/category records
  separated by 3-28 meters. Structural values such as OSM element type are no
  longer treated as shared entity identifiers.
- Vietnamese UTF-8 names were preserved. No invalid-coordinate sample remained;
  out-of-city exclusions were counted separately.

This bounded review is not a labeled precision/recall evaluation and does not
establish city-wide scientific accuracy.

## Provenance And License

- Record provenance: `76,666 / 76,666` complete.
- Field provenance: `766,660 / 766,660` complete.
- Retained media license metadata: `72 / 72` complete.
- Overture upstream/source distinctions, OSM ODbL field origin, Wikidata QID
  and CC0, and Commons per-file attribution are retained.
- Twelve source media metadata conflicts were excluded from retained media and
  queued for review.
- Google Places is absent.

## Determinism And Performance

Two runs over the same snapshots, config, canonical baseline, and empty review
decision set produced identical normalized ordering, classifications, review
queue, candidate pack, license manifest, and hashes: `PASS`.

- Acquisition: `97.873 s`
- Resolution runs: `84.429 s` and `82.358 s`
- Candidate builds: `51.375 s` and `29.717 s`
- Approximate peak RSS: `3,415 MB`
- External candidate artifacts: about `965 MB` total, intentionally outside Git

Full-city execution required bounded spatial duplicate grouping and streaming
JSON hashing/writes. Default Stage 4C-F behavior remains enabled and unchanged.

## Safety

- Canonical rows: `4166`
- Canonical SHA-256: `5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae`
- Canonical/runtime/CSV runtime changed: `NO`
- Firebase/production PostgreSQL/frontend changed: `NO`
- Google bulk ingestion/second city/runtime external POIs: `NO`

## Stage 4H Recommendation

Recommend **B. Entity-resolution hardening before apply**. Priorities are an
approved administrative polygon, review-queue prioritization, threshold-edge
analysis, chain/branch handling, and a labeled manual sample. Do not begin a
canonical apply while 79,174 items remain unresolved.
