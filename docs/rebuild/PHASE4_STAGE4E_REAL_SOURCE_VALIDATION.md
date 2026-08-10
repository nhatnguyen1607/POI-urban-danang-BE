# Phase 4 Stage 4E - Controlled Real Bounded Source Validation

## Status

`PASSED_WITH_LIMITATION`

This is a small, non-random validation sample. It is a candidate-only research
artifact and does not measure city-wide source coverage or entity-resolution
accuracy.

## Boundary and sources

Bounding box, EPSG:4326:

```text
west: 108.2200
south: 16.0580
east: 108.2300
north: 16.0680
```

The boundary covers a small central Da Nang area containing food, lodging,
landmark, cultural, and public POIs.

| Source | Bounded records | Committed sample | Access | License treatment |
| --- | ---: | ---: | --- | --- |
| Overture Places | 1960 | 31 | Python client 1.0.1, release `2026-07-22.0`, bbox | CDLA-Permissive-2.0 |
| OpenStreetMap | 219 relevant POIs | 33 | OSM Map API fallback after bounded Overpass dispatcher timeouts | ODbL-1.0, isolated share-alike candidate data, attribution required |
| Wikidata | 26 | 12 | WDQS `wikibase:box` | CC0-1.0 |
| Wikimedia Commons | 4 linked files | 4 metadata records | MediaWiki `imageinfo` metadata only; no binary media | Per-file license and attribution retained |

The deterministic sample combines category quotas with disclosed
canonical-overlap stress cases. The latter deliberately exercise match
branches and therefore must not be used as an unbiased accuracy sample.

## Pipeline result

```text
adapter records: 76
valid normalized: 76
invalid normalized: 0
HIGH_CONFIDENCE_MATCH: 6
PROBABLE_MATCH: 3
AMBIGUOUS: 1
NEW_CANDIDATE: 66
SOURCE_DUPLICATE: 3
review queue: 70
candidate build records: 9
```

No ambiguous, new, or duplicate record was automatically approved. The final
build contains nine automatic high/probable matched-enrichment candidates and
zero approved new POIs.

## Manual spot-check

- High confidence: OSM `Reply 1988` matched `google_maps_1742` at 9.26 m with
  normalized exact name and compatible category. The result is plausible.
- Probable: OSM `Minori` matched `google_maps_2391` (`Minori Tea`) at 35.58 m
  with 0.60 name similarity. The result is plausible but still reviewable.
- Ambiguous: OSM `Dong Tien` found several same-name bakery branches; the
  resolver correctly withheld an automatic match.
- New candidate: OSM `McDonald's` had no credible nearby canonical match. It
  remains review-only rather than being inserted.
- Duplicate: two OSM `McDonald's` points are 21.06 m apart. Additional pairs
  were found for `Ge Cafe` and `City Hostel`; none was auto-merged.

## Field coverage

- Overture: name/category/coordinates/address/external IDs 100%; website
  45.16%; phone 80.65%; opening hours and media 0%.
- OSM: name/category/coordinates/external IDs 100%; address 39.39%; website
  9.09%; phone 6.06%; opening hours 15.15%; media 0%.
- Wikidata sample: name/category/coordinates/external IDs 100%; linked Commons
  media 33.33%; address, phone, opening hours, and website 0%.

The sample shows that Overture is strongest for contact/address enrichment,
OSM contributes sparse local operational tags, and Wikidata/Commons is useful
for notable-place identity and licensed media rather than broad local POI
coverage.

## Provenance and licensing

- Record provenance: 76/76 complete.
- Field provenance: 760/760 complete, including snapshot reference.
- Linked media license/attribution metadata: 4/4 complete.
- Snapshot SHA-256 values and retrieval metadata are recorded in
  `data/spikes/phase4/stage4e/snapshots/snapshot_manifest.json`.
- Google Places is absent.

Limitation: the Stage 4D license manifest groups normalized records by their
record source. Commons per-file metadata is retained in the Stage 4E snapshot
and media field, but is not yet promoted to separate per-file entries in the
candidate license manifest. A full City Pack build must not proceed before
that gap is hardened.

## Candidate build and determinism

The output at
`data/citypacks/candidates/danang/stage4e-real-bounded/` is explicitly:

```text
CANDIDATE
NON-RUNTIME
NOT CANONICAL
```

Two builds from the same fixed snapshots produced identical candidate-pack,
license-manifest, and review-decision hashes. Candidate validation passed.

## Safety verification

- Canonical POIs: 4166.
- Canonical SHA-256:
  `5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae`.
- Canonical CSV bytes unchanged.
- Runtime APIs and default CSV repository unchanged.
- Firestore and production databases untouched.
- No Google Places, second city, or external routing work.

## Proposed Stage 4F

Direction C only: **provenance/license hardening**.

Promote per-file Commons license and attribution into the candidate license
manifest, define ODbL-derived artifact boundaries explicitly, and fail builds
when field/media provenance cannot be represented losslessly. Do not begin
full Da Nang generation until this gate is complete and separately approved.
