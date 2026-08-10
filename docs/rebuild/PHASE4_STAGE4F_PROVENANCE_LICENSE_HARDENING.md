# Phase 4 Stage 4F - Provenance and License Manifest Hardening

Status: `PASSED`

Scope: harden the existing bounded, non-runtime Stage 4E candidate build. This
stage did not generate a full Da Nang pack, modify canonical data, expose
external POIs through runtime APIs, or grant source-license approval.

## License Entry Model

The candidate build now emits deterministic machine-readable collections for:

- source manifest entries;
- record provenance;
- field provenance;
- per-asset media license entries;
- attribution entries.

Entries support source and record identifiers, asset identifiers, license name
and URL or policy reference, attribution text and URL, contributor, snapshot
reference, field/media scope, and the provenance relationship. Stable entry IDs
and sorted output make the manifest reproducible for fixed inputs.

## Source Boundaries

### Wikimedia Commons

Each of the four bounded Commons files has an independent media license and
attribution entry. The entries retain page/file identity, creator metadata,
license name and URL, attribution text, source page, Commons snapshot reference,
and the parent Wikidata relationship. The bounded sample contains three
`CC BY-SA 3.0` files and one `CC BY 2.0` file. No media blob was downloaded.

### OpenStreetMap

All 330 OSM field-provenance entries remain marked
`OPEN_SHAREALIKE_ISOLATED`, retain `ODbL-1.0`, and identify the OSM record and
snapshot from which the field was normalized. OSM-derived fields are not
silently flattened into another source boundary.

### Wikidata

All 12 Wikidata records retain their `wikidata:<QID>` identity, CC0 provenance,
and Wikidata snapshot reference. Wikidata entity provenance and Wikimedia
Commons media licensing remain separate manifest entries.

### Overture

All 31 bounded Overture records retain the snapshot-level license notice and
their ordered upstream source metadata, including dataset, record ID, property,
license, update time, and confidence where present. This is provenance
preservation only and makes no broader legal claim about an upstream source.

## Validation Gates

Candidate build validation fails when:

- an external record lacks source provenance;
- a real Overture record lacks upstream provenance;
- an OSM field loses its source/license boundary;
- Wikidata entity or CC0 provenance is invalid;
- a Commons media item lacks its per-file license or attribution metadata;
- Google data appears in the candidate manifest.

The fixed Stage 4E snapshots produced:

| Manifest collection | Count |
| --- | ---: |
| Source entries | 3 |
| Record provenance entries | 76 |
| Field provenance entries | 760 |
| Commons media license entries | 4 |
| Attribution entries | 7 |

Two builds from identical fixed snapshots produced identical candidate packs,
manifests, ordering, and artifact hashes. Focused Phase 4C/4D/4E/4F validation
passed `31/31` tests.

## Safety Result

- Canonical POIs: `4166`
- Canonical SHA-256:
  `5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae`
- Canonical modified: `NO`
- Runtime API modified: `NO`
- External POIs exposed to runtime: `NO`
- Production database or Firebase touched: `NO`
- Google included: `NO`

## Stage 4G Readiness

The provenance/license architecture is ready to support a separately approved
`CONTROLLED FULL DA NANG CANDIDATE GENERATION`. That future run must remain
candidate-only and non-runtime, use conservative entity resolution and a review
queue, and preserve this hardened manifest. Stage 4G is not implemented or
approved by this document. The documentary license-policy gate remains distinct
from technical manifest readiness.
