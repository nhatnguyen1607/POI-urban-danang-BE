# Phase 4 Stage 4H: Incremental Entity Resolution

## Status

`CANDIDATE ONLY - NON-RUNTIME - NOT CANONICAL`

Stage 4H hardens entity resolution and adds a deterministic incremental sync
foundation. It does not apply candidates, alter runtime APIs, download media,
run multimodal models, or train model weights.

## Historical `poi_urban` Pipeline

The historical repository at HEAD
`3e70ae6a72e86a506cd15cc3a6427de6d745eefa` was inspected as read-only. Its
working tree also contains local research changes; the flow below is based on
the actual files inspected, while reproducibility claims remain limited. Its
actual flow is:

1. `dataset/raw/crawl_danang_ggmap_1.py` scans a Da Nang grid, extracts Google
   source fields and `place_id`; `dataset/raw/crawler_foody.py` extracts Foody
   `RestaurantID`, text, media URL and geocoded coordinates.
2. `dataset/processed/preprocessed.py` aggregates Foody reviews and builds model
   text. `dataset/processed/prepare_data.py` maps provider columns, drops missing
   coordinates, derives Google district text and writes separate
   `poi_processed_gmap.csv` and `poi_processed_foody.csv` domains.
3. `src/precompute/pds_sampler.py` bounds real POIs and generates deterministic
   negative/urban-void records. `scripts/rebuild_clean_dataset_v2.py` parses
   coordinates, applies the `15.90..16.20 / 108.00..108.35` bbox, optionally
   applies cached land geometry, preserves old IDs where an exact source
   record/coordinate/category key exists, and emits manifests and hashes.
4. `src/precompute/crop_buildings.py` filters points outside cached land geometry
   and generates building-context PNGs. Image files are resolved by historical
   source identity through `src/data/image_lookup.py`.
5. `src/data/dataset.py` joins text, POI images, geometry images and modality
   masks for model input. CLIP/ResNet/fusion, descriptions, spatial neighbors,
   road matrices and training affect model features only, not POI identity.

Historical processed inputs contain 5,305 Google rows and 262 Foody rows. The
historical final research files contain 4,928 Google real rows and 225 Foody
real rows, plus 16,424 and 1,125 urban voids respectively. Google has only
3,947 unique non-empty `RestaurantID` values across 4,928 real rows. The old
name-only leakage-removal step is not a product entity merge and contains a
buggy Foody-side normalization of `RestaurantID` rather than name.

The historical repository therefore explains source collection, filtering,
ID/image conventions and research preparation, but cannot reproduce the 4,166
merged UrbanAgent entities. It lacks the approved canonical merge decision,
complete row-level source-to-`Global_ID` lineage and the alias decisions needed
to explain the canonical 3,946 Google-compatible and 225 Foody-compatible
views. Those counts must not be forced to equal the separate research domains.

## Reuse Decision

| Classification | Historical components |
| --- | --- |
| `REUSE_AS_IS` | deterministic hashing/manifests, numeric coordinate validation, original-value retention, modality-presence semantics |
| `ADAPT_AND_REUSE` | Unicode/Vietnamese folding, conservative ad-text removal, source-key identity, bbox/land checks, image lookup/cache intent, processing-version metadata |
| `REFERENCE_ONLY` | void sampling, building/road context, image descriptions, CLIP/ResNet fusion, contrastive training and train/test domain split |
| `DO_NOT_REUSE` | live Google scraping, embedded credentials, online geocoding, filename-exists-only media cache, global name-only deduplication, LLM summarization in core cleaning, row-index identity as cross-source identity |

## Resolver Hardening

The Stage 4G resolver remains the default behavior. Stage 4H adds explicit
`hardened` mode for non-runtime preparation:

- NFKC/NFD normalization, Vietnamese accent folding, punctuation and whitespace
  normalization, and narrowly scoped URL/hotline/promotion removal.
- Original names remain intact. Business descriptors are handled only at name
  edges or as explicit phrases. Branch/store tokens remain identity evidence.
- Same chain base with conflicting branch/location tokens is a hard negative.
- Known incompatible categories block same-building matches.
- Address token overlap supports, but cannot independently establish, a match.
- Exact full names at the measured 150-225 meter threshold edge may become
  `PROBABLE_MATCH`; distant same names stay separate.
- Shared entity identifiers remain stronger than fuzzy evidence. Structural
  fields such as OSM element type remain non-identity metadata.
- Spatial blocking is mandatory for full-city mode; no all-pairs resolver is
  introduced.

### Fixed Stage 4G Before/After

The Stage 4G snapshot hashes were reverified and the same 76,666 normalized
records were evaluated against the unchanged 4,166 canonical records.

| Outcome | Stage 4G | Stage 4H hardened |
| --- | ---: | ---: |
| `HIGH_CONFIDENCE_MATCH` | 393 | 413 |
| `PROBABLE_MATCH` | 253 | 283 |
| `AMBIGUOUS` | 52 | 21 |
| `NEW_CANDIDATE` | 75,968 | 75,949 |
| `SOURCE_DUPLICATE` evidence edges | 3,112 | 3,108 |
| Review queue | 79,174 | 79,078 |
| Resolution time | 84.429 s | 225.436 s |

Hardened spatial blocking evaluated 22,402,751 canonical candidate pairs versus
319,390,556 naive pairs, a 92.9858% reduction. This is not labeled
precision/recall evidence. The higher processing time and broad bbox remain
limitations; no threshold was globally loosened and no result was auto-applied.

## Incremental Sync

`npm run citypack:sync -- sync` accepts an adapter snapshot, snapshot ID,
optional prior JSONL state/media registry, and optional checkpoint. State is
keyed by `(source, sourceId)` and records:

- first/last seen snapshots, source update time and last processed snapshot;
- raw, normalized, identity, media and provenance hashes;
- `NEW`, `CHANGED`, `UNCHANGED`, `MISSING` or `INVALID`;
- canonical match/classification and resolution/provenance/multimodal versions.

Only identity/version changes invoke entity resolution. Media and provenance
changes invalidate their own stages. Retrieval timestamps do not invalidate
identity. Missing source records become retirement review candidates and never
delete canonical data. Generated work belongs under the ignored
`data/spikes/phase4/stage4h/work/` path.

The deterministic A/B replay covers unchanged, identity-changed, new, missing
and media-changed records. A retrieval-metadata-only record remains
`UNCHANGED`; Snapshot B processes two resolution records and one changed media
asset. Repeating identical Snapshot B performs zero resolution,
media or provenance work for present records; four records are skipped and the
historical missing record remains review-only. Interrupted processing resumes
from sorted completed keys, partial state, failed IDs and processing versions.

## Media And Multimodal Cache Metadata

The media registry stores source/record/media identity, reference, content hash,
ETag, Last-Modified, first/last seen, license/provenance and processing versions.
Unchanged assets are reused; changed assets invalidate only vision/embedding
metadata; missing assets retain history. No media binary is downloaded.

Future multimodal cache keys combine normalized input hash, media hash set and
model version. Vision cache design additionally requires content hash, model and
prompt version. Vision remains disabled and optional; no credentials, model
weights, embeddings or bulk Google media are used.

## Source Capabilities And Automation

- Overture: `SNAPSHOT` release comparison.
- OSM current Overpass access: `POLLING` plus stable-ID/hash comparison.
- Wikidata and Wikimedia Commons: `POLLING` plus stable-ID/hash comparison.
- `CHANGE_FEED` and `DELTA` are represented but not falsely claimed for current
  adapters.

The command is suitable for later cron, Task Scheduler, CI or worker invocation.
It performs no traveler-time provider call and requires explicit `--write-state`
before writing final non-runtime state.

## Safety And Limitations

- Canonical rows: `4166`
- Canonical SHA-256:
  `5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae`
- Canonical/runtime/CSV runtime/frontend changed: `NO`
- Production PostgreSQL/Firebase touched: `NO`
- Google bulk ingestion/second city/candidate apply: `NO`

Stage 4H does not supply an approved administrative polygon or labeled entity
resolution ground truth. Full-city hardened processing is deterministic by
algorithm and fixed input; incremental replay/checkpoint output is verified
byte-stable, but only one final full-city hardened measurement was required
after the recorded deterministic Stage 4G baseline.

## Stage 4I Recommendation

Recommend **controlled review/apply preparation without canonical apply**:
define an approved Da Nang polygon, label a stratified sample of threshold-edge,
chain, same-building and duplicate cases, then set measurable precision gates
before any candidate application is proposed.
