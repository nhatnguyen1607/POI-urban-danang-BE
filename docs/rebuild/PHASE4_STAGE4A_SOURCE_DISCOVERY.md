# Phase 4 Stage 4A - Source Discovery Spike

Updated: 2026-08-10.

Status: `SOURCE_DISCOVERY_SPIKE_COMPLETE`.

Implementation status: `DOCUMENTATION_ONLY`.

Approval basis: the user explicitly issued `APPROVED MULTI-SOURCE POI SPIKE`
for this bounded Stage 4A research task.

## 1. Baseline Invariants

- Current runtime City Pack remains Da Nang only.
- Current canonical runtime dataset remains
  `data/canonical/urbanagent_poi_master_v1.csv`.
- Current application POIs remain `4166`.
- Current canonical SHA-256 remains
  `5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae`.
- CSV remains the default runtime repository.
- PostgreSQL/PostGIS remains explicit opt-in.
- No external source was ingested into runtime storage.
- No city-scale source download was performed.
- No second City Pack was created.
- No production database or Firebase data was touched.

## 2. Discovery Method

This Stage 4A pass used source documentation and license/policy inspection.
No provider payload was committed, persisted, imported into runtime, or used to
change UrbanAgent canonical POIs.

The evaluated candidate sources were limited to:

- Overture Maps Places.
- OpenStreetMap.
- Wikidata and Wikimedia Commons.
- Google Places as an optional live-enrichment candidate only.

## 3. Source Comparison

| Source | Access method | License and attribution | Commercial-use implications | API/download limits | Expected fields | Coordinate quality | Category coverage | Address quality | Image/media | Update frequency | Duplicate risk | Complexity | Operating cost | Offline City Pack suitability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Overture Maps Places | Public cloud GeoParquet on S3/Azure; bbox extraction via Overture Explorer, Python client, or DuckDB. | Mixed source-level licensing. Overture Places docs list CC0, CDLA-Permissive-2.0, and Apache-2.0 contributors. Foursquare-derived rows require Apache notice handling. | Generally suitable for commercial use if source license, notice, and attribution metadata are preserved. Must track per-record `sources` and license class. | Bulk files are public, but full dataset is large; bounded bbox/sample extraction is required for spikes. | Place ID/GERS, point geometry, names, categories, addresses, websites, phones, confidence, source metadata. | Strong candidate baseline, but exact Da Nang quality requires Stage 4B sample. | Broad food, shopping, services, lodging, attraction coverage. | Better than OSM in many cases, still source-dependent and must remain nullable. | No dependable product media library. | Regular public releases. | Medium: multi-source conflation helps, but source conflicts and stale records remain possible. | Medium: GeoParquet, license filtering, source registry, entity-resolution reports. | Infrastructure cost only for downloads/querying. | High, if license classes and source notices are preserved. |
| OpenStreetMap | Overpass API for bounded queries; planet/Geofabrik extracts for offline processing. | ODbL with attribution and share-alike obligations for adapted databases. | Commercial use is allowed, but derived databases may inherit ODbL obligations. Keep OSM-derived layers isolated unless product licensing accepts share-alike. | Overpass has usage limits and etiquette; large extracts should use downloads, not public API abuse. | OSM IDs, names, tags, amenities, tourism, shops, addr tags, opening_hours, phone, website, wheelchair/access tags. | Variable community quality; often good for mapped amenities and landmarks. | Strong long-tail local amenities, transport-adjacent context, parks, beaches, tourism. | Highly variable; many rows may lack full address. | Occasional `wikimedia_commons`/image tags, not reliable. | Continuous community updates; extracts can be refreshed frequently. | Medium-high: nodes/ways/relations and duplicate business mappings. | Medium: tag normalization and ODbL isolation required. | Free infrastructure aside. | Medium: useful enrichment, but ODbL governance must be explicit. |
| Wikidata / Wikimedia | SPARQL, REST/API, dumps; Wikimedia Commons APIs for media metadata. | Wikidata entity data is CC0. Commons files carry per-file licenses; many require attribution or share-alike. | Wikidata is commercially usable. Commons media is commercially usable only when each file license and non-copyright restrictions are satisfied. | Public endpoints require responsible User-Agent, rate/backoff handling, and bounded queries. Dumps are available. | QIDs, labels, aliases, descriptions, coordinates, official websites, entity relationships, P18 images, Commons categories. | Strong for notable landmarks; weak for everyday cafes/restaurants. | Excellent for attractions, museums, cultural/historic entities; sparse for common POIs. | Usually weak for street-level address. | Strong for landmarks if license/attribution is stored per asset. | Continuously edited; dumps are periodic. | Low-medium: QID matching is strong, but name/spatial matching is still needed. | Medium: SPARQL modeling, multilingual labels, media license registry. | Free infrastructure aside. | High for CC0 knowledge enrichment; conditional for media due per-file licenses. |
| Google Places | Places API request-time lookup only. | Google Maps Platform terms restrict prefetching, caching, and storage; place IDs can be stored indefinitely. Attribution/logo/display rules apply. | Commercial use requires billing and compliance. It is not a permissive offline canonical source. | Quotas and pricing apply; fields and photos are controlled by API policy. | Place ID, current address, coordinates, phone, website, opening hours, rating, review count, photos/reviews/status depending fields requested. | High live quality. | Strong commercial POI coverage. | Strong live address quality. | Photo references and provider-controlled media; not an offline media library. | Live provider data, subject to API freshness. | Low for Google identifiers, but entity matching to UrbanAgent POIs remains required. | Medium-high: billing, field masks, cache expiry, attribution, restricted storage. | Paid per use. | Low. Use only as optional live enrichment where policy permits. |

## 4. Preferred Base Source

Preferred base source: `Overture Maps Places`.

Rationale:

- It is designed for broad place coverage and geographic extraction.
- It is suitable for offline City Pack candidate generation when source-level
  license metadata is retained.
- It provides structured place geometry, categories, names, addresses, contact
  fields, confidence, and source metadata.
- It avoids making the future City Pack dependent on a paid live-only provider.

The recommendation is conditional on Stage 4B verifying Da Nang sample quality
and implementing per-source license filtering. Overture rows with stricter
notice obligations, including Foursquare-sourced Apache rows, must not be
silently mixed into an unrestricted canonical layer.

## 5. Recommended Enrichment Sources

Recommended enrichment sources:

1. OpenStreetMap for local amenities, tourism tags, opening hours,
   accessibility tags, websites, phones, and mapped public-space context.
2. Wikidata for landmarks, multilingual names, official websites, cultural
   metadata, and stable QID identifiers.
3. Wikimedia Commons for landmark media only when file-level license,
   attribution, author, source URL, and derivative restrictions are stored.
4. Google Places only as optional live/request-time enrichment for selected
   fields where terms permit. It should not be used as the offline canonical
   store for place details, ratings, reviews, hours, or photos.

## 6. Sources Unsuitable For Canonical Storage

- Google Places content other than permitted retained identifiers such as
  `place_id`, unless a current contract explicitly permits persistent storage.
- Google photos, reviews, ratings, opening hours, phone, website, and status as
  permanent offline canonical fields without a policy-approved cache/storage
  basis.
- Wikimedia Commons media without a verified reusable license, attribution
  record, author/source metadata, and non-copyright restriction review.
- OSM-derived data inside an unrestricted proprietary canonical database unless
  ODbL obligations are accepted and isolated.
- Any competitor product page, scraped commercial map page, review website, or
  unlicensed dataset.

## 7. Proposed Source Priority

1. UrbanAgent canonical/merchant-verified/local first-party data.
2. Overture Places as the broad open candidate base.
3. OpenStreetMap as ODbL-governed local enrichment.
4. Wikidata as CC0 landmark and knowledge enrichment.
5. Wikimedia Commons as per-license media enrichment.
6. Google Places as request-time restricted live enrichment only.

## 8. Proposed Entity-Resolution Strategy

Entity resolution should run in deterministic stages:

1. Exact namespaced external IDs:
   `overture_gers_id`, `osm_node_id`, `osm_way_id`, `osm_relation_id`,
   `wikidata_qid`, `google_place_id`, and existing UrbanAgent source IDs.
2. Exact canonical website domain or phone match.
3. Normalized name similarity plus spatial distance.
4. Address similarity plus category compatibility.
5. Source confidence, operating status, and local verification evidence.
6. Manual review for ambiguous candidates.

Initial candidate thresholds for Stage 4B dry-run:

- `exact_match`: same external ID or same website/phone with compatible
  category.
- `high_confidence_match`: distance <= 40 meters, normalized name similarity
  >= 0.88, category compatibility >= 0.70.
- `manual_review_required`: distance <= 100 meters with partial name/category
  agreement or conflicting source facts.
- `probable_new_entity`: no strong match to the 4166 canonical POIs.

No ambiguous POI may be automatically merged. Alias identifiers must remain
source evidence, not independent traveler POIs.

## 9. Sample-Quality Conclusion

Stage 4A did not store or process a city-scale dataset. Based on source
capabilities and licensing:

- Overture is the best base candidate for future offline City Pack discovery.
- OSM is valuable for local detail, hours, accessibility, and public-space
  context, but ODbL handling is a major governance requirement.
- Wikidata/Wikimedia are strong for landmarks, culture, multilingual labels,
  and legally traceable media, but they will not cover everyday restaurants and
  cafes at UrbanAgent product depth.
- Google has high live quality but is unsuitable as the offline canonical
  source because storage, caching, display, and cost restrictions are central.

Stage 4B should validate these conclusions with a tiny Da Nang bounded sample
and reports only, not a runtime import.

## 10. Proposed Stage 4B Scope

Proposed next stage: `Phase 4 Stage 4B - Bounded Source Sample And License Dry-Run`.

Stage 4B should:

- keep the canonical 4166-POI runtime unchanged;
- use a tiny Da Nang bbox or allowlisted POI subset;
- run Overture, OSM, and Wikidata/Wikimedia discovery in temporary files only;
- optionally test Google Places on a few manually supplied known place IDs only
  if request-time policy and billing safeguards are approved;
- produce exact-match, fuzzy-match, ambiguity, probable-new-entity, duplicate,
  source-conflict, license, attribution, field-completeness, stale/removed, and
  cost reports;
- measure coverage for coordinates, names, categories, addresses, hours,
  contacts, media, source freshness, and attribution;
- decide whether Overture Foursquare-sourced Apache rows are allowed with
  notices or filtered for a simpler license profile;
- define source adapter interfaces but not implement production importers;
- avoid production DB, Firebase, frontend, second-city, and runtime changes.

## 11. References

- Overture Places guide:
  https://docs.overturemaps.org/guides/places/
- Overture attribution and license FAQ:
  https://docs.overturemaps.org/
- OpenStreetMap copyright and ODbL:
  https://www.openstreetmap.org/copyright
- Wikidata licensing:
  https://www.wikidata.org/wiki/Wikidata:Licensing
- Wikidata data access:
  https://www.wikidata.org/wiki/Help:Data_access
- Wikimedia Commons reuse:
  https://commons.wikimedia.org/wiki/Commons:Reusing_content_outside_Wikimedia/en
- Google Places API policies:
  https://developers.google.com/maps/documentation/places/web-service/policies
