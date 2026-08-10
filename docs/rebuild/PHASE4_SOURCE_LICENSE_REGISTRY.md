# Phase 4 Source License Registry

Updated: 2026-08-10.

Status: `STAGE_4A_DISCOVERY_REGISTRY`.

Implementation status: `DOCUMENTATION_ONLY`.

This registry is a source-governance artifact for Phase 4 Stage 4A. It does
not approve runtime ingestion and does not change the Da Nang canonical
dataset.

## 1. Baseline Rules

- Current canonical runtime dataset remains
  `data/canonical/urbanagent_poi_master_v1.csv`.
- Current application POIs remain `4166`.
- Current canonical SHA-256 remains
  `5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae`.
- No external payload may be committed to Git.
- No restricted provider payload may be included in public fixtures, bundles,
  exports, or public APIs.
- No source may become runtime data without a separate onboarding decision.

## 2. Usage Classes

| Usage class | Meaning | Runtime implication |
| --- | --- | --- |
| `FIRST_PARTY_VERIFIED` | UrbanAgent-owned, partner-verified, or directly licensed data. | Best canonical authority when rights and verification are recorded. |
| `OPEN_PERMISSIVE_CANDIDATE` | Open data whose license generally permits persistent use with attribution/notice obligations. | Candidate for offline City Pack storage after license review. |
| `OPEN_SHAREALIKE_ISOLATED` | Open data with database share-alike obligations. | Keep isolated unless product accepts inherited obligations. |
| `OPEN_MEDIA_ATTRIBUTION_REQUIRED` | Reusable media with file-level license and attribution requirements. | Store only with author, source URL, license, attribution, and derivative restrictions. |
| `REQUEST_TIME_RESTRICTED` | Provider-controlled content that may be queried live or cached only under explicit terms. | Not canonical offline storage unless a contract says otherwise. |
| `PROHIBITED` | Source or use is not allowed. | Must not be ingested, cached, displayed, or redistributed. |

## 3. Candidate Source Registry

| Source ID | Source | Owner | Access method | License or policy | Usage class | Persistent storage | Cache policy | Display/attribution | API redistribution | ML/derived use | Refresh/deletion | Cost | Offline City Pack suitability | Approval status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `urbanagent_canonical_v1` | Existing UrbanAgent canonical Da Nang POIs | UrbanAgent project | Local canonical CSV | Approved project dataset decision | `FIRST_PARTY_VERIFIED` for current runtime baseline | Already approved for current runtime only | Local runtime | Current app display | Current app APIs | Current app behavior only | Fixed baseline until new dataset decision | None | Current Da Nang only | `APPROVED_CURRENT_BASELINE` |
| `overture_places` | Overture Maps Places | Overture Maps Foundation and listed contributors | Cloud GeoParquet, Overture Explorer, Python client, DuckDB | Mixed source-level CC0-1.0, CDLA-Permissive-2.0, Apache-2.0 contributor data | `OPEN_PERMISSIVE_CANDIDATE` with per-record license tracking | Conditional; store only with source/license metadata and notices | Allowed by license, subject to source terms | Preserve Overture and source attribution/NOTICE where required | Conditional by license class | Conditional; inherit source license requirements | Refresh by public release; handle removed/stale records | Infrastructure only | High, after Stage 4B sample and license filtering | `DISCOVERY_ONLY` |
| `openstreetmap` | OpenStreetMap | OpenStreetMap contributors / OSMF | Overpass API, extracts, planet | ODbL | `OPEN_SHAREALIKE_ISOLATED` | Conditional; ODbL obligations may apply to adapted database | Conditional; obey API etiquette and ODbL | OSM/contributor attribution required | Allowed under ODbL obligations | Conditional; derived database obligations may apply | Frequent extracts; deletion/stale handling required | Infrastructure only | Medium; strong enrichment but governance-heavy | `DISCOVERY_ONLY` |
| `wikidata` | Wikidata | Wikimedia community / Wikimedia Foundation platform | SPARQL, APIs, dumps | Entity data CC0; attribution appreciated | `OPEN_PERMISSIVE_CANDIDATE` | Generally yes for CC0 entity data | Yes with endpoint etiquette | Attribution appreciated, not required for CC0 data | Generally yes | Generally yes for CC0 data | Dumps/API updates; handle stale statements | Infrastructure only | High for landmark/knowledge enrichment | `DISCOVERY_ONLY` |
| `wikimedia_commons` | Wikimedia Commons media | Individual creators / Wikimedia community | Commons APIs, linked file metadata | Per-file free licenses or public domain; text CC BY-SA; structured file data CC0 | `OPEN_MEDIA_ATTRIBUTION_REQUIRED` | Conditional per file license | Conditional per file license | File-level author, source, license, and attribution required unless public domain says otherwise | Conditional per file license | Conditional per file license; derivative restrictions may apply | File/license changes must be tracked | Infrastructure only | Medium; strong media enrichment with strict metadata needs | `DISCOVERY_ONLY` |
| `google_places_live` | Google Places API | Google | Places API request-time calls | Google Maps Platform terms | `REQUEST_TIME_RESTRICTED` | No for most Places content; `place_id` can be stored indefinitely | Only as terms permit; no prefetch/bulk cache | Google attribution/logo and map display rules apply | Restricted by Google terms | Restricted; do not train or derive unrestricted canonical facts from restricted content | Respect deletion/expiry/terms changes | Paid API usage | Low; live enrichment only | `NOT_APPROVED_FOR_CANONICAL_IMPORT` |

## 4. Field-Level Policy Notes

| Field class | Overture | OSM | Wikidata | Wikimedia Commons | Google Places |
| --- | --- | --- | --- | --- | --- |
| Names/categories/coordinates | Candidate persistent fields with source license | ODbL-governed enrichment | CC0 landmark enrichment | Not a POI field | Live/restricted unless contract permits |
| Address/contact | Candidate persistent fields if source license allows | ODbL-governed, often incomplete | Sparse | Not applicable | Live/restricted unless contract permits |
| Opening hours | Sparse or source-dependent | Useful but variable | Rare | Not applicable | Live/restricted unless contract permits |
| Ratings/reviews | Not core base | Not available | Not available | Not applicable | Restricted; not offline canonical by default |
| Images/media | Not dependable as media library | Occasional tags only | Links to media | Per-file license required | Restricted photo references/content |
| External IDs | Overture/GERS/source IDs | OSM node/way/relation IDs | QID | File/page IDs | `place_id` can be retained under Google policy |

## 5. Required Stage 4B Registry Fields

Every Stage 4B source sample must produce registry values for:

```text
sourceId
sourceName
sourceOwner
sourceDocumentation
licenseOrContractName
licenseOrContractVersion
termsReviewedAt
usageClass
approvedPurposes
prohibitedPurposes
allowedFields
restrictedFields
persistentStorageAllowed
cacheAllowed
maximumCacheDuration
displayAllowed
attributionRequired
attributionText
attributionLink
apiRedistributionAllowed
bulkExportAllowed
commercialUseAllowed
derivedUseAllowed
machineLearningUseAllowed
rateLimits
estimatedCost
refreshRequirement
deletionRequirement
terminationRequirement
securityClassification
privacyClassification
approvalStatus
approvalDecisionReference
```

## 6. Non-Negotiable Guardrails

- Do not change the canonical CSV or its SHA-256 during source spikes.
- Do not use Google Places as an offline canonical source unless a separate
  contract explicitly permits it.
- Do not mix OSM-derived records into an unrestricted proprietary database
  without accepting and documenting ODbL obligations.
- Do not store Wikimedia Commons media without file-level license, author,
  source URL, attribution, and derivative-use metadata.
- Do not strip Overture source/license metadata from candidate records.
- Do not automatically merge ambiguous POIs.
- Do not fabricate missing ratings, reviews, opening hours, addresses,
  coordinates, freshness, or provider identifiers.
- Do not scrape competitor products or commercial map pages as POI sources.

## 7. Stage 4B Dry-Run Registry Observations

Stage 4B uses tiny `NON_CANONICAL_SPIKE_ONLY` fixtures and generated reports
under:

`data/spikes/phase4/stage4b/`

Observed policy requirements:

- Overture-like records need per-record source/license metadata before any
  canonical import decision.
- OSM-like records must remain `OPEN_SHAREALIKE_ISOLATED` until the project
  accepts ODbL obligations for a derived database.
- Wikidata-like records can be considered CC0 knowledge enrichment, but QIDs
  must remain namespaced source identifiers.
- Wikimedia Commons-like media must not be treated as plain image URLs; every
  file requires license, author, source URL, attribution, and derivative-use
  metadata.
- Google Places was not queried in Stage 4B and remains
  `REQUEST_TIME_RESTRICTED`.

No Stage 4B record is approved runtime data.

## 8. References

- Overture Places guide:
  https://docs.overturemaps.org/guides/places/
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
