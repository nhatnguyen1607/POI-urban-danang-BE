# UrbanAgent Multi-Source POI Strategy

Updated: 2026-07-31.

Status: `PROPOSED FUTURE DATA-PLATFORM STRATEGY`.

Implementation status: `NOT STARTED`.

This document defines the future multi-source POI architecture for UrbanAgent.
It does not change the current canonical runtime dataset and does not authorize
the ingestion of any new external source.

## 1. Current Approved Baseline

The current approved runtime baseline remains:

- City Pack: Da Nang.
- Canonical dataset:
  `data/canonical/urbanagent_poi_master_v1.csv`.
- Application POIs: `4166`.
- Unique canonical `Global_ID`: `4166`.
- Canonical SHA-256:
  `5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae`.
- CSV repository remains the default runtime.
- PostgreSQL/PostGIS remains explicit opt-in.
- No production database migration or reimport is approved.
- No external multi-source pipeline is active.
- Phase 2 uses the fixed canonical dataset baseline.

The current canonical CSV must not be modified, replaced, enlarged, or
regenerated during an active Phase 2 implementation batch.

## 2. Product Position

UrbanAgent must not compete only on raw POI count or by acting as a generic
global itinerary chatbot.

UrbanAgent B2C should become a local, context-aware travel execution agent.

UrbanAgent B2B should become an evidence-backed local location-intelligence
product.

The long-term data advantage should come from verified local information,
Vietnamese semantic understanding, local suitability, source transparency,
uncertainty handling, freshness verification, itinerary feasibility,
replacement logic, field-level evidence, and business-decision signals.

A large POI catalog is infrastructure. It is not the final product
differentiator.

## 3. Multi-Source Architecture

Future POI data should be organized into four layers.

### 3.1 Layer A - Persistent Open Base

Purpose:

- provide broad geographic coverage,
- provide candidate POIs,
- provide geometry and basic categories,
- support future City Pack discovery.

Candidate source classes include:

- Overture Maps Places,
- OpenStreetMap,
- Wikidata,
- official public-sector open data,
- local tourism open datasets,
- other sources with clearly compatible licenses.

No candidate source is approved automatically. Each source requires current
license and policy review before ingestion.

### 3.2 Layer B - Licensed Or Request-Time Enrichment

Purpose:

- enrich selected POIs with data that cannot automatically become part of the
  permanent UrbanAgent canonical database,
- retrieve provider-controlled content when required by the traveler flow,
- preserve attribution and storage restrictions.

Candidate source classes include:

- Google Places,
- Foursquare Places,
- Tripadvisor content services,
- booking or accommodation providers,
- transport and activity providers,
- commercial geospatial providers.

Provider-controlled content must not be bulk-copied into the canonical
database unless the provider contract explicitly permits persistent storage
and reuse.

Restricted data should be handled through request-time retrieval, approved
cache duration, approved display rules, provider attribution, field-level
source marking, deletion rules, and expiration procedures.

### 3.3 Layer C - UrbanAgent Local Enrichment

UrbanAgent-owned or directly licensed local enrichment may include:

- verified business profiles,
- verified opening hours,
- verified operating status,
- verified price ranges,
- verified accessibility attributes,
- local survey results,
- partner submissions,
- traveler feedback,
- Vietnamese semantic tags,
- local category normalization,
- local event and weather suitability,
- recommended stay duration,
- substitution and fallback relationships,
- local business opportunity evidence.

User or partner submissions must retain submitter class, consent or agreement
status, observed timestamp, verification status, moderation status, and
deletion status.

### 3.4 Layer D - Derived Intelligence

Derived fields may include `vibeTags`, `suitableFor`, `intentMatch`,
`weatherSuitability`, `timeSuitability`, `substitutionGroup`,
`recommendationReasonCodes`, `itineraryFeasibility`,
`businessOpportunitySignals`, `confidence`, `evidenceReferences`, and
`dataQualityFlags`.

Derived intelligence must not be represented as raw source fact.

Every derived field must retain derivation method, method version, evidence
inputs, timestamp, confidence, uncertainty, and whether the value was
calculated or manually verified.

LLMs may explain evidence-backed output but must not invent missing facts.

## 4. Canonical Entity Model

The future canonical platform should distinguish source records, canonical POI
entities, external identifiers, aliases, field-level observations, derived
fields, verification records, and license/attribution records.

The existing application `Global_ID` remains the legacy canonical key.

New external source IDs do not replace `Global_ID`.

Example external identifier namespaces may include:

- `overture_gers_id`,
- `osm_node_id`,
- `osm_way_id`,
- `wikidata_qid`,
- `google_place_id`,
- `foursquare_fsq_id`,
- `tripadvisor_location_id`,
- `foody_restaurant_id`,
- `urbanagent_business_id`.

An external identifier does not automatically create a new product POI.

Aliases and source identifiers must not be exposed as independent traveler POIs
unless entity resolution concludes that they represent separate physical or
logical places.

## 5. Field-Level Provenance

Every imported or enriched field should retain:

- source name,
- source namespace,
- source record identifier,
- field name,
- observed value,
- observed timestamp,
- imported timestamp,
- verified timestamp,
- verification method,
- source license or contract class,
- attribution requirement,
- persistence class,
- expiration or refresh policy,
- confidence,
- transformation history,
- current status.

Recommended statuses include `observed`, `matched`, `verified`, `unverified`,
`conflicting`, `stale`, `expired`, `rejected`, and `removed`.

A canonical POI may contain observations from multiple sources. Source
disagreement must not be silently resolved without recording the decision and
evidence.

## 6. Entity Resolution

Entity resolution should use multiple signals, including geographic distance,
normalized name, alternate names, normalized address, phone number, website
domain, category compatibility, existing source identifiers, business
ownership evidence, and local verification evidence.

A name match alone is insufficient.

Recommended match outcomes:

- `exact_match`,
- `high_confidence_match`,
- `manual_review_required`,
- `probable_new_entity`,
- `rejected_match`.

Every merge must be reversible.

Every multi-source ingestion run must produce exact-match, fuzzy-match,
ambiguous-match, probable-new-POI, duplicate, source-conflict, license,
attribution, field-completeness, removal/stale-record, and cost reports.

## 7. Source Onboarding Gate

No new source may enter an UrbanAgent runtime until all of the following exist:

1. Source identity and owner.
2. Current license or commercial contract.
3. Allowed storage policy.
4. Allowed caching policy.
5. Allowed display policy.
6. Allowed API redistribution policy.
7. Attribution requirements.
8. Machine-learning and derived-use policy.
9. Refresh policy.
10. Deletion and termination policy.
11. Cost estimate.
12. Rate-limit estimate.
13. Data-quality sample.
14. Entity-resolution dry run.
15. Security review.
16. Approval record in `DECISIONS.md`.

A source spike may run in an isolated development environment before runtime
approval. A spike must not alter the approved canonical CSV or production
database.

## 8. Candidate Source Registry

The following are candidates, not approved runtime sources.

| Source class | Proposed role | Initial status |
| --- | --- | --- |
| Overture Maps Places | Broad open candidate POI base | `SPIKE_PROPOSED` |
| OpenStreetMap | Geometry, road and candidate POI context | `SPIKE_PROPOSED` |
| Wikidata | Landmark and knowledge enrichment | `SPIKE_PROPOSED` |
| Official local open data | Local verified or authoritative context | `SOURCE_DEPENDENT_REVIEW` |
| Google Places | Licensed/request-time enrichment | `NOT_APPROVED_FOR_CANONICAL_IMPORT` |
| Foursquare Places | Licensed enrichment candidate | `CONTRACT_REVIEW_REQUIRED` |
| Tripadvisor content | Licensed review/content candidate | `CONTRACT_REVIEW_REQUIRED` |
| Booking/activity providers | Availability and transaction context | `FUTURE_PARTNER_SCOPE` |
| Local partner submissions | Verified local enrichment | `WORKFLOW_NOT_IMPLEMENTED` |
| Traveler feedback | Product-quality and freshness signal | `PRIVACY_WORKFLOW_REQUIRED` |
| Wanderlog or competitor websites | Competitive benchmarking only | `PROHIBITED_AS_DATA_SOURCE` |

Competitor products must not be scraped or copied as POI data sources.

## 9. Phase Boundaries

### Phase 2

Phase 2 remains Traveler API v2 over the approved `4166`-POI canonical
baseline.

Phase 2 must not ingest a new POI source, alter canonical POI counts, replace
the canonical CSV, activate a global catalog, add a second city, or claim
global coverage.

### Future Data Platform / City Pack Phase

A future approved phase may include:

- Stage A - Source Discovery Spike.
- Stage B - License Registry And Source Policy.
- Stage C - Entity Resolution Prototype.
- Stage D - Local Verification Workflow.
- Stage E - Second-City Readiness.

## 10. Initial Spike Metrics

A future source spike should measure:

- total source candidates,
- candidates inside the approved city boundary,
- exact identifier matches,
- high-confidence spatial/name matches,
- ambiguous matches,
- probable new POIs,
- probable stale or closed POIs,
- duplicate rate,
- coordinate completeness,
- category completeness,
- address completeness,
- image availability,
- opening-hours availability,
- license distribution,
- attribution requirements,
- estimated refresh cost,
- estimated request cost,
- comparison with the approved 4166-POI baseline.

No source should be judged only by raw row count.

## 11. Runtime Rules

Until separately approved:

- the canonical CSV remains immutable,
- CSV remains default runtime,
- PostgreSQL remains opt-in,
- external APIs are not required for core runtime,
- failures of optional enrichment must not break core recommendations,
- provider-specific fields must not leak into public API contracts,
- traveler clients must not depend on the internal storage provider,
- missing enrichment remains null/unknown,
- restricted source data must expire according to policy.

## 12. Frontend And Mobile Boundary

This strategy does not define the final mobile interface.

Frontend and future mobile clients should consume stable UrbanAgent API
contracts rather than call external POI providers directly.

The traveler interface should receive canonical POI identity, traveler-safe
fields, source-safe attribution metadata, uncertainty warnings, recommendation
reasons, and itinerary feasibility information.

Provider credentials, raw provider payloads, and internal entity-resolution
evidence must not be exposed to clients.

Mobile product design remains a later product phase.

## 13. Non-Goals

This strategy does not approve:

- scraping Wanderlog or another competitor,
- bulk scraping Google Maps, Tripadvisor, Yelp, or similar platforms,
- replacing the Phase 2 dataset,
- production multi-source ingestion,
- a second City Pack,
- global launch claims,
- frontend redesign,
- mobile implementation,
- booking integration,
- partner monetization,
- production PostgreSQL cutover.

## 14. Approval Gate

Implementation may begin only after explicit approval of source spike scope,
source list, license review, storage policy, test environment,
entity-resolution thresholds, reporting format, and no-production-write
guardrail.

Required approval phrase:

`APPROVED MULTI-SOURCE POI SPIKE`

Until that approval exists, this document remains architecture planning only.
