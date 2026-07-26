# URBANAGENT — MASTER CONTEXT FOR CODEX

## 0. How Codex must work

You are working on an existing project, not a blank-slate demo.

Repositories:

- Frontend: `POI-urban-danang-FE-main`
- Backend: `POI-urban-danang-BE-main`
- Phase 0 canonical runtime POI dataset:
  - `data/canonical/urbanagent_poi_master_v1.csv`
  - rows: `4166`
  - unique `Global_ID`: `4166`
  - SHA-256: `5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae`
- Approved dataset decision:
  - `docs/rebuild/URBANAGENT_DATASET_DECISION.md`

Legacy/raw Google Maps and Foody CSV files are source inputs/backups only. They are not the Phase 0 traveler runtime dataset and must not be overwritten without an explicit new dataset decision.

Canonical Phase 0 data semantics:

- `Global_ID` is the canonical legacy POI key exposed by compatibility APIs.
- `Alias_Global_IDs` preserves merged source-row IDs; do not render aliases as extra product POIs.
- `RestaurantID` is a source identifier, not a verified Google `place_id`.
- `Source_IDs`, `Source`, `Merge_Status`, and `Data_Quality_Flags` carry provenance and merge context.
- `Entity_Type` must be `poi` for traveler POI, recommendation, itinerary, map, and count surfaces.
- Urban-void rows are excluded from product POIs.
- `City_ID` is required; Phase 0 supports `da-nang` only and does not implement a second City Pack.
- Missing coordinates, address, admin-boundary, rating, review count, opening hours, phone, website, or freshness values must remain unknown/null unless a verified source provides them.
- Foody rating text came from a `/10` source and is normalized separately; `Foody_Review_Sample_Count` is not total review count.

Before changing code:

1. Read both `PLANNING.md` files.
2. Audit the current routes, API endpoints, data loader, itinerary planner, Firebase auth, feedback loop, and map components.
3. Create `docs/REBUILD_PLAN.md` with:
   - current architecture,
   - problems found,
   - target architecture,
   - migration sequence,
   - files to keep,
   - files to split,
   - compatibility risks.
4. Do not delete working features.
5. Do not rewrite the whole system in one commit.
6. Implement phase by phase with tests and a runnable migration path.
7. Preserve current endpoints until equivalent `/api/v2/*` endpoints are stable.
8. Do not fabricate external data or silently replace missing values with Da Nang center coordinates.
9. Never claim a booking, ride, reservation, or payment has completed unless a real partner API confirms it.
10. Every recommendation must expose reasons, warnings, and source freshness.

---

# 1. Product direction

## 1.1 Primary product

Build **UrbanAgent Travel**, a personalized urban itinerary planner for independent travelers.

The product must let one traveler describe a trip in natural language and receive a feasible, editable, map-based itinerary personalized by:

- city or destination,
- date and time window,
- starting position or hotel,
- travel duration,
- budget,
- transport mode,
- travel party,
- interests,
- food preferences,
- pace,
- indoor/outdoor preference,
- weather,
- opening hours,
- travel time,
- previously liked/disliked places,
- places that must be included or excluded.

Core promise:

> Turn one natural-language request into a trip that can actually be followed.

Example:

> “Chiều nay tôi có 5 tiếng ở Đà Nẵng, đi cùng người yêu, muốn nơi có cảnh đẹp, ăn hải sản vừa túi tiền, không đi quá xa và tránh nơi quá đông.”

Expected result:

- a clear itinerary timeline,
- stop order,
- estimated arrival and stay time,
- travel time between stops,
- route shown on map,
- reasons for each stop,
- weather/opening-hour warnings,
- alternatives,
- one-click actions to open navigation, call, website, or partner service.

## 1.2 Geographic expansion

Da Nang is the first **City Pack**, not the permanent limit of the product.

Target expansion sequence:

1. Da Nang
2. Hoi An
3. Hue
4. Nha Trang
5. Da Lat
6. other tourism cities

A new city must not require copying business logic or rebuilding the app. A city is enabled by:

- a city configuration,
- a geographic boundary,
- a normalized POI dataset,
- category coverage,
- route and weather services,
- quality reports,
- a search/embedding index.

## 1.3 Secondary product

Keep the existing business-location functionality, but move it into a clearly separate product surface:

- Product name: `UrbanAgent Insights`
- Audience: local merchants, cafés, restaurants, hotels, tourism SMEs
- Route: `/partners` or separate partner dashboard
- It must not appear as a primary navigation item for ordinary travelers.
- It shares the POI and spatial-data core with UrbanAgent Travel.
- It has a separate user journey, pricing, onboarding, metrics, and sales strategy.

Do not present traveler planning and business site selection as one single user flow. They are two products using one intelligence core.

---

# 2. Findings in the current project that must be addressed

## 2.1 Frontend

The current project already contains useful functionality:

- traveler role,
- natural-language itinerary creation,
- map and route modal,
- itinerary editing,
- Google Maps/Grab handoff,
- feedback,
- user preferences and memory,
- seller analytics,
- admin and model metrics.

Do not rebuild features that already work.

However:

- `UrbanAgentPage.tsx` is too large and mixes API calls, state, maps, itinerary UI, weather, business panels, and helper functions.
- `RolePages.tsx` is also too large and mixes unrelated roles.
- The current UI resembles an analytics/admin dashboard more than a travel product.
- There are multiple gradient and AI-dashboard visual patterns that must be removed from the traveler experience.
- Authentication is too early in the traveler journey. Guest users should be able to generate a limited itinerary before account creation.

Split the frontend into domain modules.

Recommended target:

```text
src/
  app/
    router/
    providers/
    query/
  features/
    trip-planner/
      api/
      components/
      hooks/
      state/
      types/
      pages/
    poi-discovery/
    map/
    traveler-profile/
    feedback/
    partner-insights/
    admin/
  shared/
    components/
    design-system/
    lib/
    types/
  assets/
    brand/
    landmarks/
    video/
```

## 2.2 Backend

Current strengths:

- Express API,
- Firebase auth and persistence,
- itinerary flow,
- POI retrieval,
- reranker and memory artifacts,
- weather service,
- route expert system,
- feedback loop,
- business location scoring.

Critical fixes:

1. CSV must not remain the source of truth for a multi-city product.
2. `poiDataService` currently loads everything into process memory.
3. Missing coordinates must never default to Da Nang city center.
4. Review count mapping must support actual CSV field names such as `reviews_count`.
5. Text mapping must support actual fields such as `aggregated_reviews`.
6. Source-specific rows must be normalized through adapters.
7. Duplicate Google rows must be merged by external ID and spatial/name matching.
8. Address text must not determine current administrative units.
9. Itinerary order must not be based only on distance from the original start.
10. Fixed stay-time rules must be replaced by category-aware dwell-time estimates.
11. Route feasibility must include opening hours, time windows, weather, and transport.
12. The system needs city-aware filtering at every retrieval and planning stage.

Recommended backend structure:

```text
src/
  modules/
    cities/
    pois/
    ingestion/
    search/
    itineraries/
    routing/
    weather/
    preferences/
    feedback/
    partners/
    admin/
  infrastructure/
    db/
    firebase/
    external/
    jobs/
  shared/
    errors/
    validation/
    logging/
    types/
```

Use a modular monolith. Do not create microservices for the MVP.

---

# 3. Recommended technical architecture

## 3.1 Data storage

Use:

- PostgreSQL + PostGIS for canonical POI and spatial data.
- `pgvector` or an external vector index for POI embeddings.
- Firebase Auth for authentication.
- Firebase/Firestore may continue for lightweight user profiles, feedback events, or realtime UI, but POI geospatial truth belongs in PostGIS.
- Object storage for permitted images and generated artifacts.
- Redis is optional for live API caching and itinerary sessions.

Do not store a city as one giant CSV in production.

CSV remains supported only as:

- import format,
- export format,
- test fixture,
- compatibility deliverable.

## 3.2 Data layers

Implement three ingestion layers:

### Bronze — raw source records

Store untouched source payloads with:

- provider,
- provider record ID,
- fetched time,
- source URL,
- license/policy class,
- checksum,
- raw JSON,
- ingestion run ID.

### Silver — normalized source records

Normalize:

- names,
- coordinates,
- categories,
- contacts,
- address parts,
- opening hours,
- price,
- ratings,
- review counts,
- image references,
- status,
- text,
- source freshness.

### Gold — canonical POI entities

Create one stable internal POI entity that may link to many source records.

Gold entities power:

- traveler search,
- itinerary planning,
- map display,
- embeddings,
- partner analytics.

## 3.3 Core tables

Minimum schema:

```text
cities
admin_boundaries
poi_entities
poi_source_records
poi_external_ids
poi_categories
poi_images
poi_opening_hours
poi_reviews_summary
poi_embeddings
poi_aliases
poi_merge_candidates
ingestion_runs
data_quality_issues
trip_sessions
itineraries
itinerary_stops
user_preference_events
recommendation_feedback
partner_profiles
```

## 3.4 Canonical POI shape

```ts
type CanonicalPoi = {
  id: string;
  cityId: string;

  name: string;
  normalizedName: string;
  aliases: string[];

  categories: {
    primary: string;
    secondary: string[];
    rawBySource: Record<string, string[]>;
  };

  location: {
    lat: number;
    lng: number;
    geohash?: string;
    boundaryVersion: string;
    currentAdmin: {
      province?: string;
      ward?: string;
    };
    legacyAdmin?: {
      province?: string;
      district?: string;
      ward?: string;
    };
    originalAddresses: Array<{
      source: string;
      value: string;
    }>;
    displayAddress?: string;
  };

  contact: {
    phones: string[];
    websites: string[];
    socials: string[];
    emails: string[];
  };

  experience: {
    priceLevel?: number;
    priceMinVnd?: number;
    priceMaxVnd?: number;
    typicalVisitMinutes?: number;
    indoorOutdoor?: "indoor" | "outdoor" | "mixed";
    suitableFor: string[];
    accessibilityTags: string[];
    vibeTags: string[];
  };

  quality: {
    normalizedRating?: number;
    ratingCount?: number;
    ratingBySource: Record<string, {
      value?: number;
      scale?: number;
      count?: number;
      observedAt?: string;
    }>;
    freshnessScore: number;
    completenessScore: number;
    confidenceScore: number;
  };

  media: Array<{
    source: string;
    type: "photo" | "video";
    url?: string;
    storageKey?: string;
    attribution?: string;
    license?: string;
    expiresAt?: string;
  }>;

  semantic: {
    summaryVi?: string;
    summaryEn?: string;
    reviewSummary?: string;
    keywords: string[];
    embeddingText: string;
    embeddingVersion?: string;
  };

  externalIds: Record<string, string>;

  provenance: Record<string, {
    source: string;
    observedAt: string;
    confidence: number;
  }>;

  status: {
    operating: "open" | "temporarily_closed" | "permanently_closed" | "unknown";
    lastVerifiedAt?: string;
  };
};
```

## 3.5 Backward-compatible CSV export

Support a generated CSV with the current columns:

```text
name
address
phone
category
rating
reviews_count
price_range
aggregated_reviews
image_urls
lat
lng
google_url
place_id
LLM_Input_Text
```

But these are derived export columns. In the Phase 0 canonical dataset, `RestaurantID` remains a source identifier and must not be described as a verified Google `place_id`.

Add internal fields in the database:

```text
poi_id
city_id
source_ids
address_original
address_current
admin_boundary_version
data_freshness
data_confidence
field_provenance
operating_status
last_verified_at
```

---

# 4. Multi-source data strategy

## 4.1 Important rule

No single provider will reliably fill every column.

The pipeline must combine sources while retaining:

- field-level provenance,
- freshness,
- confidence,
- licensing restrictions,
- attribution requirements.

Never overwrite a higher-confidence value with a lower-confidence value only because it arrived later.

## 4.2 Source priority

### Baseline open data

Use Overture Maps Places as the first large-scale baseline because it supports geographic extraction by bounding box and contains stable place records with coordinates, categories, websites, phones, addresses, operating status, source metadata, and confidence.

Use OpenStreetMap/Overpass as a second baseline and enrichment source for:

- tourism,
- food,
- lodging,
- amenities,
- landmarks,
- parks,
- museums,
- opening hours,
- wheelchair tags,
- websites,
- phones,
- locally mapped features.

### Knowledge and landmark enrichment

Use Wikidata for notable attractions and landmarks:

- multilingual names,
- coordinates,
- official website,
- entity relationships,
- images through Wikimedia Commons,
- cultural/historic metadata.

Use Wikimedia Commons only when the media license and attribution are stored.

### Commercial/live enrichment

Google Places may be used for live enrichment such as:

- place ID,
- current address,
- coordinates,
- phone,
- website,
- hours,
- ratings,
- review count,
- photos,
- operating status.

However:

- do not design the permanent canonical database around unrestricted storage of Google Places content,
- store place IDs and permitted derived/internal data,
- retrieve time-sensitive fields live or under an allowed cache policy,
- display required attribution,
- do not permanently hotlink Google photo URLs as the product media library.

Foursquare may be added as an optional paid/open-source enrichment adapter if budget permits.

### Local and first-party enrichment

Prefer durable first-party data for the moat:

- merchant-verified profiles,
- tourism-department data,
- hotel/homestay partner recommendations,
- user corrections,
- completed-itinerary feedback,
- user-uploaded photos,
- local editors,
- operating-hour verification,
- seasonal/event information.

## 4.3 Source-to-field matrix

```text
Field                 Overture  OSM  Wikidata  Google live  Foursquare  First-party
name                      ✓      ✓       ✓          ✓            ✓           ✓
lat/lng                   ✓      ✓       ✓          ✓            ✓           ✓
category                  ✓      ✓       ✓          ✓            ✓           ✓
address                   ✓      ✓       partial    ✓            ✓           ✓
phone                     ✓      ✓       rare       ✓            ✓           ✓
website                   ✓      ✓       ✓          ✓            ✓           ✓
opening hours             -      ✓       rare       ✓            ✓           ✓
rating                    -      -       -          ✓            ✓           ✓
review count              -      -       -          ✓            ✓           ✓
review text               -      -       -          limited      tips        ✓
price range               -      partial -          ✓            partial     ✓
images                    -      partial Commons    refs         photos      ✓
historic/cultural text    -      partial ✓          partial      partial     ✓
accessibility             -      ✓       partial    partial      partial     ✓
vibe tags                 derived from legal text and first-party behavior
```

## 4.4 Data adapter contract

```ts
interface PoiSourceAdapter {
  provider: string;
  policyClass: "open" | "commercial-cache-limited" | "first-party";
  discover(city: CityConfig, cursor?: string): Promise<DiscoveryPage>;
  fetchDetails(externalId: string): Promise<RawPoiRecord>;
  normalize(raw: RawPoiRecord): Promise<NormalizedPoiRecord>;
}
```

Adapters:

```text
OvertureAdapter
OsmOverpassAdapter
WikidataAdapter
GooglePlacesAdapter
FoursquareAdapter
CsvLegacyAdapter
MerchantVerifiedAdapter
UserContributionAdapter
```

## 4.5 Deduplication

Deduplicate in this order:

1. Exact external ID match.
2. Exact phone or canonical website match.
3. Name similarity + spatial distance.
4. Address similarity + category compatibility.
5. Human review for uncertain matches.

Suggested merge candidate rule:

```text
distance <= 40 m
AND normalized name similarity >= 0.88
AND category compatibility >= 0.70
```

Do not merge branches of the same chain merely because names match.

Store merge confidence and allow undo.

## 4.6 Address normalization after administrative changes

Never trust formatted address text as the only source of current administrative units.

Process:

```text
valid lat/lng
  -> spatial point-in-polygon
  -> versioned administrative boundary
  -> current province/ward
```

Store:

- source address as observed,
- current normalized address,
- legacy address if known,
- boundary version,
- date normalized.

If a boundary source has not caught up with a legal change, maintain a versioned team-reviewed GeoJSON layer until an authoritative machine-readable source is available.

## 4.7 Data refresh

Recommended schedules:

- Overture full city import: each new release or monthly.
- OSM incremental refresh: weekly.
- landmark/Wikidata enrichment: monthly.
- merchant-verified fields: immediate.
- user corrections: moderation queue.
- live hours/status/rating: on demand with short TTL when provider policy permits.
- permanently closed detection: weekly candidate scan, then verification.
- embeddings: regenerate only when semantic fields materially change.

## 4.8 Quality gates

A city cannot be marked `READY` until:

- coordinate validity >= 99.5%,
- duplicate unresolved rate < 3%,
- required-category coverage passes thresholds,
- at least 80% of top attractions have images or valid visual fallback,
- at least 70% of top traveler POIs have hours or an “unknown hours” warning,
- route matrix works,
- no POI falls outside city bounds unless explicitly linked as a nearby excursion,
- address normalization has a boundary version,
- source attribution is available.

City states:

```text
DRAFT
INGESTING
VALIDATING
READY_FOR_BETA
READY
DEGRADED
```

---

# 5. City Pack system

## 5.1 City configuration

```yaml
id: da-nang
display_name_vi: Đà Nẵng
display_name_en: Da Nang
country_code: VN
timezone: Asia/Ho_Chi_Minh
currency: VND

center:
  lat: 16.0544
  lng: 108.2022

bbox:
  west: 107.80
  south: 15.80
  east: 108.50
  north: 16.30

boundary_version: VN_ADMIN_2025_V1

enabled_categories:
  - attraction
  - landmark
  - beach
  - museum
  - cafe
  - restaurant
  - local_food
  - nightlife
  - park
  - shopping
  - lodging
  - wellness
  - transportation

default_dwell_minutes:
  cafe: 60
  restaurant: 75
  attraction: 90
  museum: 90
  beach: 120
  shopping: 75

visual_identity:
  landmark_assets:
    - dragon-bridge
    - golden-bridge
    - my-khe-beach
    - marble-mountains
    - son-tra
```

## 5.2 City import command

Provide commands similar to:

```bash
npm run city:ingest -- --city=da-nang --sources=overture,osm,wikidata
npm run city:enrich -- --city=da-nang --sources=google
npm run city:dedupe -- --city=da-nang
npm run city:validate -- --city=da-nang
npm run city:index -- --city=da-nang
npm run city:export-csv -- --city=da-nang
```

A new city should be enabled through configuration and pipeline execution, not code duplication.

---

# 6. Traveler itinerary engine

## 6.1 Planning stages

Implement a deterministic, inspectable pipeline:

```text
1. Parse traveler request
2. Resolve city and time window
3. Build hard constraints
4. Retrieve candidate POIs
5. Apply semantic and preference ranking
6. Check opening hours and weather
7. Compute route matrix
8. Optimize order and time allocation
9. Generate explanations
10. Return alternatives and warnings
```

The LLM may:

- parse natural language,
- explain evidence,
- generate traveler-friendly summaries.

The LLM must not:

- invent POIs,
- invent opening hours,
- invent route times,
- invent ratings,
- invent booking confirmation.

## 6.2 Traveler intent contract

```ts
type TravelerIntent = {
  cityId: string;
  startLocation?: { lat: number; lng: number };
  startPlaceId?: string;
  startAt: string;
  endAt: string;
  dates?: string[];
  transportModes: Array<"walk" | "motorbike" | "car" | "taxi" | "public_transit">;
  party: {
    adults: number;
    children?: number;
    elderly?: number;
    relationship?: "solo" | "couple" | "friends" | "family" | "business";
  };
  budget?: {
    minVnd?: number;
    maxVnd?: number;
    level?: "budget" | "moderate" | "premium";
  };
  interests: string[];
  foodPreferences: string[];
  pace: "slow" | "balanced" | "packed";
  avoid: string[];
  mustInclude: string[];
  indoorOutdoor?: "indoor" | "outdoor" | "mixed";
  maximumTravelMinutesBetweenStops?: number;
  accessibilityNeeds: string[];
};
```

## 6.3 Feasibility rules

Every itinerary must satisfy or warn about:

- POI open at planned arrival,
- enough time for stay,
- travel time,
- total trip duration,
- weather suitability,
- transport compatibility,
- no duplicates,
- meal timing,
- start/end constraints,
- maximum detour,
- mandatory places,
- user exclusions.

## 6.4 Route optimization

For MVP:

- retrieve 20–60 candidates,
- score each candidate,
- select category-diverse candidates,
- use a route matrix,
- run nearest-neighbor insertion + local search,
- respect opening-hour time windows,
- calculate category-specific dwell time,
- provide one backup POI for each vulnerable stop.

Later, consider OR-Tools or another constraint solver.

## 6.5 Replanning

Add a key premium/useful feature:

> Replan from here.

Triggers:

- user is late,
- rain begins,
- place is closed,
- user removes a stop,
- user wants cheaper/closer/quieter options.

Replanning preserves:

- remaining time,
- current location,
- completed stops,
- must-visit stops,
- current budget.

## 6.6 API v2

```text
POST /api/v2/trips/preview
POST /api/v2/trips
GET  /api/v2/trips/:tripId
PATCH /api/v2/trips/:tripId
POST /api/v2/trips/:tripId/replan
POST /api/v2/trips/:tripId/stops
DELETE /api/v2/trips/:tripId/stops/:stopId
POST /api/v2/trips/:tripId/feedback
GET  /api/v2/cities
GET  /api/v2/cities/:cityId/status
GET  /api/v2/pois/:poiId
GET  /api/v2/pois/search
```

Response example:

```json
{
  "tripId": "trip_123",
  "city": {
    "id": "da-nang",
    "name": "Đà Nẵng"
  },
  "summary": {
    "startAt": "2026-07-25T14:00:00+07:00",
    "endAt": "2026-07-25T19:00:00+07:00",
    "distanceKm": 18.4,
    "travelMinutes": 63,
    "visitMinutes": 210,
    "estimatedSpendVnd": {
      "min": 450000,
      "max": 750000
    }
  },
  "stops": [],
  "route": {},
  "warnings": [],
  "alternatives": [],
  "explanation": {
    "intentSummary": "",
    "selectionSignals": []
  },
  "dataFreshness": {
    "generatedAt": "",
    "liveSourcesUsed": []
  }
}
```

---

# 7. Frontend experience

## 7.1 Product layout

Reference interaction patterns:

- Wanderlog: itinerary and map shown together, day-by-day organization, travel time between places.
- Roadtrippers: map-first exploration, editable waypoints, route customization.
- Do not copy branding or assets.
- Use the interaction pattern, not the exact appearance.

Desktop planner:

```text
┌──────────────────────────────────────────────────────────────┐
│ Brand | City | Trip dates | Profile                         │
├───────────────────────────────┬──────────────────────────────┤
│ Itinerary / assistant         │ Map                          │
│ 40% width                     │ 60% width                    │
│                               │                              │
│ Natural-language request      │ Route + numbered markers     │
│ Day tabs                      │ POI preview on marker click   │
│ Timeline cards                │                              │
│ Add/remove/reorder            │                              │
│ Warnings and alternatives     │                              │
└───────────────────────────────┴──────────────────────────────┘
```

Mobile:

- map and itinerary switch with a bottom segmented control,
- sticky “Tạo lịch trình” or “Cập nhật lộ trình” button,
- itinerary cards are compact,
- map opens full-screen,
- no permanent desktop sidebar,
- guest flow works without login.

## 7.2 Main screens

### Landing page

Purpose:

- explain value quickly,
- show one realistic itinerary example,
- allow immediate city/request input,
- establish tourism identity.

Sections:

1. Header
2. Hero with 3D-like Da Nang landmark animation
3. “Describe your day” input
4. Three-step explanation
5. Live itinerary preview
6. Supported City Packs
7. Traveler testimonials/pilot proof
8. Hotel/partner section
9. Pricing
10. Footer

### Planner

- city selector,
- date/time,
- natural-language input,
- preference drawer,
- itinerary timeline,
- map,
- route stats,
- warnings,
- alternatives,
- share/save/export.

### Trip library

- upcoming,
- completed,
- shared,
- draft.

### Preferences

Keep it simple:

- travel style,
- pace,
- budget,
- food,
- transport,
- accessibility,
- dislikes.

### Partner dashboard

Separate visual shell and route. Do not mix it with traveler navigation.

---

# 8. Visual direction

## 8.1 Brand vibe

The website must feel like:

- coastal travel,
- local discovery,
- sunny and human,
- trustworthy,
- modern but not futuristic,
- premium enough for tourism partners.

It must not feel like:

- generic AI SaaS,
- crypto/Web3,
- neon cyberpunk,
- developer dashboard,
- purple-blue AI gradient,
- excessive glassmorphism.

## 8.2 Color system

Use a restrained palette.

```css
--color-ocean: #0B3B60;       /* primary */
--color-ocean-dark: #082A44;
--color-sand: #F4EDE2;        /* page background */
--color-surface: #FFFDF9;
--color-coral: #E76F51;       /* CTA/accent */
--color-seafoam: #2A9D8F;     /* success/secondary */
--color-sun: #E9C46A;         /* limited highlight */
--color-ink: #1F2933;
--color-muted: #68737D;
--color-border: #DDD7CD;
```

Rules:

- Ocean blue is the dominant brand color.
- Coral is the main CTA color.
- Sand/ivory is the dominant background.
- Seafoam is secondary.
- Do not use gradients in standard cards, buttons, headings, or navigation.
- A subtle natural sky/light transition is allowed only inside the hero media itself, not as an AI-style UI gradient.
- Avoid pure white across the whole app; use warm surfaces.
- Avoid more than one accent color in the same component.

## 8.3 Typography

Preferred:

- `Be Vietnam Pro` for Vietnamese UI.
- Fallback: `Inter`, system sans-serif.

Use:

- strong but not oversized headings,
- normal sentence case,
- no all-caps dashboard headers,
- readable 15–17 px body text,
- 1.5–1.7 line height.

## 8.4 Components

- border radius: 12–18 px,
- subtle shadow only,
- 1 px warm-gray border,
- real travel photos or original 3D landmark art,
- numbered itinerary markers,
- simple category icons,
- no robot mascots in core planning screens,
- no glowing borders,
- no glass cards over maps,
- no animated particles.

---

# 9. 3D-like animation requirement

Create 3D-like website animations from simple videos, similar to an Emergent/Spline-style cinematic landing page, but keep the implementation performant and tourism-focused.

## 9.1 Assets

Use original stylized 3D assets or rendered video loops representing Da Nang:

- Dragon Bridge,
- Golden Bridge/Ba Na Hills,
- My Khe Beach,
- Marble Mountains,
- Son Tra Peninsula,
- Han River skyline.

Do not copy copyrighted commercial 3D models without license.

## 9.2 Hero composition

Suggested hero:

- foreground: phone/card showing a generated itinerary,
- midground: stylized 3D Dragon Bridge,
- background: sea, skyline, sun,
- small map pins moving along a route,
- 6–10 second seamless loop.

Implementation options:

### Preferred MVP

- pre-rendered transparent WebM or regular MP4,
- responsive `<video autoplay muted loop playsInline>`,
- poster fallback,
- CSS parallax layers,
- small cursor/scroll movement,
- no Three.js required.

### Optional richer version

- Spline scene embedded only in the landing hero,
- lazy-loaded after main content,
- static fallback for mobile and reduced-motion users.

## 9.3 Performance and accessibility

- hero media <= 3–5 MB if practical,
- use WebM + MP4 fallback,
- lazy-load below-the-fold videos,
- respect `prefers-reduced-motion`,
- mobile uses static poster if performance is weak,
- no autoplay audio,
- do not let animation block the planner input,
- Lighthouse performance target >= 85 on a realistic mobile test.

---

# 10. Monetization

## 10.1 Traveler revenue

Start with freemium.

Free:

- limited active trips,
- basic itinerary generation,
- map and route,
- manual editing,
- share link.

Paid traveler features:

- unlimited trips,
- smart “replan from here,”
- offline itinerary,
- multiple-day planning,
- group collaboration,
- advanced budget planning,
- hidden-gem mode,
- weather/closure alternatives,
- export PDF/calendar/map,
- priority support.

Treat initial prices as hypotheses to test, not fixed truths.

Suggested experiments:

- one-trip premium pack: 29.000–59.000 VNĐ,
- monthly plan: 49.000–99.000 VNĐ,
- annual plan only after repeated use is proven.

## 10.2 Transaction/referral revenue

Potential:

- tours and activities,
- hotel/homestay referrals,
- transport handoff,
- restaurant booking,
- attraction tickets,
- local experience vouchers.

Rules:

- label sponsored placements,
- do not let payment override itinerary relevance,
- keep organic and sponsored ranking separate,
- track conversion transparently.

## 10.3 B2B revenue

Best early B2B offer:

### White-label hotel/homestay concierge

A guest scans a QR code and opens a co-branded planner already anchored at the hotel.

Partner value:

- better guest experience,
- fewer repetitive front-desk questions,
- local partner referrals,
- usage analytics,
- curated hotel recommendations.

Pilot pricing hypothesis:

- free pilot for 30 days,
- then 500.000–2.000.000 VNĐ/month depending on property size and features,
- custom setup/integration fee for larger partners.

### UrbanAgent Insights

Keep for later:

- merchant profile verification,
- area/competition reports,
- partner demand insights,
- campaign recommendations.

Do not make this the main acquisition engine until traveler usage creates better first-party demand signals.

---

# 11. First-user strategy

## 11.1 Initial target

Do not target “all tourists.”

Start with:

> Independent Vietnamese travelers aged roughly 18–35 visiting Da Nang for 2–4 days as couples or small friend groups.

This group:

- searches on TikTok/Facebook/Google,
- changes plans frequently,
- cares about cafés, food, photos, beaches, and convenience,
- can test the natural-language itinerary value quickly.

## 11.2 First 100 users

Channels:

1. 30–50 students and friends planning real weekend trips.
2. Da Nang travel Facebook groups.
3. TikTok/Reels demos: one prompt → complete itinerary.
4. QR cards at 5–10 hostels/homestays.
5. Partnerships with local cafés or small tour hosts.
6. Tourism clubs at universities.
7. Shareable “48 hours in Da Nang” templates.

Acquisition offer:

- generate one itinerary without login,
- save/share requires lightweight sign-in,
- invite one friend to unlock an extra smart replan,
- collect structured feedback after the trip.

## 11.3 First partner pilot

Recruit:

- 5 homestays/hostels,
- 2 boutique hotels,
- 5 cafés/restaurants willing to verify data,
- 1 local tour operator.

Give each lodging partner:

- unique QR code,
- co-branded landing page,
- hotel as default starting point,
- suggested nearby itineraries,
- basic usage report.

## 11.4 Metrics

North-star candidate:

> Number of itineraries actually acted on.

Proxy events:

```text
trip_preview_generated
trip_saved
stop_added
stop_removed
route_opened
navigation_opened
trip_shared
trip_started
stop_completed
replan_used
post_trip_rating
```

Early funnel:

- landing → preview generation,
- preview → save,
- save → route open,
- route open → completed-stop feedback,
- week-4 repeat planning.

Do not optimize for raw page views.

## 11.5 Four-week validation sprint

### Week 1

- guest itinerary preview,
- fix data quality,
- new landing page,
- analytics events,
- one strong Da Nang demo scenario.

### Week 2

- recruit 50 testers,
- interview 10,
- measure generation/save/edit/map-open rates,
- fix top three failures.

### Week 3

- launch 5 lodging QR pilots,
- create 3 ready-made itinerary templates,
- enable share links,
- collect merchant corrections.

### Week 4

- test one paid traveler pack,
- test one paid partner proposal,
- decide which willingness-to-pay signal is stronger.

---

# 12. Implementation phases

## Phase 0 — Audit and safety

Deliverables:

- `docs/REBUILD_PLAN.md`
- architecture diagram
- endpoint inventory
- data-field inventory
- data-policy inventory
- current test baseline
- feature flags

Fix immediately:

- no coordinate fallback,
- correct actual CSV field mapping,
- city filtering,
- data validation report,
- duplicate report,
- guest-safe preview endpoint.

## Phase 1 — Data platform foundation

Deliverables:

- Postgres/PostGIS setup,
- migrations,
- legacy CSV importer,
- canonical POI schema,
- source adapters,
- provenance and freshness,
- city table and config,
- Da Nang City Pack import,
- quality dashboard/report.

Keep existing API working through a repository adapter.

## Phase 2 — Traveler API v2

Deliverables:

- intent schema,
- candidate retrieval,
- route matrix abstraction,
- time-window planning,
- opening-hour checks,
- category dwell times,
- itinerary explanation,
- replan endpoint,
- itinerary persistence.

## Phase 3 — Travel UI rebuild

Deliverables:

- tourism landing page,
- guest preview,
- split map/timeline planner,
- mobile map/itinerary switch,
- preference drawer,
- save/share,
- alternatives,
- warnings,
- loading/empty/error states,
- 3D-like Da Nang hero media.

## Phase 4 — City Pack automation

Deliverables:

- Overture importer,
- OSM importer,
- Wikidata/Wikimedia enrichment,
- optional Google live enrichment,
- dedupe,
- address spatial join,
- city validation,
- Hue or Hoi An pilot pack.

## Phase 5 — Monetization and pilot

Deliverables:

- premium feature flags,
- partner QR landing pages,
- referral tracking,
- sponsored-result labeling,
- analytics funnel,
- pilot dashboard,
- pricing experiment.

## Phase 6 — Separate partner product

Deliverables:

- `/partners`,
- merchant verification,
- hotel concierge setup,
- business insights,
- separate partner auth/onboarding.

---

# 13. Frontend component target

```text
features/trip-planner/
  pages/
    TripPlannerPage.tsx
    TripDetailsPage.tsx
  components/
    TripPromptBar.tsx
    TripConstraintDrawer.tsx
    DayTabs.tsx
    ItineraryTimeline.tsx
    ItineraryStopCard.tsx
    RouteSummary.tsx
    RouteMap.tsx
    StopAlternativesSheet.tsx
    TripWarnings.tsx
    ReplanButton.tsx
    ShareTripDialog.tsx
  hooks/
    useTripPreview.ts
    useTrip.ts
    useReplanTrip.ts
  state/
    tripDraftStore.ts
  api/
    tripApi.ts
  types/
    trip.ts

features/map/
  MapShell.tsx
  NumberedPoiMarker.tsx
  RoutePolyline.tsx
  MapPoiPreview.tsx

shared/design-system/
  Button.tsx
  Card.tsx
  Chip.tsx
  Input.tsx
  Drawer.tsx
  Dialog.tsx
  Skeleton.tsx
  EmptyState.tsx
```

Avoid files larger than roughly 400–500 lines unless there is a strong reason.

---

# 14. Testing

Backend tests:

- CSV legacy mapping,
- coordinate validation,
- city boundary filtering,
- dedup exact IDs,
- dedup name + distance,
- no branch over-merge,
- address spatial join,
- itinerary within requested duration,
- opening-hour conflict warning,
- duplicate stop prevention,
- replan preserves completed stops,
- external provider fallback,
- attribution metadata.

Frontend tests:

- guest can generate preview,
- user can add/remove/reorder,
- map and timeline stay synchronized,
- mobile view switch works,
- warning is visible,
- failed API has retry,
- reduced-motion disables hero motion,
- partner dashboard is absent from traveler navigation.

End-to-end scenario:

```text
Guest selects Da Nang
→ enters 5-hour couple itinerary request
→ receives 3–4 feasible stops
→ sees route and reasons
→ removes one stop
→ chooses a replacement
→ opens navigation
→ saves by signing in
→ submits post-trip feedback
```

---

# 15. Acceptance criteria

The rebuild is acceptable only when:

1. Existing traveler functionality is preserved or improved.
2. The primary product is clearly a traveler planner.
3. Business insights are separated from traveler UX.
4. A POI always belongs to a city.
5. No missing coordinate is silently replaced with a city center.
6. POI fields expose provenance and freshness.
7. Da Nang data imports from legacy CSV successfully.
8. One open-data adapter imports a second city without code duplication.
9. The itinerary respects requested time and warns on uncertainty.
10. The planner works before login in limited guest mode.
11. The UI uses the defined tourism palette and no generic AI gradients.
12. The landing hero uses lightweight Da Nang 3D-like media with fallbacks.
13. The app emits activation and itinerary-action analytics events.
14. README contains setup, migration, ingestion, and run instructions.
15. Each phase builds and passes tests.

---

# 16. First Codex execution prompt

Use this prompt after adding this file to the repositories:

> Read `URBANAGENT_CODEX_CONTEXT.md`, both existing `PLANNING.md` files, the frontend and backend source, and the two real POI CSV schemas. Do not code immediately. First create `docs/REBUILD_PLAN.md` containing: (1) current architecture, (2) concrete bugs and technical debt, (3) proposed target architecture, (4) database schema and migration plan, (5) API v2 contracts, (6) frontend component split, (7) data-source adapter strategy, (8) phased implementation plan, (9) risks, and (10) acceptance tests. Preserve existing working features. Explicitly identify every place where the current implementation is hard-coded to Da Nang or assumes CSV-only data. After writing the plan, stop and summarize the first implementation batch, limited to Phase 0 fixes.

# 17. Second Codex execution prompt — Phase 0

> Implement only Phase 0 from `docs/REBUILD_PLAN.md`. Do not start the full UI redesign or external-source ingestion. Fix unsafe POI normalization, remove coordinate fallback, correctly map both CSV schemas, add `cityId`, add duplicate/data-quality reports, add guest itinerary preview behind a feature flag, split no more than the most critical oversized frontend module, add tests, and keep existing endpoints compatible. Run build/tests and report changed files, migrations, remaining risks, and exact commands to verify locally.

# 18. Third Codex execution prompt — data foundation

> Implement Phase 1 data foundation. Add PostgreSQL/PostGIS migrations, canonical POI repositories, source records, external IDs, provenance, freshness, city configuration, a legacy CSV importer, dedup candidate generation, and Da Nang City Pack validation. Keep Firebase Auth. Make the existing recommendation service read through a repository abstraction so it can use either legacy CSV or PostGIS during migration. Add seed/import commands, tests, and rollback instructions. Do not add Google scraping.

# 19. Fourth Codex execution prompt — traveler experience

> Implement the traveler-focused API v2 and UI rebuild. Use a split itinerary/map layout inspired by mature trip planners, but apply the UrbanAgent coastal tourism design system. Add guest preview, editable timeline, route stats, warnings, alternatives, share/save, and mobile map/itinerary switching. Remove generic AI gradients and dashboard styling from traveler pages. Add the lightweight Da Nang 3D-like hero media component with video/poster/reduced-motion fallbacks. Keep `/partners` separate and do not surface seller tools in traveler navigation.
