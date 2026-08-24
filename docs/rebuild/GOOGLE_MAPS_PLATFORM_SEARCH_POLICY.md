# Google Maps Platform Search Policy

Status: implemented behind configuration; live acceptance pending.

## Scope

Google Maps Platform is the preferred request-time provider for place search,
category discovery, autocomplete, Place Details, and address geocoding. The
UrbanAgent canonical repository remains first-party and CSV remains the default
runtime. Photon/OpenStreetMap remains a bounded fallback. This work does not use
Google Routes API and does not alter the existing OSRM route engine.

## Required Configuration

Enable only these Google Cloud APIs:

- Maps JavaScript API
- Places API (New)
- Geocoding API

Runtime variables:

- `VITE_GOOGLE_MAPS_API_KEY`: browser key, restricted to Maps JavaScript API
  and approved production/local HTTP referrers.
- `GOOGLE_MAPS_SERVER_API_KEY`: server key, restricted to Places API (New) and
  Geocoding API. It must be stored as a deployment secret and never exposed to
  the browser or logs.

Billing and quota alerts must be configured before live acceptance. No key is
stored in this repository.

## Search Policy

- Queries are classified as `EXACT_ADDRESS`, `NAMED_PLACE`,
  `CATEGORY_NEARBY`, or `GENERAL_PLACE_QUERY`.
- Origin priority is reliable live GPS, active-trip context, selected map/trip
  context, then an explicit Da Nang scope fallback.
- An explicit near-me query without reliable GPS fails with
  `SEARCH_ORIGIN_REQUIRED`; the Da Nang center is never represented as the
  user's position.
- Category search expands deterministically through 3 km, 5 km, and 10 km.
  Normal local search is capped at 20 km. Results outside the active radius are
  rejected before ranking.
- Category ranking is deterministic: geographic eligibility, query/category
  match, distance, business status, then stable identity.
- Text Search, Nearby Search, Autocomplete, and Place Details request bounded
  result counts and minimal field masks. Autocomplete is debounced and uses a
  session token.
- Canonical and request-time results are deduplicated for display. Canonical
  identity wins; provider provenance is retained internally.

## Address Exactness

An address is silently accepted only when all of the following are true:

- requested house number equals returned `street_number`;
- requested street matches returned `route`;
- returned city/administrative context is consistent with Da Nang; and
- Google reports `ROOFTOP` precision.

Other results are classified as `INTERPOLATED_ADDRESS`, `STREET_LEVEL`, or
`APPROXIMATE` and require a user-confirmed map pin. A nearby business never
masquerades as the requested house address. The confirmed manual-pin
coordinates override provider approximations for itinerary, routing,
navigation, ride links, and saved-trip state.

## Map And Attribution

Google Places content is shown on the discovery map only when the Google Maps
JavaScript API is configured. Native Google attribution must remain visible.
Canonical, Photon, and manual-pin results may continue to use the existing
Leaflet/OpenStreetMap discovery fallback when no Google content is displayed.
Itinerary and route maps remain on the existing UrbanAgent/OSRM architecture.

## Persistence

- Google results remain noncanonical and `request_time_only`.
- The provider Place ID is retained as the stable refresh key where permitted.
- Raw Google response payloads, reviews, photos, and bulk provider content are
  not stored, indexed into canonical data, or redistributed.
- The saved-trip contract may retain the user's selected trip label, address,
  and location as a bounded trip snapshot; it is not a reusable POI catalog.
- Manual-pin-confirmed coordinates and user-authored labels are first-party
  trip state and carry `first_party_confirmed` policy metadata.
- Production activation requires a final review against the then-current
  Google Maps Platform agreement, including refresh/deletion requirements. A
  Place ID older than twelve months should be refreshed before provider reuse.

`AUTO_CREATE_NEW` remains false. No request-time result may become an approved
canonical POI automatically.

## Failure Order

When Google is unavailable or not configured, the product uses canonical
results, then bounded Photon/OpenStreetMap results, then manual pin according
to query type. A fallback result is never upgraded to rooftop exactness.

## Authoritative References

- [Places Text Search (New)](https://developers.google.com/maps/documentation/places/web-service/text-search)
- [Places Nearby Search (New)](https://developers.google.com/maps/documentation/places/web-service/nearby-search)
- [Places Autocomplete (New)](https://developers.google.com/maps/documentation/places/web-service/place-autocomplete)
- [Geocoding requests and precision](https://developers.google.com/maps/documentation/geocoding/requests-geocoding)
- [Places API policies](https://developers.google.com/maps/documentation/places/web-service/policies)
- [Maps JavaScript API policies](https://developers.google.com/maps/documentation/javascript/policies)
- [API security best practices](https://developers.google.com/maps/api-security-best-practices)
- [Place IDs](https://developers.google.com/maps/documentation/places/web-service/place-id)

## Current Gate

The code boundary and mock suite are complete, but neither required key is
configured in the validation environment. Live Google search, billing/quota,
referrer restriction, API restriction, attribution, and address acceptance
cannot be declared passed until staging configuration is supplied.

