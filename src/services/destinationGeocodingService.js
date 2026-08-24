const DEFAULT_DANANG_BIAS = { lat: 16.0544, lon: 108.2022 };
const DEFAULT_GEOCODER_URL = 'https://photon.komoot.io/api';
const GOOGLE_PLACES_BASE_URL = 'https://places.googleapis.com/v1';
const GOOGLE_GEOCODING_URL = 'https://maps.googleapis.com/maps/api/geocode/json';
const SUPPORTED_CITY_ID = 'da-nang';
const LOCAL_RADIUS_STEPS_M = [3_000, 5_000, 10_000];
const MAX_LOCAL_RADIUS_M = 20_000;
const GOOGLE_PLACE_FIELD_MASK = [
  'places.id',
  'places.displayName',
  'places.formattedAddress',
  'places.location',
  'places.primaryType',
  'places.types',
  'places.businessStatus',
  'places.googleMapsUri',
].join(',');

const CATEGORY_TYPES = [
  { patterns: ['quan com', 'quan an', 'nha hang', 'hai san', 'an uong'], types: ['restaurant', 'meal_takeaway'] },
  { patterns: ['cafe', 'ca phe', 'coffee'], types: ['cafe', 'coffee_shop'] },
  { patterns: ['tra sua', 'bubble tea'], types: ['cafe', 'beverage_store'] },
];

function serviceError(message, status, code) {
  const error = new Error(message);
  error.status = status;
  error.code = code;
  return error;
}

function cleanText(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function foldText(value) {
  return cleanText(value)
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/đ/gi, 'd')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
}

function normalizeCityId(value) {
  const cityId = cleanText(value) || SUPPORTED_CITY_ID;
  if (cityId !== SUPPORTED_CITY_ID) {
    throw serviceError('Destination search currently supports Da Nang only.', 400, 'UNSUPPORTED_GEOCODER_CITY');
  }
  return cityId;
}

function isNearMeQuery(query) {
  return /(?:^| )(?:gan toi|gan day|quanh day|near me)(?: |$)/.test(foldText(query));
}

function parseAddressQuery(query) {
  const original = cleanText(query);
  const match = original.match(/^\s*(\d+[a-z]?(?:[\/-]\d+[a-z]?)?)\s+(.+)$/iu);
  if (!match) return null;
  let street = match[2].trim().replace(/,+\s*$/, '');
  street = street
    .replace(/\s*,\s*(?:Đà Nẵng|Da Nang|Danang)\s*$/iu, '')
    .replace(/\s+(?:Đà Nẵng|Da Nang|Danang)\s*$/iu, '')
    .trim();
  if (!street) return null;
  return { original, houseNumber: match[1], street, city: 'Đà Nẵng', country: 'VN' };
}

function categoryTypesForQuery(query) {
  const folded = foldText(query);
  return CATEGORY_TYPES.find((entry) => entry.patterns.some((pattern) => folded.includes(pattern)))?.types || [];
}

function classifySearchIntent(query) {
  if (parseAddressQuery(query)) return 'EXACT_ADDRESS';
  const folded = foldText(query);
  const categoryTypes = categoryTypesForQuery(query);
  if (categoryTypes.length && (
    isNearMeQuery(query)
    || folded.split(' ').length <= 4
    || /(?:^| )(?:yen tinh|gan bien|dia phuong|binh dan)(?: |$)/.test(folded)
  )) return 'CATEGORY_NEARBY';
  if (/\b(?:ba|ong|co|tiem|quan|nha hang|hotel|homestay)\b/.test(folded) && folded.split(' ').length >= 3) {
    return 'NAMED_PLACE';
  }
  return 'GENERAL_PLACE_QUERY';
}

function validCoordinate(lat, lon) {
  return Number.isFinite(lat) && Number.isFinite(lon) && !(lat === 0 && lon === 0)
    && lat >= -90 && lat <= 90 && lon >= -180 && lon <= 180;
}

function resolveSearchOrigin({ query, lat, lon, accuracy, originSource } = {}) {
  const parsedLat = Number(lat);
  const parsedLon = Number(lon);
  const parsedAccuracy = accuracy === undefined || accuracy === null || accuracy === '' ? null : Number(accuracy);
  const source = cleanText(originSource);
  const reliableGps = source !== 'live_gps'
    || (Number.isFinite(parsedAccuracy) && parsedAccuracy > 0 && parsedAccuracy <= 150);
  if (validCoordinate(parsedLat, parsedLon) && reliableGps) {
    return {
      lat: parsedLat,
      lon: parsedLon,
      accuracy: Number.isFinite(parsedAccuracy) ? parsedAccuracy : null,
      source: source || 'map_context',
      scopeFallback: false,
    };
  }
  if (isNearMeQuery(query)) {
    throw serviceError('Bật vị trí để tìm địa điểm gần bạn.', 422, 'SEARCH_ORIGIN_REQUIRED');
  }
  return { ...DEFAULT_DANANG_BIAS, accuracy: null, source: 'da_nang_scope', scopeFallback: true };
}

function haversineMeters(from, to) {
  const radians = (value) => value * Math.PI / 180;
  const earthRadiusM = 6_371_000;
  const deltaLat = radians(to.lat - from.lat);
  const deltaLon = radians(to.lon - from.lon);
  const startLat = radians(from.lat);
  const endLat = radians(to.lat);
  const value = Math.sin(deltaLat / 2) ** 2
    + Math.cos(startLat) * Math.cos(endLat) * Math.sin(deltaLon / 2) ** 2;
  return 2 * earthRadiusM * Math.asin(Math.sqrt(value));
}

function radiusRectangle(origin, radiusM) {
  const latitudeDelta = radiusM / 111_320;
  const longitudeDelta = radiusM / (111_320 * Math.max(Math.cos(origin.lat * Math.PI / 180), 0.2));
  return {
    low: { latitude: origin.lat - latitudeDelta, longitude: origin.lon - longitudeDelta },
    high: { latitude: origin.lat + latitudeDelta, longitude: origin.lon + longitudeDelta },
  };
}

function providerUrl(endpoint, allowedHosts = process.env.URBANAGENT_GEOCODER_ALLOWED_HOSTS) {
  let url;
  try {
    url = new URL(endpoint);
  } catch {
    throw serviceError('Destination geocoding provider is misconfigured.', 503, 'GEOCODER_INVALID_CONFIG');
  }
  const hosts = cleanText(allowedHosts || 'photon.komoot.io')
    .split(',').map((host) => host.trim().toLowerCase()).filter(Boolean);
  const localDevelopmentHost = process.env.NODE_ENV !== 'production'
    && ['127.0.0.1', 'localhost', '::1'].includes(url.hostname.toLowerCase());
  if (url.username || url.password || (!localDevelopmentHost && url.protocol !== 'https:')
    || (!localDevelopmentHost && !hosts.includes(url.hostname.toLowerCase()))) {
    throw serviceError('Destination geocoding provider is not allowed.', 503, 'GEOCODER_PROVIDER_NOT_ALLOWED');
  }
  return url;
}

function photonResult(feature, index) {
  const properties = feature && typeof feature.properties === 'object' ? feature.properties : {};
  const coordinates = Array.isArray(feature?.geometry?.coordinates) ? feature.geometry.coordinates : [];
  const lon = Number(coordinates[0]);
  const lat = Number(coordinates[1]);
  if (!validCoordinate(lat, lon)) return null;
  const name = cleanText(properties.name);
  const streetAddress = [cleanText(properties.housenumber), cleanText(properties.street)].filter(Boolean).join(' ');
  const locality = [properties.district, properties.city, properties.state, properties.country]
    .map(cleanText).filter(Boolean);
  const address = [streetAddress, ...locality].filter(Boolean).join(', ');
  const label = name || streetAddress || address;
  if (!label) return null;
  const sourceId = [cleanText(properties.osm_type), cleanText(String(properties.osm_id || ''))]
    .filter(Boolean).join(':') || `result:${index}`;
  const category = cleanText(properties.osm_value || properties.type);
  const addressLike = Boolean(streetAddress || properties.type === 'house' || properties.osm_key === 'highway');
  return {
    id: `photon:${sourceId}`,
    type: addressLike ? 'address' : 'place',
    label,
    address: address || label,
    category: category || null,
    lat,
    lon,
    source: 'photon',
    attribution: '© OpenStreetMap contributors',
    exactness: addressLike ? 'APPROXIMATE' : 'NAMED_PLACE',
    autoConfirmed: false,
    requiresConfirmation: addressLike,
  };
}

async function requestJson(url, options, fetchImpl, failurePrefix) {
  let response;
  try {
    response = await fetchImpl(url, { ...options, signal: AbortSignal.timeout(8_000) });
  } catch (error) {
    const timedOut = error?.name === 'AbortError' || error?.name === 'TimeoutError';
    throw serviceError(
      timedOut ? `${failurePrefix} timed out.` : `${failurePrefix} is unavailable.`,
      502,
      timedOut ? 'GOOGLE_UPSTREAM_TIMEOUT' : 'GOOGLE_UPSTREAM_FAILED',
    );
  }
  if (!response.ok) throw serviceError(`${failurePrefix} returned ${response.status}.`, 502, 'GOOGLE_UPSTREAM_FAILED');
  try {
    return await response.json();
  } catch {
    throw serviceError(`${failurePrefix} returned an invalid response.`, 502, 'GOOGLE_INVALID_RESPONSE');
  }
}

function googlePlaceResult(place, origin) {
  const lat = Number(place?.location?.latitude);
  const lon = Number(place?.location?.longitude);
  if (!validCoordinate(lat, lon)) return null;
  const label = cleanText(place?.displayName?.text);
  const address = cleanText(place?.formattedAddress);
  if (!label && !address) return null;
  return {
    id: `google:${cleanText(place.id)}`,
    providerPlaceId: cleanText(place.id),
    type: 'place',
    label: label || address,
    address,
    category: cleanText(place.primaryType) || null,
    types: Array.isArray(place.types) ? place.types.map(cleanText).filter(Boolean) : [],
    lat,
    lon,
    source: 'google_places',
    attribution: 'Google Maps',
    googleMapsUri: cleanText(place.googleMapsUri) || null,
    businessStatus: cleanText(place.businessStatus) || null,
    distanceMeters: haversineMeters(origin, { lat, lon }),
    exactness: 'NAMED_PLACE',
    autoConfirmed: true,
    requiresConfirmation: false,
  };
}

function addressComponent(result, type) {
  const component = Array.isArray(result?.address_components)
    ? result.address_components.find((item) => Array.isArray(item.types) && item.types.includes(type))
    : null;
  return cleanText(component?.long_name);
}

function streetMatches(requested, returned) {
  const left = foldText(requested).replace(/\b(?:duong|street|road)\b/g, '').trim();
  const right = foldText(returned).replace(/\b(?:duong|street|road)\b/g, '').trim();
  return Boolean(left && right && (left === right || left.includes(right) || right.includes(left)));
}

function mapGoogleGranularity(locationType) {
  if (locationType === 'ROOFTOP') return 'EXACT_ROOFTOP';
  if (locationType === 'RANGE_INTERPOLATED') return 'INTERPOLATED_ADDRESS';
  if (locationType === 'GEOMETRIC_CENTER') return 'STREET_LEVEL';
  return 'APPROXIMATE';
}

function googleGeocodeResult(result, parsedAddress, index = 0) {
  const lat = Number(result?.geometry?.location?.lat);
  const lon = Number(result?.geometry?.location?.lng);
  if (!validCoordinate(lat, lon)) return null;
  const returnedHouseNumber = addressComponent(result, 'street_number');
  const returnedRoute = addressComponent(result, 'route');
  const localityCandidates = [
    addressComponent(result, 'locality'),
    addressComponent(result, 'sublocality'),
    addressComponent(result, 'administrative_area_level_2'),
    addressComponent(result, 'administrative_area_level_1'),
  ].filter(Boolean);
  const locationType = cleanText(result?.geometry?.location_type) || 'APPROXIMATE';
  const houseMatches = foldText(parsedAddress.houseNumber) === foldText(returnedHouseNumber);
  const routeMatches = streetMatches(parsedAddress.street, returnedRoute);
  const cityMatches = localityCandidates.some((value) => foldText(value).includes('da nang'));
  const autoConfirmed = houseMatches && routeMatches && cityMatches && locationType === 'ROOFTOP';
  return {
    id: `google-geocode:${cleanText(result.place_id) || index}`,
    providerPlaceId: cleanText(result.place_id),
    type: 'address',
    label: cleanText(result.formatted_address) || parsedAddress.original,
    address: cleanText(result.formatted_address) || parsedAddress.original,
    category: 'street_address',
    lat,
    lon,
    source: 'google_geocoding',
    attribution: 'Google Maps',
    exactness: mapGoogleGranularity(locationType),
    googleGranularity: locationType,
    autoConfirmed,
    requiresConfirmation: !autoConfirmed,
    addressMatch: {
      requestedHouseNumber: parsedAddress.houseNumber,
      returnedHouseNumber: returnedHouseNumber || null,
      requestedStreet: parsedAddress.street,
      returnedRoute: returnedRoute || null,
      cityConsistent: cityMatches,
    },
  };
}

function googleHeaders(apiKey, fieldMask = GOOGLE_PLACE_FIELD_MASK) {
  return { 'Content-Type': 'application/json', 'X-Goog-Api-Key': apiKey, 'X-Goog-FieldMask': fieldMask };
}

async function googleTextSearch({ query, origin, radiusM, limit, apiKey, fetchImpl }) {
  const payload = await requestJson(`${GOOGLE_PLACES_BASE_URL}/places:searchText`, {
    method: 'POST',
    headers: googleHeaders(apiKey),
    body: JSON.stringify({
      textQuery: query,
      pageSize: Math.min(limit, 20),
      languageCode: 'vi',
      regionCode: 'VN',
      locationRestriction: { rectangle: radiusRectangle(origin, radiusM) },
    }),
  }, fetchImpl, 'Google Text Search');
  return (Array.isArray(payload?.places) ? payload.places : [])
    .map((place) => googlePlaceResult(place, origin)).filter(Boolean)
    .filter((place) => place.distanceMeters <= radiusM);
}

async function googleNearbySearch({ query, origin, radiusM, limit, apiKey, fetchImpl }) {
  const payload = await requestJson(`${GOOGLE_PLACES_BASE_URL}/places:searchNearby`, {
    method: 'POST',
    headers: googleHeaders(apiKey),
    body: JSON.stringify({
      includedTypes: categoryTypesForQuery(query),
      maxResultCount: Math.min(limit, 20),
      languageCode: 'vi',
      regionCode: 'VN',
      locationRestriction: { circle: { center: { latitude: origin.lat, longitude: origin.lon }, radius: radiusM } },
      rankPreference: 'DISTANCE',
    }),
  }, fetchImpl, 'Google Nearby Search');
  return (Array.isArray(payload?.places) ? payload.places : [])
    .map((place) => googlePlaceResult(place, origin)).filter(Boolean)
    .filter((place) => place.distanceMeters <= radiusM)
    .sort((a, b) => a.distanceMeters - b.distanceMeters || a.id.localeCompare(b.id));
}

async function googleGeocode({ query, apiKey, fetchImpl }) {
  const parsedAddress = parseAddressQuery(query);
  if (!parsedAddress) return [];
  const url = new URL(GOOGLE_GEOCODING_URL);
  url.searchParams.set('address', parsedAddress.original);
  url.searchParams.set('components', 'country:VN|administrative_area:Da Nang');
  url.searchParams.set('language', 'vi');
  url.searchParams.set('region', 'vn');
  url.searchParams.set('key', apiKey);
  const payload = await requestJson(url, { headers: { Accept: 'application/json' } }, fetchImpl, 'Google Geocoding');
  return (Array.isArray(payload?.results) ? payload.results : [])
    .map((result, index) => googleGeocodeResult(result, parsedAddress, index)).filter(Boolean);
}

async function searchGoogleDestinations({ query, intent, origin, limit, apiKey, fetchImpl }) {
  if (intent === 'EXACT_ADDRESS') return googleGeocode({ query, apiKey, fetchImpl });
  if (intent === 'CATEGORY_NEARBY') {
    const simpleCategory = categoryTypesForQuery(query).length > 0
      && foldText(query).replace(/\b(?:gan toi|gan day|quanh day|o da nang)\b/g, '').trim().split(' ').length <= 3;
    let results = [];
    for (const radiusM of LOCAL_RADIUS_STEPS_M) {
      results = simpleCategory
        ? await googleNearbySearch({ query, origin, radiusM, limit, apiKey, fetchImpl })
        : await googleTextSearch({ query, origin, radiusM, limit, apiKey, fetchImpl });
      if (results.length >= Math.min(3, limit)) return { results: results.slice(0, limit), radiusM };
    }
    return { results: results.slice(0, limit), radiusM: LOCAL_RADIUS_STEPS_M.at(-1) };
  }
  const results = await googleTextSearch({ query, origin, radiusM: MAX_LOCAL_RADIUS_M, limit, apiKey, fetchImpl });
  return { results: results.slice(0, limit), radiusM: MAX_LOCAL_RADIUS_M };
}

async function searchDestinations({
  query,
  cityId = SUPPORTED_CITY_ID,
  limit = 8,
  origin = DEFAULT_DANANG_BIAS,
  maxDistanceM = MAX_LOCAL_RADIUS_M,
  fetchImpl = global.fetch,
  endpoint = process.env.URBANAGENT_GEOCODER_URL || DEFAULT_GEOCODER_URL,
  allowedHosts = process.env.URBANAGENT_GEOCODER_ALLOWED_HOSTS,
} = {}) {
  normalizeCityId(cityId);
  const normalizedQuery = cleanText(query);
  if (normalizedQuery.length < 3 || normalizedQuery.length > 160) {
    throw serviceError('Search query must contain 3 to 160 characters.', 400, 'INVALID_GEOCODE_QUERY');
  }
  if (!endpoint) throw serviceError('Destination geocoding is not configured.', 503, 'GEOCODER_NOT_CONFIGURED');
  if (typeof fetchImpl !== 'function') {
    throw serviceError('Geocoding transport is unavailable.', 503, 'GEOCODER_TRANSPORT_UNAVAILABLE');
  }
  const safeLimit = Math.min(Math.max(Number(limit) || 8, 1), 10);
  const url = providerUrl(endpoint, allowedHosts);
  url.searchParams.set('q', normalizedQuery);
  url.searchParams.set('limit', String(safeLimit));
  url.searchParams.set('lat', String(origin.lat));
  url.searchParams.set('lon', String(origin.lon));
  url.searchParams.set('lang', 'default');
  let response;
  try {
    response = await fetchImpl(url, {
      headers: { Accept: 'application/json', 'User-Agent': 'UrbanAgent/1.0 (+https://github.com/nhatnguyen1607/POI-urban-danang-BE)' },
      signal: AbortSignal.timeout(8_000),
    });
  } catch (error) {
    const timedOut = error?.name === 'AbortError' || error?.name === 'TimeoutError';
    throw serviceError(
      timedOut ? 'Destination search timed out.' : 'Destination search provider is unavailable.',
      502,
      timedOut ? 'GEOCODER_UPSTREAM_TIMEOUT' : 'GEOCODER_UPSTREAM_FAILED',
    );
  }
  if (!response.ok) throw serviceError(`Geocoding provider returned ${response.status}.`, 502, 'GEOCODER_UPSTREAM_FAILED');
  let payload;
  try {
    payload = await response.json();
  } catch {
    throw serviceError('Destination search provider returned an invalid response.', 502, 'GEOCODER_INVALID_RESPONSE');
  }
  return (Array.isArray(payload?.features) ? payload.features : [])
    .map(photonResult).filter(Boolean)
    .map((result) => ({ ...result, distanceMeters: haversineMeters(origin, result) }))
    .filter((result) => result.distanceMeters <= maxDistanceM)
    .slice(0, safeLimit);
}

async function resolveDestinationSearch({
  query,
  cityId = SUPPORTED_CITY_ID,
  limit = 8,
  lat,
  lon,
  accuracy,
  originSource,
  googleMapCompliant = false,
  fetchImpl = global.fetch,
  googleApiKey = process.env.GOOGLE_MAPS_SERVER_API_KEY,
  photonEndpoint = process.env.URBANAGENT_GEOCODER_URL || DEFAULT_GEOCODER_URL,
  photonAllowedHosts = process.env.URBANAGENT_GEOCODER_ALLOWED_HOSTS,
} = {}) {
  normalizeCityId(cityId);
  const normalizedQuery = cleanText(query);
  if (normalizedQuery.length < 3 || normalizedQuery.length > 160) {
    throw serviceError('Search query must contain 3 to 160 characters.', 400, 'INVALID_GEOCODE_QUERY');
  }
  const intent = classifySearchIntent(normalizedQuery);
  const origin = resolveSearchOrigin({ query: normalizedQuery, lat, lon, accuracy, originSource });
  const safeLimit = Math.min(Math.max(Number(limit) || 8, 1), 10);
  const googleConfigured = Boolean(cleanText(googleApiKey));
  const mayDisplayGoogle = googleConfigured && googleMapCompliant === true;
  let googleFailure = null;
  if (mayDisplayGoogle) {
    try {
      const googleResponse = await searchGoogleDestinations({
        query: normalizedQuery, intent, origin, limit: safeLimit, apiKey: googleApiKey, fetchImpl,
      });
      const results = Array.isArray(googleResponse) ? googleResponse : googleResponse.results;
      if (results.length) {
        return {
          results,
          meta: {
            cityId: SUPPORTED_CITY_ID,
            source: intent === 'EXACT_ADDRESS' ? 'google_geocoding' : 'google_places',
            queryIntent: intent,
            requestTimeOnly: true,
            canonicalDataChanged: false,
            googleConfigured: true,
            googleMapRequired: true,
            origin,
            radiusM: Array.isArray(googleResponse) ? null : googleResponse.radiusM,
          },
        };
      }
    } catch (error) {
      googleFailure = error?.code || 'GOOGLE_UPSTREAM_FAILED';
    }
  }
  const maxDistanceM = intent === 'CATEGORY_NEARBY' ? LOCAL_RADIUS_STEPS_M.at(-1) : MAX_LOCAL_RADIUS_M;
  const fallback = await searchDestinations({
    query: normalizedQuery,
    cityId,
    limit: safeLimit,
    origin,
    maxDistanceM,
    fetchImpl,
    endpoint: photonEndpoint,
    allowedHosts: photonAllowedHosts,
  });
  return {
    results: fallback,
    meta: {
      cityId: SUPPORTED_CITY_ID,
      source: 'photon',
      queryIntent: intent,
      requestTimeOnly: true,
      canonicalDataChanged: false,
      googleConfigured,
      googleMapRequired: googleConfigured,
      configurationRequired: !googleConfigured || !googleMapCompliant,
      origin,
      radiusM: maxDistanceM,
      fallbackReason: googleFailure
        || (!googleConfigured ? 'GOOGLE_MAPS_PLATFORM_CONFIGURATION_REQUIRED' : 'GOOGLE_MAP_DISPLAY_REQUIRED'),
    },
  };
}

async function autocompleteGooglePlaces({
  input,
  sessionToken,
  lat,
  lon,
  accuracy,
  originSource,
  googleMapCompliant = false,
  fetchImpl = global.fetch,
  googleApiKey = process.env.GOOGLE_MAPS_SERVER_API_KEY,
} = {}) {
  const query = cleanText(input);
  if (query.length < 3 || query.length > 160) return [];
  if (!cleanText(googleApiKey) || googleMapCompliant !== true) {
    throw serviceError('Google Maps Platform configuration is required.', 503, 'GOOGLE_MAPS_PLATFORM_CONFIGURATION_REQUIRED');
  }
  const origin = resolveSearchOrigin({ query, lat, lon, accuracy, originSource });
  const payload = await requestJson(`${GOOGLE_PLACES_BASE_URL}/places:autocomplete`, {
    method: 'POST',
    headers: googleHeaders(googleApiKey, 'suggestions.placePrediction.placeId,suggestions.placePrediction.text'),
    body: JSON.stringify({
      input: query,
      sessionToken: cleanText(sessionToken) || undefined,
      includedRegionCodes: ['vn'],
      languageCode: 'vi',
      regionCode: 'VN',
      origin: { latitude: origin.lat, longitude: origin.lon },
      locationRestriction: {
        circle: { center: { latitude: origin.lat, longitude: origin.lon }, radius: MAX_LOCAL_RADIUS_M },
      },
    }),
  }, fetchImpl, 'Google Autocomplete');
  return (Array.isArray(payload?.suggestions) ? payload.suggestions : []).flatMap((suggestion) => {
    const prediction = suggestion?.placePrediction;
    const placeId = cleanText(prediction?.placeId);
    const text = cleanText(prediction?.text?.text);
    return placeId && text ? [{ placeId, text, attribution: 'Google Maps' }] : [];
  }).slice(0, 5);
}

async function resolveGooglePlace({
  placeId,
  sessionToken,
  lat,
  lon,
  accuracy,
  originSource,
  googleMapCompliant = false,
  fetchImpl = global.fetch,
  googleApiKey = process.env.GOOGLE_MAPS_SERVER_API_KEY,
} = {}) {
  const safePlaceId = cleanText(placeId);
  if (!/^[A-Za-z0-9_-]{8,256}$/.test(safePlaceId)) {
    throw serviceError('Invalid Google Place ID.', 400, 'INVALID_GOOGLE_PLACE_ID');
  }
  if (!cleanText(googleApiKey) || googleMapCompliant !== true) {
    throw serviceError('Google Maps Platform configuration is required.', 503, 'GOOGLE_MAPS_PLATFORM_CONFIGURATION_REQUIRED');
  }
  const origin = resolveSearchOrigin({ query: 'Đà Nẵng', lat, lon, accuracy, originSource });
  const url = new URL(`${GOOGLE_PLACES_BASE_URL}/places/${encodeURIComponent(safePlaceId)}`);
  url.searchParams.set('languageCode', 'vi');
  url.searchParams.set('regionCode', 'VN');
  if (cleanText(sessionToken)) url.searchParams.set('sessionToken', cleanText(sessionToken));
  const detailMask = GOOGLE_PLACE_FIELD_MASK.replaceAll('places.', '');
  const payload = await requestJson(
    url,
    { headers: googleHeaders(googleApiKey, detailMask) },
    fetchImpl,
    'Google Place Details',
  );
  const result = googlePlaceResult(payload, origin);
  if (!result) throw serviceError('Google Place Details returned no usable location.', 502, 'GOOGLE_INVALID_RESPONSE');
  return result;
}

module.exports = {
  DEFAULT_DANANG_BIAS,
  DEFAULT_GEOCODER_URL,
  GOOGLE_GEOCODING_URL,
  GOOGLE_PLACES_BASE_URL,
  LOCAL_RADIUS_STEPS_M,
  MAX_LOCAL_RADIUS_M,
  autocompleteGooglePlaces,
  categoryTypesForQuery,
  classifySearchIntent,
  googleGeocodeResult,
  googlePlaceResult,
  haversineMeters,
  isNearMeQuery,
  mapGoogleGranularity,
  normalizeCityId,
  parseAddressQuery,
  photonResult,
  providerUrl,
  resolveDestinationSearch,
  resolveGooglePlace,
  resolveSearchOrigin,
  searchDestinations,
  streetMatches,
};
