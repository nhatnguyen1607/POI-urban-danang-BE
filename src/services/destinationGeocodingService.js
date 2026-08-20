const DEFAULT_DANANG_BIAS = { lat: 16.0544, lon: 108.2022 };
const SUPPORTED_CITY_ID = 'da-nang';

function serviceError(message, status, code) {
  const error = new Error(message);
  error.status = status;
  error.code = code;
  return error;
}

function cleanText(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function normalizeCityId(value) {
  const cityId = cleanText(value) || SUPPORTED_CITY_ID;
  if (cityId !== SUPPORTED_CITY_ID) {
    throw serviceError('Destination search currently supports Da Nang only.', 400, 'UNSUPPORTED_GEOCODER_CITY');
  }
  return cityId;
}

function providerUrl(endpoint, allowedHosts = process.env.URBANAGENT_GEOCODER_ALLOWED_HOSTS) {
  let url;
  try {
    url = new URL(endpoint);
  } catch {
    throw serviceError('Destination geocoding provider is misconfigured.', 503, 'GEOCODER_INVALID_CONFIG');
  }

  const hosts = cleanText(allowedHosts || 'photon.komoot.io')
    .split(',')
    .map((host) => host.trim().toLowerCase())
    .filter(Boolean);
  const localDevelopmentHost = process.env.NODE_ENV !== 'production'
    && ['127.0.0.1', 'localhost', '::1'].includes(url.hostname.toLowerCase());
  if (
    url.username
    || url.password
    || (!localDevelopmentHost && url.protocol !== 'https:')
    || (!localDevelopmentHost && !hosts.includes(url.hostname.toLowerCase()))
  ) {
    throw serviceError('Destination geocoding provider is not allowed.', 503, 'GEOCODER_PROVIDER_NOT_ALLOWED');
  }
  return url;
}

function photonResult(feature, index) {
  const properties = feature && typeof feature.properties === 'object' ? feature.properties : {};
  const coordinates = Array.isArray(feature?.geometry?.coordinates) ? feature.geometry.coordinates : [];
  const lon = Number(coordinates[0]);
  const lat = Number(coordinates[1]);
  if (
    !Number.isFinite(lat)
    || !Number.isFinite(lon)
    || (lat === 0 && lon === 0)
    || lat < -90
    || lat > 90
    || lon < -180
    || lon > 180
  ) return null;

  const name = cleanText(properties.name);
  const streetAddress = [cleanText(properties.housenumber), cleanText(properties.street)]
    .filter(Boolean)
    .join(' ');
  const locality = [
    cleanText(properties.district),
    cleanText(properties.city),
    cleanText(properties.state),
    cleanText(properties.country),
  ].filter(Boolean);
  const address = [streetAddress, ...locality].filter(Boolean).join(', ');
  const label = name || streetAddress || address;
  if (!label) return null;
  const osmType = cleanText(properties.osm_type);
  const osmId = cleanText(String(properties.osm_id || ''));
  const sourceId = [osmType, osmId].filter(Boolean).join(':') || `result:${index}`;
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
  };
}

async function searchDestinations({
  query,
  cityId = SUPPORTED_CITY_ID,
  limit = 8,
  fetchImpl = global.fetch,
  endpoint = process.env.URBANAGENT_GEOCODER_URL,
  allowedHosts = process.env.URBANAGENT_GEOCODER_ALLOWED_HOSTS,
} = {}) {
  normalizeCityId(cityId);
  const normalizedQuery = cleanText(query);
  if (normalizedQuery.length < 3 || normalizedQuery.length > 160) {
    throw serviceError('Search query must contain 3 to 160 characters.', 400, 'INVALID_GEOCODE_QUERY');
  }
  if (!endpoint) {
    throw serviceError(
      'Destination geocoding is not configured.',
      503,
      'GEOCODER_NOT_CONFIGURED',
    );
  }
  if (typeof fetchImpl !== 'function') {
    throw serviceError('Geocoding transport is unavailable.', 503, 'GEOCODER_TRANSPORT_UNAVAILABLE');
  }

  const safeLimit = Math.min(Math.max(Number(limit) || 8, 1), 10);
  const url = providerUrl(endpoint, allowedHosts);
  url.searchParams.set('q', normalizedQuery);
  url.searchParams.set('limit', String(safeLimit));
  url.searchParams.set('lat', String(DEFAULT_DANANG_BIAS.lat));
  url.searchParams.set('lon', String(DEFAULT_DANANG_BIAS.lon));
  url.searchParams.set('lang', 'default');

  let response;
  try {
    response = await fetchImpl(url, {
      headers: {
        Accept: 'application/json',
        'User-Agent': 'UrbanAgent/1.0 (+https://github.com/nhatnguyen1607/POI-urban-danang-BE)',
      },
      signal: AbortSignal.timeout(8000),
    });
  } catch (error) {
    const timedOut = error?.name === 'AbortError' || error?.name === 'TimeoutError';
    throw serviceError(
      timedOut ? 'Destination search timed out.' : 'Destination search provider is unavailable.',
      502,
      timedOut ? 'GEOCODER_UPSTREAM_TIMEOUT' : 'GEOCODER_UPSTREAM_FAILED',
    );
  }
  if (!response.ok) {
    throw serviceError(`Geocoding provider returned ${response.status}.`, 502, 'GEOCODER_UPSTREAM_FAILED');
  }
  let payload;
  try {
    payload = await response.json();
  } catch {
    throw serviceError('Destination search provider returned an invalid response.', 502, 'GEOCODER_INVALID_RESPONSE');
  }
  const features = Array.isArray(payload?.features) ? payload.features : [];
  return features.map(photonResult).filter(Boolean).slice(0, safeLimit);
}

module.exports = {
  photonResult,
  normalizeCityId,
  providerUrl,
  searchDestinations,
};
