const DEFAULT_DANANG_BIAS = { lat: 16.0544, lon: 108.2022 };

function serviceError(message, status, code) {
  const error = new Error(message);
  error.status = status;
  error.code = code;
  return error;
}

function cleanText(value) {
  return typeof value === 'string' ? value.trim() : '';
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
  limit = 8,
  fetchImpl = global.fetch,
  endpoint = process.env.URBANAGENT_GEOCODER_URL,
} = {}) {
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
  const url = new URL(endpoint);
  url.searchParams.set('q', normalizedQuery);
  url.searchParams.set('limit', String(safeLimit));
  url.searchParams.set('lat', String(DEFAULT_DANANG_BIAS.lat));
  url.searchParams.set('lon', String(DEFAULT_DANANG_BIAS.lon));
  url.searchParams.set('lang', 'vi');

  const response = await fetchImpl(url, {
    headers: {
      Accept: 'application/json',
      'User-Agent': 'UrbanAgent/1.0 (+https://github.com/nhatnguyen1607/POI-urban-danang-BE)',
    },
    signal: AbortSignal.timeout(8000),
  });
  if (!response.ok) {
    throw serviceError(`Geocoding provider returned ${response.status}.`, 502, 'GEOCODER_UPSTREAM_FAILED');
  }
  const payload = await response.json();
  const features = Array.isArray(payload?.features) ? payload.features : [];
  return features.map(photonResult).filter(Boolean).slice(0, safeLimit);
}

module.exports = {
  photonResult,
  searchDestinations,
};
