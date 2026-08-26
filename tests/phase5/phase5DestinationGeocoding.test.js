const assert = require('node:assert/strict');
const test = require('node:test');

const {
  autocompleteGooglePlaces,
  classifySearchIntent,
  googleGeocodeResult,
  parseAddressQuery,
  photonResult,
  providerUrl,
  resolveDestinationSearch,
  resolveGooglePlace,
  resolveSearchOrigin,
  searchDestinations,
} = require('../../src/services/destinationGeocodingService');
const { createGeocoderRateLimit } = require('../../src/middleware/geocoderRateLimit');

test('destination geocoder remains disabled without an explicitly configured provider', async () => {
  await assert.rejects(
    () => searchDestinations({ query: '110 Phuoc Tuong 5', endpoint: '' }),
    (error) => error.code === 'GEOCODER_NOT_CONFIGURED' && error.status === 503,
  );
  await assert.rejects(
    () => searchDestinations({ query: 'Hanoi address', cityId: 'ha-noi', endpoint: 'https://provider.invalid/api' }),
    (error) => error.code === 'UNSUPPORTED_GEOCODER_CITY' && error.status === 400,
  );
});

test('destination geocoder normalizes bounded request-time Photon results without POI IDs', async () => {
  let requestedUrl;
  const results = await searchDestinations({
    query: '110 Phuoc Tuong 5, Da Nang',
    endpoint: 'https://geocoder.invalid/api',
    allowedHosts: 'geocoder.invalid',
    fetchImpl: async (url) => {
      requestedUrl = url;
      return {
        ok: true,
        json: async () => ({
          features: [{
            geometry: { coordinates: [108.181, 16.048] },
            properties: {
              osm_type: 'W',
              osm_id: 123,
              osm_key: 'highway',
              osm_value: 'residential',
              housenumber: '110',
              street: 'Phước Tường 5',
              district: 'Cẩm Lệ',
              city: 'Đà Nẵng',
              country: 'Việt Nam',
            },
          }],
        }),
      };
    },
  });

  assert.equal(requestedUrl.searchParams.get('q'), '110 Phuoc Tuong 5, Da Nang');
  assert.equal(requestedUrl.searchParams.get('lat'), '16.0544');
  assert.equal(requestedUrl.searchParams.get('lon'), '108.2022');
  assert.equal(requestedUrl.searchParams.get('lang'), 'default');
  assert.equal(results.length, 1);
  assert.equal(results[0].type, 'address');
  assert.equal(results[0].label, '110 Phước Tường 5');
  assert.equal(results[0].lat, 16.048);
  assert.equal(results[0].lon, 108.181);
  assert.equal(Object.hasOwn(results[0], 'poiId'), false);
  assert.equal(results[0].attribution, '© OpenStreetMap contributors');
});

test('destination geocoder restricts provider URLs and sanitizes upstream failures', async () => {
  assert.throws(
    () => providerUrl('http://metadata.internal/api', 'metadata.internal'),
    (error) => error.code === 'GEOCODER_PROVIDER_NOT_ALLOWED',
  );
  assert.throws(
    () => providerUrl('https://user:secret@photon.komoot.io/api'),
    (error) => error.code === 'GEOCODER_PROVIDER_NOT_ALLOWED',
  );
  await assert.rejects(
    () => searchDestinations({
      query: 'Cầu Rồng Đà Nẵng',
      endpoint: 'https://photon.komoot.io/api',
      fetchImpl: async () => { throw new Error('secret upstream details'); },
    }),
    (error) => error.code === 'GEOCODER_UPSTREAM_FAILED'
      && !error.message.includes('secret upstream details'),
  );
});

test('destination geocoder proxy applies a bounded in-memory request limit', () => {
  let currentTime = 1_000;
  const middleware = createGeocoderRateLimit({
    maxRequests: 2,
    windowMs: 1_000,
    now: () => currentTime,
  });
  const responses = [];
  const response = {
    set: (name, value) => responses.push({ name, value }),
    status(status) {
      this.statusCode = status;
      return this;
    },
    json(payload) {
      this.payload = payload;
      return this;
    },
  };
  let accepted = 0;
  middleware({}, response, () => { accepted += 1; });
  middleware({}, response, () => { accepted += 1; });
  middleware({}, response, () => { accepted += 1; });
  assert.equal(accepted, 2);
  assert.equal(response.statusCode, 429);
  assert.equal(response.payload.error, 'GEOCODER_RATE_LIMITED');
  assert.equal(responses[0].name, 'Retry-After');

  currentTime += 1_000;
  middleware({}, response, () => { accepted += 1; });
  assert.equal(accepted, 3);
});

test('destination geocoder rejects invalid coordinates and keeps provider IDs noncanonical', () => {
  assert.equal(photonResult({ geometry: { coordinates: [0, 0] }, properties: { name: 'Invalid' } }, 0), null);
  const result = photonResult({
    geometry: { coordinates: [108.22, 16.06] },
    properties: { osm_type: 'N', osm_id: 42, name: 'Cầu Rồng', osm_value: 'attraction' },
  }, 0);
  assert.equal(result.id, 'photon:N:42');
  assert.equal(result.type, 'place');
  assert.equal(Object.hasOwn(result, 'poiId'), false);
});

test('search intent distinguishes exact addresses, named places, and local categories', () => {
  assert.equal(classifySearchIntent('470 Trần Đại Nghĩa, Đà Nẵng'), 'EXACT_ADDRESS');
  assert.equal(classifySearchIntent('470 tran dai nghia da nang'), 'EXACT_ADDRESS');
  assert.equal(classifySearchIntent('Bánh xèo Bà Dưỡng'), 'NAMED_PLACE');
  assert.equal(classifySearchIntent('quán cơm gần tôi'), 'CATEGORY_NEARBY');
  assert.equal(classifySearchIntent('cafe yên tĩnh gần biển'), 'CATEGORY_NEARBY');
  assert.equal(classifySearchIntent('cầu ngắm hoàng hôn'), 'GENERAL_PLACE_QUERY');
  assert.deepEqual(parseAddressQuery('110 phuoc tuong 5 da nang'), {
    original: '110 phuoc tuong 5 da nang',
    houseNumber: '110',
    street: 'phuoc tuong 5',
    city: 'Đà Nẵng',
    country: 'VN',
  });
});

test('near-me search requires reliable GPS and never substitutes Da Nang center for the user', () => {
  assert.throws(
    () => resolveSearchOrigin({ query: 'quán cơm gần tôi' }),
    (error) => error.code === 'SEARCH_ORIGIN_REQUIRED' && error.status === 422,
  );
  assert.throws(
    () => resolveSearchOrigin({
      query: 'quán cơm gần tôi', lat: 16.06, lon: 108.22, accuracy: 900, originSource: 'live_gps',
    }),
    (error) => error.code === 'SEARCH_ORIGIN_REQUIRED',
  );
  assert.equal(resolveSearchOrigin({ query: 'quán cơm ở Đà Nẵng' }).source, 'da_nang_scope');
});

function geocodePayload({
  houseNumber,
  route,
  locationType,
  lat = 16.02,
  lon = 108.24,
  placeId = 'ChIJTestAddress123',
} = {}) {
  return {
    place_id: placeId,
    formatted_address: `${houseNumber || ''} ${route}, Đà Nẵng`.trim(),
    geometry: { location: { lat, lng: lon }, location_type: locationType },
    address_components: [
      ...(houseNumber ? [{ long_name: houseNumber, types: ['street_number'] }] : []),
      { long_name: route, types: ['route'] },
      { long_name: 'Ngũ Hành Sơn', types: ['administrative_area_level_2'] },
      { long_name: 'Đà Nẵng', types: ['administrative_area_level_1'] },
    ],
  };
}

test('Google geocoding silently confirms only matching rooftop addresses', () => {
  const requested = parseAddressQuery('470 Trần Đại Nghĩa, Đà Nẵng');
  const exact = googleGeocodeResult(geocodePayload({
    houseNumber: '470', route: 'Trần Đại Nghĩa', locationType: 'ROOFTOP',
  }), requested);
  assert.equal(exact.exactness, 'EXACT_ROOFTOP');
  assert.equal(exact.autoConfirmed, true);
  assert.equal(exact.requiresConfirmation, false);

  for (const variant of [
    geocodePayload({ houseNumber: '306', route: 'Trần Đại Nghĩa', locationType: 'ROOFTOP' }),
    geocodePayload({ houseNumber: '470', route: 'Trần Đại Nghĩa', locationType: 'RANGE_INTERPOLATED' }),
    geocodePayload({ route: 'Trần Đại Nghĩa', locationType: 'GEOMETRIC_CENTER' }),
    geocodePayload({ houseNumber: '470', route: 'Lê Duẩn', locationType: 'ROOFTOP' }),
  ]) {
    const result = googleGeocodeResult(variant, requested);
    assert.equal(result.autoConfirmed, false);
    assert.equal(result.requiresConfirmation, true);
  }
});

test('Google address resolver preserves requested house number and reports returned components', async () => {
  const cases = [
    ['470 Trần Đại Nghĩa, Đà Nẵng', '470', 'Trần Đại Nghĩa', 'ROOFTOP', true],
    ['470 tran dai nghia da nang', '470', 'Trần Đại Nghĩa', 'ROOFTOP', true],
    ['306 Trần Đại Nghĩa, Đà Nẵng', '306', 'Trần Đại Nghĩa', 'RANGE_INTERPOLATED', false],
    ['110 Phước Tường 5, Đà Nẵng', '110', 'Phước Tường 5', 'ROOFTOP', true],
    ['110 phuoc tuong 5 da nang', '110', 'Phước Tường 5', 'ROOFTOP', true],
    ['25 Bạch Đằng, Đà Nẵng', '25', 'Bạch Đằng', 'ROOFTOP', true],
    ['99999 Hoàng Diệu, Đà Nẵng', null, 'Hoàng Diệu', 'GEOMETRIC_CENTER', false],
  ];
  for (const [query, returnedHouse, route, granularity, expectedExact] of cases) {
    const response = await resolveDestinationSearch({
      query,
      googleApiKey: 'test-key',
      googleMapCompliant: true,
      fetchImpl: async () => ({
        ok: true,
        json: async () => ({
          results: [geocodePayload({
            houseNumber: returnedHouse,
            route,
            locationType: granularity,
            placeId: `ChIJ${String(returnedHouse || 'missing').padEnd(8, '0')}`,
          })],
        }),
      }),
    });
    assert.equal(response.meta.source, 'google_geocoding');
    assert.equal(response.results[0].addressMatch.requestedHouseNumber, parseAddressQuery(query).houseNumber);
    assert.equal(response.results[0].googleGranularity, granularity);
    assert.equal(response.results[0].autoConfirmed, expectedExact);
    assert.equal(response.results[0].requiresConfirmation, !expectedExact);
  }
});

test('nearby Google category search expands deterministically and rejects out-of-radius places', async () => {
  const radii = [];
  const fetchImpl = async (url, options) => {
    assert.match(String(url), /places:searchNearby$/);
    const body = JSON.parse(options.body);
    radii.push(body.locationRestriction.circle.radius);
    const radius = body.locationRestriction.circle.radius;
    const nearPlaces = radius === 3_000 ? [{
      id: 'near-one', displayName: { text: 'Cơm nhà gần' }, formattedAddress: 'Đà Nẵng',
      location: { latitude: 16.0601, longitude: 108.2201 }, primaryType: 'restaurant', businessStatus: 'OPERATIONAL',
    }] : [
      { id: 'near-one', displayName: { text: 'Cơm nhà gần' }, formattedAddress: 'Đà Nẵng', location: { latitude: 16.0601, longitude: 108.2201 }, primaryType: 'restaurant' },
      { id: 'near-two', displayName: { text: 'Cơm quê' }, formattedAddress: 'Đà Nẵng', location: { latitude: 16.065, longitude: 108.225 }, primaryType: 'restaurant' },
      { id: 'near-three', displayName: { text: 'Cơm biển' }, formattedAddress: 'Đà Nẵng', location: { latitude: 16.07, longitude: 108.23 }, primaryType: 'restaurant' },
      { id: 'far-away', displayName: { text: 'Cơm rất xa' }, formattedAddress: 'Huế', location: { latitude: 16.7, longitude: 107.6 }, primaryType: 'restaurant' },
    ];
    return { ok: true, json: async () => ({ places: nearPlaces }) };
  };
  const response = await resolveDestinationSearch({
    query: 'quán cơm gần tôi',
    lat: 16.06,
    lon: 108.22,
    accuracy: 12,
    originSource: 'live_gps',
    googleApiKey: 'test-key',
    googleMapCompliant: true,
    fetchImpl,
  });
  assert.deepEqual(radii, [3_000, 5_000]);
  assert.equal(response.meta.radiusM, 5_000);
  assert.equal(response.results.length, 3);
  assert.ok(response.results.every((result) => result.distanceMeters <= 5_000));
  assert.deepEqual(response.results.map((result) => result.id), ['google:near-one', 'google:near-two', 'google:near-three']);
});

test('Google autocomplete and Place Details use bounded fields and require map-compliant configuration', async () => {
  await assert.rejects(
    () => autocompleteGooglePlaces({ input: '470 Trần', googleApiKey: '', googleMapCompliant: false }),
    (error) => error.code === 'GOOGLE_MAPS_PLATFORM_CONFIGURATION_REQUIRED',
  );
  const calls = [];
  const fetchImpl = async (url, options) => {
    calls.push({ url: String(url), options });
    if (String(url).includes(':autocomplete')) {
      return {
        ok: true,
        json: async () => ({ suggestions: [{ placePrediction: { placeId: 'ChIJAutocomplete1', text: { text: '470 Trần Đại Nghĩa' } } }] }),
      };
    }
    return {
      ok: true,
      json: async () => ({
        id: 'ChIJAutocomplete1',
        displayName: { text: 'Bánh xèo Bà Dưỡng' },
        formattedAddress: 'Đà Nẵng',
        location: { latitude: 16.05, longitude: 108.21 },
        primaryType: 'restaurant',
      }),
    };
  };
  const suggestions = await autocompleteGooglePlaces({
    input: '470 Trần', sessionToken: 'session-1', googleApiKey: 'test-key', googleMapCompliant: true, fetchImpl,
  });
  assert.equal(suggestions[0].placeId, 'ChIJAutocomplete1');
  const place = await resolveGooglePlace({
    placeId: 'ChIJAutocomplete1', sessionToken: 'session-1', googleApiKey: 'test-key', googleMapCompliant: true, fetchImpl,
  });
  assert.equal(place.source, 'google_places');
  assert.equal(place.providerPlaceId, 'ChIJAutocomplete1');
  assert.match(calls[0].options.headers['X-Goog-FieldMask'], /placePrediction\.placeId/);
  assert.doesNotMatch(calls[1].options.headers['X-Goog-FieldMask'], /reviews|photos/);
});
