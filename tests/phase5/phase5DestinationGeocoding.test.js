const assert = require('node:assert/strict');
const test = require('node:test');

const {
  photonResult,
  providerUrl,
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
