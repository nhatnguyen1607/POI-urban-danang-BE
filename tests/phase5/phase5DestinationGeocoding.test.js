const assert = require('node:assert/strict');
const test = require('node:test');

const {
  photonResult,
  searchDestinations,
} = require('../../src/services/destinationGeocodingService');

test('destination geocoder remains disabled without an explicitly configured provider', async () => {
  await assert.rejects(
    () => searchDestinations({ query: '110 Phuoc Tuong 5', endpoint: '' }),
    (error) => error.code === 'GEOCODER_NOT_CONFIGURED' && error.status === 503,
  );
});

test('destination geocoder normalizes bounded request-time Photon results without POI IDs', async () => {
  let requestedUrl;
  const results = await searchDestinations({
    query: '110 Phuoc Tuong 5, Da Nang',
    endpoint: 'https://geocoder.invalid/api',
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
