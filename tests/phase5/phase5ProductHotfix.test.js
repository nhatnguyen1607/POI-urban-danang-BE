const assert = require('node:assert/strict');
const test = require('node:test');

const {
  DEFAULT_GEOCODER_URL,
  searchDestinations,
} = require('../../src/services/destinationGeocodingService');
const { buildTripPreview } = require('../../src/modules/travelerApiV2/tripPreview');
const { validateTripPreviewRequest } = require('../../src/modules/travelerApiV2/tripPreviewValidation');

function requestWithTemporaryPlace(overrides = {}) {
  return {
    cityId: 'da-nang',
    query: 'cafe và địa điểm đã chọn',
    trip: {
      date: '2026-08-22',
      dayCount: 1,
      dailyWindow: { startTime: '09:00', endTime: '18:00' },
      transport: 'motorbike',
      pace: 'balanced',
      budget: 'unknown',
    },
    constraints: {
      maxStopsPerDay: 3,
      mustIncludePoiIds: [],
      excludePoiIds: [],
      temporaryPlaces: [{
        id: 'temporary:photon:W:123',
        name: '110 Phước Tường 5',
        address: '110 Phước Tường 5, Đà Nẵng',
        category: 'Địa chỉ',
        lat: 16.038014,
        lon: 108.1742342,
        source: 'photon',
        canonical: false,
        attribution: '© OpenStreetMap contributors',
        ...overrides,
      }],
    },
  };
}

test('product hotfix geocoder uses the approved request-time Photon endpoint by default', async () => {
  let requestedUrl;
  const results = await searchDestinations({
    query: '110 phuoc tuong 5 da nang',
    fetchImpl: async (url) => {
      requestedUrl = url;
      return {
        ok: true,
        json: async () => ({
          features: [{
            geometry: { coordinates: [108.1742342, 16.038014] },
            properties: {
              osm_type: 'W',
              osm_id: 123,
              name: 'Đường Phước Tường 5',
              city: 'Đà Nẵng',
              country: 'Việt Nam',
            },
          }],
        }),
      };
    },
  });

  assert.equal(requestedUrl.origin + requestedUrl.pathname, DEFAULT_GEOCODER_URL);
  assert.equal(requestedUrl.searchParams.get('q'), '110 phuoc tuong 5 da nang');
  assert.equal(results.length, 1);
  assert.equal(results[0].source, 'photon');
});

test('product hotfix validates temporary places without turning them into canonical POIs', () => {
  const validation = validateTripPreviewRequest(requestWithTemporaryPlace());
  assert.equal(validation.errors, undefined);
  assert.deepEqual(validation.value.constraints.mustIncludePoiIds, ['temporary:photon:W:123']);
  assert.equal(validation.value.constraints.temporaryPlaces[0].canonical, false);

  const outside = validateTripPreviewRequest(requestWithTemporaryPlace({ lat: 21.0285, lon: 105.8542 }));
  assert.ok(outside.errors.some((error) => error.field.endsWith('.location')));
  const fabricatedCanonical = validateTripPreviewRequest(requestWithTemporaryPlace({ canonical: true }));
  assert.ok(fabricatedCanonical.errors.some((error) => error.field.endsWith('.canonical')));
});

test('product hotfix scheduler includes a temporary place and reports request-time provenance', async () => {
  const request = requestWithTemporaryPlace();
  request.trip.dayCount = 2;
  request.constraints.maxStopsPerDay = 4;
  request.constraints.mustIncludePoiIds = ['google_maps_2712', 'google_maps_1749', 'google_maps_1851'];
  const validation = validateTripPreviewRequest(request);
  assert.equal(validation.errors, undefined);
  const result = await buildTripPreview(validation.value);
  assert.equal(result.error, undefined);
  const scheduled = result.trip.stops.find((stop) => stop.poi.globalId === 'temporary:photon:W:123');
  assert.ok(scheduled);
  assert.equal(scheduled.poi.canonical, false);
  assert.equal(scheduled.poi.temporary, true);
  assert.equal(scheduled.poi.location.lat, 16.038014);
  for (const requiredId of validation.value.constraints.mustIncludePoiIds) {
    assert.ok(result.trip.stops.some((stop) => stop.poi.globalId === requiredId));
  }
  assert.equal(result.trip.provenance.externalLiveDataUsed, true);
});
