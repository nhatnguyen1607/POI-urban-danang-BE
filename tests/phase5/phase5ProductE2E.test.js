const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');

const { getProfile } = require('../../src/services/agentMemoryService');
const {
  categoryMatchScore,
  detectIntents,
} = require('../../src/services/intentService');
const { loadPOIs, getPoiDataQualityReport } = require('../../src/services/poiDataService');
const { applyReranker } = require('../../src/services/rerankerService');
const { malformedJsonErrorHandler } = require('../../src/middleware/malformedJsonError');
const {
  getTravelerRecommendations,
  rankRecommendationItemsForQuery,
} = require('../../src/modules/travelerApiV2/recommendations');
const { buildTripPreview } = require('../../src/modules/travelerApiV2/tripPreview');
const { validateTripPreviewRequest } = require('../../src/modules/travelerApiV2/tripPreviewValidation');

const CANONICAL_PATH = path.resolve(__dirname, '..', '..', 'data', 'canonical', 'urbanagent_poi_master_v1.csv');
const EXPECTED_SHA = 'dcb404cc8b5c7a9b5fd70df63039ab8f828c504270e22b12a671fa4ed61583f4';
const MARBLE_MOUNTAINS_ID = 'candidate:da-nang:c017ebe6c5f87ec2';

function previewRequest(query, excludePoiIds = []) {
  const validation = validateTripPreviewRequest({
    cityId: 'da-nang',
    query,
    trip: {
      date: '2026-08-15',
      dayCount: 2,
      dailyWindow: { startTime: '09:00', endTime: '20:00' },
      dayWindows: [],
      transport: 'motorbike',
      pace: 'relaxed',
      budget: 'unknown',
    },
    constraints: {
      maxStopsPerDay: 3,
      mustIncludePoiIds: [],
      excludePoiIds,
    },
    recommendationOptions: { limit: 9 },
  });
  assert.equal(validation.error, undefined);
  return validation.value;
}

test('Phase 5A intent matching uses word boundaries and recognizes multi-intent Vietnamese input', () => {
  const intents = detectIntents('Tôi muốn uống cafe yên tĩnh rồi ăn hải sản.');
  assert.deepEqual(intents.map((intent) => intent.id).slice(0, 2), ['cafe', 'seafood']);
  const seafood = intents.find((intent) => intent.id === 'seafood');
  assert.equal(categoryMatchScore({
    name: 'BonPas Bakery & Cafe - Điện Biên Phủ',
    category: 'Café/Dessert',
    text: '',
  }, seafood), 0.12);
});

test('Phase 5A anonymous requests do not inherit global demo feedback memory', () => {
  assert.equal(getProfile({}), null);
  assert.ok(getProfile({ useGlobalMemory: true }));
});

test('Phase 5A reranker artifact cannot saturate a moderate base score', () => {
  const reranked = applyReranker({
    poi: { id: 'phase5a-synthetic-cafe', category: 'Café/Dessert' },
    score: 0.5,
  }, 'cafe', {});
  assert.ok(reranked.score >= 0.5);
  assert.ok(reranked.score <= 0.62);
});

test('Phase 5A malformed JSON handler returns a public-safe v2 error envelope', () => {
  const error = new SyntaxError('Unexpected token');
  error.status = 400;
  error.type = 'entity.parse.failed';
  const req = { originalUrl: '/api/v2/recommendations', travelerApiV2: { requestId: 'req_phase5' } };
  const response = {};
  const res = {
    status(status) {
      response.status = status;
      return this;
    },
    json(body) {
      response.body = body;
      return body;
    },
  };
  malformedJsonErrorHandler(error, req, res, () => assert.fail('Malformed JSON should be handled'));
  assert.equal(response.status, 400);
  assert.equal(response.body.error.code, 'MALFORMED_JSON');
  assert.equal(response.body.meta.requestId, 'req_phase5');
  assert.ok(!JSON.stringify(response.body).includes('Unexpected token'));
});

test('Phase 5A query ranking prioritizes an explicitly named POI and preserves intent diversity', () => {
  const items = [
    { id: 'cafe-1', name: 'Quiet Cafe', category: 'Quán cà phê', score: 90, scoreRaw: 0.9 },
    { id: 'seafood-1', name: 'Seafood House', category: 'Nhà hàng hải sản', score: 89, scoreRaw: 0.89 },
    { id: MARBLE_MOUNTAINS_ID, name: 'Di tích danh thắng Ngũ Hành Sơn, Đà Nẵng', category: 'attraction', score: 50, scoreRaw: 0.5 },
  ];
  const named = rankRecommendationItemsForQuery(items, 'Tôi muốn đi Ngũ Hành Sơn rồi uống cafe.', 3);
  assert.equal(named[0].id, MARBLE_MOUNTAINS_ID);
  const mixed = rankRecommendationItemsForQuery(items, 'Uống cafe rồi ăn hải sản.', 2);
  assert.deepEqual(new Set(mixed.map((item) => item.category)), new Set(['Quán cà phê', 'Nhà hàng hải sản']));
});

test('Phase 5A product smoke covers 4173 loader, recommendations, preview and replan', async () => {
  const canonicalSha = crypto.createHash('sha256').update(fs.readFileSync(CANONICAL_PATH)).digest('hex');
  assert.equal(canonicalSha, EXPECTED_SHA);

  const [pois, quality] = await Promise.all([loadPOIs(), getPoiDataQualityReport()]);
  assert.equal(pois.length, 4173);
  assert.equal(new Set(pois.map((poi) => poi.globalId)).size, 4173);
  assert.equal(quality.totals.applicationPois, 4173);
  assert.equal(quality.totals.invalidRows, 0);
  assert.ok(pois.some((poi) => poi.globalId === MARBLE_MOUNTAINS_ID));

  const cafeSeafood = await getTravelerRecommendations({
    cityId: 'da-nang',
    query: 'Tối nay tôi muốn uống cafe yên tĩnh gần biển rồi ăn hải sản.',
    context: {},
    limit: 6,
  });
  assert.ok(cafeSeafood.recommendations.length > 0);
  assert.equal(new Set(cafeSeafood.recommendations.map((item) => item.poi.id)).size, cafeSeafood.recommendations.length);
  assert.ok(cafeSeafood.recommendations.some((item) => /cafe|cà phê|dessert/i.test(item.poi.category)));
  assert.ok(cafeSeafood.recommendations.some((item) => /hải sản|seafood/i.test(item.poi.category)));
  assert.ok(cafeSeafood.recommendations.every((item) => (
    item.poi.id && item.poi.name && item.poi.category
      && Number.isFinite(item.poi.location.lat) && Number.isFinite(item.poi.location.lon)
  )));

  const named = await getTravelerRecommendations({
    cityId: 'da-nang',
    query: 'Tôi muốn đi Ngũ Hành Sơn rồi tìm quán ăn hoặc cafe phù hợp gần đó.',
    context: {},
    limit: 6,
  });
  assert.ok(named.recommendations.some((item) => item.poi.id === MARBLE_MOUNTAINS_ID));

  const query = 'Cafe yên tĩnh gần biển rồi ăn hải sản.';
  const initial = await buildTripPreview(previewRequest(query));
  assert.ok(initial.trip.stops.length > 0);
  assert.equal(new Set(initial.trip.stops.map((stop) => stop.poi.id)).size, initial.trip.stops.length);
  assert.ok(initial.trip.stops.every((stop) => !stop.departureTime || stop.departureTime <= '20:00'));

  const removedPoiId = initial.trip.stops[0].poi.id;
  const replanned = await buildTripPreview(previewRequest(`${query} Ưu tiên ít di chuyển.`, [removedPoiId]));
  assert.ok(replanned.trip.stops.length > 0);
  assert.ok(replanned.trip.stops.every((stop) => stop.poi.id !== removedPoiId));
});
