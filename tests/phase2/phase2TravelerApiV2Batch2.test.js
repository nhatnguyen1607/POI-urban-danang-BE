const assert = require('node:assert/strict');
const { spawn } = require('node:child_process');
const http = require('node:http');
const path = require('node:path');
const test = require('node:test');

const fixture = require('../fixtures/phase2/recommendationQueries.json');
const {
  DEFAULT_RECOMMENDATION_LIMIT,
  MAX_RECOMMENDATION_LIMIT,
} = require('../../src/modules/travelerApiV2/constants');
const {
  getTravelerRecommendations,
  parseRecommendationLimit,
  rankRecommendationItems,
  serializeRecommendationItem,
  validateRecommendationRequest,
} = require('../../src/modules/travelerApiV2/recommendations');
const { recommendPOIs } = require('../../src/services/poiRetrievalService');

const DEFAULT_CITY_ID = 'da-nang';

function requestJson({ port, method = 'GET', path: requestPath, headers = {}, body }) {
  return new Promise((resolve, reject) => {
    const payload = body ? JSON.stringify(body) : null;
    const req = http.request({
      hostname: '127.0.0.1',
      port,
      method,
      path: requestPath,
      headers: {
        ...headers,
        ...(payload ? { 'content-type': 'application/json', 'content-length': Buffer.byteLength(payload) } : {}),
      },
    }, (res) => {
      const chunks = [];
      res.on('data', (chunk) => chunks.push(chunk));
      res.on('end', () => {
        const text = Buffer.concat(chunks).toString('utf8');
        try {
          resolve({
            statusCode: res.statusCode,
            body: text ? JSON.parse(text) : null,
          });
        } catch (error) {
          reject(new Error(`Failed to parse JSON from ${requestPath}: ${error.message}. Body: ${text.slice(0, 300)}`));
        }
      });
    });
    req.on('error', reject);
    if (payload) req.write(payload);
    req.end();
  });
}

async function waitForTravelerApi(port, child) {
  const startedAt = Date.now();
  let lastError = null;
  while (Date.now() - startedAt < 45000) {
    if (child.exitCode !== null) {
      throw new Error(`Server exited before readiness check completed with code ${child.exitCode}`);
    }
    try {
      const response = await requestJson({ port, path: '/api/v2/cities' });
      if (response.statusCode === 200) return;
      lastError = new Error(`Readiness returned HTTP ${response.statusCode}`);
    } catch (error) {
      lastError = error;
    }
    await new Promise((resolve) => setTimeout(resolve, 750));
  }
  throw lastError || new Error('Timed out waiting for traveler API readiness');
}

async function stopServer(child) {
  if (!child || child.exitCode !== null) return;
  child.kill();
  const exited = await new Promise((resolve) => {
    const timeout = setTimeout(() => resolve(false), 5000);
    child.once('exit', () => {
      clearTimeout(timeout);
      resolve(true);
    });
  });
  if (exited) return;
  child.kill('SIGKILL');
  await new Promise((resolve) => child.once('exit', resolve));
}

test('Phase 2 Batch 2 recommendation fixture is a smoke foundation, not a quality claim', () => {
  assert.equal(fixture.fixtureVersion, 'phase2-recommendation-smoke-v1');
  assert.equal(
    fixture.datasetSha256,
    '5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae',
  );
  assert.equal(Array.isArray(fixture.queries), true);
  assert.equal(fixture.queries.length > 0, true);
  assert.ok(fixture.notes.some((note) => note.includes('must not be used for quality superiority claims')));
});

test('Phase 2 Batch 2 recommendation request validation is city-scoped and bounded', () => {
  assert.deepEqual(parseRecommendationLimit(undefined), { limit: DEFAULT_RECOMMENDATION_LIMIT });
  assert.deepEqual(parseRecommendationLimit(String(MAX_RECOMMENDATION_LIMIT)), { limit: MAX_RECOMMENDATION_LIMIT });
  assert.equal(parseRecommendationLimit('0').error.rule, `integer_between_1_and_${MAX_RECOMMENDATION_LIMIT}`);
  assert.equal(parseRecommendationLimit('1.5').error.rule, `integer_between_1_and_${MAX_RECOMMENDATION_LIMIT}`);
  assert.equal(validateRecommendationRequest({ query: 'cafe' }).error.field, 'cityId');
  assert.equal(validateRecommendationRequest({ cityId: DEFAULT_CITY_ID }).error.field, 'query');
  assert.equal(validateRecommendationRequest({
    cityId: DEFAULT_CITY_ID,
    query: 'cafe',
    context: { location: { lat: 999, lon: 108.22 } },
  }).error.field, 'context.location.lat');

  assert.deepEqual(validateRecommendationRequest({
    cityId: DEFAULT_CITY_ID,
    query: ' quan cafe yen tinh ',
    limit: '5',
    context: { location: { lat: 16.06, lng: 108.22 } },
  }), {
    cityId: DEFAULT_CITY_ID,
    query: 'quan cafe yen tinh',
    limit: 5,
    context: { location: { lat: 16.06, lng: 108.22 } },
  });
});

test('Phase 2 Batch 2 recommendation serialization removes raw scoring fields', () => {
  const serialized = serializeRecommendationItem({
    id: 'DN_TEST_2',
    globalId: 'DN_TEST_2',
    cityId: DEFAULT_CITY_ID,
    title: 'Beta Cafe',
    name: 'Beta Cafe',
    category: 'Cafe',
    lat: 16.06,
    lon: 108.22,
    hasCoordinates: true,
    coordinateStatus: 'valid',
    source: 'google_maps+foody',
    sourceIds: ['restaurant_id:123'],
    score: 88,
    scoreRaw: 0.88,
    reason: 'Test reason.',
    signals: {
      semantic: 1,
      category: 1,
      preference: 0.5,
      rating: 0.8,
      distance: 0.9,
      review: 0.2,
      distanceKm: 1.2,
    },
    warnings: ['freshness_unknown'],
  });

  assert.equal(serialized.poi.id, 'DN_TEST_2');
  assert.equal(serialized.score, 88);
  assert.equal(serialized.reason, 'Test reason.');
  assert.ok(serialized.reasonCodes.includes('intent_match'));
  assert.ok(serialized.reasonCodes.includes('category_match'));
  assert.ok(serialized.reasonCodes.includes('query_text_match'));
  assert.equal(serialized.provenance.source, 'google_maps+foody');

  const text = JSON.stringify(serialized);
  for (const forbidden of ['signals', 'scoreRaw', 'sourceIds', 'placeId']) {
    assert.equal(text.includes(forbidden), false, `${forbidden} must not be public`);
  }
});

test('Phase 2 Batch 2 recommendation ranking uses deterministic tie-breaks', () => {
  const ranked = rankRecommendationItems([
    { globalId: 'DN_3', name: 'Beta Cafe', score: 90 },
    { globalId: 'DN_2', name: 'Alpha Cafe', score: 90 },
    { globalId: 'DN_1', name: 'Alpha Cafe', score: 90 },
    { globalId: 'DN_4', name: 'Gamma Cafe', score: 80 },
  ]);

  assert.deepEqual(ranked.map((item) => item.globalId), ['DN_1', 'DN_2', 'DN_3', 'DN_4']);
});

test('Phase 2 Batch 2 recommendation service wraps current recommender with public v2 semantics', async () => {
  const query = fixture.queries[0];
  const legacy = await recommendPOIs({
    query: query.query,
    context: { cityId: query.cityId },
    limit: MAX_RECOMMENDATION_LIMIT,
  });
  const v2 = await getTravelerRecommendations({
    query: query.query,
    context: {},
    limit: query.limit,
    cityId: query.cityId,
  });

  assert.ok(legacy.results.length > 0);
  assert.equal(v2.cityId, DEFAULT_CITY_ID);
  assert.equal(v2.recommendations.length, query.limit);
  assert.ok(v2.recommendations.every((item) => item.poi.cityId === DEFAULT_CITY_ID));
  assert.ok(v2.recommendations.every((item) => typeof item.score === 'number'));
  assert.ok(v2.recommendations.every((item) => typeof item.reason === 'string' && item.reason.length > 0));
  assert.ok(v2.recommendations.every((item) => Array.isArray(item.reasonCodes) && item.reasonCodes.length > 0));
  assert.ok(v2.recommendations.every((item) => Array.isArray(item.warnings)));
  assert.ok(v2.recommendations.every((item) => item.provenance && typeof item.provenance.source !== 'undefined'));

  const repeated = await getTravelerRecommendations({
    query: query.query,
    context: {},
    limit: query.limit,
    cityId: query.cityId,
  });
  assert.deepEqual(v2.recommendations.map((item) => item.poi.id), repeated.recommendations.map((item) => item.poi.id));

  const text = JSON.stringify(v2);
  for (const forbidden of query.mustNotExposeFields) {
    assert.equal(text.includes(forbidden), false, `${forbidden} must not be public`);
  }
});

test('Phase 2 Batch 2 recommendation endpoint validates requests and returns deterministic public output', {
  timeout: 90000,
}, async () => {
  const port = 22000 + Math.floor(Math.random() * 1000);
  const child = spawn(process.execPath, ['src/server.js'], {
    cwd: path.join(__dirname, '..', '..'),
    env: {
      ...process.env,
      PORT: String(port),
      FEATURE_GUEST_ITINERARY_PREVIEW: 'false',
      URBANAGENT_POI_REPOSITORY: '',
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  const logs = [];
  child.stdout.on('data', (chunk) => logs.push(chunk.toString('utf8')));
  child.stderr.on('data', (chunk) => logs.push(chunk.toString('utf8')));

  try {
    await waitForTravelerApi(port, child);

    const missingCity = await requestJson({
      port,
      method: 'POST',
      path: '/api/v2/recommendations',
      body: { query: 'quan cafe yen tinh' },
    });
    assert.equal(missingCity.statusCode, 400);
    assert.equal(missingCity.body.error.code, 'VALIDATION_ERROR');
    assert.equal(missingCity.body.error.details[0].field, 'cityId');

    const unsupportedCity = await requestJson({
      port,
      method: 'POST',
      path: '/api/v2/recommendations',
      body: { cityId: 'hue', query: 'quan cafe yen tinh' },
    });
    assert.equal(unsupportedCity.statusCode, 422);
    assert.equal(unsupportedCity.body.error.code, 'CITY_NOT_SUPPORTED');

    const missingQuery = await requestJson({
      port,
      method: 'POST',
      path: '/api/v2/recommendations',
      body: { cityId: DEFAULT_CITY_ID, query: '   ' },
    });
    assert.equal(missingQuery.statusCode, 400);
    assert.equal(missingQuery.body.error.details[0].field, 'query');

    const invalidLimit = await requestJson({
      port,
      method: 'POST',
      path: '/api/v2/recommendations',
      body: { cityId: DEFAULT_CITY_ID, query: 'quan cafe yen tinh', limit: MAX_RECOMMENDATION_LIMIT + 1 },
    });
    assert.equal(invalidLimit.statusCode, 400);
    assert.equal(invalidLimit.body.error.details[0].field, 'limit');

    const body = {
      cityId: DEFAULT_CITY_ID,
      query: 'quan cafe yen tinh',
      context: {
        location: {
          lat: 16.06,
          lon: 108.22,
        },
        maxDistanceKm: 14,
      },
      limit: 5,
    };
    const recommendation = await requestJson({
      port,
      method: 'POST',
      path: '/api/v2/recommendations',
      headers: { 'x-request-id': 'phase2_batch2:recommendations-1' },
      body,
    });
    assert.equal(recommendation.statusCode, 200, logs.join('').slice(-1000));
    assert.equal(recommendation.body.ok, true);
    assert.equal(recommendation.body.meta.apiVersion, 'v2');
    assert.equal(recommendation.body.meta.cityId, DEFAULT_CITY_ID);
    assert.equal(recommendation.body.meta.requestId, 'phase2_batch2:recommendations-1');
    assert.equal(recommendation.body.data.cityId, DEFAULT_CITY_ID);
    assert.equal(recommendation.body.data.limit, 5);
    assert.ok(recommendation.body.data.recommendations.length > 0);
    assert.ok(recommendation.body.data.recommendations.length <= 5);

    for (const item of recommendation.body.data.recommendations) {
      assert.equal(item.poi.cityId, DEFAULT_CITY_ID);
      assert.equal(typeof item.score, 'number');
      assert.ok(item.score >= 0 && item.score <= 100);
      assert.equal(typeof item.reason, 'string');
      assert.ok(item.reason.length > 0);
      assert.equal(Array.isArray(item.reasonCodes), true);
      assert.ok(item.reasonCodes.length > 0);
      assert.equal(Array.isArray(item.warnings), true);
      assert.ok(item.provenance);
      assert.equal(item.provenance.source, item.poi.provenance.source);
      assert.equal(Object.prototype.hasOwnProperty.call(item.poi, 'placeId'), false);
      assert.equal(Object.prototype.hasOwnProperty.call(item.poi, 'sourceIds'), false);
    }

    const repeated = await requestJson({
      port,
      method: 'POST',
      path: '/api/v2/recommendations',
      headers: { 'x-request-id': 'phase2_batch2:recommendations-2' },
      body,
    });
    assert.equal(repeated.statusCode, 200);
    assert.deepEqual(
      repeated.body.data.recommendations.map((item) => item.poi.id),
      recommendation.body.data.recommendations.map((item) => item.poi.id),
    );

    const text = JSON.stringify(recommendation.body);
    for (const forbidden of [
      'signals',
      'scoreRaw',
      'sourceIds',
      'placeId',
      'DATABASE_URL',
      'PostgresPoiRepository',
      'CanonicalCsvPoiRepository',
      'poi_entities',
      'data/canonical',
      'urbanagent_poi_master_v1.csv',
      'stack',
      'Bearer ',
      'token',
    ]) {
      assert.equal(text.includes(forbidden), false, `${forbidden} leaked in response body`);
    }
  } finally {
    await stopServer(child);
  }
});
