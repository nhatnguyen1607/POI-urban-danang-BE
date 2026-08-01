const assert = require('node:assert/strict');
const { spawn } = require('node:child_process');
const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');
const test = require('node:test');

const { CONTRACT_VERSION, DEFAULT_LIMIT, MAX_LIMIT } = require('../../src/modules/travelerApiV2/constants');
const { CanonicalCsvPoiRepository } = require('../../src/services/canonicalCsvPoiRepository');
const { decodeCursor, encodeCursor, parsePagination } = require('../../src/modules/travelerApiV2/pagination');
const { rankPois } = require('../../src/modules/travelerApiV2/poiSearch');
const { resolveRequestId } = require('../../src/modules/travelerApiV2/requestContext');
const { serializePoi } = require('../../src/modules/travelerApiV2/serializers');

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

test('Phase 2 request IDs are echoed only when valid', () => {
  const valid = 'trace_ABC-123.456:request';
  assert.equal(resolveRequestId(valid), valid);

  const invalid = 'bad/request';
  const generated = resolveRequestId(invalid);
  assert.notEqual(generated, invalid);
  assert.match(generated, /^req_[0-9a-f-]{36}$/);

  const tooLong = 'x'.repeat(129);
  assert.match(resolveRequestId(tooLong), /^req_[0-9a-f-]{36}$/);
});

test('Phase 2 pagination uses safe defaults, maximums, and opaque cursors', () => {
  assert.deepEqual(parsePagination({}), { limit: DEFAULT_LIMIT, offset: 0 });
  assert.deepEqual(parsePagination({ limit: String(MAX_LIMIT) }), { limit: MAX_LIMIT, offset: 0 });
  assert.equal(parsePagination({ limit: '0' }).error.rule, `integer_between_1_and_${MAX_LIMIT}`);
  assert.equal(parsePagination({ limit: '101' }).error.rule, `integer_between_1_and_${MAX_LIMIT}`);
  assert.equal(parsePagination({ limit: '1.5' }).error.rule, `integer_between_1_and_${MAX_LIMIT}`);
  assert.equal(parsePagination({ cursor: 'not-json' }).error.rule, 'opaque_cursor');

  const cursor = encodeCursor(40);
  assert.equal(typeof cursor, 'string');
  assert.equal(decodeCursor(cursor), 40);
  assert.deepEqual(parsePagination({ limit: '10', cursor }), { limit: 10, offset: 40 });
});

test('Phase 2 POI ordering is deterministic and uses Global_ID as tie-break', () => {
  const pois = [
    { globalId: 'poi_2', name: 'Beta Cafe', category: 'Cafe', categoryNormalized: 'cafe' },
    { globalId: 'poi_1', name: 'Alpha Cafe', category: 'Cafe', categoryNormalized: 'cafe' },
    { globalId: 'poi_3', name: 'Alpha Cafe', category: 'Restaurant', categoryNormalized: 'restaurant' },
  ];

  assert.deepEqual(rankPois(pois).map((poi) => poi.globalId), ['poi_1', 'poi_3', 'poi_2']);
  assert.deepEqual(rankPois(pois, { q: 'beta' }).map((poi) => poi.globalId), ['poi_2']);
  assert.deepEqual(rankPois(pois, { category: 'restaurant' }).map((poi) => poi.globalId), ['poi_3']);
});

test('Phase 2 POI serializer preserves canonical public semantics', async () => {
  const repository = new CanonicalCsvPoiRepository();
  const pois = await repository.loadAll();

  assert.equal(pois.length, 4166);
  assert.equal(pois.some((poi) => poi.entityType && poi.entityType !== 'poi'), false);

  const merged = pois.find((poi) => poi.source === 'google_maps+foody');
  const nullRating = pois.find((poi) => poi.rating === null);
  const nullReviewCount = pois.find((poi) => poi.reviewCount === null);
  const unknownDistrict = { ...pois[0], district: '' };
  const foodyRated = pois.find((poi) => (
    poi.foodyRating10 !== null &&
    poi.foodyReviewSampleCount !== null &&
    poi.reviewCount === null
  ));
  const googleRated = pois.find((poi) => poi.googleRating !== null);
  const multipleImages = pois.find((poi) => poi.imageUrls.length > 1);
  const aliasPoi = pois.find((poi) => poi.aliasGlobalIds.length > 0);

  assert.ok(merged, 'canonical data should include merged Google/Foody provenance');
  assert.ok(nullRating, 'canonical data should include unknown rating values');
  assert.ok(nullReviewCount, 'canonical data should include unknown review count values');
  assert.ok(foodyRated, 'canonical data should include Foody source ratings');
  assert.ok(googleRated, 'canonical data should include Google source ratings');
  assert.ok(multipleImages, 'canonical data should include multiple image URLs');
  assert.ok(aliasPoi, 'canonical data should include alias Global_IDs');

  const serializedMerged = serializePoi(merged);
  assert.equal(serializedMerged.id, merged.globalId);
  assert.equal(serializedMerged.globalId, merged.globalId);
  assert.equal(serializedMerged.provenance.source, 'google_maps+foody');
  assert.ok(serializedMerged.provenance.sourceIdentifiers.every((identifier) => (
    typeof identifier.namespace === 'string' &&
    typeof identifier.value === 'string' &&
    Object.prototype.hasOwnProperty.call(identifier, 'source')
  )));
  assert.equal(Object.prototype.hasOwnProperty.call(serializedMerged, 'placeId'), false);
  assert.equal(Object.prototype.hasOwnProperty.call(serializedMerged, 'sourceIds'), false);

  const serializedNullRating = serializePoi(nullRating);
  assert.equal(serializedNullRating.rating.normalized.value, null);
  assert.equal(serializedNullRating.rating.normalized.status, 'unknown');
  assert.equal(serializedNullRating.rating.normalized.scale, 5);

  const serializedNullReviewCount = serializePoi(nullReviewCount);
  assert.equal(serializedNullReviewCount.rating.reviewCount.value, null);
  assert.equal(serializedNullReviewCount.rating.reviewCount.status, 'unknown');

  const serializedUnknownDistrict = serializePoi(unknownDistrict);
  assert.equal(serializedUnknownDistrict.address.district, null);

  const serializedFoodyRated = serializePoi(foodyRated);
  assert.equal(serializedFoodyRated.rating.foody.scale, 10);
  assert.notEqual(serializedFoodyRated.rating.foody.value, null);
  assert.notEqual(serializedFoodyRated.rating.foody.sampleReviewCount, serializedFoodyRated.rating.reviewCount.value);

  const serializedGoogleRated = serializePoi(googleRated);
  assert.equal(serializedGoogleRated.rating.google.scale, 5);
  assert.notEqual(serializedGoogleRated.rating.google.value, null);

  const serializedImages = serializePoi(multipleImages);
  assert.ok(Array.isArray(serializedImages.images.imageUrls));
  assert.equal(serializedImages.images.imageUrl, serializedImages.images.imageUrls[0]);

  const serializedAlias = serializePoi(aliasPoi);
  assert.deepEqual(serializedAlias.provenance.aliasGlobalIds, aliasPoi.aliasGlobalIds);
  assert.ok(serializedAlias.provenance.aliasGlobalIds.every((alias) => alias !== serializedAlias.globalId));
});

test('Phase 2 OpenAPI draft is limited to approved implemented core endpoints', () => {
  const openApiPath = path.join(__dirname, '..', '..', 'docs', 'rebuild', 'PHASE2_TRAVELER_API_V2_OPENAPI_DRAFT.json');
  const artifactText = fs.readFileSync(openApiPath, 'utf8');
  const artifact = JSON.parse(artifactText);
  const paths = Object.keys(artifact.paths).sort();

  assert.equal(artifact.openapi, '3.1.0');
  assert.equal(artifact.info.version, CONTRACT_VERSION);
  assert.deepEqual(paths, [
    '/api/v2/cities',
    '/api/v2/cities/{cityId}/status',
    '/api/v2/pois/search',
    '/api/v2/pois/{poiId}',
    '/api/v2/recommendations',
  ]);

  for (const forbidden of [
    'contractHash',
    'phase2-contract-draft-v1',
    'DATABASE_URL',
    'data/canonical',
    'urbanagent_poi_master_v1.csv',
    'PostgresPoiRepository',
    'poi_entities',
    'RestaurantID',
    '/api/v2/trips',
  ]) {
    assert.equal(artifactText.includes(forbidden), false, `${forbidden} must not appear in the public artifact`);
  }
});

test('Phase 2 Batch 1 traveler API endpoints expose CSV-backed city and POI contracts', {
  timeout: 90000,
}, async () => {
  const port = 21000 + Math.floor(Math.random() * 1000);
  const child = spawn(process.execPath, ['src/server.js'], {
    cwd: process.cwd(),
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

    const cities = await requestJson({
      port,
      path: '/api/v2/cities',
      headers: { 'x-request-id': 'phase2_test:valid-1' },
    });
    assert.equal(cities.statusCode, 200, logs.join('').slice(-1000));
    assert.equal(cities.body.ok, true);
    assert.equal(cities.body.meta.apiVersion, 'v2');
    assert.equal(cities.body.meta.requestId, 'phase2_test:valid-1');
    assert.equal(Object.prototype.hasOwnProperty.call(cities.body.meta, 'cityId'), false);
    assert.equal(cities.body.data.cities.length, 1);
    assert.equal(cities.body.data.cities[0].cityId, DEFAULT_CITY_ID);
    assert.equal(cities.body.data.cities[0].capabilityStatus.poiSearch, 'experimental');
    assert.equal(cities.body.data.cities[0].capabilityStatus.recommendations, 'experimental');
    assert.equal(cities.body.data.cities[0].capabilityStatus.liveBooking, 'unavailable');

    const generatedRequestId = await requestJson({ port, path: '/api/v2/cities' });
    assert.equal(generatedRequestId.statusCode, 200);
    assert.match(generatedRequestId.body.meta.requestId, /^req_[0-9a-f-]{36}$/);

    const invalidRequestId = await requestJson({
      port,
      path: '/api/v2/cities',
      headers: { 'x-request-id': 'invalid/request/id' },
    });
    assert.equal(invalidRequestId.statusCode, 200);
    assert.notEqual(invalidRequestId.body.meta.requestId, 'invalid/request/id');
    assert.match(invalidRequestId.body.meta.requestId, /^req_[0-9a-f-]{36}$/);

    const status = await requestJson({ port, path: `/api/v2/cities/${DEFAULT_CITY_ID}/status` });
    assert.equal(status.statusCode, 200);
    assert.equal(status.body.meta.cityId, DEFAULT_CITY_ID);
    assert.equal(status.body.data.dataset.applicationPoiCount, 4166);
    assert.equal(status.body.data.dataset.contractVersion, CONTRACT_VERSION);
    assert.equal(status.body.data.qualitySummary.adminBoundary.status, 'pending_spatial_join');

    const unsupportedCity = await requestJson({ port, path: '/api/v2/cities/hue/status' });
    assert.equal(unsupportedCity.statusCode, 422);
    assert.equal(unsupportedCity.body.error.code, 'CITY_NOT_SUPPORTED');
    assert.equal(unsupportedCity.body.meta.apiVersion, 'v2');
    assert.match(unsupportedCity.body.meta.requestId, /^req_[0-9a-f-]{36}$/);

    const missingCity = await requestJson({ port, path: '/api/v2/pois/search?source=all' });
    assert.equal(missingCity.statusCode, 400);
    assert.equal(missingCity.body.error.code, 'VALIDATION_ERROR');
    assert.equal(missingCity.body.meta.apiVersion, 'v2');
    assert.match(missingCity.body.meta.requestId, /^req_[0-9a-f-]{36}$/);

    const invalidLimit = await requestJson({ port, path: `/api/v2/pois/search?cityId=${DEFAULT_CITY_ID}&limit=101` });
    assert.equal(invalidLimit.statusCode, 400);
    assert.equal(invalidLimit.body.error.details[0].field, 'limit');

    const invalidIntegerLimit = await requestJson({ port, path: `/api/v2/pois/search?cityId=${DEFAULT_CITY_ID}&limit=1.5` });
    assert.equal(invalidIntegerLimit.statusCode, 400);
    assert.equal(invalidIntegerLimit.body.error.details[0].field, 'limit');

    const invalidCursor = await requestJson({ port, path: `/api/v2/pois/search?cityId=${DEFAULT_CITY_ID}&cursor=not-json` });
    assert.equal(invalidCursor.statusCode, 400);
    assert.equal(invalidCursor.body.error.details[0].field, 'cursor');

    const google = await requestJson({
      port,
      path: `/api/v2/pois/search?cityId=${DEFAULT_CITY_ID}&source=google_maps&limit=1`,
    });
    assert.equal(google.statusCode, 200);
    assert.equal(google.body.meta.cityId, DEFAULT_CITY_ID);
    assert.equal(google.body.data.page.total, 3946);
    assert.equal(google.body.data.page.limit, 1);
    assert.equal(typeof google.body.data.page.nextCursor, 'string');
    assert.equal(google.body.data.pois.length, 1);
    assert.equal(Object.prototype.hasOwnProperty.call(google.body.data.pois[0], 'placeId'), false);
    assert.equal(Object.prototype.hasOwnProperty.call(google.body.data.pois[0], 'sourceIds'), false);
    assert.ok(Array.isArray(google.body.data.pois[0].provenance.sourceIdentifiers));

    const secondGooglePage = await requestJson({
      port,
      path: `/api/v2/pois/search?cityId=${DEFAULT_CITY_ID}&source=google_maps&limit=1&cursor=${google.body.data.page.nextCursor}`,
    });
    assert.equal(secondGooglePage.statusCode, 200);
    assert.notEqual(secondGooglePage.body.data.pois[0].id, google.body.data.pois[0].id);

    const firstTwo = await requestJson({
      port,
      path: `/api/v2/pois/search?cityId=${DEFAULT_CITY_ID}&source=google_maps&limit=2`,
    });
    assert.equal(firstTwo.statusCode, 200);
    assert.deepEqual(
      [google.body.data.pois[0].id, secondGooglePage.body.data.pois[0].id],
      firstTwo.body.data.pois.map((poi) => poi.id),
    );

    const repeatedFirstTwo = await requestJson({
      port,
      path: `/api/v2/pois/search?cityId=${DEFAULT_CITY_ID}&source=google_maps&limit=2`,
    });
    assert.deepEqual(repeatedFirstTwo.body.data, firstTwo.body.data);
    assert.equal(new Set(firstTwo.body.data.pois.map((poi) => poi.globalId)).size, firstTwo.body.data.pois.length);

    const foody = await requestJson({
      port,
      path: `/api/v2/pois/search?cityId=${DEFAULT_CITY_ID}&source=foody&limit=1`,
    });
    assert.equal(foody.statusCode, 200);
    assert.equal(foody.body.data.page.total, 225);

    const all = await requestJson({
      port,
      path: `/api/v2/pois/search?cityId=${DEFAULT_CITY_ID}&source=all&limit=1`,
    });
    assert.equal(all.statusCode, 200);
    assert.equal(all.body.data.page.total, 4166);

    const canonical = await requestJson({
      port,
      path: `/api/v2/pois/search?cityId=${DEFAULT_CITY_ID}&source=canonical&limit=1`,
    });
    assert.equal(canonical.statusCode, 200);
    assert.equal(canonical.body.data.page.total, 4166);

    const detailId = google.body.data.pois[0].id;
    const detail = await requestJson({ port, path: `/api/v2/pois/${detailId}?cityId=${DEFAULT_CITY_ID}` });
    assert.equal(detail.statusCode, 200);
    assert.equal(detail.body.meta.cityId, DEFAULT_CITY_ID);
    assert.equal(detail.body.data.poi.id, detailId);
    assert.equal(Object.prototype.hasOwnProperty.call(detail.body.data.poi, 'placeId'), false);
    assert.equal(Object.prototype.hasOwnProperty.call(detail.body.data.poi, 'sourceIds'), false);

    const missingDetail = await requestJson({ port, path: `/api/v2/pois/not-a-real-poi?cityId=${DEFAULT_CITY_ID}` });
    assert.equal(missingDetail.statusCode, 404);
    assert.equal(missingDetail.body.error.code, 'NOT_FOUND');

    for (const response of [
      cities,
      invalidRequestId,
      status,
      unsupportedCity,
      missingCity,
      invalidLimit,
      google,
      foody,
      all,
      detail,
      missingDetail,
    ]) {
      const text = JSON.stringify(response.body);
      for (const forbidden of [
        'DATABASE_URL',
        'data/canonical',
        'urbanagent_poi_master_v1.csv',
        'PostgresPoiRepository',
        'CanonicalCsvPoiRepository',
        'poi_entities',
        'SELECT ',
        'stack',
        'Bearer ',
        'token',
      ]) {
        assert.equal(text.includes(forbidden), false, `${forbidden} leaked in response body`);
      }
    }
  } finally {
    await stopServer(child);
  }
});
