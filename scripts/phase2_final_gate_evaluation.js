const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');
const { performance } = require('node:perf_hooks');
const { spawn } = require('node:child_process');

const { CanonicalCsvPoiRepository } = require('../src/services/canonicalCsvPoiRepository');
const { getPoiRepository, setPoiRepositoryForTests } = require('../src/services/poiRepository');

const ROOT_DIR = path.resolve(__dirname, '..');
const CANONICAL_CSV = path.join(ROOT_DIR, 'data', 'canonical', 'urbanagent_poi_master_v1.csv');
const EXPECTED_SHA = '5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae';
const EXPECTED_POIS = 4166;
const PERF_RUNS = Number.parseInt(process.env.URBANAGENT_PHASE2_PERF_RUNS || '10', 10);

function sha256(filePath) {
  return crypto.createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');
}

function percentile(values, ratio) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.ceil(sorted.length * ratio) - 1);
  return Number(sorted[index].toFixed(2));
}

function requestJson({ port, method = 'GET', path: requestPath, headers = {}, body }) {
  return new Promise((resolve, reject) => {
    const payload = body === undefined ? null : JSON.stringify(body);
    const startedAt = performance.now();
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
        let parsed = null;
        try {
          parsed = text ? JSON.parse(text) : null;
        } catch (_) {
          parsed = null;
        }
        resolve({
          statusCode: res.statusCode,
          body: parsed,
          text,
          latencyMs: performance.now() - startedAt,
        });
      });
    });
    req.on('error', reject);
    if (payload) req.write(payload);
    req.end();
  });
}

async function waitForTravelerApi(port, child) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < 45000) {
    if (child.exitCode !== null) {
      throw new Error(`Server exited before readiness check completed with code ${child.exitCode}`);
    }
    try {
      const response = await requestJson({ port, path: '/api/v2/cities' });
      if (response.statusCode === 200) return;
    } catch (_) {
      // Keep polling until the child server is ready.
    }
    await new Promise((resolve) => setTimeout(resolve, 750));
  }
  throw new Error('Timed out waiting for traveler API readiness');
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

function unsignedDevToken(uid) {
  const encode = (value) => Buffer.from(JSON.stringify(value)).toString('base64url');
  return `${encode({ alg: 'none', typ: 'JWT' })}.${encode({
    sub: uid,
    user_id: uid,
    email: `${uid}@urbanagent.local`,
    role: 'customer',
  })}.dev-signature`;
}

function authHeader(uid) {
  return { authorization: `Bearer ${unsignedDevToken(uid)}` };
}

async function measureEndpoint(port, label, request, runs = PERF_RUNS) {
  const latencies = [];
  for (let i = 0; i < runs; i += 1) {
    const response = await requestJson({ port, ...request });
    assert.ok(response.statusCode >= 200 && response.statusCode < 300, `${label} returned ${response.statusCode}`);
    latencies.push(response.latencyMs);
  }
  return {
    label,
    runs,
    p50Ms: percentile(latencies, 0.5),
    p95Ms: percentile(latencies, 0.95),
  };
}

function savedTripPayloadFromPreview(previewTrip, request) {
  return {
    title: 'Phase 2 final gate saved trip',
    cityId: previewTrip.cityId,
    query: previewTrip.query,
    startDate: previewTrip.date,
    dayCount: previewTrip.dayCount,
    dailyWindow: previewTrip.dailyWindow,
    dayWindows: previewTrip.dayWindows || [],
    pace: previewTrip.pace,
    transport: previewTrip.transport,
    includedPoiIds: [],
    excludedPoiIds: [],
    request,
    preview: previewTrip,
    itinerary: previewTrip.stops,
    warnings: previewTrip.warnings,
  };
}

async function run() {
  const recommendationFixture = require('../tests/fixtures/phase2/recommendationQueries.json');
  const tripFixture = require('../tests/fixtures/phase2/tripPreviewQueries.json');
  const previewInput = tripFixture.cases.find((item) => item.id === 'multi-day-balanced-trip').input;
  const previewForSaveInput = tripFixture.cases.find((item) => item.id === 'one-day-balanced-trip').input;

  const canonicalSha = sha256(CANONICAL_CSV);
  assert.equal(canonicalSha, EXPECTED_SHA);

  const csvRepository = new CanonicalCsvPoiRepository();
  const pois = await csvRepository.findByCity('da-nang');
  const quality = await csvRepository.getQualityReport();
  assert.equal(pois.length, EXPECTED_POIS);
  assert.equal(quality.totals.applicationPois, EXPECTED_POIS);
  assert.equal(quality.totals.invalidRows, 0);
  assert.equal(quality.headerMatchesExpected, true);

  setPoiRepositoryForTests(null);
  const defaultRepositoryName = getPoiRepository().constructor.name;
  assert.equal(defaultRepositoryName, 'CanonicalCsvPoiRepository');

  const port = 27000 + Math.floor(Math.random() * 1000);
  const child = spawn(process.execPath, ['src/server.js'], {
    cwd: ROOT_DIR,
    env: {
      ...process.env,
      NODE_ENV: 'test',
      PORT: String(port),
      URBANAGENT_POI_REPOSITORY: '',
      URBANAGENT_SAVED_TRIPS_STORE: 'memory',
      DISABLE_DEV_AUTH_FALLBACK: 'false',
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  const logs = [];
  child.stdout.on('data', (chunk) => logs.push(chunk.toString('utf8')));
  child.stderr.on('data', (chunk) => logs.push(chunk.toString('utf8')));

  try {
    await waitForTravelerApi(port, child);

    const legacyGoogle = await requestJson({ port, path: '/api/eda?source=google_maps' });
    const legacyFoody = await requestJson({ port, path: '/api/eda?source=foody' });
    const legacyAll = await requestJson({ port, path: '/api/eda?source=all' });
    assert.equal(legacyGoogle.body.metrics.totalPOIs, 3946);
    assert.equal(legacyFoody.body.metrics.totalPOIs, 225);
    assert.equal(legacyAll.body.metrics.totalPOIs, EXPECTED_POIS);

    const v2Google = await requestJson({ port, path: '/api/v2/pois/search?cityId=da-nang&source=google_maps&limit=1' });
    const v2Foody = await requestJson({ port, path: '/api/v2/pois/search?cityId=da-nang&source=foody&limit=1' });
    const v2All = await requestJson({ port, path: '/api/v2/pois/search?cityId=da-nang&source=all&limit=1' });
    assert.equal(v2Google.body.data.page.total, 3946);
    assert.equal(v2Foody.body.data.page.total, 225);
    assert.equal(v2All.body.data.page.total, EXPECTED_POIS);

    const recommendation = await requestJson({
      port,
      method: 'POST',
      path: '/api/v2/recommendations',
      body: recommendationFixture.queries[0],
    });
    assert.equal(recommendation.statusCode, 200, logs.join('').slice(-1000));
    assert.ok(recommendation.body.data.recommendations.length > 0);
    assert.ok(recommendation.body.data.recommendations.every((item) => item.poi.cityId === 'da-nang'));

    const preview = await requestJson({
      port,
      method: 'POST',
      path: '/api/v2/trips/preview',
      body: previewInput,
    });
    assert.equal(preview.statusCode, 200, preview.text);
    assert.ok(preview.body.data.trip.stops.length > 0);
    assert.equal(preview.body.data.trip.persisted, false);

    const missingOrigin = await requestJson({
      port,
      method: 'POST',
      path: '/api/v2/trips/preview',
      body: tripFixture.cases.find((item) => item.id === 'missing-origin').input,
    });
    const firstLeg = missingOrigin.body.data.trip.stops[0].travelFromPrevious;
    assert.equal(firstLeg.distanceMeters, null);
    assert.equal(firstLeg.travelDurationMinutes, null);
    assert.equal(firstLeg.distanceKnown, false);
    assert.equal(firstLeg.calculationSource, 'missing-origin');

    const previewForSave = await requestJson({
      port,
      method: 'POST',
      path: '/api/v2/trips/preview',
      body: previewForSaveInput,
    });
    const created = await requestJson({
      port,
      method: 'POST',
      path: '/api/v2/trips',
      headers: authHeader('phase2-final-user-a'),
      body: savedTripPayloadFromPreview(previewForSave.body.data.trip, previewForSaveInput),
    });
    assert.equal(created.statusCode, 201, created.text);
    const tripId = created.body.data.trip.tripId;
    const userBGet = await requestJson({
      port,
      path: `/api/v2/trips/${tripId}`,
      headers: authHeader('phase2-final-user-b'),
    });
    assert.equal(userBGet.statusCode, 404);
    const replanned = await requestJson({
      port,
      method: 'POST',
      path: `/api/v2/trips/${tripId}/replan`,
      headers: authHeader('phase2-final-user-a'),
    });
    assert.equal(replanned.statusCode, 200, replanned.text);
    assert.equal(replanned.body.data.trip.tripId, tripId);

    const feedback = await requestJson({
      port,
      method: 'POST',
      path: `/api/v2/trips/${tripId}/feedback`,
      headers: authHeader('phase2-final-user-a'),
      body: { rating: 5 },
    });
    assert.equal(feedback.statusCode, 404);

    const performanceResults = [
      await measureEndpoint(port, 'city status', { path: '/api/v2/cities/da-nang/status' }),
      await measureEndpoint(port, 'POI search first page', { path: '/api/v2/pois/search?cityId=da-nang&limit=10' }),
      await measureEndpoint(port, 'POI search q', { path: '/api/v2/pois/search?cityId=da-nang&q=cafe&limit=10' }),
      await measureEndpoint(port, 'recommendation', {
        method: 'POST',
        path: '/api/v2/recommendations',
        body: recommendationFixture.queries[0],
      }),
      await measureEndpoint(port, 'itinerary preview', {
        method: 'POST',
        path: '/api/v2/trips/preview',
        body: previewInput,
      }),
    ];

    const report = {
      verdict: 'PASS',
      timestamp: new Date().toISOString(),
      node: process.version,
      repositoryMode: {
        defaultRepositoryName,
        postgresOptInEnv: 'URBANAGENT_POI_REPOSITORY=postgres',
      },
      canonical: {
        path: 'data/canonical/urbanagent_poi_master_v1.csv',
        sha256: canonicalSha,
        applicationPois: pois.length,
        invalidRows: quality.totals.invalidRows,
        headerMatchesExpected: quality.headerMatchesExpected,
      },
      compatibility: {
        legacyEdaCounts: {
          google: legacyGoogle.body.metrics.totalPOIs,
          foody: legacyFoody.body.metrics.totalPOIs,
          all: legacyAll.body.metrics.totalPOIs,
        },
        v2PoiSearchCounts: {
          google: v2Google.body.data.page.total,
          foody: v2Foody.body.data.page.total,
          all: v2All.body.data.page.total,
        },
        recommendationCount: recommendation.body.data.recommendations.length,
        tripPreviewStops: preview.body.data.trip.stops.length,
        savedTripReplanSameTripId: replanned.body.data.trip.tripId === tripId,
        feedbackEndpointStatus: 'DEFERRED_NOT_IMPLEMENTED_404',
      },
      missingOrigin: {
        distanceMeters: firstLeg.distanceMeters,
        travelDurationMinutes: firstLeg.travelDurationMinutes,
        distanceKnown: firstLeg.distanceKnown,
        calculationSource: firstLeg.calculationSource,
      },
      performance: {
        runsPerEndpoint: PERF_RUNS,
        metrics: performanceResults,
        threshold: null,
        thresholdStatus: 'NO_DOCUMENTED_NUMERIC_THRESHOLD',
      },
      scientificOfflineEvaluation: {
        recommendationFixtureVersion: recommendationFixture.fixtureVersion,
        tripPreviewFixtureVersion: tripFixture.fixtureVersion,
        recommendationCases: recommendationFixture.queries.length,
        tripPreviewCases: tripFixture.cases.length,
        status: 'STRUCTURAL_FIXTURE_ONLY_NO_QUALITY_CLAIM',
        limitation: 'No curated scored relevance benchmark exists, so Recall@k, nDCG, MRR, statistical significance, and user-quality claims are not valid.',
      },
      productionSafety: {
        savedTripsStore: 'memory',
        firebaseProductionTouched: false,
        productionDatabaseTouched: false,
        externalPoiSourcesTouched: false,
      },
    };

    console.log(JSON.stringify(report, null, 2));
  } finally {
    await stopServer(child);
    setPoiRepositoryForTests(null);
  }
}

run().catch((error) => {
  console.error(error.stack || error.message);
  process.exitCode = 1;
});
