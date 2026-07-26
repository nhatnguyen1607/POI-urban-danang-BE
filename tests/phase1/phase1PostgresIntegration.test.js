const assert = require('node:assert/strict');
const { spawn } = require('node:child_process');
const http = require('node:http');
const test = require('node:test');

const { createPostgresPool, withTransaction } = require('../../src/infrastructure/db/postgresClient');
const {
  listExistingPhase1Tables,
  migratePhase1,
  rollbackPhase1,
} = require('../../src/infrastructure/db/phase1MigrationRunner');
const {
  assertExpectedPhase1Diagnostics,
  getPhase1DbDiagnostics,
} = require('../../src/modules/pois/postgresDiagnostics');
const { buildCanonicalPoiImportPlan } = require('../../src/modules/pois/canonicalPoiImportPlan');
const { writeImportPlan } = require('../../scripts/phase1_import_canonical_pois');
const { PostgresPoiRepository } = require('../../src/modules/pois/postgresPoiRepository');
const { CanonicalCsvPoiRepository } = require('../../src/services/canonicalCsvPoiRepository');
const {
  filterPoisForEdaSource,
  loadPOIs,
} = require('../../src/services/poiDataService');
const { recommendPOIs } = require('../../src/services/poiRetrievalService');
const { createItinerary } = require('../../src/services/itineraryPlannerService');
const { setPoiRepositoryForTests } = require('../../src/services/poiRepository');

const DEFAULT_CITY_ID = 'da-nang';
const EXPECTED_COUNTS = {
  poi_entities: 4166,
  source_records: 4166,
  external_ids: 8337,
  aliases: 985,
  images: 16246,
  review_summaries: 4166,
};

function requestJson({ port, method = 'GET', path, body, token }) {
  return new Promise((resolve, reject) => {
    const payload = body ? JSON.stringify(body) : null;
    const req = http.request({
      hostname: '127.0.0.1',
      port,
      method,
      path,
      headers: {
        ...(payload ? { 'content-type': 'application/json', 'content-length': Buffer.byteLength(payload) } : {}),
        ...(token ? { authorization: `Bearer ${token}` } : {}),
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
          reject(new Error(`Failed to parse JSON from ${path}: ${error.message}. Body: ${text.slice(0, 300)}`));
        }
      });
    });
    req.on('error', reject);
    if (payload) req.write(payload);
    req.end();
  });
}

async function waitForServer(port, child) {
  const startedAt = Date.now();
  let lastError = null;
  while (Date.now() - startedAt < 45000) {
    if (child.exitCode !== null) {
      throw new Error(`Server exited before readiness check completed with code ${child.exitCode}`);
    }
    try {
      const response = await requestJson({ port, path: '/api/eda?source=all' });
      if (response.statusCode === 200) return;
      lastError = new Error(`Readiness returned HTTP ${response.statusCode}`);
    } catch (error) {
      lastError = error;
    }
    await new Promise((resolve) => setTimeout(resolve, 750));
  }
  throw lastError || new Error('Timed out waiting for server readiness');
}

async function stopServer(child) {
  if (!child || child.exitCode !== null) return;
  child.kill();
  const exited = await new Promise((resolve) => {
    const timeout = setTimeout(resolve, 5000);
    child.once('exit', () => {
      clearTimeout(timeout);
      resolve(true);
    });
  });
  if (exited) return;
  child.kill('SIGKILL');
  await new Promise((resolve) => child.once('exit', resolve));
}

async function runPostgresEndpointSmoke({ databaseUrl }) {
  assert.ok(databaseUrl, 'Endpoint smoke requires DATABASE_URL or POSTGRES_URL');
  const port = 19000 + Math.floor(Math.random() * 1000);
  const child = spawn(process.execPath, ['src/server.js'], {
    cwd: process.cwd(),
    env: {
      ...process.env,
      DATABASE_URL: databaseUrl,
      PORT: String(port),
      URBANAGENT_POI_REPOSITORY: 'postgres',
      DISABLE_DEV_AUTH_FALLBACK: 'false',
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  const logs = [];
  child.stdout.on('data', (chunk) => logs.push(chunk.toString('utf8')));
  child.stderr.on('data', (chunk) => logs.push(chunk.toString('utf8')));

  try {
    await waitForServer(port, child);

    const google = await requestJson({ port, path: '/api/eda?source=google_maps' });
    assert.equal(google.statusCode, 200, logs.join('').slice(-1000));
    assert.equal(google.body.metrics.totalPOIs, 3946);
    assert.equal(google.body.quality.totals.applicationPois, EXPECTED_COUNTS.poi_entities);

    const foody = await requestJson({ port, path: '/api/eda?source=foody' });
    assert.equal(foody.statusCode, 200);
    assert.equal(foody.body.metrics.totalPOIs, 225);

    const quality = await requestJson({ port, path: '/api/pois/data-quality' });
    assert.equal(quality.statusCode, 200);
    assert.equal(quality.body.totals.applicationPois, EXPECTED_COUNTS.poi_entities);
    assert.equal(quality.body.headerMatchesExpected, true);

    const recommendation = await requestJson({
      port,
      method: 'POST',
      path: '/api/agent/recommend-poi',
      body: { query: 'quan cafe yen tinh', context: { cityId: DEFAULT_CITY_ID }, limit: 3 },
    });
    assert.equal(recommendation.statusCode, 200);
    assert.ok(recommendation.body.results.length > 0);
    assert.equal(recommendation.body.results[0].cityId, DEFAULT_CITY_ID);

    const itinerary = await requestJson({
      port,
      method: 'POST',
      path: '/api/agent/create-itinerary',
      token: 'local-admin-dev-token',
      body: {
        query: 'quan cafe yen tinh',
        context: { cityId: DEFAULT_CITY_ID },
        limit: 3,
        durationMinutes: 180,
      },
    });
    assert.equal(itinerary.statusCode, 200, JSON.stringify(itinerary.body));
    assert.equal(itinerary.body.cityId, DEFAULT_CITY_ID);
    assert.ok(itinerary.body.itinerary.length > 0);
    assert.equal(itinerary.body.itinerary[0].travelFromPrevious.distanceKm, null);
    assert.equal(itinerary.body.itinerary[0].travelFromPrevious.distanceKnown, false);
  } finally {
    await stopServer(child);
  }
}

function coreCounts(diagnostics) {
  return Object.fromEntries(Object.keys(EXPECTED_COUNTS).map((key) => [key, diagnostics.counts[key]]));
}

function byId(pois) {
  return new Map(pois.map((poi) => [poi.globalId, poi]));
}

function pickCsvExamples(csvPois) {
  return {
    googleOnly: csvPois.find((poi) => poi.source === 'google_maps'),
    foodyOnly: csvPois.find((poi) => poi.source === 'foody'),
    merged: csvPois.find((poi) => poi.source === 'google_maps+foody'),
    nullRating: csvPois.find((poi) => poi.rating === null),
    multipleImages: csvPois.find((poi) => poi.imageUrls.length > 1),
    aliases: csvPois.find((poi) => poi.aliasGlobalIds.length > 0),
  };
}

const integrationEnabled = process.env.URBANAGENT_PHASE1_INTEGRATION === 'true';

test('Phase 1 disposable Postgres migration/import/rollback/repository integration', {
  skip: integrationEnabled ? false : 'Set URBANAGENT_PHASE1_INTEGRATION=true with disposable PostGIS DATABASE_URL to run.',
  timeout: 180000,
}, async () => {
  const pool = createPostgresPool();
  const csvRepository = new CanonicalCsvPoiRepository();
  const plan = await buildCanonicalPoiImportPlan();

  try {
    await rollbackPhase1();
    assert.deepEqual(await listExistingPhase1Tables(pool), []);

    await migratePhase1();
    const firstImportResult = await withTransaction(pool, (client) => writeImportPlan(client, plan));

    assert.ok(firstImportResult.ingestionRunId);
    const firstDiagnostics = await getPhase1DbDiagnostics(pool);
    assertExpectedPhase1Diagnostics(firstDiagnostics);
    assert.deepEqual(coreCounts(firstDiagnostics), EXPECTED_COUNTS);

    await withTransaction(pool, (client) => writeImportPlan(client, plan));

    const secondDiagnostics = await getPhase1DbDiagnostics(pool);
    assertExpectedPhase1Diagnostics(secondDiagnostics);
    assert.deepEqual(coreCounts(secondDiagnostics), coreCounts(firstDiagnostics));

    const pgRepository = new PostgresPoiRepository({ pool });
    const [csvPois, pgPois] = await Promise.all([
      csvRepository.findByCity(DEFAULT_CITY_ID),
      pgRepository.findByCity(DEFAULT_CITY_ID),
    ]);
    assert.equal(pgPois.length, csvPois.length);
    assert.equal(filterPoisForEdaSource(pgPois, 'google_maps').length, 3946);
    assert.equal(filterPoisForEdaSource(pgPois, 'foody').length, 225);
    assert.equal(filterPoisForEdaSource(pgPois, 'all').length, 4166);

    const pgMap = byId(pgPois);
    for (const [label, csvPoi] of Object.entries(pickCsvExamples(csvPois))) {
      assert.ok(csvPoi, `Missing CSV example for ${label}`);
      const pgPoi = pgMap.get(csvPoi.globalId);
      assert.ok(pgPoi, `Missing Postgres example for ${label}`);
      assert.equal(pgPoi.globalId, csvPoi.globalId);
      assert.equal(pgPoi.cityId, csvPoi.cityId);
      assert.equal(pgPoi.source, csvPoi.source);
      assert.equal(pgPoi.rating, csvPoi.rating);
      assert.equal(pgPoi.reviewCount, csvPoi.reviewCount);
      assert.deepEqual(pgPoi.aliasGlobalIds, csvPoi.aliasGlobalIds);
      assert.deepEqual(pgPoi.imageUrls, csvPoi.imageUrls);
      assert.equal(Object.prototype.hasOwnProperty.call(pgPoi, 'placeId'), false);
    }

    const lookupExample = pickCsvExamples(csvPois).merged;
    const lookedUp = await pgRepository.findById(lookupExample.globalId);
    assert.equal(lookedUp.globalId, lookupExample.globalId);
    assert.equal(lookedUp.source, 'google_maps+foody');

    process.env.URBANAGENT_POI_REPOSITORY = 'postgres';
    setPoiRepositoryForTests(pgRepository);
    const runtimePois = await loadPOIs({ cityId: DEFAULT_CITY_ID });
    assert.equal(runtimePois.length, EXPECTED_COUNTS.poi_entities);
    const recommendation = await recommendPOIs({
      query: 'quan cafe yen tinh',
      context: { cityId: DEFAULT_CITY_ID },
      limit: 3,
    });
    assert.ok(recommendation.results.length > 0);
    const itinerary = await createItinerary({
      query: 'quan cafe yen tinh',
      context: { cityId: DEFAULT_CITY_ID },
      limit: 3,
      durationMinutes: 180,
    });
    assert.ok(itinerary.itinerary.length > 0);
    assert.equal(itinerary.itinerary[0].travelFromPrevious.distanceKm, null);
    assert.equal(itinerary.itinerary[0].travelFromPrevious.estimatedMinutes, null);
    assert.equal(itinerary.itinerary[0].travelFromPrevious.distanceKnown, false);

    delete process.env.URBANAGENT_POI_REPOSITORY;
    setPoiRepositoryForTests(null);
    const csvDefaultRuntimePois = await loadPOIs({ cityId: DEFAULT_CITY_ID });
    assert.equal(csvDefaultRuntimePois.length, EXPECTED_COUNTS.poi_entities);

    await rollbackPhase1();
    assert.deepEqual(await listExistingPhase1Tables(pool), []);

    await migratePhase1();
    await withTransaction(pool, (client) => writeImportPlan(client, plan));
    const finalDiagnostics = await getPhase1DbDiagnostics(pool);
    assertExpectedPhase1Diagnostics(finalDiagnostics);
    assert.deepEqual(coreCounts(finalDiagnostics), EXPECTED_COUNTS);

    await runPostgresEndpointSmoke({ databaseUrl: process.env.DATABASE_URL || process.env.POSTGRES_URL });
  } finally {
    delete process.env.URBANAGENT_POI_REPOSITORY;
    setPoiRepositoryForTests(null);
    await pool.end();
  }
});
