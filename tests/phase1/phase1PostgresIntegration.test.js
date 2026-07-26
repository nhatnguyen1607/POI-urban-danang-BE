const assert = require('node:assert/strict');
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
  } finally {
    delete process.env.URBANAGENT_POI_REPOSITORY;
    setPoiRepositoryForTests(null);
    await pool.end();
  }
});
