const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');

const {
  CANONICAL_POI_CSV_PATH,
  EXPECTED_CANONICAL_COLUMNS,
  CanonicalCsvPoiRepository,
  normalizeCanonicalPoiRow,
  parseImageUrls,
  readCSV,
} = require('../../src/services/canonicalCsvPoiRepository');
const { normalizePoi } = require('../../src/services/firestorePersistenceService');
const { createItinerary } = require('../../src/services/itineraryPlannerService');
const { filterPoisForEdaSource, loadPOIs, normalizeEdaSource } = require('../../src/services/poiDataService');
const { recommendPOIs } = require('../../src/services/poiRetrievalService');

const EXPECTED_HASH = '5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae';
const EXPECTED_ROWS = 4166;
const DEFAULT_CITY_ID = 'da-nang';

function sha256(filePath) {
  return crypto.createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');
}

test('canonical CSV file is the approved immutable Phase 0 dataset', async () => {
  assert.equal(path.basename(CANONICAL_POI_CSV_PATH), 'urbanagent_poi_master_v1.csv');
  assert.equal(sha256(CANONICAL_POI_CSV_PATH), EXPECTED_HASH);

  const source = await readCSV(CANONICAL_POI_CSV_PATH);
  assert.equal(source.rows.length, EXPECTED_ROWS);
  assert.deepEqual(source.headers, EXPECTED_CANONICAL_COLUMNS);
  assert.equal(source.headers[0], 'Global_ID');

  const ids = new Set(source.rows.map((row) => row.Global_ID));
  assert.equal(ids.size, EXPECTED_ROWS);
  assert.equal(source.rows.filter((row) => row.Entity_Type !== 'poi').length, 0);
});

test('canonical repository loads all application POIs and honors city filtering', async () => {
  const repository = new CanonicalCsvPoiRepository();
  const pois = await repository.loadAll();
  const report = await repository.getQualityReport();

  assert.equal(pois.length, EXPECTED_ROWS);
  assert.equal(report.headerMatchesExpected, true);
  assert.equal(report.totals.rows, EXPECTED_ROWS);
  assert.equal(report.totals.applicationPois, EXPECTED_ROWS);
  assert.equal(report.totals.invalidRows, 0);
  assert.equal(report.totals.uniqueGlobalIds, EXPECTED_ROWS);
  assert.equal(report.totals.duplicateGlobalIdExtraRows, 0);
  assert.equal(report.totals.missingCoordinates, 0);
  assert.equal(report.totals.invalidCoordinates, 0);
  assert.equal(report.entityTypeDistribution.poi, EXPECTED_ROWS);

  const daNangPois = await repository.findByCity(DEFAULT_CITY_ID);
  const otherCityPois = await repository.findByCity('hue');
  assert.equal(daNangPois.length, EXPECTED_ROWS);
  assert.equal(otherCityPois.length, 0);
});

test('canonical POI semantics preserve source IDs, aliases, nulls, and address uncertainty', async () => {
  const repository = new CanonicalCsvPoiRepository();
  const pois = await repository.loadAll();

  assert.equal(new Set(pois.map((poi) => poi.id)).size, EXPECTED_ROWS);
  assert.ok(pois.every((poi) => poi.id === poi.globalId));
  assert.ok(pois.every((poi) => !Object.prototype.hasOwnProperty.call(poi, 'placeId')));
  assert.ok(pois.some((poi) => poi.sourceId && poi.sourceIds.length > 0));
  assert.ok(pois.some((poi) => poi.aliasGlobalIds.length > 0));
  assert.ok(pois.some((poi) => poi.rating === null));
  assert.ok(pois.some((poi) => poi.foodyReviewSampleCount !== null && poi.reviewCount === null));
  assert.ok(pois.every((poi) => poi.hasCoordinates === true));
  assert.ok(pois.every((poi) => poi.lat !== 16.0544 || poi.lon !== 108.2022));
  assert.ok(pois.every((poi) => !poi.address || poi.address !== poi.district));
  assert.ok(pois.every((poi) => Array.isArray(poi.imageUrls)));
  assert.ok(pois.some((poi) => poi.imageUrls.length > 0 && poi.imageUrl === poi.imageUrls[0]));
});

test('canonical image URL parsing keeps only unique http URLs in source order', () => {
  assert.deepEqual(parseImageUrls('https://example.com/only.jpg'), ['https://example.com/only.jpg']);
  assert.deepEqual(
    parseImageUrls(' https://example.com/a.jpg,not-a-url,http://example.com/b.jpg, https://example.com/a.jpg, ftp://example.com/c.jpg, '),
    ['https://example.com/a.jpg', 'http://example.com/b.jpg'],
  );
  assert.deepEqual(parseImageUrls(''), []);
  assert.deepEqual(parseImageUrls(null), []);
});

test('normalizer rejects invalid rows instead of fabricating POI fields', () => {
  const invalid = normalizeCanonicalPoiRow({
    Global_ID: '',
    City_ID: DEFAULT_CITY_ID,
    Entity_Type: 'poi',
    'Restaurant Name': 'Missing coordinates',
    District: 'Hai Chau',
    Lat: '',
    Lon: '',
  });

  assert.equal(invalid.validForApplication, false);
  assert.equal(invalid.poi, null);
  assert.ok(invalid.issues.includes('missing_global_id'));
  assert.ok(invalid.issues.includes('invalid_coordinates'));
});

test('Firestore POI normalization keeps missing rating and review count as unknown', () => {
  const poi = normalizePoi({
    poiId: 'test_poi',
    name: 'No numeric semantics',
    category: 'cafe',
    cityId: DEFAULT_CITY_ID,
    location: { lat: 16.06, lng: 108.22, address: '' },
    rating: '',
    reviewCount: null,
  });

  assert.equal(poi.rating, null);
  assert.equal(poi.reviewCount, null);
  assert.equal(poi.ratingSum, null);
  assert.equal(poi.location.hasCoordinates, true);
  assert.equal(poi.location.address, '');
});

test('Firestore POI normalization preserves merged Google/Foody provenance', () => {
  const poi = normalizePoi({
    poiId: 'merged_poi',
    name: 'Merged source',
    category: 'restaurant',
    cityId: DEFAULT_CITY_ID,
    source: 'google_maps+foody',
    location: { lat: 16.06, lng: 108.22 },
  });

  assert.equal(poi.source, 'google_maps+foody');
});

test('EDA source compatibility keeps merged-source POIs in Google and Foody views', async () => {
  const pois = await loadPOIs({ cityId: DEFAULT_CITY_ID });

  const googleAliases = [undefined, '', 'missing', 'ggmap', 'google', 'google_maps', 'unknown-provider'];
  for (const source of googleAliases) {
    assert.equal(normalizeEdaSource(source), 'google_maps');
    assert.equal(filterPoisForEdaSource(pois, source).length, 3946);
  }

  assert.equal(normalizeEdaSource('foody'), 'foody');
  assert.equal(filterPoisForEdaSource(pois, 'foody').length, 225);

  for (const source of ['all', 'canonical']) {
    assert.equal(normalizeEdaSource(source), 'all');
    assert.equal(filterPoisForEdaSource(pois, source).length, 4166);
  }
});

test('recommendation and itinerary services return canonical POIs after header normalization', async () => {
  const recommendation = await recommendPOIs({
    query: 'quan cafe yen tinh',
    context: { cityId: DEFAULT_CITY_ID },
    limit: 5,
  });

  assert.ok(recommendation.results.length > 0);
  assert.ok(recommendation.results.every((poi) => poi.cityId === DEFAULT_CITY_ID));
  assert.ok(recommendation.results.every((poi) => poi.globalId));
  assert.ok(recommendation.results.every((poi) => !Object.prototype.hasOwnProperty.call(poi, 'placeId')));

  const itinerary = await createItinerary({
    query: 'quan cafe yen tinh',
    context: { cityId: DEFAULT_CITY_ID },
    limit: 3,
    durationMinutes: 180,
  });

  assert.ok(itinerary.itinerary.length > 0);
  assert.ok(itinerary.itinerary.every((stop) => stop.poi.cityId === DEFAULT_CITY_ID));
  assert.equal(itinerary.itinerary[0].travelFromPrevious.distanceKm, null);
  assert.equal(itinerary.itinerary[0].travelFromPrevious.estimatedMinutes, null);
  assert.equal(itinerary.itinerary[0].travelFromPrevious.distanceKnown, false);
  assert.equal(itinerary.itinerary[0].travelFromPrevious.source, 'missing-origin');
  if (itinerary.itinerary.length > 1) {
    assert.equal(typeof itinerary.itinerary[1].travelFromPrevious.distanceKm, 'number');
    assert.equal(typeof itinerary.itinerary[1].travelFromPrevious.estimatedMinutes, 'number');
    assert.equal(itinerary.itinerary[1].travelFromPrevious.distanceKnown, true);
    assert.equal(itinerary.itinerary[1].travelFromPrevious.source, 'local-haversine-estimate');
  }
  assert.ok(itinerary.warnings.length > 0);
});

test('legacy root expert-system density engine uses canonical POIs instead of old runtime CSVs', async () => {
  const enginePath = path.join(__dirname, '..', '..', 'ES-system', 'poi_density_engine.js');
  const source = fs.readFileSync(enginePath, 'utf8');
  assert.equal(source.includes('poi_data_foody.csv'), false);
  assert.equal(source.includes('poi_data_ggmap.csv'), false);

  const POIDensityEngine = require('../../ES-system/poi_density_engine');
  const engine = new POIDensityEngine();
  await engine.load();
  assert.equal(engine.allPois.length, EXPECTED_ROWS);
});
