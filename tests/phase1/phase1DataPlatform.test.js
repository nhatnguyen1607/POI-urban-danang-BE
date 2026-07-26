const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');

const {
  buildCanonicalPoiImportPlan,
  toPoiEntityRecord,
} = require('../../src/modules/pois/canonicalPoiImportPlan');
const { getCityConfig, bboxToPolygonWkt } = require('../../src/modules/cities/cityConfig');
const { mapPostgresPoiRow } = require('../../src/modules/pois/postgresPoiRepository');

const DEFAULT_CITY_ID = 'da-nang';
const EXPECTED_ROWS = 4166;
const EXPECTED_HASH = '5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae';

test('Phase 1 migration defines the required PostGIS canonical POI foundation', () => {
  const migrationPath = path.join(__dirname, '..', '..', 'migrations', 'phase1', '001_core_postgis_schema.sql');
  const sql = fs.readFileSync(migrationPath, 'utf8');

  assert.match(sql, /CREATE EXTENSION IF NOT EXISTS postgis/i);
  assert.match(sql, /CREATE EXTENSION IF NOT EXISTS pgcrypto/i);
  for (const table of [
    'cities',
    'ingestion_runs',
    'poi_entities',
    'poi_source_records',
    'poi_external_ids',
    'poi_aliases',
    'poi_images',
    'poi_reviews_summary',
    'poi_merge_candidates',
    'data_quality_issues',
  ]) {
    assert.match(sql, new RegExp(`CREATE TABLE IF NOT EXISTS ${table}`, 'i'));
  }
  assert.doesNotMatch(sql, /\bDROP\s+TABLE\b/i);
  assert.doesNotMatch(sql, /\bTRUNCATE\b/i);
});

test('Da Nang City Pack config is explicit and can become a PostGIS bbox polygon', () => {
  const city = getCityConfig(DEFAULT_CITY_ID);

  assert.equal(city.cityId, DEFAULT_CITY_ID);
  assert.equal(city.countryCode, 'VN');
  assert.equal(city.timezone, 'Asia/Ho_Chi_Minh');
  assert.equal(city.currency, 'VND');
  assert.equal(city.status, 'READY_FOR_BETA');
  assert.equal(
    bboxToPolygonWkt(city.bbox),
    'POLYGON(( 107.8 15.8, 108.5 15.8, 108.5 16.3, 107.8 16.3, 107.8 15.8 ))',
  );
});

test('legacy canonical importer plans Bronze, Silver, Gold records without changing CSV bytes', async () => {
  const plan = await buildCanonicalPoiImportPlan();

  assert.equal(plan.city.cityId, DEFAULT_CITY_ID);
  assert.equal(plan.source.sha256, EXPECTED_HASH);
  assert.equal(plan.quality.totals.applicationPois, EXPECTED_ROWS);
  assert.equal(plan.quality.totals.invalidRows, 0);
  assert.equal(plan.quality.headerMatchesExpected, true);
  assert.equal(plan.records.poiEntities.length, EXPECTED_ROWS);
  assert.equal(plan.records.poiSourceRecords.length, EXPECTED_ROWS);
  assert.equal(plan.records.poiReviewsSummary.length, EXPECTED_ROWS);
  assert.ok(plan.records.poiExternalIds.length >= EXPECTED_ROWS);
  assert.ok(plan.records.poiImages.length > 0);
});

test('Phase 1 POI entity records preserve null semantics and provenance', () => {
  const record = toPoiEntityRecord({
    globalId: 'poi_test',
    cityId: DEFAULT_CITY_ID,
    entityType: 'poi',
    name: 'No Rating Cafe',
    category: 'Cafe',
    categoryNormalized: 'cafe',
    lat: 16.06,
    lon: 108.22,
    addressRaw: null,
    addressCurrent: null,
    districtRaw: '',
    district: '',
    adminNormalizationStatus: 'pending_spatial_join',
    openingHoursRaw: null,
    price: null,
    priceMinVnd: null,
    priceMaxVnd: null,
    source: 'google_maps+foody',
    mergeStatus: 'merged',
    dataQualityFlags: [],
    text: '',
    coordinateStatus: 'valid',
    hasCoordinates: true,
    rating: null,
    reviewCount: null,
  });

  assert.equal(record.poi_id, 'poi_test');
  assert.equal(record.source, 'google_maps+foody');
  assert.equal(record.address_raw, null);
  assert.equal(record.address_current, null);
  assert.equal(record.freshness.observedAt, null);
  assert.equal(record.freshness.lastVerifiedAt, null);
  assert.equal(record.quality.ratingKnown, false);
  assert.equal(record.quality.reviewCountKnown, false);
  assert.equal(record.field_provenance.location.source, 'legacy_canonical_csv');
});

test('Postgres repository mapper keeps the legacy API-compatible POI shape', () => {
  const poi = mapPostgresPoiRow({
    poi_id: 'poi_1',
    city_id: DEFAULT_CITY_ID,
    entity_type: 'poi',
    source: 'google_maps+foody',
    source_ids: ['src_1'],
    alias_global_ids: ['old_1'],
    name: 'Mapped Cafe',
    primary_category: 'Cafe',
    category_normalized: 'cafe',
    district: '',
    district_raw: '',
    admin_normalization_status: 'pending_spatial_join',
    address_raw: null,
    address_current: null,
    lat: 16.06,
    lon: 108.22,
    quality: { coordinateStatus: 'valid' },
    normalized_rating: null,
    rating_scale: null,
    rating_count: null,
    review_sample_count: null,
    google_rating: null,
    google_rating_count: null,
    foody_rating_10: 8.5,
    foody_review_sample_count: 2,
    primary_rating_source: 'foody',
    price_text: null,
    price_min_vnd: null,
    price_max_vnd: null,
    opening_hours_raw: null,
    image_urls: ['https://example.com/a.jpg'],
    semantic_text: 'quiet cafe',
    merge_status: 'merged',
    data_quality_flags: [],
  });

  assert.equal(poi.id, 'poi_1');
  assert.equal(poi.globalId, 'poi_1');
  assert.equal(poi.cityId, DEFAULT_CITY_ID);
  assert.equal(poi.source, 'google_maps+foody');
  assert.equal(poi.sourceId, 'src_1');
  assert.deepEqual(poi.aliasGlobalIds, ['old_1']);
  assert.equal(poi.address, '');
  assert.equal(poi.rating, null);
  assert.equal(poi.reviewCount, null);
  assert.equal(poi.foodyReviewSampleCount, 2);
  assert.equal(poi.imageUrl, 'https://example.com/a.jpg');
  assert.equal(Object.prototype.hasOwnProperty.call(poi, 'placeId'), false);
});
