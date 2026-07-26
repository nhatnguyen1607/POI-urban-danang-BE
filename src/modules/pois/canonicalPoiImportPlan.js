const crypto = require('crypto');
const path = require('path');

const {
  CANONICAL_POI_CSV_PATH,
  DEFAULT_CITY_ID,
  CanonicalCsvPoiRepository,
  fileSha256,
  normalizeText,
} = require('../../services/canonicalCsvPoiRepository');
const { getCityConfig } = require('../cities/cityConfig');

const LEGACY_CANONICAL_PROVIDER = 'legacy_canonical_csv';
const LEGACY_CANONICAL_POLICY_CLASS = 'legacy-canonical';

function stableHash(value) {
  return crypto.createHash('sha256').update(JSON.stringify(value)).digest('hex');
}

function integerOrNull(value) {
  if (value === null || value === undefined) return null;
  const number = Number(value);
  return Number.isInteger(number) ? number : null;
}

function moneyOrNull(value) {
  if (value === null || value === undefined) return null;
  const number = Number(value);
  return Number.isFinite(number) ? Math.round(number) : null;
}

function buildFieldProvenance(poi) {
  return {
    id: { source: LEGACY_CANONICAL_PROVIDER, field: 'Global_ID' },
    name: { source: LEGACY_CANONICAL_PROVIDER, field: 'Restaurant Name' },
    category: { source: LEGACY_CANONICAL_PROVIDER, field: 'Category' },
    location: { source: LEGACY_CANONICAL_PROVIDER, fields: ['Lat', 'Lon'] },
    address: { source: LEGACY_CANONICAL_PROVIDER, fields: ['Address_Raw', 'Address_Current'] },
    rating: { source: LEGACY_CANONICAL_PROVIDER, fields: ['Overall Rating', 'Rating_Count'] },
    images: { source: LEGACY_CANONICAL_PROVIDER, field: 'Image_URL' },
  };
}

function buildFreshness() {
  return {
    observedAt: null,
    lastVerifiedAt: null,
    sourceFreshness: 'unknown',
  };
}

function toPoiEntityRecord(poi) {
  return {
    poi_id: poi.globalId,
    city_id: poi.cityId,
    entity_type: poi.entityType,
    name: poi.name,
    normalized_name: normalizeText(poi.name),
    primary_category: poi.category,
    category_normalized: poi.categoryNormalized,
    lat: poi.lat,
    lon: poi.lon,
    address_raw: poi.addressRaw,
    address_current: poi.addressCurrent,
    district_raw: poi.districtRaw || null,
    district: poi.district || null,
    admin_normalization_status: poi.adminNormalizationStatus,
    opening_hours_raw: poi.openingHoursRaw,
    price_text: poi.price,
    price_min_vnd: moneyOrNull(poi.priceMinVnd),
    price_max_vnd: moneyOrNull(poi.priceMaxVnd),
    source: poi.source,
    merge_status: poi.mergeStatus,
    data_quality_flags: poi.dataQualityFlags,
    semantic_text: poi.text || null,
    field_provenance: buildFieldProvenance(poi),
    freshness: buildFreshness(),
    quality: {
      coordinateStatus: poi.coordinateStatus,
      hasCoordinates: poi.hasCoordinates,
      ratingKnown: poi.rating !== null,
      reviewCountKnown: poi.reviewCount !== null,
    },
  };
}

function toSourceRecord(poi, { sourcePath, ingestionRunId = null } = {}) {
  const rawPayload = poi.raw || {};
  const providerRecordId = poi.sourceId || poi.globalId;
  return {
    source_record_id: `${LEGACY_CANONICAL_PROVIDER}:${poi.globalId}`,
    poi_id: poi.globalId,
    ingestion_run_id: ingestionRunId,
    city_id: poi.cityId,
    provider: LEGACY_CANONICAL_PROVIDER,
    provider_record_id: providerRecordId,
    policy_class: LEGACY_CANONICAL_POLICY_CLASS,
    source_path: sourcePath,
    source_row_index: poi.rowIndex,
    checksum: stableHash(rawPayload),
    observed_at: null,
    raw_payload: rawPayload,
    normalized_payload: {
      id: poi.globalId,
      cityId: poi.cityId,
      source: poi.source,
      name: poi.name,
      category: poi.category,
      lat: poi.lat,
      lon: poi.lon,
    },
  };
}

function toExternalIdRecords(poi) {
  const ids = new Set([poi.sourceId, ...poi.sourceIds].filter(Boolean));
  return Array.from(ids).map((externalId) => ({
    poi_id: poi.globalId,
    provider: poi.source || LEGACY_CANONICAL_PROVIDER,
    external_id: externalId,
    source_record_id: `${LEGACY_CANONICAL_PROVIDER}:${poi.globalId}`,
    confidence: 1,
  }));
}

function toAliasRecords(poi) {
  return poi.aliasGlobalIds.map((aliasGlobalId) => ({
    poi_id: poi.globalId,
    alias_global_id: aliasGlobalId,
    source: LEGACY_CANONICAL_PROVIDER,
  }));
}

function toImageRecords(poi) {
  return poi.imageUrls.map((imageUrl, index) => ({
    poi_id: poi.globalId,
    source: LEGACY_CANONICAL_PROVIDER,
    image_url: imageUrl,
    storage_key: null,
    attribution: null,
    license: null,
    sort_order: index,
  }));
}

function toReviewSummaryRecord(poi) {
  return {
    poi_id: poi.globalId,
    normalized_rating: poi.rating,
    rating_scale: poi.ratingScale,
    rating_count: integerOrNull(poi.ratingCount),
    review_sample_count: integerOrNull(poi.reviewSampleCount),
    google_rating: poi.googleRating,
    google_rating_count: integerOrNull(poi.googleRatingCount),
    foody_rating_10: poi.foodyRating10,
    foody_review_sample_count: integerOrNull(poi.foodyReviewSampleCount),
    primary_rating_source: poi.primaryRatingSource,
    rating_by_source: {
      google_maps: {
        value: poi.googleRating,
        scale: poi.googleRating === null ? null : 5,
        count: integerOrNull(poi.googleRatingCount),
      },
      foody: {
        value: poi.foodyRating10,
        scale: poi.foodyRating10 === null ? null : 10,
        sampleCount: integerOrNull(poi.foodyReviewSampleCount),
      },
    },
  };
}

async function buildCanonicalPoiImportPlan({
  filePath = CANONICAL_POI_CSV_PATH,
  cityId = DEFAULT_CITY_ID,
} = {}) {
  const city = getCityConfig(cityId);
  if (!city) {
    const error = new Error(`Unsupported City Pack: ${cityId}`);
    error.code = 'UNSUPPORTED_CITY';
    throw error;
  }

  const repository = new CanonicalCsvPoiRepository({ filePath });
  const [pois, qualityReport] = await Promise.all([
    repository.findByCity(cityId),
    repository.getQualityReport(),
  ]);
  const sourcePath = path.relative(path.resolve(__dirname, '..', '..', '..'), filePath).replace(/\\/g, '/');

  const entityRecords = pois.map(toPoiEntityRecord);
  const sourceRecords = pois.map((poi) => toSourceRecord(poi, { sourcePath }));
  const externalIdRecords = pois.flatMap(toExternalIdRecords);
  const aliasRecords = pois.flatMap(toAliasRecords);
  const imageRecords = pois.flatMap(toImageRecords);
  const reviewSummaryRecords = pois.map(toReviewSummaryRecord);

  return {
    city,
    source: {
      provider: LEGACY_CANONICAL_PROVIDER,
      policyClass: LEGACY_CANONICAL_POLICY_CLASS,
      path: sourcePath,
      sha256: fileSha256(filePath),
      rows: qualityReport.totals.rows,
    },
    quality: qualityReport,
    records: {
      cities: [city],
      poiEntities: entityRecords,
      poiSourceRecords: sourceRecords,
      poiExternalIds: externalIdRecords,
      poiAliases: aliasRecords,
      poiImages: imageRecords,
      poiReviewsSummary: reviewSummaryRecords,
    },
    totals: {
      applicationPois: entityRecords.length,
      sourceRecords: sourceRecords.length,
      externalIds: externalIdRecords.length,
      aliases: aliasRecords.length,
      images: imageRecords.length,
      reviewSummaries: reviewSummaryRecords.length,
    },
  };
}

module.exports = {
  LEGACY_CANONICAL_POLICY_CLASS,
  LEGACY_CANONICAL_PROVIDER,
  buildCanonicalPoiImportPlan,
  toAliasRecords,
  toExternalIdRecords,
  toImageRecords,
  toPoiEntityRecord,
  toReviewSummaryRecord,
  toSourceRecord,
};
