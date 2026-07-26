const { DEFAULT_CITY_ID, normalizeText } = require('../../services/canonicalCsvPoiRepository');
const { createPostgresPool } = require('../../infrastructure/db/postgresClient');

function arrayOrEmpty(value) {
  return Array.isArray(value) ? value : [];
}

function numberOrNull(value) {
  if (value === null || value === undefined) return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function mapPostgresPoiRow(row) {
  const imageUrls = arrayOrEmpty(row.image_urls);
  const sourceIds = arrayOrEmpty(row.source_ids);
  const aliasGlobalIds = arrayOrEmpty(row.alias_global_ids);

  return {
    id: row.poi_id,
    globalId: row.poi_id,
    legacyId: row.poi_id,
    cityId: row.city_id,
    entityType: row.entity_type,
    source: row.source,
    sourceId: sourceIds[0] || null,
    sourceIds,
    aliasGlobalIds,
    name: row.name,
    category: row.primary_category || row.category_normalized || 'unknown',
    categoryNormalized: row.category_normalized || null,
    district: row.district || '',
    districtRaw: row.district_raw || '',
    adminNormalizationStatus: row.admin_normalization_status || null,
    address: row.address_current || row.address_raw || '',
    addressRaw: row.address_raw || null,
    addressCurrent: row.address_current || null,
    lat: numberOrNull(row.lat),
    lon: numberOrNull(row.lon),
    hasCoordinates: row.lat !== null && row.lon !== null,
    coordinateStatus: row.quality?.coordinateStatus || 'valid',
    rating: numberOrNull(row.normalized_rating),
    ratingScale: numberOrNull(row.rating_scale),
    ratingCount: numberOrNull(row.rating_count),
    reviewCount: numberOrNull(row.rating_count),
    reviewSampleCount: numberOrNull(row.review_sample_count),
    googleRating: numberOrNull(row.google_rating),
    googleRatingCount: numberOrNull(row.google_rating_count),
    foodyRating10: numberOrNull(row.foody_rating_10),
    foodyReviewSampleCount: numberOrNull(row.foody_review_sample_count),
    primaryRatingSource: row.primary_rating_source || null,
    price: row.price_text || null,
    priceMinVnd: numberOrNull(row.price_min_vnd),
    priceMaxVnd: numberOrNull(row.price_max_vnd),
    openingHoursRaw: row.opening_hours_raw || null,
    imageUrls,
    imageUrl: imageUrls[0] || null,
    text: row.semantic_text || '',
    normalized: normalizeText([
      row.name,
      row.primary_category,
      row.category_normalized,
      row.district,
      row.address_current,
      row.address_raw,
      row.semantic_text,
    ].join(' ')),
    mergeStatus: row.merge_status || null,
    dataQualityFlags: arrayOrEmpty(row.data_quality_flags),
    rowIndex: null,
    sourceFile: 'postgres',
    raw: row,
  };
}

class PostgresPoiRepository {
  constructor({ pool = createPostgresPool() } = {}) {
    this.pool = pool;
  }

  async loadAll() {
    return this.findByCity(DEFAULT_CITY_ID);
  }

  async findByCity(cityId = DEFAULT_CITY_ID) {
    const sql = `
      SELECT
        p.*,
        r.normalized_rating,
        r.rating_scale,
        r.rating_count,
        r.review_sample_count,
        r.google_rating,
        r.google_rating_count,
        r.foody_rating_10,
        r.foody_review_sample_count,
        r.primary_rating_source,
        COALESCE(ext.source_ids, ARRAY[]::TEXT[]) AS source_ids,
        COALESCE(alias.alias_global_ids, ARRAY[]::TEXT[]) AS alias_global_ids,
        COALESCE(img.image_urls, ARRAY[]::TEXT[]) AS image_urls
      FROM poi_entities p
      LEFT JOIN poi_reviews_summary r ON r.poi_id = p.poi_id
      LEFT JOIN LATERAL (
        SELECT array_agg(external_id ORDER BY external_id) AS source_ids
        FROM poi_external_ids
        WHERE poi_id = p.poi_id
      ) ext ON true
      LEFT JOIN LATERAL (
        SELECT array_agg(alias_global_id ORDER BY alias_global_id) AS alias_global_ids
        FROM poi_aliases
        WHERE poi_id = p.poi_id
      ) alias ON true
      LEFT JOIN LATERAL (
        SELECT array_agg(image_url ORDER BY sort_order, poi_image_id) AS image_urls
        FROM poi_images
        WHERE poi_id = p.poi_id AND image_url IS NOT NULL
      ) img ON true
      WHERE p.city_id = $1 AND p.entity_type = 'poi'
      ORDER BY p.poi_id
    `;
    const result = await this.pool.query(sql, [cityId]);
    return result.rows.map(mapPostgresPoiRow);
  }

  async findById(id, { cityId = DEFAULT_CITY_ID } = {}) {
    const pois = await this.findByCity(cityId);
    return pois.find((poi) => poi.globalId === id) || null;
  }

  async getQualityReport() {
    const result = await this.pool.query(`
      SELECT
        COUNT(*)::int AS application_pois,
        COUNT(DISTINCT poi_id)::int AS unique_global_ids,
        COUNT(*) FILTER (WHERE lat IS NULL OR lon IS NULL)::int AS missing_coordinates
      FROM poi_entities
      WHERE entity_type = 'poi'
    `);
    const row = result.rows[0] || {};
    return {
      generatedAt: new Date().toISOString(),
      dataset: {
        path: 'postgres',
        sha256: null,
        expectedColumns: [],
      },
      cityId: DEFAULT_CITY_ID,
      totals: {
        rows: row.application_pois || 0,
        columns: 0,
        applicationPois: row.application_pois || 0,
        invalidRows: 0,
        uniqueGlobalIds: row.unique_global_ids || 0,
        missingCoordinates: row.missing_coordinates || 0,
      },
      columns: [],
      categoryDistribution: {},
      entityTypeDistribution: { poi: row.application_pois || 0 },
      duplicateGlobalIds: [],
      nonMergeDuplicateCandidates: [],
      headerMatchesExpected: true,
    };
  }

  clearCache() {}
}

module.exports = {
  PostgresPoiRepository,
  mapPostgresPoiRow,
};
