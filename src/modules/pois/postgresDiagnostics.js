const { DA_NANG_BBOX } = require('../../services/canonicalCsvPoiRepository');

async function singleValue(pool, sql, params = []) {
  const result = await pool.query(sql, params);
  const row = result.rows[0] || {};
  return Number(Object.values(row)[0] || 0);
}

async function getPhase1DbDiagnostics(pool) {
  const countsResult = await pool.query(`
    SELECT
      (SELECT COUNT(*)::int FROM cities) AS cities,
      (SELECT COUNT(*)::int FROM poi_entities) AS poi_entities,
      (SELECT COUNT(*)::int FROM poi_source_records) AS source_records,
      (SELECT COUNT(*)::int FROM poi_external_ids) AS external_ids,
      (SELECT COUNT(*)::int FROM poi_aliases) AS aliases,
      (SELECT COUNT(*)::int FROM poi_images) AS images,
      (SELECT COUNT(*)::int FROM poi_reviews_summary) AS review_summaries,
      (SELECT COUNT(*)::int FROM ingestion_runs) AS ingestion_runs
  `);
  const counts = countsResult.rows[0];

  const duplicateChecks = {
    sourceRecords: await singleValue(pool, `
      SELECT COUNT(*) FROM (
        SELECT source_record_id
        FROM poi_source_records
        GROUP BY source_record_id
        HAVING COUNT(*) > 1
      ) d
    `),
    externalIds: await singleValue(pool, `
      SELECT COUNT(*) FROM (
        SELECT poi_id, provider, external_id
        FROM poi_external_ids
        GROUP BY poi_id, provider, external_id
        HAVING COUNT(*) > 1
      ) d
    `),
    aliases: await singleValue(pool, `
      SELECT COUNT(*) FROM (
        SELECT poi_id, alias_global_id
        FROM poi_aliases
        GROUP BY poi_id, alias_global_id
        HAVING COUNT(*) > 1
      ) d
    `),
    imageAssociations: await singleValue(pool, `
      SELECT COUNT(*) FROM (
        SELECT poi_id, source, image_url
        FROM poi_images
        WHERE image_url IS NOT NULL
        GROUP BY poi_id, source, image_url
        HAVING COUNT(*) > 1
      ) d
    `),
  };

  const orphanChecks = {
    sourceRecords: await singleValue(pool, `
      SELECT COUNT(*)
      FROM poi_source_records s
      LEFT JOIN poi_entities p ON p.poi_id = s.poi_id
      WHERE s.poi_id IS NOT NULL AND p.poi_id IS NULL
    `),
    externalIds: await singleValue(pool, `
      SELECT COUNT(*)
      FROM poi_external_ids e
      LEFT JOIN poi_entities p ON p.poi_id = e.poi_id
      WHERE p.poi_id IS NULL
    `),
    aliases: await singleValue(pool, `
      SELECT COUNT(*)
      FROM poi_aliases a
      LEFT JOIN poi_entities p ON p.poi_id = a.poi_id
      WHERE p.poi_id IS NULL
    `),
    images: await singleValue(pool, `
      SELECT COUNT(*)
      FROM poi_images i
      LEFT JOIN poi_entities p ON p.poi_id = i.poi_id
      WHERE p.poi_id IS NULL
    `),
    reviews: await singleValue(pool, `
      SELECT COUNT(*)
      FROM poi_reviews_summary r
      LEFT JOIN poi_entities p ON p.poi_id = r.poi_id
      WHERE p.poi_id IS NULL
    `),
  };

  const geometryChecks = {
    invalidLongitude: await singleValue(pool, 'SELECT COUNT(*) FROM poi_entities WHERE lon < -180 OR lon > 180'),
    invalidLatitude: await singleValue(pool, 'SELECT COUNT(*) FROM poi_entities WHERE lat < -90 OR lat > 90'),
    nullGeometry: await singleValue(pool, 'SELECT COUNT(*) FROM poi_entities WHERE location IS NULL'),
    wrongSrid: await singleValue(pool, 'SELECT COUNT(*) FROM poi_entities WHERE ST_SRID(location::geometry) <> 4326'),
    coordinateMismatch: await singleValue(pool, `
      SELECT COUNT(*)
      FROM poi_entities
      WHERE abs(ST_X(location::geometry) - lon) > 0.0000001
         OR abs(ST_Y(location::geometry) - lat) > 0.0000001
    `),
    outsideDaNangEnvelope: await singleValue(pool, `
      SELECT COUNT(*)
      FROM poi_entities
      WHERE lon < $1 OR lon > $2 OR lat < $3 OR lat > $4
    `, [DA_NANG_BBOX.west, DA_NANG_BBOX.east, DA_NANG_BBOX.south, DA_NANG_BBOX.north]),
  };

  const semanticChecks = {
    missingGlobalId: await singleValue(pool, "SELECT COUNT(*) FROM poi_entities WHERE poi_id IS NULL OR poi_id = ''"),
    urbanVoidRecords: await singleValue(pool, "SELECT COUNT(*) FROM poi_entities WHERE entity_type <> 'poi'"),
    lostMergedProvenance: await singleValue(pool, "SELECT COUNT(*) FROM poi_entities WHERE source = 'google_maps+foody'"),
    invalidRatingRanges: await singleValue(pool, `
      SELECT COUNT(*)
      FROM poi_reviews_summary
      WHERE (normalized_rating IS NOT NULL AND (normalized_rating < 0 OR normalized_rating > 5))
         OR (foody_rating_10 IS NOT NULL AND (foody_rating_10 < 0 OR foody_rating_10 > 10))
    `),
    imageOrderingAnomalies: await singleValue(pool, `
      SELECT COUNT(*)
      FROM (
        SELECT poi_id, COUNT(*) AS image_count, COUNT(DISTINCT sort_order) AS sort_order_count
        FROM poi_images
        GROUP BY poi_id
      ) i
      WHERE image_count <> sort_order_count
    `),
  };

  const indexesResult = await pool.query(`
    SELECT indexname
    FROM pg_indexes
    WHERE schemaname = 'public'
      AND tablename = 'poi_entities'
      AND indexdef ILIKE '%gist%'
    ORDER BY indexname
  `);
  const explainResult = await pool.query(`
    EXPLAIN
    SELECT poi_id
    FROM poi_entities
    WHERE location && ST_MakeEnvelope($1, $2, $3, $4, 4326)::geography
    LIMIT 10
  `, [DA_NANG_BBOX.west, DA_NANG_BBOX.south, DA_NANG_BBOX.east, DA_NANG_BBOX.north]);

  return {
    counts: Object.fromEntries(Object.entries(counts).map(([key, value]) => [key, Number(value)])),
    duplicateChecks,
    orphanChecks,
    geometryChecks,
    semanticChecks,
    gistIndexes: indexesResult.rows.map((row) => row.indexname),
    explainPlan: explainResult.rows.map((row) => row['QUERY PLAN']),
  };
}

function assertExpectedPhase1Diagnostics(diagnostics) {
  const expectedCounts = {
    poi_entities: 4166,
    source_records: 4166,
    external_ids: 8337,
    aliases: 985,
    images: 16246,
    review_summaries: 4166,
  };
  for (const [key, expected] of Object.entries(expectedCounts)) {
    if (diagnostics.counts[key] !== expected) {
      throw new Error(`Unexpected ${key}: expected ${expected}, got ${diagnostics.counts[key]}`);
    }
  }

  for (const group of [diagnostics.duplicateChecks, diagnostics.orphanChecks, diagnostics.geometryChecks]) {
    for (const [key, value] of Object.entries(group)) {
      if (key === 'outsideDaNangEnvelope') continue;
      if (value !== 0) throw new Error(`Invariant failed: ${key}=${value}`);
    }
  }

  if (diagnostics.semanticChecks.missingGlobalId !== 0) throw new Error('Missing Global_ID in imported POIs.');
  if (diagnostics.semanticChecks.urbanVoidRecords !== 0) throw new Error('Unexpected urban-void records imported.');
  if (diagnostics.semanticChecks.lostMergedProvenance <= 0) throw new Error('Merged provenance google_maps+foody was not preserved.');
  if (diagnostics.semanticChecks.invalidRatingRanges !== 0) throw new Error('Invalid rating ranges found.');
  if (diagnostics.semanticChecks.imageOrderingAnomalies !== 0) throw new Error('Image ordering anomalies found.');
  if (!diagnostics.gistIndexes.includes('poi_entities_location_gix')) throw new Error('Missing GiST spatial index.');
}

module.exports = {
  assertExpectedPhase1Diagnostics,
  getPhase1DbDiagnostics,
};
