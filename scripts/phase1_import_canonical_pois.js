#!/usr/bin/env node

const { buildCanonicalPoiImportPlan } = require('../src/modules/pois/canonicalPoiImportPlan');
const { createPostgresPool, withTransaction } = require('../src/infrastructure/db/postgresClient');
const { assertPhase1DbWriteAllowed, sanitizeDatabaseUrl } = require('../src/infrastructure/db/phase1MigrationRunner');
const { bboxToPolygonWkt } = require('../src/modules/cities/cityConfig');

function hasFlag(name) {
  return process.argv.includes(name);
}

async function upsertCity(client, city) {
  await client.query(`
    INSERT INTO cities (
      city_id,
      display_name_vi,
      display_name_en,
      country_code,
      timezone,
      currency,
      center,
      bbox,
      status,
      config
    )
    VALUES (
      $1, $2, $3, $4, $5, $6,
      ST_SetSRID(ST_MakePoint($7, $8), 4326)::geography,
      ST_GeogFromText($9),
      $10,
      $11::jsonb
    )
    ON CONFLICT (city_id) DO UPDATE SET
      display_name_vi = EXCLUDED.display_name_vi,
      display_name_en = EXCLUDED.display_name_en,
      timezone = EXCLUDED.timezone,
      currency = EXCLUDED.currency,
      center = EXCLUDED.center,
      bbox = EXCLUDED.bbox,
      status = EXCLUDED.status,
      config = EXCLUDED.config,
      updated_at = now()
  `, [
    city.cityId,
    city.displayNameVi,
    city.displayNameEn,
    city.countryCode,
    city.timezone,
    city.currency,
    city.center.lon,
    city.center.lat,
    bboxToPolygonWkt(city.bbox),
    city.status,
    JSON.stringify(city),
  ]);
}

async function writeImportPlan(client, plan) {
  await upsertCity(client, plan.city);

  const ingestionRun = await client.query(`
    INSERT INTO ingestion_runs (
      city_id,
      provider,
      source_path,
      dataset_sha256,
      row_count,
      status,
      finished_at,
      metadata
    )
    VALUES ($1, $2, $3, $4, $5, 'succeeded', now(), $6::jsonb)
    RETURNING ingestion_run_id
  `, [
    plan.city.cityId,
    plan.source.provider,
    plan.source.path,
    plan.source.sha256,
    plan.source.rows,
    JSON.stringify({ policyClass: plan.source.policyClass, phase: 'phase1' }),
  ]);
  const ingestionRunId = ingestionRun.rows[0].ingestion_run_id;

  for (const record of plan.records.poiEntities) {
    await client.query(`
      INSERT INTO poi_entities (
        poi_id,
        city_id,
        entity_type,
        name,
        normalized_name,
        primary_category,
        category_normalized,
        location,
        lat,
        lon,
        address_raw,
        address_current,
        district_raw,
        district,
        admin_normalization_status,
        opening_hours_raw,
        price_text,
        price_min_vnd,
        price_max_vnd,
        source,
        merge_status,
        data_quality_flags,
        semantic_text,
        field_provenance,
        freshness,
        quality
      )
      VALUES (
        $1, $2, $3, $4, $5, $6, $7,
        ST_SetSRID(ST_MakePoint($8, $9), 4326)::geography,
        $9, $8, $10, $11, $12, $13, $14, $15, $16, $17, $18,
        $19, $20, $21, $22, $23::jsonb, $24::jsonb, $25::jsonb
      )
      ON CONFLICT (poi_id) DO UPDATE SET
        name = EXCLUDED.name,
        normalized_name = EXCLUDED.normalized_name,
        primary_category = EXCLUDED.primary_category,
        category_normalized = EXCLUDED.category_normalized,
        location = EXCLUDED.location,
        lat = EXCLUDED.lat,
        lon = EXCLUDED.lon,
        address_raw = EXCLUDED.address_raw,
        address_current = EXCLUDED.address_current,
        district_raw = EXCLUDED.district_raw,
        district = EXCLUDED.district,
        admin_normalization_status = EXCLUDED.admin_normalization_status,
        opening_hours_raw = EXCLUDED.opening_hours_raw,
        price_text = EXCLUDED.price_text,
        price_min_vnd = EXCLUDED.price_min_vnd,
        price_max_vnd = EXCLUDED.price_max_vnd,
        source = EXCLUDED.source,
        merge_status = EXCLUDED.merge_status,
        data_quality_flags = EXCLUDED.data_quality_flags,
        semantic_text = EXCLUDED.semantic_text,
        field_provenance = EXCLUDED.field_provenance,
        freshness = EXCLUDED.freshness,
        quality = EXCLUDED.quality,
        updated_at = now()
    `, [
      record.poi_id,
      record.city_id,
      record.entity_type,
      record.name,
      record.normalized_name,
      record.primary_category,
      record.category_normalized,
      record.lon,
      record.lat,
      record.address_raw,
      record.address_current,
      record.district_raw,
      record.district,
      record.admin_normalization_status,
      record.opening_hours_raw,
      record.price_text,
      record.price_min_vnd,
      record.price_max_vnd,
      record.source,
      record.merge_status,
      record.data_quality_flags,
      record.semantic_text,
      JSON.stringify(record.field_provenance),
      JSON.stringify(record.freshness),
      JSON.stringify(record.quality),
    ]);
  }

  for (const record of plan.records.poiSourceRecords) {
    await client.query(`
      INSERT INTO poi_source_records (
        source_record_id,
        poi_id,
        ingestion_run_id,
        city_id,
        provider,
        provider_record_id,
        policy_class,
        source_path,
        source_row_index,
        checksum,
        observed_at,
        raw_payload,
        normalized_payload
      )
      VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12::jsonb, $13::jsonb)
      ON CONFLICT (source_record_id) DO UPDATE SET
        poi_id = EXCLUDED.poi_id,
        ingestion_run_id = EXCLUDED.ingestion_run_id,
        checksum = EXCLUDED.checksum,
        raw_payload = EXCLUDED.raw_payload,
        normalized_payload = EXCLUDED.normalized_payload
    `, [
      record.source_record_id,
      record.poi_id,
      ingestionRunId,
      record.city_id,
      record.provider,
      record.provider_record_id,
      record.policy_class,
      record.source_path,
      record.source_row_index,
      record.checksum,
      record.observed_at,
      JSON.stringify(record.raw_payload),
      JSON.stringify(record.normalized_payload),
    ]);
  }

  for (const record of plan.records.poiExternalIds) {
    await client.query(`
      INSERT INTO poi_external_ids (poi_id, provider, external_id, source_record_id, confidence)
      VALUES ($1, $2, $3, $4, $5)
      ON CONFLICT (poi_id, provider, external_id) DO UPDATE SET
        poi_id = EXCLUDED.poi_id,
        source_record_id = EXCLUDED.source_record_id,
        confidence = EXCLUDED.confidence
    `, [record.poi_id, record.provider, record.external_id, record.source_record_id, record.confidence]);
  }

  for (const record of plan.records.poiAliases) {
    await client.query(`
      INSERT INTO poi_aliases (poi_id, alias_global_id, source)
      VALUES ($1, $2, $3)
      ON CONFLICT (poi_id, alias_global_id) DO NOTHING
    `, [record.poi_id, record.alias_global_id, record.source]);
  }

  for (const record of plan.records.poiImages) {
    await client.query(`
      INSERT INTO poi_images (
        poi_id,
        source,
        image_url,
        storage_key,
        attribution,
        license,
        sort_order
      )
      VALUES ($1, $2, $3, $4, $5, $6, $7)
      ON CONFLICT DO NOTHING
    `, [
      record.poi_id,
      record.source,
      record.image_url,
      record.storage_key,
      record.attribution,
      record.license,
      record.sort_order,
    ]);
  }

  for (const record of plan.records.poiReviewsSummary) {
    await client.query(`
      INSERT INTO poi_reviews_summary (
        poi_id,
        normalized_rating,
        rating_scale,
        rating_count,
        review_sample_count,
        google_rating,
        google_rating_count,
        foody_rating_10,
        foody_review_sample_count,
        primary_rating_source,
        rating_by_source
      )
      VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb)
      ON CONFLICT (poi_id) DO UPDATE SET
        normalized_rating = EXCLUDED.normalized_rating,
        rating_scale = EXCLUDED.rating_scale,
        rating_count = EXCLUDED.rating_count,
        review_sample_count = EXCLUDED.review_sample_count,
        google_rating = EXCLUDED.google_rating,
        google_rating_count = EXCLUDED.google_rating_count,
        foody_rating_10 = EXCLUDED.foody_rating_10,
        foody_review_sample_count = EXCLUDED.foody_review_sample_count,
        primary_rating_source = EXCLUDED.primary_rating_source,
        rating_by_source = EXCLUDED.rating_by_source,
        updated_at = now()
    `, [
      record.poi_id,
      record.normalized_rating,
      record.rating_scale,
      record.rating_count,
      record.review_sample_count,
      record.google_rating,
      record.google_rating_count,
      record.foody_rating_10,
      record.foody_review_sample_count,
      record.primary_rating_source,
      JSON.stringify(record.rating_by_source),
    ]);
  }

  return { ingestionRunId };
}

async function main() {
  const write = hasFlag('--write');
  const dryRun = hasFlag('--dry-run') || !write;
  const plan = await buildCanonicalPoiImportPlan();

  if (dryRun) {
    console.log(JSON.stringify({
      mode: 'dry-run',
      cityId: plan.city.cityId,
      source: plan.source,
      quality: {
        applicationPois: plan.quality.totals.applicationPois,
        invalidRows: plan.quality.totals.invalidRows,
        headerMatchesExpected: plan.quality.headerMatchesExpected,
      },
      totals: plan.totals,
      nextStep: 'Run database migration, then rerun with --write and DATABASE_URL to import.',
    }, null, 2));
    return;
  }

  assertPhase1DbWriteAllowed();
  const pool = createPostgresPool();
  try {
    const result = await withTransaction(pool, (client) => writeImportPlan(client, plan));
    console.log(JSON.stringify({
      mode: 'write',
      database: sanitizeDatabaseUrl(),
      cityId: plan.city.cityId,
      ingestionRunId: result.ingestionRunId,
      totals: plan.totals,
    }, null, 2));
  } finally {
    await pool.end();
  }
}

if (require.main === module) {
  main().catch((error) => {
    console.error(error.message);
    process.exit(1);
  });
}

module.exports = {
  writeImportPlan,
};
