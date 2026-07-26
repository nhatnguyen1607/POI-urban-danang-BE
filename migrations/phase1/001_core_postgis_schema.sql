-- Phase 1 UrbanAgent data platform foundation.
-- This migration defines the Postgres/PostGIS schema only. Do not run it
-- against production without an explicit deployment approval.

BEGIN;

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS cities (
  city_id TEXT PRIMARY KEY,
  display_name_vi TEXT NOT NULL,
  display_name_en TEXT NOT NULL,
  country_code CHAR(2) NOT NULL,
  timezone TEXT NOT NULL,
  currency TEXT NOT NULL,
  center GEOGRAPHY(POINT, 4326) NOT NULL,
  bbox GEOGRAPHY(POLYGON, 4326) NOT NULL,
  status TEXT NOT NULL DEFAULT 'DRAFT',
  config JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT cities_status_check CHECK (
    status IN ('DRAFT', 'INGESTING', 'VALIDATING', 'READY_FOR_BETA', 'READY', 'DEGRADED')
  )
);

CREATE TABLE IF NOT EXISTS ingestion_runs (
  ingestion_run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  city_id TEXT NOT NULL REFERENCES cities(city_id),
  provider TEXT NOT NULL,
  source_path TEXT,
  dataset_sha256 TEXT,
  row_count INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'planned',
  started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  finished_at TIMESTAMPTZ,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  CONSTRAINT ingestion_runs_status_check CHECK (
    status IN ('planned', 'running', 'succeeded', 'failed', 'rolled_back')
  )
);

CREATE TABLE IF NOT EXISTS poi_entities (
  poi_id TEXT PRIMARY KEY,
  city_id TEXT NOT NULL REFERENCES cities(city_id),
  entity_type TEXT NOT NULL DEFAULT 'poi',
  name TEXT NOT NULL,
  normalized_name TEXT NOT NULL,
  primary_category TEXT NOT NULL,
  category_normalized TEXT,
  location GEOGRAPHY(POINT, 4326) NOT NULL,
  lat DOUBLE PRECISION NOT NULL,
  lon DOUBLE PRECISION NOT NULL,
  address_raw TEXT,
  address_current TEXT,
  district_raw TEXT,
  district TEXT,
  admin_normalization_status TEXT,
  opening_hours_raw TEXT,
  price_text TEXT,
  price_min_vnd INTEGER,
  price_max_vnd INTEGER,
  source TEXT NOT NULL,
  merge_status TEXT,
  data_quality_flags TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
  semantic_text TEXT,
  field_provenance JSONB NOT NULL DEFAULT '{}'::jsonb,
  freshness JSONB NOT NULL DEFAULT '{}'::jsonb,
  quality JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT poi_entities_entity_type_check CHECK (entity_type = 'poi'),
  CONSTRAINT poi_entities_lat_check CHECK (lat BETWEEN -90 AND 90),
  CONSTRAINT poi_entities_lon_check CHECK (lon BETWEEN -180 AND 180)
);

CREATE INDEX IF NOT EXISTS poi_entities_city_idx ON poi_entities(city_id);
CREATE INDEX IF NOT EXISTS poi_entities_category_idx ON poi_entities(city_id, primary_category);
CREATE INDEX IF NOT EXISTS poi_entities_location_gix ON poi_entities USING GIST(location);

CREATE TABLE IF NOT EXISTS poi_source_records (
  source_record_id TEXT PRIMARY KEY,
  poi_id TEXT REFERENCES poi_entities(poi_id) ON DELETE SET NULL,
  ingestion_run_id UUID REFERENCES ingestion_runs(ingestion_run_id),
  city_id TEXT NOT NULL REFERENCES cities(city_id),
  provider TEXT NOT NULL,
  provider_record_id TEXT,
  policy_class TEXT NOT NULL,
  source_path TEXT,
  source_row_index INTEGER,
  checksum TEXT,
  observed_at TIMESTAMPTZ,
  raw_payload JSONB NOT NULL,
  normalized_payload JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT poi_source_records_policy_check CHECK (
    policy_class IN ('legacy-canonical', 'open', 'commercial-cache-limited', 'first-party')
  )
);

CREATE INDEX IF NOT EXISTS poi_source_records_poi_idx ON poi_source_records(poi_id);
CREATE INDEX IF NOT EXISTS poi_source_records_provider_idx ON poi_source_records(provider, provider_record_id);

CREATE TABLE IF NOT EXISTS poi_external_ids (
  poi_id TEXT NOT NULL REFERENCES poi_entities(poi_id) ON DELETE CASCADE,
  provider TEXT NOT NULL,
  external_id TEXT NOT NULL,
  source_record_id TEXT REFERENCES poi_source_records(source_record_id) ON DELETE SET NULL,
  confidence NUMERIC(4, 3) NOT NULL DEFAULT 1.0,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (poi_id, provider, external_id),
  CONSTRAINT poi_external_ids_confidence_check CHECK (confidence >= 0 AND confidence <= 1)
);

CREATE INDEX IF NOT EXISTS poi_external_ids_poi_idx ON poi_external_ids(poi_id);
CREATE INDEX IF NOT EXISTS poi_external_ids_provider_value_idx ON poi_external_ids(provider, external_id);

CREATE TABLE IF NOT EXISTS poi_aliases (
  poi_id TEXT NOT NULL REFERENCES poi_entities(poi_id) ON DELETE CASCADE,
  alias_global_id TEXT NOT NULL,
  source TEXT NOT NULL DEFAULT 'canonical_csv',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (poi_id, alias_global_id)
);

CREATE TABLE IF NOT EXISTS poi_images (
  poi_image_id BIGSERIAL PRIMARY KEY,
  poi_id TEXT NOT NULL REFERENCES poi_entities(poi_id) ON DELETE CASCADE,
  source TEXT NOT NULL,
  image_url TEXT,
  storage_key TEXT,
  attribution TEXT,
  license TEXT,
  sort_order INTEGER NOT NULL DEFAULT 0,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT poi_images_one_reference_check CHECK (image_url IS NOT NULL OR storage_key IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS poi_images_poi_idx ON poi_images(poi_id, sort_order);
CREATE UNIQUE INDEX IF NOT EXISTS poi_images_unique_url_idx
  ON poi_images(poi_id, source, image_url)
  WHERE image_url IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS poi_images_unique_storage_idx
  ON poi_images(poi_id, source, storage_key)
  WHERE storage_key IS NOT NULL;

CREATE TABLE IF NOT EXISTS poi_reviews_summary (
  poi_id TEXT PRIMARY KEY REFERENCES poi_entities(poi_id) ON DELETE CASCADE,
  normalized_rating NUMERIC(4, 2),
  rating_scale NUMERIC(4, 2),
  rating_count INTEGER,
  review_sample_count INTEGER,
  google_rating NUMERIC(4, 2),
  google_rating_count INTEGER,
  foody_rating_10 NUMERIC(4, 2),
  foody_review_sample_count INTEGER,
  primary_rating_source TEXT,
  rating_by_source JSONB NOT NULL DEFAULT '{}'::jsonb,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT poi_reviews_summary_rating_count_check CHECK (rating_count IS NULL OR rating_count >= 0),
  CONSTRAINT poi_reviews_summary_review_sample_check CHECK (review_sample_count IS NULL OR review_sample_count >= 0)
);

CREATE TABLE IF NOT EXISTS poi_merge_candidates (
  merge_candidate_id BIGSERIAL PRIMARY KEY,
  city_id TEXT NOT NULL REFERENCES cities(city_id),
  left_poi_id TEXT NOT NULL REFERENCES poi_entities(poi_id) ON DELETE CASCADE,
  right_poi_id TEXT NOT NULL REFERENCES poi_entities(poi_id) ON DELETE CASCADE,
  candidate_reason TEXT NOT NULL,
  distance_meters DOUBLE PRECISION,
  name_similarity NUMERIC(4, 3),
  category_compatibility NUMERIC(4, 3),
  confidence NUMERIC(4, 3),
  status TEXT NOT NULL DEFAULT 'needs_review',
  evidence JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT poi_merge_candidates_distinct_check CHECK (left_poi_id <> right_poi_id),
  CONSTRAINT poi_merge_candidates_status_check CHECK (
    status IN ('needs_review', 'accepted', 'rejected', 'superseded')
  )
);

CREATE INDEX IF NOT EXISTS poi_merge_candidates_city_status_idx
  ON poi_merge_candidates(city_id, status);

CREATE TABLE IF NOT EXISTS data_quality_issues (
  data_quality_issue_id BIGSERIAL PRIMARY KEY,
  city_id TEXT NOT NULL REFERENCES cities(city_id),
  poi_id TEXT REFERENCES poi_entities(poi_id) ON DELETE SET NULL,
  source_record_id TEXT REFERENCES poi_source_records(source_record_id) ON DELETE SET NULL,
  issue_code TEXT NOT NULL,
  severity TEXT NOT NULL DEFAULT 'warning',
  field_name TEXT,
  message TEXT NOT NULL,
  evidence JSONB NOT NULL DEFAULT '{}'::jsonb,
  status TEXT NOT NULL DEFAULT 'open',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  resolved_at TIMESTAMPTZ,
  CONSTRAINT data_quality_issues_severity_check CHECK (severity IN ('info', 'warning', 'error')),
  CONSTRAINT data_quality_issues_status_check CHECK (status IN ('open', 'acknowledged', 'resolved'))
);

CREATE INDEX IF NOT EXISTS data_quality_issues_city_status_idx
  ON data_quality_issues(city_id, status, severity);

COMMIT;
