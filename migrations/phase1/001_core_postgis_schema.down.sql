-- Phase 1 UrbanAgent data platform rollback.
-- DESTRUCTIVE: disposable Phase 1 verification database only.
-- Do not run against production or shared databases.

BEGIN;

DROP TABLE IF EXISTS data_quality_issues;
DROP TABLE IF EXISTS poi_merge_candidates;
DROP TABLE IF EXISTS poi_reviews_summary;
DROP TABLE IF EXISTS poi_images;
DROP TABLE IF EXISTS poi_aliases;
DROP TABLE IF EXISTS poi_external_ids;
DROP TABLE IF EXISTS poi_source_records;
DROP TABLE IF EXISTS poi_entities;
DROP TABLE IF EXISTS ingestion_runs;
DROP TABLE IF EXISTS cities;

-- Do not drop postgis or pgcrypto. Extensions may be shared by unrelated
-- objects in non-disposable databases.

COMMIT;
