const fs = require('fs');
const path = require('path');

const { createPostgresPool } = require('./postgresClient');

const ROOT_DIR = path.resolve(__dirname, '..', '..', '..');
const PHASE1_MIGRATIONS = [
  {
    name: '001_core_postgis_schema.sql',
    path: path.join(ROOT_DIR, 'migrations', 'phase1', '001_core_postgis_schema.sql'),
  },
];
const PHASE1_ROLLBACKS = [
  {
    name: '001_core_postgis_schema.down.sql',
    path: path.join(ROOT_DIR, 'migrations', 'phase1', '001_core_postgis_schema.down.sql'),
  },
];
const PHASE1_TABLES = [
  'data_quality_issues',
  'poi_merge_candidates',
  'poi_reviews_summary',
  'poi_images',
  'poi_aliases',
  'poi_external_ids',
  'poi_source_records',
  'poi_entities',
  'ingestion_runs',
  'cities',
];

function parseDatabaseUrl(value = process.env.DATABASE_URL || process.env.POSTGRES_URL) {
  if (!value) return null;
  try {
    return new URL(value);
  } catch (_) {
    return null;
  }
}

function sanitizeDatabaseUrl(value = process.env.DATABASE_URL || process.env.POSTGRES_URL) {
  const parsed = parseDatabaseUrl(value);
  if (!parsed) return null;
  if (parsed.password) parsed.password = '***';
  return parsed.toString();
}

function assertPhase1DbWriteAllowed() {
  if (process.env.URBANAGENT_ALLOW_PHASE1_DB_WRITE !== 'true') {
    const error = new Error('Refusing Phase 1 DB write: URBANAGENT_ALLOW_PHASE1_DB_WRITE=true is required.');
    error.code = 'PHASE1_DB_WRITE_NOT_ALLOWED';
    throw error;
  }

  const parsed = parseDatabaseUrl();
  if (!parsed) {
    const error = new Error('Refusing Phase 1 DB write: DATABASE_URL or POSTGRES_URL is required.');
    error.code = 'PHASE1_DB_URL_MISSING';
    throw error;
  }

  const host = parsed.hostname.toLowerCase();
  const port = parsed.port;
  const dbName = parsed.pathname.replace(/^\//, '').toLowerCase();
  const localHost = ['localhost', '127.0.0.1', '::1'].includes(host);
  const disposableName = dbName.includes('phase1') || dbName.includes('test') || dbName.includes('disposable');

  if (!localHost || port !== '55432' || !disposableName) {
    const error = new Error('Refusing Phase 1 DB write: database must be disposable localhost:55432 and named for phase1/test.');
    error.code = 'PHASE1_DB_URL_NOT_DISPOSABLE';
    throw error;
  }
}

async function executeSqlFiles(files) {
  assertPhase1DbWriteAllowed();
  const pool = createPostgresPool();
  try {
    const executed = [];
    for (const migration of files) {
      const sql = fs.readFileSync(migration.path, 'utf8');
      await pool.query(sql);
      executed.push(migration.name);
    }
    return {
      database: sanitizeDatabaseUrl(),
      executed,
    };
  } finally {
    await pool.end();
  }
}

async function migratePhase1() {
  return executeSqlFiles(PHASE1_MIGRATIONS);
}

async function rollbackPhase1() {
  return executeSqlFiles(PHASE1_ROLLBACKS);
}

async function listExistingPhase1Tables(pool) {
  const result = await pool.query(`
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'
      AND table_name = ANY($1::text[])
    ORDER BY table_name
  `, [PHASE1_TABLES]);
  return result.rows.map((row) => row.table_name);
}

module.exports = {
  PHASE1_MIGRATIONS,
  PHASE1_ROLLBACKS,
  PHASE1_TABLES,
  assertPhase1DbWriteAllowed,
  listExistingPhase1Tables,
  migratePhase1,
  rollbackPhase1,
  sanitizeDatabaseUrl,
};
