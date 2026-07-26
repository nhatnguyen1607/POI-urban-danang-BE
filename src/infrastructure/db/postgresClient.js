let pgModule = null;

function loadPg() {
  if (pgModule) return pgModule;
  try {
    // Lazy-load so the default CSV runtime still works without a database.
    pgModule = require('pg');
    return pgModule;
  } catch (error) {
    const wrapped = new Error('Postgres repository selected but the "pg" package is not installed.');
    wrapped.code = 'POSTGRES_DRIVER_MISSING';
    wrapped.cause = error;
    throw wrapped;
  }
}

function createPostgresPool({
  connectionString = process.env.DATABASE_URL || process.env.POSTGRES_URL,
  ssl = process.env.PGSSLMODE === 'require' ? { rejectUnauthorized: false } : undefined,
} = {}) {
  if (!connectionString) {
    const error = new Error('Postgres repository selected but DATABASE_URL or POSTGRES_URL is not set.');
    error.code = 'POSTGRES_CONFIG_MISSING';
    throw error;
  }

  const { Pool } = loadPg();
  return new Pool({ connectionString, ssl });
}

async function withTransaction(pool, callback) {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');
    const result = await callback(client);
    await client.query('COMMIT');
    return result;
  } catch (error) {
    await client.query('ROLLBACK');
    throw error;
  } finally {
    client.release();
  }
}

module.exports = {
  createPostgresPool,
  withTransaction,
};
