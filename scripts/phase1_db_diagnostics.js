#!/usr/bin/env node

const { createPostgresPool } = require('../src/infrastructure/db/postgresClient');
const {
  assertExpectedPhase1Diagnostics,
  getPhase1DbDiagnostics,
} = require('../src/modules/pois/postgresDiagnostics');

async function main() {
  const pool = createPostgresPool();
  try {
    const diagnostics = await getPhase1DbDiagnostics(pool);
    assertExpectedPhase1Diagnostics(diagnostics);
    console.log(JSON.stringify({
      mode: 'diagnostics',
      status: 'pass',
      diagnostics,
    }, null, 2));
  } finally {
    await pool.end();
  }
}

main().catch((error) => {
  console.error(error.message);
  process.exit(1);
});
