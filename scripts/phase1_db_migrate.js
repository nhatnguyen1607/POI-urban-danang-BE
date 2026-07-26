#!/usr/bin/env node

const { migratePhase1 } = require('../src/infrastructure/db/phase1MigrationRunner');

migratePhase1()
  .then((result) => {
    console.log(JSON.stringify({
      mode: 'migrate',
      database: result.database,
      executed: result.executed,
    }, null, 2));
  })
  .catch((error) => {
    console.error(error.message);
    process.exit(1);
  });
