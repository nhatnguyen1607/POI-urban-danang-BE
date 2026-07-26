#!/usr/bin/env node

const { rollbackPhase1 } = require('../src/infrastructure/db/phase1MigrationRunner');

rollbackPhase1()
  .then((result) => {
    console.log(JSON.stringify({
      mode: 'rollback',
      database: result.database,
      executed: result.executed,
    }, null, 2));
  })
  .catch((error) => {
    console.error(error.message);
    process.exit(1);
  });
