const { verifyRuntimeDataset } = require('../src/services/runtimeDatasetVerifier');

async function main() {
  const result = await verifyRuntimeDataset();
  console.log(JSON.stringify({ status: 'PASS', ...result }, null, 2));
}

if (require.main === module) {
  main().catch((error) => {
    console.error(`${error.code || 'RUNTIME_DATA_VERIFICATION_FAILED'}: ${error.message}`);
    process.exitCode = 1;
  });
}

module.exports = { main };
