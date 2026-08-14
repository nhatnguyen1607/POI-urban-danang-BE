const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const { prepareRuntimePayload } = require('../../scripts/prepare_hf_space_payload');
const { verifyRuntimeDataset } = require('../../src/services/runtimeDatasetVerifier');

const CANONICAL_SHA = 'dcb404cc8b5c7a9b5fd70df63039ab8f828c504270e22b12a671fa4ed61583f4';

test('Hugging Face payload is deterministic, complete, and excludes legacy LFS assets', async () => {
  const tempBase = path.resolve(process.env.URBANAGENT_TEST_TEMP_ROOT || os.tmpdir());
  fs.mkdirSync(tempBase, { recursive: true });
  const runRoot = fs.mkdtempSync(path.join(tempBase, 'urbanagent-hf-payload-'));
  const firstPath = path.join(runRoot, 'first');
  const secondPath = path.join(runRoot, 'second');

  try {
    const first = prepareRuntimePayload({ outputPath: firstPath });
    const second = prepareRuntimePayload({ outputPath: secondPath });
    assert.deepEqual(first.files, second.files);

    const paths = new Set(first.files.map((item) => item.path));
    for (const required of [
      'Dockerfile',
      'package-lock.json',
      'src/server.js',
      'scripts/verify_runtime_dataset.js',
      'data/canonical/runtime_dataset_manifest.json',
      'data/canonical/urbanagent_poi_master_v1.csv',
      'artifacts/memory/agent_memory_v1.json',
      'artifacts/reranker/agent_reranker_v1.json',
      'ES-system/rule/cam_queo.csv',
    ]) {
      assert.equal(paths.has(required), true, `Missing runtime file: ${required}`);
    }

    for (const forbiddenPrefix of [
      '.git/', '.github/', 'node_modules/', 'data/spikes/', 'data/citypacks/',
      'data/poi_data_', 'functions/', 'migrations/', 'model/', 'tests/',
      'src/modules/cityPackPreparation/',
    ]) {
      assert.equal(
        [...paths].some((file) => file.startsWith(forbiddenPrefix)),
        false,
        `Forbidden payload prefix: ${forbiddenPrefix}`,
      );
    }

    const result = await verifyRuntimeDataset({
      datasetPath: path.join(firstPath, 'data', 'canonical', 'urbanagent_poi_master_v1.csv'),
      manifestPath: path.join(firstPath, 'data', 'canonical', 'runtime_dataset_manifest.json'),
    });
    assert.equal(result.poiCount, 4173);
    assert.equal(result.sha256, CANONICAL_SHA);
  } finally {
    const resolvedRunRoot = path.resolve(runRoot);
    if (!resolvedRunRoot.startsWith(`${tempBase}${path.sep}`)) {
      throw new Error(`Refusing to clean unexpected test path: ${resolvedRunRoot}`);
    }
    fs.rmSync(resolvedRunRoot, { recursive: true, force: true });
  }
});
