const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const {
  assertLocalDependencyClosure,
  listTrackedRuntimeFiles,
  readRuntimeEntries,
} = require('../../scripts/prepare_hf_space_payload');

const { createCorsOptions, parseAllowedOrigins } = require('../../src/config/corsOptions');
const { CanonicalCsvPoiRepository } = require('../../src/services/canonicalCsvPoiRepository');
const {
  DEFAULT_RUNTIME_DATASET_MANIFEST,
  RuntimeDatasetError,
  verifyRuntimeDataset,
} = require('../../src/services/runtimeDatasetVerifier');

const ROOT = path.resolve(__dirname, '..', '..');
const CANONICAL = path.join(ROOT, 'data', 'canonical', 'urbanagent_poi_master_v1.csv');
const EXPECTED_SHA = 'dcb404cc8b5c7a9b5fd70df63039ab8f828c504270e22b12a671fa4ed61583f4';

function corsDecision(options, origin) {
  return new Promise((resolve) => {
    options.origin(origin, (error, allowed) => resolve({ error, allowed }));
  });
}

test('HF runtime manifest includes Admin and closes every static local dependency', () => {
  const tracked = listTrackedRuntimeFiles(readRuntimeEntries());
  assert.ok(tracked.includes('src/modules/admin/adminRouter.js'));
  assert.ok(tracked.includes('src/modules/admin/adminReadService.js'));
  assert.doesNotThrow(() => assertLocalDependencyClosure(tracked));
  assert.throws(
    () => assertLocalDependencyClosure(
      tracked.filter((file) => file !== 'src/modules/admin/adminReadService.js'),
    ),
    /adminRouter\.js -> src\/modules\/admin\/adminReadService\.js/,
  );
});

test('Stage 5B runtime manifest verifies exact canonical bytes and data quality', async () => {
  const result = await verifyRuntimeDataset();
  assert.equal(result.packagingMode, 'git-blob');
  assert.equal(result.byteSize, 7220189);
  assert.equal(result.sha256, EXPECTED_SHA);
  assert.equal(result.poiCount, 4173);
  assert.equal(result.duplicateCanonicalIds, 0);
  assert.equal(result.invalidCoreRecords, 0);
  assert.equal(result.requiredColumnsReadable, true);

  const stage4sState = JSON.parse(fs.readFileSync(path.join(
    ROOT,
    'data',
    'citypacks',
    'enrichments',
    'danang',
    'stage4s_create_new_state.json',
  ), 'utf8'));
  const pois = await new CanonicalCsvPoiRepository({ filePath: CANONICAL }).loadAll();
  const canonicalIds = new Set(pois.map((poi) => poi.globalId));
  assert.equal(stage4sState.appliedCanonicalIds.length, 7);
  assert.ok(stage4sState.appliedCanonicalIds.every((id) => canonicalIds.has(id)));
});

test('Stage 5B verifier rejects missing data and Git LFS pointer text with actionable codes', async (t) => {
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'urbanagent-stage5b-'));
  t.after(() => fs.rmSync(temporary, { recursive: true, force: true }));
  const missing = path.join(temporary, 'missing.csv');
  await assert.rejects(
    verifyRuntimeDataset({ datasetPath: missing, manifestPath: DEFAULT_RUNTIME_DATASET_MANIFEST }),
    (error) => error instanceof RuntimeDatasetError && error.code === 'RUNTIME_DATA_MISSING',
  );

  const pointer = path.join(temporary, 'pointer.csv');
  fs.writeFileSync(pointer, [
    'version https://git-lfs.github.com/spec/v1',
    `oid sha256:${EXPECTED_SHA}`,
    'size 7220189',
    '',
  ].join('\n'));
  await assert.rejects(
    verifyRuntimeDataset({ datasetPath: pointer, manifestPath: DEFAULT_RUNTIME_DATASET_MANIFEST }),
    (error) => error instanceof RuntimeDatasetError && error.code === 'RUNTIME_DATA_IS_LFS_POINTER',
  );
});

test('Stage 5B verifier rejects changed canonical bytes without mutating the source', async (t) => {
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'urbanagent-stage5b-'));
  t.after(() => fs.rmSync(temporary, { recursive: true, force: true }));
  const changed = path.join(temporary, 'changed.csv');
  fs.copyFileSync(CANONICAL, changed);
  fs.appendFileSync(changed, 'changed');
  await assert.rejects(
    verifyRuntimeDataset({ datasetPath: changed, manifestPath: DEFAULT_RUNTIME_DATASET_MANIFEST }),
    (error) => error instanceof RuntimeDatasetError && error.code === 'RUNTIME_DATA_SIZE_MISMATCH',
  );
  assert.equal(fs.statSync(CANONICAL).size, 7220189);
});

test('Stage 5B CORS uses explicit production origins and safe local development defaults', async () => {
  assert.deepEqual(parseAllowedOrigins('https://travel.example, https://admin.example/'), [
    'https://travel.example',
    'https://admin.example',
  ]);
  const production = createCorsOptions({
    NODE_ENV: 'production',
    URBANAGENT_CORS_ALLOWED_ORIGINS: 'https://travel.example',
  });
  assert.equal((await corsDecision(production, 'https://travel.example')).allowed, true);
  assert.equal((await corsDecision(production, undefined)).allowed, true);
  assert.equal((await corsDecision(production, 'https://other.example')).error.code, 'CORS_ORIGIN_NOT_ALLOWED');

  const development = createCorsOptions({ NODE_ENV: 'development' });
  assert.equal((await corsDecision(development, 'http://localhost:5173')).allowed, true);
  const productionWithoutOrigins = createCorsOptions({ NODE_ENV: 'production' });
  assert.equal(
    (await corsDecision(productionWithoutOrigins, 'http://localhost:5173')).error.code,
    'CORS_ORIGIN_NOT_ALLOWED',
  );
});
