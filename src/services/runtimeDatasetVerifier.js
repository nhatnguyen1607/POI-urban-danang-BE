const fs = require('node:fs');
const path = require('node:path');

const {
  CANONICAL_POI_CSV_PATH,
  EXPECTED_CANONICAL_COLUMNS,
  buildQualityReport,
  fileSha256,
  readCSV,
} = require('./canonicalCsvPoiRepository');

const ROOT_DIR = path.resolve(__dirname, '..', '..');
const DEFAULT_RUNTIME_DATASET_MANIFEST = path.join(
  ROOT_DIR,
  'data',
  'canonical',
  'runtime_dataset_manifest.json',
);

class RuntimeDatasetError extends Error {
  constructor(code, message) {
    super(message);
    this.name = 'RuntimeDatasetError';
    this.code = code;
  }
}

function fail(code, message) {
  throw new RuntimeDatasetError(code, message);
}

function readRuntimeDatasetManifest(manifestPath = DEFAULT_RUNTIME_DATASET_MANIFEST) {
  if (!fs.existsSync(manifestPath)) {
    fail('RUNTIME_DATA_MANIFEST_MISSING', 'Runtime dataset manifest is missing. Restore the versioned manifest.');
  }

  let manifest;
  try {
    manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
  } catch (_) {
    fail('RUNTIME_DATA_MANIFEST_INVALID', 'Runtime dataset manifest is not valid JSON.');
  }

  const required = ['datasetName', 'datasetVersion', 'poiCount', 'sha256', 'byteSize', 'runtimePath', 'packagingMode'];
  if (required.some((key) => manifest[key] === undefined || manifest[key] === null)) {
    fail('RUNTIME_DATA_MANIFEST_INVALID', 'Runtime dataset manifest is missing required fields.');
  }
  if (!Number.isInteger(manifest.poiCount) || manifest.poiCount < 1
    || !Number.isInteger(manifest.byteSize) || manifest.byteSize < 1
    || !/^[a-f0-9]{64}$/i.test(String(manifest.sha256))) {
    fail('RUNTIME_DATA_MANIFEST_INVALID', 'Runtime dataset manifest has invalid count, size, or SHA-256 values.');
  }
  return manifest;
}

function isGitLfsPointer(filePath) {
  if (!fs.existsSync(filePath)) return false;
  const descriptor = fs.openSync(filePath, 'r');
  try {
    const buffer = Buffer.alloc(200);
    const bytesRead = fs.readSync(descriptor, buffer, 0, buffer.length, 0);
    return buffer.subarray(0, bytesRead).toString('utf8')
      .startsWith('version https://git-lfs.github.com/spec/v1');
  } finally {
    fs.closeSync(descriptor);
  }
}

async function verifyRuntimeDataset({
  datasetPath = CANONICAL_POI_CSV_PATH,
  manifestPath = DEFAULT_RUNTIME_DATASET_MANIFEST,
} = {}) {
  const manifest = readRuntimeDatasetManifest(manifestPath);
  if (!fs.existsSync(datasetPath)) {
    fail('RUNTIME_DATA_MISSING', 'Runtime POI dataset is missing. Restore the versioned runtime dataset.');
  }
  if (isGitLfsPointer(datasetPath)) {
    fail('RUNTIME_DATA_IS_LFS_POINTER', 'Runtime POI dataset is a Git LFS pointer, not CSV data.');
  }

  const byteSize = fs.statSync(datasetPath).size;
  if (byteSize !== manifest.byteSize) {
    fail('RUNTIME_DATA_SIZE_MISMATCH', `Runtime POI dataset size is ${byteSize}; expected ${manifest.byteSize}.`);
  }
  const sha256 = fileSha256(datasetPath);
  if (sha256 !== String(manifest.sha256).toLowerCase()) {
    fail('RUNTIME_DATA_HASH_MISMATCH', 'Runtime POI dataset SHA-256 does not match the versioned manifest.');
  }

  const source = await readCSV(datasetPath);
  const quality = buildQualityReport(source, datasetPath);
  if (!quality.headerMatchesExpected
    || JSON.stringify(source.headers) !== JSON.stringify(EXPECTED_CANONICAL_COLUMNS)) {
    fail('RUNTIME_DATA_SCHEMA_MISMATCH', 'Runtime POI dataset columns do not match the canonical schema.');
  }
  if (quality.totals.rows !== manifest.poiCount
    || quality.totals.applicationPois !== manifest.poiCount
    || quality.totals.uniqueGlobalIds !== manifest.poiCount
    || quality.totals.duplicateGlobalIdExtraRows !== 0
    || quality.totals.invalidRows !== 0) {
    fail('RUNTIME_DATA_QUALITY_MISMATCH', 'Runtime POI count, IDs, or core-record quality do not match the manifest.');
  }

  return {
    datasetName: manifest.datasetName,
    datasetVersion: manifest.datasetVersion,
    runtimePath: manifest.runtimePath,
    packagingMode: manifest.packagingMode,
    byteSize,
    sha256,
    poiCount: quality.totals.applicationPois,
    duplicateCanonicalIds: quality.totals.duplicateGlobalIdExtraRows,
    invalidCoreRecords: quality.totals.invalidRows,
    requiredColumnsReadable: true,
  };
}

module.exports = {
  DEFAULT_RUNTIME_DATASET_MANIFEST,
  RuntimeDatasetError,
  isGitLfsPointer,
  readRuntimeDatasetManifest,
  verifyRuntimeDataset,
};
