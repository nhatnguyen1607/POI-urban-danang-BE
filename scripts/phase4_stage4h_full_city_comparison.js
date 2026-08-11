const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const { performance } = require('node:perf_hooks');

const { inspectCanonicalDataset, readCanonicalPois } = require('../src/modules/cityPackPreparation/canonicalDataset');
const { resolveSourceRecords } = require('../src/modules/cityPackPreparation/entityResolution');
const { loadRealSnapshotRecords } = require('../src/modules/cityPackPreparation/realSnapshot');
const { normalizeWithAdapters } = require('../src/modules/cityPackPreparation/dryRun');
const { buildReviewQueue } = require('../src/modules/cityPackPreparation/reviewQueue');

const ROOT = path.resolve(__dirname, '..');
const CANONICAL_PATH = path.join(ROOT, 'data', 'canonical', 'urbanagent_poi_master_v1.csv');
const CONFIG_PATH = path.join(ROOT, 'config', 'phase4_stage4g_sources.json');
const MANIFEST_PATH = path.join(ROOT, 'data', 'spikes', 'phase4', 'stage4g', 'snapshot_manifest.json');
const STAGE4G_SUMMARY_PATH = path.join(ROOT, 'data', 'spikes', 'phase4', 'stage4g', 'stage4g_summary.json');

function fileHash(filePath) {
  return crypto.createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');
}

function snapshotPaths(cacheDir) {
  return {
    overture: path.join(cacheDir, 'overture_full_danang.json'),
    osm: path.join(cacheDir, 'osm_full_danang.json'),
    wikidata: path.join(cacheDir, 'wikidata_full_danang.json'),
    commons: path.join(cacheDir, 'wikimedia_commons_full_danang_metadata.json'),
  };
}

function insideBoundary(record, boundary) {
  return Number.isFinite(record.latitude)
    && Number.isFinite(record.longitude)
    && record.longitude >= boundary.west
    && record.longitude <= boundary.east
    && record.latitude >= boundary.south
    && record.latitude <= boundary.north;
}

function verifySnapshots(manifest, paths) {
  const bySource = new Map(manifest.snapshots.map((snapshot) => [snapshot.source, snapshot]));
  for (const [key, filePath] of Object.entries(paths)) {
    const source = key === 'commons' ? 'wikimedia_commons' : key;
    const expected = bySource.get(source)?.sha256;
    if (!expected || fileHash(filePath) !== expected) {
      throw new Error(`Stage 4G snapshot hash mismatch: ${source}`);
    }
  }
}

function runResolver(records, canonicalPois, hardened) {
  const started = performance.now();
  const result = resolveSourceRecords(records, canonicalPois, undefined, {
    useSpatialIndex: true,
    useSpatialDuplicateIndex: true,
    includeCrossSourceDuplicates: true,
    maxCandidateDistanceMeters: 800,
    hardened,
    includeMetrics: true,
    resolutionVersion: hardened ? 'stage4h-v1' : 'stage4g-v1',
  });
  return {
    result,
    reviewQueueCount: buildReviewQueue(result).length,
    seconds: Number(((performance.now() - started) / 1000).toFixed(3)),
  };
}

function runComparison({ cacheDir, outputPath = null }) {
  const config = JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf8'));
  const manifest = JSON.parse(fs.readFileSync(MANIFEST_PATH, 'utf8'));
  const paths = snapshotPaths(cacheDir);
  verifySnapshots(manifest, paths);
  const canonicalBefore = inspectCanonicalDataset(CANONICAL_PATH);
  const canonicalPois = readCanonicalPois(CANONICAL_PATH);
  const rawRecords = loadRealSnapshotRecords(paths);
  const records = normalizeWithAdapters(
    rawRecords.filter((record) => insideBoundary(record, config.boundary)),
  );
  const recordedBaseline = JSON.parse(fs.readFileSync(STAGE4G_SUMMARY_PATH, 'utf8'));
  if (recordedBaseline.normalizedRecords !== records.length || recordedBaseline.deterministic !== true) {
    throw new Error('Recorded Stage 4G baseline does not match the verified fixed snapshots.');
  }
  const hardened = runResolver(records, canonicalPois, true);
  const comparisons = hardened.result.summary.candidateComparisons;
  const naive = hardened.result.summary.naiveCandidateComparisons;
  const summary = {
    status: 'CANDIDATE_NON_RUNTIME_NOT_CANONICAL',
    snapshotsVerified: true,
    normalizedRecords: records.length,
    canonical: canonicalBefore,
    baseline: {
      resolution: recordedBaseline.resolution,
      reviewQueueCount: recordedBaseline.reviewQueueCount,
      seconds: recordedBaseline.timings.entityResolutionSeconds,
      source: 'verified_stage4g_fixed_snapshot_run',
    },
    hardened: {
      resolution: hardened.result.summary,
      reviewQueueCount: hardened.reviewQueueCount,
      seconds: hardened.seconds,
    },
    candidateComparisonReduction: Number((1 - comparisons / Math.max(naive, 1)).toFixed(6)),
    deterministicInputOrderingHash: crypto.createHash('sha256')
      .update(JSON.stringify(records.map((record) => `${record.source}:${record.sourceId}`)))
      .digest('hex'),
    canonicalChanged: fileHash(CANONICAL_PATH) !== canonicalBefore.sha256,
    runtimeChanged: false,
    candidateApplyAuthorized: false,
  };
  if (outputPath) {
    fs.mkdirSync(path.dirname(outputPath), { recursive: true });
    fs.writeFileSync(outputPath, `${JSON.stringify(summary, null, 2)}\n`, 'utf8');
  }
  if (summary.canonicalChanged) throw new Error('Canonical CSV changed during Stage 4H comparison.');
  return summary;
}

if (require.main === module) {
  const cacheDir = process.argv[2];
  const outputPath = process.argv[3] || null;
  if (!cacheDir) {
    console.error('Usage: node scripts/phase4_stage4h_full_city_comparison.js <stage4g-cache-dir> [output-json]');
    process.exitCode = 1;
  } else {
    console.log(JSON.stringify(runComparison({ cacheDir: path.resolve(cacheDir), outputPath }), null, 2));
  }
}

module.exports = { runComparison, verifySnapshots };
