const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const { inspectCanonicalDataset, EXPECTED_CANONICAL_SHA } = require('../../src/modules/cityPackPreparation/canonicalDataset');
const {
  DECISIONS,
  classifySourceRecordHardened,
} = require('../../src/modules/cityPackPreparation/entityResolution');
const {
  buildIdentityProfile,
  hasBranchConflict,
  normalizeVietnameseText,
} = require('../../src/modules/cityPackPreparation/historicalNormalization');
const {
  DEFAULT_SOURCE_CAPABILITIES,
  SOURCE_CAPABILITIES,
  STATUSES,
  runIncrementalSync,
  stageFingerprints,
} = require('../../src/modules/cityPackPreparation/incrementalSync');

const ROOT = path.resolve(__dirname, '..', '..');
const CANONICAL_PATH = path.join(ROOT, 'data', 'canonical', 'urbanagent_poi_master_v1.csv');

function source(overrides = {}) {
  return {
    source: 'osm',
    sourceId: 'osm:node:1',
    name: 'Cà phê Mộc',
    category: 'cafe',
    latitude: 16.05,
    longitude: 108.2,
    address: 'Hải Châu, Đà Nẵng',
    externalIds: { osm_id: '1' },
    media: null,
    license: { license: 'ODbL-1.0', attribution: 'OpenStreetMap contributors' },
    provenance: { snapshotRef: 'A', retrievedAt: '2026-08-01T00:00:00Z' },
    ...overrides,
  };
}

function canonical(overrides = {}) {
  return {
    id: 'canonical:1',
    name: 'Cà phê Mộc',
    category: 'cafe',
    latitude: 16.05001,
    longitude: 108.20001,
    address: 'Hải Châu, Đà Nẵng',
    district: 'Hải Châu',
    ...overrides,
  };
}

test('Stage 4H adopts deterministic Vietnamese normalization without losing original values', () => {
  assert.equal(normalizeVietnameseText('  CÀ-PHÊ  Mộc!  '), 'ca phe moc');
  assert.equal(
    normalizeVietnameseText('Cà phê Mộc - khuyến mãi 20% - https://example.test'),
    'ca phe moc',
  );
  assert.equal(buildIdentityProfile('Highlands Coffee Nguyễn Văn Linh').baseName, 'highlands');
  assert.equal(buildIdentityProfile('Highlands Coffee Nguyễn Văn Linh').branchName, 'nguyen van linh');
  assert.equal(hasBranchConflict(
    'Highlands Coffee Nguyễn Văn Linh',
    'Highlands Coffee Bạch Đằng',
  ), true);
});

test('Stage 4H keeps chain branches and same-building incompatible businesses separate', () => {
  const chain = classifySourceRecordHardened(
    source({ name: 'Highlands Coffee Nguyễn Văn Linh' }),
    [canonical({ name: 'Highlands Coffee Bạch Đằng' })],
    undefined,
    { hardened: true },
  );
  assert.equal(chain.decision, DECISIONS.NEW_CANDIDATE);
  assert.ok(chain.reasonCodes.includes('branch_location_conflict'));

  const categoryCollision = classifySourceRecordHardened(
    source({ name: 'Sunrise', category: 'cafe' }),
    [canonical({ name: 'Sunrise', category: 'hotel' })],
    undefined,
    { hardened: true },
  );
  assert.equal(categoryCollision.decision, DECISIONS.NEW_CANDIDATE);
  assert.ok(categoryCollision.reasonCodes.includes('incompatible_category'));
});

test('Stage 4H recovers exact-name threshold edges but rejects distant same names', () => {
  const thresholdEdge = classifySourceRecordHardened(
    source({ latitude: 16.05, longitude: 108.2 }),
    [canonical({ latitude: 16.05162, longitude: 108.2 })],
    undefined,
    { hardened: true },
  );
  assert.equal(thresholdEdge.decision, DECISIONS.PROBABLE_MATCH);
  assert.ok(thresholdEdge.reasonCodes.includes('exact_name_threshold_edge'));

  const distant = classifySourceRecordHardened(
    source(),
    [canonical({ latitude: 16.06, longitude: 108.2 })],
    undefined,
    { hardened: true },
  );
  assert.equal(distant.decision, DECISIONS.NEW_CANDIDATE);
});

test('Stage 4H fingerprints ignore retrieval timestamps and isolate identity, media, provenance changes', () => {
  const first = stageFingerprints(source());
  const timestampOnly = stageFingerprints(source({
    provenance: { snapshotRef: 'A', retrievedAt: '2026-08-02T00:00:00Z' },
  }));
  assert.equal(first.identityHash, timestampOnly.identityHash);
  assert.equal(first.mediaHash, timestampOnly.mediaHash);

  const mediaChange = stageFingerprints(source({
    media: { mediaId: 'm1', url: 'https://example.test/a.jpg', contentHash: 'changed' },
  }));
  assert.equal(first.identityHash, mediaChange.identityHash);
  assert.notEqual(first.mediaHash, mediaChange.mediaHash);

  const identityChange = stageFingerprints(source({ name: 'Cà phê Mộc 2' }));
  assert.notEqual(first.identityHash, identityChange.identityHash);
});

test('Stage 4H incremental replay processes only required stages and marks missing records for review', () => {
  const snapshotA = [
    source({ sourceId: 'a' }),
    source({ sourceId: 'b' }),
    source({ sourceId: 'c' }),
    source({ sourceId: 'd', media: { mediaId: 'm1', url: 'https://example.test/a.jpg', contentHash: 'h1' } }),
  ];
  const resolveRecord = (record) => ({
    decision: DECISIONS.PROBABLE_MATCH,
    bestCandidate: { canonicalPoiId: `canonical:${record.sourceId}` },
  });
  const first = runIncrementalSync({ snapshotId: 'A', records: snapshotA, resolveRecord });
  assert.equal(first.metrics.processed, 4);
  assert.equal(first.metrics.resolutionProcessed, 4);

  const snapshotB = [
    source({ sourceId: 'a', provenance: { snapshotRef: 'B', retrievedAt: 'later' } }),
    source({ sourceId: 'b', name: 'Cà phê Mộc mới' }),
    source({ sourceId: 'd', media: { mediaId: 'm1', url: 'https://example.test/a.jpg', contentHash: 'h2' } }),
    source({ sourceId: 'e' }),
  ];
  const second = runIncrementalSync({
    snapshotId: 'B',
    records: snapshotB,
    previousState: first.state,
    previousMediaRegistry: first.mediaRegistry,
    resolveRecord,
  });
  const statuses = Object.fromEntries(second.state.map((item) => [item.sourceId, item.status]));
  assert.deepEqual(statuses, {
    a: STATUSES.CHANGED,
    b: STATUSES.CHANGED,
    c: STATUSES.MISSING,
    d: STATUSES.CHANGED,
    e: STATUSES.NEW,
  });
  assert.equal(second.metrics.resolutionProcessed, 2);
  assert.equal(second.metrics.mediaInvalidated, 1);
  assert.equal(second.reviewCandidates[0].autoDeleteAuthorized, false);

  const repeated = runIncrementalSync({
    snapshotId: 'B',
    records: snapshotB,
    previousState: second.state,
    previousMediaRegistry: second.mediaRegistry,
    resolveRecord,
  });
  assert.equal(repeated.metrics.processed, 0);
  assert.equal(repeated.metrics.skipped, 4);
  assert.equal(repeated.metrics.resolutionProcessed, 0);
  assert.equal(repeated.metrics.mediaProcessed, 0);
  assert.equal(repeated.metrics.provenanceProcessed, 0);
  assert.equal(repeated.metrics.mediaReused, 1);
  const repeatedAgain = runIncrementalSync({
    snapshotId: 'B',
    records: snapshotB,
    previousState: second.state,
    previousMediaRegistry: second.mediaRegistry,
    resolveRecord,
  });
  assert.equal(repeated.deterministicHash, repeatedAgain.deterministicHash);
});

test('Stage 4H invalid records never enter entity resolution', () => {
  let resolutions = 0;
  const result = runIncrementalSync({
    snapshotId: 'A',
    records: [source({ sourceId: 'invalid', latitude: null })],
    resolveRecord: () => { resolutions += 1; },
  });
  assert.equal(result.state[0].status, STATUSES.INVALID);
  assert.equal(result.metrics.invalid, 1);
  assert.equal(resolutions, 0);
});

test('Stage 4H checkpoint resumes deterministically instead of restarting', () => {
  const records = ['a', 'b', 'c', 'd'].map((sourceId) => source({ sourceId }));
  const resolveRecord = (record) => ({ decision: DECISIONS.NEW_CANDIDATE, sourceId: record.sourceId });
  const direct = runIncrementalSync({ snapshotId: 'A', records, resolveRecord });
  const stopped = runIncrementalSync({ snapshotId: 'A', records, resolveRecord, stopAfter: 2 });
  assert.equal(stopped.complete, false);
  assert.equal(stopped.checkpoint.completedKeys.length, 2);
  const resumed = runIncrementalSync({
    snapshotId: 'A',
    records,
    resolveRecord,
    checkpoint: stopped.checkpoint,
  });
  assert.equal(resumed.complete, true);
  assert.deepEqual(resumed.state, direct.state);
  assert.deepEqual(resumed.metrics, direct.metrics);
});

test('Stage 4H processing-version changes invalidate only declared cache stages', () => {
  const first = runIncrementalSync({ snapshotId: 'A', records: [source()], resolveRecord: () => null });
  const changed = runIncrementalSync({
    snapshotId: 'B',
    records: [source()],
    previousState: first.state,
    previousMediaRegistry: first.mediaRegistry,
    versions: { multimodalVersion: 'stage4h-cache-v2' },
    resolveRecord: () => null,
  });
  assert.equal(changed.metrics.resolutionProcessed, 0);
  assert.equal(changed.metrics.mediaProcessed, 1);
  assert.equal(changed.state[0].processing.identity, false);
  assert.equal(changed.state[0].processing.media, true);
});

test('Stage 4H source capability model is explicit and conservative', () => {
  assert.deepEqual(DEFAULT_SOURCE_CAPABILITIES, {
    overture: SOURCE_CAPABILITIES.SNAPSHOT,
    osm: SOURCE_CAPABILITIES.POLLING,
    wikidata: SOURCE_CAPABILITIES.POLLING,
    wikimedia_commons: SOURCE_CAPABILITIES.POLLING,
  });
});

test('Stage 4H leaves the canonical baseline immutable', () => {
  const before = inspectCanonicalDataset(CANONICAL_PATH);
  assert.equal(before.rows, 4166);
  assert.equal(before.sha256, EXPECTED_CANONICAL_SHA);
  assert.equal(fs.existsSync(CANONICAL_PATH), true);
});

test.after(() => {
  const accidental = path.join(os.tmpdir(), 'urbanagent-stage4h-unused');
  if (fs.existsSync(accidental)) fs.rmSync(accidental, { recursive: true, force: true });
});
