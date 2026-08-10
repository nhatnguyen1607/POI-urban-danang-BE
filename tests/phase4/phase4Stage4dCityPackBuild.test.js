const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const { inspectCanonicalDataset, EXPECTED_CANONICAL_SHA } = require('../../src/modules/cityPackPreparation/canonicalDataset');
const { buildCandidateCityPack } = require('../../src/modules/cityPackPreparation/cityPackBuild/cityPackBuilder');

const ROOT = path.resolve(__dirname, '..', '..');
const CANONICAL_PATH = path.join(ROOT, 'data', 'canonical', 'urbanagent_poi_master_v1.csv');
const SAMPLE_PATH = path.join(
  ROOT,
  'data',
  'spikes',
  'phase4',
  'stage4b',
  'source_samples',
  'phase4_stage4b_source_samples.json',
);
const DECISIONS_PATH = path.join(
  ROOT,
  'data',
  'spikes',
  'phase4',
  'stage4d',
  'review_decisions',
  'stage4d_review_decisions.fixture.json',
);

function tempDir(prefix) {
  return fs.mkdtempSync(path.join(os.tmpdir(), prefix));
}

function writeDecisionFile(document) {
  const dir = tempDir('urbanagent-stage4d-decisions-');
  const filePath = path.join(dir, 'decisions.json');
  fs.writeFileSync(filePath, `${JSON.stringify(document)}\n`);
  return filePath;
}

function runBuild({ reviewDecisionPath = DECISIONS_PATH, buildId = 'stage4d-test-build' } = {}) {
  return buildCandidateCityPack({
    cityId: 'da-nang',
    buildId,
    canonicalPath: CANONICAL_PATH,
    samplePath: SAMPLE_PATH,
    reviewDecisionPath,
    outputDir: tempDir('urbanagent-stage4d-output-'),
  });
}

function fileHash(filePath) {
  return crypto.createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');
}

test('Stage 4D builds a valid non-runtime candidate City Pack', () => {
  const result = runBuild();

  assert.equal(result.summary.status, 'CANDIDATE_NON_RUNTIME_NOT_CANONICAL');
  assert.equal(result.summary.canonicalBaselineCount, 4166);
  assert.equal(result.summary.canonicalSha256, EXPECTED_CANONICAL_SHA);
  assert.equal(result.summary.matchedEnriched, 5);
  assert.equal(result.summary.approvedNewCandidates, 1);
  assert.equal(result.summary.rejected, 1);
  assert.equal(result.summary.deferred, 3);
  assert.equal(result.summary.unresolved, 7);
  assert.equal(result.summary.sourceDuplicateGroups, 1);
  assert.equal(result.summary.candidateTotal, 6);
  assert.equal(result.summary.runtimeChanged, false);
});

test('Stage 4D rejects invalid review decisions', () => {
  const reviewDecisionPath = writeDecisionFile({
    decisions: [
      {
        queueId: 'UNKNOWN:overture:overture:stage4b:missing',
        action: 'CREATE_NEW',
      },
    ],
  });

  assert.throws(
    () => runBuild({ reviewDecisionPath }),
    (error) => error.validation.errors.some((item) => item.code === 'UNKNOWN_QUEUE_ID'),
  );
});

test('Stage 4D rejects conflicting duplicate review decisions', () => {
  const duplicated = {
    queueId: 'NEW_CANDIDATE:overture:overture:stage4b:dragon-bridge',
    action: 'CREATE_NEW',
  };
  const reviewDecisionPath = writeDecisionFile({
    decisions: [duplicated, duplicated],
  });

  assert.throws(
    () => runBuild({ reviewDecisionPath }),
    (error) => error.validation.errors.some((item) => item.code === 'DUPLICATE_OR_CONFLICTING_DECISION'),
  );
});

test('Stage 4D rejects unsafe ambiguous acceptance with invalid canonical ID', () => {
  const reviewDecisionPath = writeDecisionFile({
    decisions: [
      {
        queueId: 'AMBIGUOUS:overture:overture:stage4b:anh-quan-bakery-a',
        action: 'ACCEPT_MATCH',
        canonicalPoiId: 'not_a_real_canonical_id',
      },
    ],
  });

  assert.throws(
    () => runBuild({ reviewDecisionPath }),
    (error) => error.validation.errors.some((item) => item.code === 'INVALID_CANONICAL_ID'),
  );
});

test('Stage 4D excludes unresolved ambiguous records from automatic application', () => {
  const result = runBuild();
  const deferredQueueIds = new Set(result.pack.deferredReviewItems.map((item) => item.queueId));

  assert.ok(deferredQueueIds.has('AMBIGUOUS:overture:overture:stage4b:anh-quan-bakery-b'));
  assert.ok(deferredQueueIds.has('AMBIGUOUS:osm:osm:node:stage4b:anh-quan'));
  assert.equal(
    result.pack.matchedEnrichment.some((record) => record.sourceId === 'overture:stage4b:anh-quan-bakery-b'),
    false,
  );
});

test('Stage 4D applies approved match enrichment only to candidate output', () => {
  const result = runBuild();
  const accepted = result.pack.matchedEnrichment.find(
    (record) => record.sourceId === 'overture:stage4b:anh-quan-bakery-a',
  );

  assert.equal(accepted.canonicalPoiId, 'google_maps_553');
  assert.equal(accepted.decisionSource, 'review_decision_accept_match');
  assert.equal(accepted.provenance.policyClass, 'OPEN_PERMISSIVE_CANDIDATE');
});

test('Stage 4D applies approved CREATE_NEW only to candidate output', () => {
  const result = runBuild();
  const created = result.pack.approvedNewCandidates.find(
    (record) => record.sourceId === 'overture:stage4b:dragon-bridge',
  );

  assert.ok(created.candidateId.startsWith('candidate:da-nang:'));
  assert.equal(created.status, 'CANDIDATE_NON_RUNTIME_NOT_CANONICAL');
  assert.equal(created.decisionSource, 'review_decision_create_new');
});

test('Stage 4D handles source duplicate review groups without canonical merge', () => {
  const result = runBuild();

  assert.equal(result.pack.sourceDuplicateGroups.length, 1);
  assert.equal(result.pack.sourceDuplicateGroups[0].decisionSource, 'review_decision_merge_source_duplicate');
  assert.equal(result.pack.sourceDuplicateGroups[0].records.length, 2);
});

test('Stage 4D writes provenance and license manifest', () => {
  const result = runBuild();

  assert.equal(result.licenseManifest.status, 'CANDIDATE_NON_RUNTIME_NOT_CANONICAL');
  assert.equal(result.licenseManifest.googlePlacesIncluded, false);
  assert.ok(result.licenseManifest.sources.some((source) => source.policyClass === 'OPEN_SHAREALIKE_ISOLATED'));
  assert.ok(result.licenseManifest.sources.every((source) => source.attribution !== undefined));
});

test('Stage 4D build is deterministic for fixed inputs', () => {
  const first = runBuild({ buildId: 'stage4d-deterministic' });
  const second = runBuild({ buildId: 'stage4d-deterministic' });

  assert.deepEqual(first.summary, second.summary);
  assert.equal(first.summary.artifactHashes.candidatePackHash, second.summary.artifactHashes.candidatePackHash);
  assert.equal(first.summary.artifactHashes.licenseManifestHash, second.summary.artifactHashes.licenseManifestHash);
});

test('Stage 4D keeps canonical dataset integrity unchanged', () => {
  const before = fileHash(CANONICAL_PATH);
  runBuild();
  const after = fileHash(CANONICAL_PATH);
  const canonical = inspectCanonicalDataset(CANONICAL_PATH);

  assert.equal(before, after);
  assert.equal(canonical.rows, 4166);
  assert.equal(canonical.sha256, EXPECTED_CANONICAL_SHA);
});
