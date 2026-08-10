const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');

const { inspectCanonicalDataset, EXPECTED_CANONICAL_SHA } = require('../../src/modules/cityPackPreparation/canonicalDataset');
const { loadRealSnapshotRecords } = require('../../src/modules/cityPackPreparation/realSnapshot');
const { normalizeCategory } = require('../../src/modules/cityPackPreparation/sourceRecord');
const { runStage4eValidation } = require('../../scripts/phase4_stage4e_real_source_validation');

const ROOT = path.resolve(__dirname, '..', '..');
const SNAPSHOT_DIR = path.join(ROOT, 'data', 'spikes', 'phase4', 'stage4e', 'snapshots');
const CANONICAL_PATH = path.join(ROOT, 'data', 'canonical', 'urbanagent_poi_master_v1.csv');

function fileHash(filePath) {
  return crypto.createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');
}

const canonicalHashBeforeValidation = fileHash(CANONICAL_PATH);
const stage4e = runStage4eValidation();

test('Stage 4E maps real source schemas into the Stage 4C adapter contract', () => {
  const records = loadRealSnapshotRecords({
    overture: path.join(SNAPSHOT_DIR, 'overture_places_2026-07-22.0_bounded.json'),
    osm: path.join(SNAPSHOT_DIR, 'osm_2026-08-10_bounded.json'),
    wikidata: path.join(SNAPSHOT_DIR, 'wikidata_2026-08-10_bounded.json'),
  });

  assert.equal(records.length, 76);
  assert.equal(records.filter((record) => record.source === 'overture').length, 31);
  assert.equal(records.filter((record) => record.source === 'osm').length, 33);
  assert.equal(records.filter((record) => record.source === 'wikidata').length, 12);
  assert.ok(records.every((record) => record.snapshotRef && record.license?.policyClass));
});

test('Stage 4E fixes real Vietnamese and provider category semantics', () => {
  assert.equal(normalizeCategory('Quán cà phê'), 'cafe');
  assert.equal(normalizeCategory('Nhà hàng Việt Nam'), 'restaurant');
  assert.equal(normalizeCategory('Khách sạn'), 'accommodation');
  assert.equal(normalizeCategory('coffee_shop'), 'cafe');
  assert.equal(normalizeCategory('landmark_and_historical_building'), 'attraction');
});

test('Stage 4E produces bounded resolver metrics and a review queue', () => {
  const { summary } = stage4e;

  assert.deepEqual(summary.matchSummary, {
    highConfidenceMatches: 6,
    probableMatches: 3,
    ambiguous: 1,
    newCandidates: 66,
    invalid: 0,
    sourceDuplicates: 3,
  });
  assert.equal(summary.reviewQueueCount, 70);
  assert.equal(summary.candidateCount, 9);
});

test('Stage 4E keeps record, field, media, and license provenance complete', () => {
  const { summary } = stage4e;

  assert.equal(summary.provenance.completeRecordProvenanceRate, 1);
  assert.equal(summary.provenance.completeFieldProvenanceRate, 1);
  assert.equal(summary.provenance.completePerFileMediaLicenseRate, 1);
  assert.equal(summary.googlePlacesIncluded, false);
  assert.equal(summary.candidateBuild.status, 'CANDIDATE_NON_RUNTIME_NOT_CANONICAL');
});

test('Stage 4E fixed-snapshot candidate build is deterministic', () => {
  const { summary } = stage4e;

  assert.equal(summary.deterministic, true);
  assert.equal(summary.candidateBuild.matchedEnriched, 9);
  assert.equal(summary.candidateBuild.approvedNewCandidates, 0);
  assert.equal(summary.candidateBuild.runtimeChanged, false);
});

test('Stage 4E leaves canonical data unchanged', () => {
  const after = fileHash(CANONICAL_PATH);
  const canonical = inspectCanonicalDataset(CANONICAL_PATH);

  assert.equal(canonicalHashBeforeValidation, after);
  assert.equal(canonical.rows, 4166);
  assert.equal(canonical.sha256, EXPECTED_CANONICAL_SHA);
});
