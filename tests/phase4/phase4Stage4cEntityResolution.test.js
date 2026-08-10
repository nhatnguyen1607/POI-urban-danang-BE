const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const overtureAdapter = require('../../src/modules/cityPackPreparation/adapters/overtureAdapter');
const osmAdapter = require('../../src/modules/cityPackPreparation/adapters/osmAdapter');
const wikidataWikimediaAdapter = require('../../src/modules/cityPackPreparation/adapters/wikidataWikimediaAdapter');
const { EXPECTED_CANONICAL_SHA, inspectCanonicalDataset, readCanonicalPois } = require('../../src/modules/cityPackPreparation/canonicalDataset');
const { DECISIONS, resolveSourceRecords } = require('../../src/modules/cityPackPreparation/entityResolution');
const { buildReviewQueue } = require('../../src/modules/cityPackPreparation/reviewQueue');
const { readStage4bFixture, runStage4cDryRun } = require('../../src/modules/cityPackPreparation/dryRun');

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

function fileSha(filePath) {
  return crypto.createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');
}

test('Stage 4C adapters normalize bounded fixtures and retain provenance', () => {
  const rawRecords = readStage4bFixture(SAMPLE_PATH);
  const overtureRecords = overtureAdapter.normalize(rawRecords);
  const osmRecords = osmAdapter.normalize(rawRecords);
  const wikiRecords = wikidataWikimediaAdapter.normalize(rawRecords);

  assert.equal(overtureRecords.length, 8);
  assert.equal(osmRecords.length, 5);
  assert.equal(wikiRecords.length, 4);

  const overtureRecord = overtureRecords.find((record) => record.sourceId === 'overture:stage4b:thanh-tam-bakery');
  assert.equal(overtureRecord.provenance.source, 'overture');
  assert.equal(overtureRecord.provenance.policyClass, 'OPEN_PERMISSIVE_CANDIDATE');
  assert.equal(overtureRecord.provenance.fields.openingHours.sourceId, overtureRecord.sourceId);

  const osmRecord = osmRecords[0];
  assert.equal(osmRecord.provenance.policyClass, 'OPEN_SHAREALIKE_ISOLATED');

  const commonsRecord = wikiRecords.find((record) => record.source === 'wikimedia_commons');
  assert.equal(commonsRecord.provenance.policyClass, 'OPEN_MEDIA_ATTRIBUTION_REQUIRED');
  assert.ok(commonsRecord.externalIds.commons_file);
});

test('Stage 4C entity resolution classifies conservative match outcomes', () => {
  const rawRecords = readStage4bFixture(SAMPLE_PATH);
  const sourceRecords = [
    ...overtureAdapter.normalize(rawRecords),
    ...osmAdapter.normalize(rawRecords),
    ...wikidataWikimediaAdapter.normalize(rawRecords),
  ];
  const canonicalPois = readCanonicalPois(CANONICAL_PATH);
  const result = resolveSourceRecords(sourceRecords, canonicalPois);

  assert.equal(result.summary.highConfidenceMatches, 3);
  assert.equal(result.summary.probableMatches, 1);
  assert.equal(result.summary.ambiguous, 3);
  assert.equal(result.summary.newCandidates, 9);
  assert.equal(result.summary.invalid, 1);
  assert.equal(result.summary.sourceDuplicates, 1);

  const probable = result.matches.find((match) => match.decision === DECISIONS.PROBABLE_MATCH);
  assert.ok(probable.reasonCodes.includes('usable_name_similarity'));

  const ambiguous = result.matches.find((match) => match.decision === DECISIONS.AMBIGUOUS);
  assert.ok(ambiguous.reasonCodes.some((code) => code.includes('collision') || code.includes('multiple')));

  const invalid = result.matches.find((match) => match.decision === DECISIONS.INVALID);
  assert.deepEqual(invalid.reasonCodes, ['invalid_or_missing_coordinates']);
});

test('Stage 4C review queue captures ambiguous, new, invalid, and source duplicate records', () => {
  const rawRecords = readStage4bFixture(SAMPLE_PATH);
  const sourceRecords = [
    ...overtureAdapter.normalize(rawRecords),
    ...osmAdapter.normalize(rawRecords),
    ...wikidataWikimediaAdapter.normalize(rawRecords),
  ];
  const result = resolveSourceRecords(sourceRecords, readCanonicalPois(CANONICAL_PATH));
  const queue = buildReviewQueue(result);

  assert.equal(queue.length, 14);
  assert.ok(queue.some((item) => item.decision === DECISIONS.AMBIGUOUS));
  assert.ok(queue.some((item) => item.decision === DECISIONS.NEW_CANDIDATE));
  assert.ok(queue.some((item) => item.decision === DECISIONS.INVALID));
  assert.ok(queue.some((item) => item.decision === DECISIONS.SOURCE_DUPLICATE));
  assert.ok(queue.every((item) => item.suggestedActions.length > 0));
  assert.ok(queue.every((item) => item.license));
});

test('Stage 4C dry-run output is deterministic for fixed fixtures', () => {
  const outputA = fs.mkdtempSync(path.join(os.tmpdir(), 'urbanagent-stage4c-a-'));
  const outputB = fs.mkdtempSync(path.join(os.tmpdir(), 'urbanagent-stage4c-b-'));

  const first = runStage4cDryRun({
    canonicalPath: CANONICAL_PATH,
    samplePath: SAMPLE_PATH,
    outputDir: outputA,
  });
  const second = runStage4cDryRun({
    canonicalPath: CANONICAL_PATH,
    samplePath: SAMPLE_PATH,
    outputDir: outputB,
  });

  assert.deepEqual(first.summary, second.summary);
  assert.equal(
    fileSha(path.join(outputA, 'stage4c_review_queue.json')),
    fileSha(path.join(outputB, 'stage4c_review_queue.json')),
  );
});

test('Stage 4C keeps canonical dataset integrity unchanged', () => {
  const canonical = inspectCanonicalDataset(CANONICAL_PATH);

  assert.equal(canonical.rows, 4166);
  assert.equal(canonical.sha256, EXPECTED_CANONICAL_SHA);
  assert.equal(canonical.shaMatchesExpected, true);
});
