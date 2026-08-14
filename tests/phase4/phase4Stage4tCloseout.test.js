const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');

const config = require('../../config/phase4_stage4t_closeout.json');
const stage4l = require('../../config/phase4_stage4l_policy.json');
const stage4m = require('../../config/phase4_stage4m_auto_pr.json');
const stage4n = require('../../config/phase4_stage4n_scheduled_sync.json');
const stage4o = require('../../config/phase4_stage4o_operations.json');
const stage4p = require('../../config/phase4_stage4p_create_new_policy.json');
const stage4q = require('../../config/phase4_stage4q_create_new_evidence.json');
const stage4s = require('../../config/phase4_stage4s_controlled_create_new.json');
const { CanonicalCsvPoiRepository } = require('../../src/services/canonicalCsvPoiRepository');
const { getTravelerRecommendations } = require('../../src/modules/travelerApiV2/recommendations');
const { buildTripPreview } = require('../../src/modules/travelerApiV2/tripPreview');
const { validateTripPreviewRequest } = require('../../src/modules/travelerApiV2/tripPreviewValidation');
const {
  buildCloseoutStatus,
  replayAppliedSources,
  sha256File,
  validateCanonicalBaseline,
  validateSchedulerAlignment,
  verifyAutomationSafety,
  verifyDecisionMemory,
  verifySidecarLinkage,
} = require('../../src/modules/cityPackPreparation/phase4Closeout');
const { readCsvObjects } = require('../../scripts/phase4_stage4k_controlled_canary_apply');
const { readCanonicalEvidencePois } = require('../../scripts/phase4_stage4p_create_new_policy_research');
const { readHistoricalPois } = require('../../scripts/phase4_stage4q_create_new_evidence');
const { loadTargetRecords } = require('../../scripts/phase4_stage4r_create_new_canary_review');

const ROOT = path.resolve(__dirname, '..', '..');
const CANONICAL = path.join(ROOT, 'data', 'canonical', 'urbanagent_poi_master_v1.csv');
const SIDECAR = path.join(ROOT, 'data', 'citypacks', 'enrichments', 'danang',
  'stage4l-enrichment-v1.json');
const BACKUP = 'D:\\UrbanAgent-artifacts\\stage4s-create-new\\stage4s-20260813-204149\\backup\\urbanagent_poi_master_v1.before_stage4s.csv';
const MEMORY = 'D:\\UrbanAgent-artifacts\\stage4s-create-new\\stage4s-20260813-204149\\decision_memory.json';
const PACK = 'D:\\UrbanAgent-spikes\\phase4-stage4g-20260810\\candidate\\candidate_city_pack.json';
const HISTORICAL = [
  'D:\\POI-urban-danang-BE\\data\\raw\\legacy\\poi_urban_ggmap.csv',
  'D:\\POI-urban-danang-BE\\data\\raw\\legacy\\poi_urban_foody.csv',
];
const rows = readCsvObjects(CANONICAL);
const baseline = validateCanonicalBaseline({
  canonicalPath: CANONICAL, baselineBackupPath: null, rows, config,
});
const hasHistoryBackup = fs.existsSync(BACKUP);
const hasDecisionMemory = fs.existsSync(MEMORY);
const hasReplayEvidence = fs.existsSync(PACK) && HISTORICAL.every(fs.existsSync);
const historyTest = hasHistoryBackup ? test : test.skip;
const memoryTest = hasDecisionMemory ? test : test.skip;
const replayTest = hasReplayEvidence ? test : test.skip;
const memory = hasDecisionMemory ? verifyDecisionMemory({
  memory: JSON.parse(fs.readFileSync(MEMORY, 'utf8')),
  authoritativeDecisions: stage4s.authoritativeHumanDecisions,
  config,
}) : null;
let replayPromise;
function loadReplay() {
  if (!replayPromise) replayPromise = (async () => {
    const sourceRecords = await loadTargetRecords(PACK, new Set(config.canary.map((item) => item.caseId)));
    return replayAppliedSources({
      sourceRecords,
      canonicalPois: readCanonicalEvidencePois(CANONICAL),
      historicalPois: readHistoricalPois(HISTORICAL),
      thresholds: stage4q.thresholds,
      config,
    });
  })();
  return replayPromise;
}

test('Stage 4T promotes the exact 4173-row post-canary baseline', () => {
  assert.equal(baseline.rows, 4173);
  assert.equal(baseline.sha256, config.officialBaseline.sha256);
});

test('Stage 4T resolves exactly seven new canonical entities', () => {
  assert.equal(baseline.entities.length, 7);
  assert.deepEqual(new Set(baseline.entities.map((item) => item.canonicalId)),
    new Set(config.canary.map((item) => item.canonicalId)));
});

historyTest('Stage 4T preserves the original 4166 canonical bytes', () => {
  const historicalBaseline = validateCanonicalBaseline({
    canonicalPath: CANONICAL, baselineBackupPath: BACKUP, rows, config,
  });
  assert.equal(historicalBaseline.oldRowsPreserved, true);
  assert.equal(historicalBaseline.oldRowsDeleted, 0);
  assert.equal(historicalBaseline.oldRowsMutated, 0);
});

test('Stage 4T canary core fields are complete and valid', () => {
  assert.ok(baseline.entities.every((item) => item.name && item.category && item.address
    && Number.isFinite(item.latitude) && Number.isFinite(item.longitude)
    && item.sourceLineageReference));
});

test('Stage 4T canonical IDs remain unique', () => {
  assert.equal(baseline.uniqueIds, 4173);
});

replayTest('Stage 4T seven-source replay resolves every source to its canonical entity', async () => {
  const replay = await loadReplay();
  assert.equal(replay.recognizedExisting, 7);
  assert.ok(replay.results.every((item) => item.exactMatch));
});

replayTest('Stage 4T seven-source replay proposes zero CREATE_NEW', async () => {
  assert.equal((await loadReplay()).createNewProposals, 0);
});

replayTest('Stage 4T seven-source replay repeats zero human CREATE_NEW reviews', async () => {
  assert.equal((await loadReplay()).repeatHumanCreateNewReview, 0);
});

replayTest('Stage 4T post-merge duplicate guard finds no unresolved canary pair', async () => {
  assert.equal((await loadReplay()).duplicateProposals, 0);
});

memoryTest('Stage 4T decision memory retains seven applied human approvals', () => {
  assert.equal(memory.appliedHumanApprovals, 7);
  assert.equal(memory.generalAutoCreateAuthorization, false);
});

memoryTest('Stage 4T decision memory retains four reusable unchanged rejects', () => {
  assert.equal(memory.reusableRejects, 4);
});

memoryTest('Stage 4T decision memory suppresses fifteen unchanged defers', () => {
  assert.equal(memory.suppressedUnchangedDefers, 15);
});

test('Stage 4T sidecar links nineteen unique records without orphans', () => {
  const result = verifySidecarLinkage({
    sidecarPath: SIDECAR, canonicalIds: new Set(rows.map((row) => row.Global_ID)), config,
  });
  assert.deepEqual([result.records, result.orphanReferences, result.duplicateIdentities], [19, 0, 0]);
});

test('Stage 4T sidecar provenance, license and deterministic serialization pass', () => {
  const result = verifySidecarLinkage({
    sidecarPath: SIDECAR, canonicalIds: new Set(rows.map((row) => row.Global_ID)), config,
  });
  assert.deepEqual([result.provenance, result.license, result.deterministic], ['PASS', 'PASS', true]);
});

test('Stage 4T baseline promotion is aligned across Stage 4M, 4N and 4O', () => {
  assert.deepEqual([stage4m.expectedCanonicalRows, stage4n.expectedCanonicalRows,
    stage4o.expectedCanonicalRows], [4173, 4173, 4173]);
  assert.equal(require('../../src/modules/cityPackPreparation/canonicalDataset').EXPECTED_CANONICAL_SHA,
    config.officialBaseline.sha256);
});

test('Stage 4T scheduler alignment validator requires main commit, enabled task and self-test', () => {
  assert.equal(validateSchedulerAlignment({ operationalCommit: 'abc', expectedCommit: 'abc',
    enabled: true, selfTest: { ok: true } }).aligned, true);
  assert.throws(() => validateSchedulerAlignment({ operationalCommit: 'old', expectedCommit: 'new',
    enabled: true, selfTest: { ok: true } }), /scheduler operational commit/);
});

test('Stage 4T no-change validation leaves canonical and sidecar bytes unchanged', async () => {
  const before = [sha256File(CANONICAL), sha256File(SIDECAR)];
  await loadReplay();
  assert.deepEqual([sha256File(CANONICAL), sha256File(SIDECAR)], before);
});

test('Stage 4T final automation policy keeps destructive and automatic creation disabled', () => {
  const safety = verifyAutomationSafety({ stage4l, stage4m, stage4n, stage4o, stage4s });
  assert.deepEqual([safety.autoCreateNew, safety.automaticDelete, safety.automaticMerge,
    safety.prAutoMerge], [false, false, false, false]);
});

test('Stage 4T runtime loader exposes 4173 unique valid POIs', async () => {
  const repository = new CanonicalCsvPoiRepository({ filePath: CANONICAL });
  const pois = await repository.loadAll();
  const quality = await repository.getQualityReport();
  assert.deepEqual([pois.length, new Set(pois.map((poi) => poi.id)).size,
    quality.totals.invalidRows], [4173, 4173, 0]);
});

replayTest('Stage 4T recommendation, trip preview and closeout artifact contracts pass', async () => {
  const recommendation = await getTravelerRecommendations({
    query: 'cafe va diem tham quan o Da Nang', context: {}, limit: 5, cityId: 'da-nang',
  });
  const validation = validateTripPreviewRequest({
    cityId: 'da-nang', query: 'cafe va diem tham quan o Da Nang',
    trip: { dayCount: 1, date: '2026-08-20', transport: 'motorbike', pace: 'balanced',
      dailyWindow: { start: '08:00', end: '18:00' } },
    startLocation: { lat: 16.0544, lon: 108.2022 },
    constraints: { maxStopsPerDay: 3 }, recommendationOptions: { limit: 8 },
  });
  const preview = await buildTripPreview(validation.value);
  assert.ok(recommendation.recommendations.length > 0 && preview.trip.stops.length > 0);
  const closeout = buildCloseoutStatus({ mainCommit: 'main', canonical: baseline,
    scheduler: { operationalCommit: 'main', enabled: true }, replay: await loadReplay(),
    duplicateResult: { unresolved: 0 }, decisionMemory: memory,
    runtime: { loader: { applicationPois: 4173 } }, soak: { status: 'IN_PROGRESS' },
    tests: { passed: true }, artifactReferences: {} });
  assert.equal(closeout.status, 'PHASE_4_IMPLEMENTATION_CLOSED');
  assert.equal(closeout.autoCreateNew, false);
  assert.equal(stage4p.soakGate.requireDueSourceCheck, true);
});
