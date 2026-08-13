const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const config = require('../../config/phase4_stage4p_create_new_policy.json');
const {
  CREATE_NEW_STATES,
  assertResearchOnly,
  branchAssessment,
  buildCanaryReviewSample,
  createResearchDecisionRecord,
  evaluateCandidate,
  evaluateCreateNewResearch,
  findReusableResearchDecision,
  prepareCandidateRecord,
  provenanceAssessment,
} = require('../../src/modules/cityPackPreparation/createNewResearchPolicy');
const {
  aggregateOperationalSoak,
  readOperationalManifests,
} = require('../../src/modules/cityPackPreparation/operationalSoakEvaluator');
const { DECISIONS } = require('../../src/modules/cityPackPreparation/entityResolution');
const { NEW_TIERS } = require('../../src/modules/cityPackPreparation/reviewApplyPreparation');
const { inspectCanonicalDataset } = require('../../src/modules/cityPackPreparation/canonicalDataset');
const { writeCanaryCsv } = require('../../scripts/phase4_stage4p_create_new_policy_research');

const ROOT = path.resolve(__dirname, '..', '..');
const CANONICAL_SHA = '39647b29308813a7ec19e4695fd2b95ffa27743db8e8cb46800c45d3a3fe6ded';
const NOW = '2026-08-13T07:00:00.000Z';

function record(overrides = {}) {
  return prepareCandidateRecord({
    source: 'overture', sourceId: 'overture:one', name: 'Bao tang du lich Da Nang',
    category: 'museum', latitude: 16.06, longitude: 108.22,
    address: '12 Bach Dang', phone: '+842361234567', website: 'https://example.vn',
    externalIds: { overture_id: 'one' },
    provenance: {
      source: 'overture', sourceId: 'overture:one', snapshotRef: 'snapshot-1',
      policyClass: 'OPEN_PERMISSIVE_CANDIDATE', license: 'CDLA-Permissive-2.0',
      attribution: 'Overture Maps Foundation',
      upstreamSources: [{ dataset: 'Overture', updateTime: '2026-07-14T00:00:00Z', confidence: 0.95 }],
    },
    ...overrides,
  });
}

function match(overrides = {}) {
  return {
    source: 'overture', sourceId: 'overture:one', decision: DECISIONS.NEW_CANDIDATE,
    bestCandidate: { canonicalPoiId: 'poi-1', canonicalName: 'Khac', distanceMeters: 800,
      nameSimilarity: 0.1 },
    ...overrides,
  };
}

function evaluate(overrides = {}) {
  return evaluateCandidate({
    record: overrides.record || record(),
    match: overrides.match || match(),
    triage: overrides.triage || { tier: NEW_TIERS.A, categoryEligibility: 'TRAVELER_RELEVANT',
      reason: 'strong_review_candidate_not_auto_create' },
    duplicateContext: overrides.duplicateContext || { representative: 'overture:one', duplicateRisk: 0,
      supportSources: ['osm', 'overture'], independentSupport: true, componentSize: 2 },
    config,
    now: NOW,
  });
}

test('soak aggregation distinguishes scheduled from manual runs and remains in progress', () => {
  const manifests = [
    { runId: 'manual', triggerType: 'MANUAL', finalStatus: 'COMPLETED_NO_PR',
      startedAt: '2026-08-13T00:00:00Z', endedAt: '2026-08-13T00:00:02Z', sourceChecks: [{ due: true }] },
    { runId: 'scheduled-1', triggerType: 'SCHEDULED', finalStatus: 'NO_WORK',
      startedAt: '2026-08-13T01:00:00Z', endedAt: '2026-08-13T01:00:01Z', sourceChecks: [{ source: 'osm', due: false }] },
  ];
  const soak = aggregateOperationalSoak({ manifests, gate: config.soakGate,
    artifactRoot: path.join(os.tmpdir(), 'missing-stage4p-runs'),
    logRoot: path.join(os.tmpdir(), 'missing-stage4p-logs') });
  assert.equal(soak.scheduledInvocationCount, 1);
  assert.equal(soak.successfulCount, 1);
  assert.equal(soak.noWorkCount, 1);
  assert.equal(soak.sourceDueCount, 0);
  assert.equal(soak.status, 'IN_PROGRESS');
});

test('soak manifest reader aggregates nested operational artifacts', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'stage4p-soak-'));
  const run = path.join(root, 'run-1');
  fs.mkdirSync(run, { recursive: true });
  fs.writeFileSync(path.join(run, 'operational_run_manifest.json'), JSON.stringify({
    runId: 'run-1', triggerType: 'SCHEDULED', finalStatus: 'NO_WORK', startedAt: NOW, endedAt: NOW,
  }));
  assert.equal(readOperationalManifests(root).length, 1);
});

test('false-new second pass routes credible canonical proximity to possible existing', () => {
  const result = evaluate({ match: match({ decision: DECISIONS.PROBABLE_MATCH }) });
  assert.equal(result.recommendation, CREATE_NEW_STATES.POSSIBLE_EXISTING);
  assert.ok(result.reasonCodes.includes('possible_existing_canonical_second_pass'));
});

test('cross-source support can make a permissive strong candidate canary eligible', () => {
  const result = evaluate();
  assert.equal(result.recommendation, CREATE_NEW_STATES.CANARY);
  assert.ok(result.crossSourceSupport.includes('osm'));
  assert.equal(result.reviewerDecision, 'DEFER');
});

test('non-representative duplicate is rejected without canonical action', () => {
  const result = evaluate({ duplicateContext: { representative: 'osm:other', duplicateRisk: 1,
    supportSources: ['osm', 'overture'], independentSupport: true, componentSize: 2 } });
  assert.equal(result.recommendation, CREATE_NEW_STATES.REJECT);
  assert.ok(result.reasonCodes.includes('non_representative_source_duplicate'));
});

test('traveler-ineligible and weak records stay low confidence or rejected', () => {
  const weak = record({ category: 'dentist', phone: null, website: null });
  const result = evaluate({ record: weak, triage: { tier: NEW_TIERS.C,
    categoryEligibility: 'LOW_VALUE', reason: 'non_traveler_priority' } });
  assert.ok([CREATE_NEW_STATES.LOW_CONFIDENCE, CREATE_NEW_STATES.REJECT].includes(result.recommendation));
});

test('chain branches require branch-specific identity evidence', () => {
  assert.equal(branchAssessment(record({ name: 'Highlands Coffee', address: null,
    phone: null, website: null })).branchSpecific, false);
  assert.equal(branchAssessment(record({ name: 'Highlands Coffee', address: '12 Bach Dang' })).branchSpecific, true);
});

test('incomplete provenance is rejected and OSM remains share-alike review only', () => {
  assert.equal(provenanceAssessment(record({ provenance: {} })).rejected, true);
  const osm = record({ source: 'osm', sourceId: 'osm:node:1', provenance: {
    source: 'osm', sourceId: 'osm:node:1', snapshotRef: 'osm-1', license: 'ODbL-1.0',
    policyClass: 'OPEN_SHAREALIKE_ISOLATED', attribution: 'OpenStreetMap contributors',
  } });
  assert.equal(provenanceAssessment(osm).canaryStorageEligible, false);
});

test('CREATE_NEW decision memory suppresses unchanged and invalidates changed identity', () => {
  const result = evaluate();
  const memory = [createResearchDecisionRecord(result, 'DEFER', 'human-review-1')];
  assert.equal(findReusableResearchDecision(memory, result).outcome, 'SUPPRESS_UNCHANGED_DEFER');
  assert.equal(findReusableResearchDecision(memory, { ...result, identityFingerprint: 'changed' }).reusable, false);
  assert.equal(memory[0].canonicalWriteAuthorized, false);
});

test('canary artifact is deterministic, compact and defaults every decision to DEFER', () => {
  const results = Array.from({ length: 80 }, (_, index) => ({
    ...evaluate(), caseId: `CREATE_NEW:overture:${String(index).padStart(3, '0')}`,
    sourceId: `overture:${index}`,
  }));
  const first = buildCanaryReviewSample(results, config);
  const second = buildCanaryReviewSample([...results].reverse(), config);
  assert.deepEqual(first, second);
  assert.equal(first.length, 75);
  assert.ok(first.every((item) => item.reviewerDecision === 'DEFER'));
  const output = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'stage4p-csv-')), 'review.csv');
  writeCanaryCsv(output, first);
  assert.match(fs.readFileSync(output, 'utf8'), /^case_id,source,source_id/);
});

test('research output is deterministic regardless of candidate input order', () => {
  const candidates = [
    record({ sourceId: 'overture:one', externalIds: { overture_id: 'one' } }),
    record({ source: 'osm', sourceId: 'osm:node:two', name: 'Bao tang song Han',
      longitude: 108.23, externalIds: { osm_id: 'node:two' }, provenance: {
        source: 'osm', sourceId: 'osm:node:two', snapshotRef: 'osm-1', license: 'ODbL-1.0',
        policyClass: 'OPEN_SHAREALIKE_ISOLATED', attribution: 'OpenStreetMap contributors',
      } }),
  ];
  const boundary = { type: 'Polygon', coordinates: [[
    [108, 15.9], [108.4, 15.9], [108.4, 16.3], [108, 16.3], [108, 15.9],
  ]] };
  const canonicalPois = [{ id: 'canonical-1', sourceId: 'canonical-1', name: 'Khac',
    category: 'park', latitude: 16.1, longitude: 108.3, address: '', externalIds: {} }];
  const first = evaluateCreateNewResearch({ candidates, canonicalPois, boundary, config, now: NOW });
  const second = evaluateCreateNewResearch({ candidates: [...candidates].reverse(), canonicalPois,
    boundary, config, now: NOW });
  assert.equal(first.summary.deterministicHash, second.summary.deterministicHash);
  assert.deepEqual(first.canary, second.canary);
});

test('AUTO_CREATE_NEW cannot be enabled by configuration', () => {
  assert.doesNotThrow(() => assertResearchOnly(config));
  assert.throws(() => assertResearchOnly({ ...config, autoCreateNew: true }), /must remain disabled/);
});

test('canonical baseline remains immutable at 4166 approved rows', () => {
  const canonical = inspectCanonicalDataset(path.join(ROOT, 'data', 'canonical', 'urbanagent_poi_master_v1.csv'));
  assert.equal(canonical.rows, 4166);
  assert.equal(canonical.sha256, CANONICAL_SHA);
  assert.equal(config.canonicalWrites, 0);
  assert.equal(config.runtimeExposure, false);
});
