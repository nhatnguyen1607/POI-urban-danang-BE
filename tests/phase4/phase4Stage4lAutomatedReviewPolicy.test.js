const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const config = require('../../config/phase4_stage4l_policy.json');
const {
  POLICY_OUTCOMES,
  applySafetyBudgets,
  evaluateRunAnomalies,
  evaluateReviewCase,
  runAutomatedReviewPolicy,
} = require('../../src/modules/cityPackPreparation/automatedReviewPolicy');
const {
  DECISION_SOURCES,
  buildFieldFingerprints,
  buildIdentityFingerprint,
  createDecisionRecord,
  findReusableDecision,
} = require('../../src/modules/cityPackPreparation/decisionMemory');
const {
  buildEnrichmentRecord,
  recoverStage4kSkippedEnrichments,
} = require('../../src/modules/cityPackPreparation/cityPackEnrichment');
const { runCli } = require('../../scripts/citypack_sync');
const {
  buildDeltaReplay,
  importHumanMemory,
} = require('../../scripts/phase4_stage4l_automated_review_policy');

function reviewCase(overrides = {}) {
  return {
    caseId: 'MATCH:overture:1',
    syncStatus: 'CHANGED',
    resolver: { classification: 'HIGH_CONFIDENCE_MATCH', confidence: 0.99 },
    source: {
      source: 'overture', sourceId: 'overture:1', normalizedName: 'cafe one',
      category: 'cafe', latitude: 16.06, longitude: 108.22,
    },
    canonical: {
      id: 'google_maps_1', normalizedName: 'cafe one', category: 'cafe',
      latitude: 16.0601, longitude: 108.2201,
    },
    evidence: {
      confidence: 'STRONG', independent: true,
      checks: { resolverDistanceMeters: 20 },
    },
    provenanceComplete: true,
    geographicEligibility: 'INSIDE_DANANG',
    categoryEligibility: 'TRAVELER_RELEVANT',
    provenance: { snapshotRef: 'fixed-snapshot' },
    fieldChanges: [{
      field: 'address', oldValue: null, newValue: '1 Bach Dang', state: 'SAFE_ADDITION',
      provenance: { source: 'overture', sourceId: 'overture:1', license: 'CDLA-Permissive-2.0' },
    }],
    ...overrides,
  };
}

function memoryFor(item, decision = 'APPROVE') {
  return createDecisionRecord({
    reviewCase: item,
    decision,
    decisionSource: DECISION_SOURCES.HUMAN,
    decisionReference: 'test-review',
    policyVersion: 'stage4l-review-policy-v1',
    resolverVersion: 'stage4h-v1',
    evidenceVersion: 'stage4j-independent-evidence-v1',
    boundaryVersion: 'osm-relation-1891418-v72-product-scope-v1',
    approvedFields: decision === 'APPROVE' ? ['address'] : [],
  });
}

test('supports all seven deterministic policy outcomes', () => {
  assert.deepEqual(POLICY_OUTCOMES, [
    'SKIP_UNCHANGED', 'REUSE_APPROVAL', 'REUSE_REJECTION', 'AUTO_ACCEPT_SAFE',
    'AUTO_REJECT', 'HUMAN_REVIEW', 'DEFER',
  ]);
});

test('identity fingerprint is deterministic and excludes retrieval timestamps', () => {
  const item = reviewCase();
  assert.equal(buildIdentityFingerprint(item), buildIdentityFingerprint({ ...item, retrievedAt: 'later' }));
});

test('field fingerprints invalidate only changed fields', () => {
  const before = reviewCase();
  const after = reviewCase({ fieldChanges: [{ ...before.fieldChanges[0], newValue: '2 Bach Dang' }] });
  assert.notEqual(buildFieldFingerprints(before).address, buildFieldFingerprints(after).address);
});

test('human approval is reusable for unchanged identity and approved fields', () => {
  const item = reviewCase();
  assert.equal(evaluateReviewCase(item, { memory: [memoryFor(item)], config }).outcome, 'REUSE_APPROVAL');
});

test('human rejection is reusable for unchanged case', () => {
  const item = reviewCase();
  assert.equal(evaluateReviewCase(item, { memory: [memoryFor(item, 'REJECT')], config }).outcome, 'REUSE_REJECTION');
});

test('unchanged human defer is suppressed without reopening review', () => {
  const item = reviewCase();
  assert.equal(evaluateReviewCase(item, { memory: [memoryFor(item, 'DEFER')], config }).outcome, 'DEFER');
});

test('changed evidence invalidates a deferred decision', () => {
  const item = reviewCase();
  const memory = [memoryFor(item, 'DEFER')];
  const changed = { ...item, evidence: { ...item.evidence, confidence: 'MODERATE' } };
  assert.equal(findReusableDecision(memory, changed, {
    policy: 'stage4l-review-policy-v1', resolver: 'stage4h-v1',
    evidence: 'stage4j-independent-evidence-v1', boundary: 'osm-relation-1891418-v72-product-scope-v1',
  }).reason, 'EVIDENCE_CHANGED');
});

test('unchanged sync records skip every expensive policy stage', () => {
  assert.equal(evaluateReviewCase(reviewCase({ syncStatus: 'UNCHANGED' }), { memory: [], config }).outcome, 'SKIP_UNCHANGED');
});

test('strict high-confidence existing match auto-accepts safe fields', () => {
  const result = evaluateReviewCase(reviewCase(), { memory: [], config });
  assert.equal(result.outcome, 'AUTO_ACCEPT_SAFE');
  assert.deepEqual(result.applyOperations.map((item) => item.field), ['address']);
});

test('auto-accept never applies locked identity fields', () => {
  const item = reviewCase({ fieldChanges: [
    ...reviewCase().fieldChanges,
    { field: 'name', oldValue: 'Old', newValue: 'New', state: 'CONFLICT_REVIEW' },
  ] });
  assert.deepEqual(evaluateReviewCase(item, { memory: [], config }).applyOperations.map((op) => op.field), ['address']);
});

test('probable matches require human review', () => {
  const item = reviewCase({ resolver: { classification: 'PROBABLE_MATCH', confidence: 0.9 } });
  assert.equal(evaluateReviewCase(item, { memory: [], config }).outcome, 'HUMAN_REVIEW');
});

test('ambiguous matches require human review', () => {
  const item = reviewCase({ resolver: { classification: 'AMBIGUOUS', confidence: 0.5 } });
  assert.equal(evaluateReviewCase(item, { memory: [], config }).outcome, 'HUMAN_REVIEW');
});

test('new tier A candidates require human approval and cannot auto-create', () => {
  const item = reviewCase({ resolver: { classification: 'NEW_CANDIDATE' }, newCandidateTier: 'NEW_TIER_A_STRONG' });
  assert.equal(evaluateReviewCase(item, { memory: [], config }).outcome, 'HUMAN_REVIEW');
});

test('new tier B candidates defer', () => {
  const item = reviewCase({ resolver: { classification: 'NEW_CANDIDATE' }, newCandidateTier: 'NEW_TIER_B_REVIEW' });
  assert.equal(evaluateReviewCase(item, { memory: [], config }).outcome, 'DEFER');
});

test('invalid or excluded new candidates auto-reject', () => {
  const item = reviewCase({ resolver: { classification: 'NEW_CANDIDATE' }, newCandidateTier: 'NEW_TIER_D_INVALID_OR_EXCLUDE' });
  assert.equal(evaluateReviewCase(item, { memory: [], config }).outcome, 'AUTO_REJECT');
});

test('outside-boundary record auto-rejects without deletion', () => {
  const result = evaluateReviewCase(reviewCase({ geographicEligibility: 'OUTSIDE_DANANG' }), { memory: [], config });
  assert.equal(result.outcome, 'AUTO_REJECT');
  assert.equal(result.applyOperations, undefined);
});

test('hard provenance failure auto-rejects', () => {
  assert.equal(evaluateReviewCase(reviewCase({ provenanceComplete: false }), { memory: [], config }).outcome, 'AUTO_REJECT');
});

test('source duplicates auto-reject and never create a POI', () => {
  const item = reviewCase({ resolver: { classification: 'SOURCE_DUPLICATE' } });
  assert.equal(evaluateReviewCase(item, { memory: [], config }).outcome, 'AUTO_REJECT');
});

test('safety budget trips when auto-accept field budget is exceeded', () => {
  const result = applySafetyBudgets([
    { outcome: 'AUTO_ACCEPT_SAFE', applyOperations: [{}, {}] },
  ], { budgets: { maximumAutoAcceptCases: 2, maximumAutoAcceptFields: 1, maximumAutoRejectRate: 1 } });
  assert.equal(result.tripped, true);
  assert.deepEqual(result.reasons, ['AUTO_ACCEPT_FIELD_BUDGET']);
});

test('circuit breaker suppresses the entire apply plan', () => {
  const tight = { ...config, budgets: { ...config.budgets, maximumAutoAcceptCases: 0 } };
  const result = runAutomatedReviewPolicy([reviewCase()], { memory: [], config: tight });
  assert.equal(result.circuitBreaker.tripped, true);
  assert.deepEqual(result.applyPlan, []);
});

test('policy output is deterministic regardless of input order', () => {
  const a = reviewCase({ caseId: 'MATCH:overture:a' });
  const b = reviewCase({ caseId: 'MATCH:overture:b', source: { ...reviewCase().source, sourceId: 'overture:b' } });
  assert.equal(
    runAutomatedReviewPolicy([a, b], { memory: [], config }).deterministicHash,
    runAutomatedReviewPolicy([b, a], { memory: [], config }).deterministicHash,
  );
});

test('policy apply plan contains rollback and provenance metadata', () => {
  const result = runAutomatedReviewPolicy([reviewCase()], { memory: [], config });
  assert.equal(result.applyPlan[0].rollback.field, 'address');
  assert.equal(result.applyPlan[0].provenance.source, 'overture');
  assert.equal(result.applyPlan[0].policyDecision, 'AUTO_ACCEPT_SAFE');
  assert.equal(result.applyPlan[0].policyVersion, 'stage4l-review-policy-v1');
  assert.deepEqual(result.applyPlan[0].reasonCodes, ['strict_existing_match_safe_fields_only']);
});

test('enrichment schema rejects locked fields', () => {
  assert.throws(() => buildEnrichmentRecord({ field: 'name' }), /Unsupported enrichment field/);
});

test('Stage 4K skipped phone and website fields recover into non-runtime sidecar', () => {
  const skipped = ['phone', 'website'].map((field) => ({
    status: 'SKIPPED_UNSUPPORTED_CANONICAL_FIELD', field, canonical_id: 'google_maps_1',
    case_id: 'MATCH:overture:1', source: 'overture', source_id: 'overture:1',
    new_value: field === 'phone' ? '+84123' : 'https://example.com', human_decision: 'APPROVE',
    provenance: { source: 'overture', sourceId: 'overture:1', snapshotRef: 'fixed' },
  }));
  const result = recoverStage4kSkippedEnrichments({ skipped });
  assert.equal(result.recordCount, 2);
  assert.equal(result.runtimeInstalled, false);
  assert.ok(result.records.every((item) => item.status === 'CANDIDATE_NON_RUNTIME_NOT_CANONICAL'));
});

test('policy version change invalidates a reusable decision', () => {
  const item = reviewCase();
  const reuse = findReusableDecision([memoryFor(item)], item, {
    policy: 'v2', resolver: 'stage4h-v1', evidence: 'stage4j-independent-evidence-v1',
    boundary: 'osm-relation-1891418-v72-product-scope-v1',
  });
  assert.equal(reuse.reason, 'POLICY_VERSION_CHANGED');
});

test('boundary change invalidates geography only', () => {
  const item = reviewCase();
  const reuse = findReusableDecision([memoryFor(item)], item, {
    policy: 'stage4l-review-policy-v1', resolver: 'stage4h-v1', evidence: 'stage4j-independent-evidence-v1',
    boundary: 'new-boundary',
  });
  assert.deepEqual(reuse.invalidatedFields, ['geographicEligibility']);
});

test('resolver version change invalidates identity resolution only', () => {
  const item = reviewCase();
  const reuse = findReusableDecision([memoryFor(item)], item, {
    policy: 'stage4l-review-policy-v1', resolver: 'stage4h-v2',
    evidence: 'stage4j-independent-evidence-v1', boundary: 'osm-relation-1891418-v72-product-scope-v1',
  });
  assert.deepEqual(reuse.invalidatedFields, ['identityResolution']);
});

test('new website reuses approved identity and validates only website', () => {
  const item = reviewCase();
  const changed = { ...item, fieldChanges: [...item.fieldChanges, {
    field: 'website', oldValue: null, newValue: 'https://example.com', state: 'SAFE_ADDITION',
    provenance: { source: 'overture', sourceId: 'overture:1', license: 'CDLA-Permissive-2.0' },
  }] };
  const result = evaluateReviewCase(changed, { memory: [memoryFor(item)], config });
  assert.equal(result.outcome, 'AUTO_ACCEPT_SAFE');
  assert.deepEqual(result.applyOperations.map((operation) => operation.field), ['website']);
  assert.ok(result.reasonCodes.includes('entity_approval_reused'));
});

test('run anomaly gate catches canonical, snapshot, provenance and locked-field failures', () => {
  const result = evaluateRunAnomalies({
    canonicalSha256: 'unexpected', snapshotVersionRegressed: true,
    provenanceFailureRate: 0.5, lockedFieldMutationAttempts: 1,
  }, config);
  assert.equal(result.tripped, true);
  assert.deepEqual(result.reasons, [
    'PROVENANCE_FAILURE_SPIKE', 'SNAPSHOT_VERSION_REGRESSION',
    'UNEXPECTED_CANONICAL_BASELINE', 'LOCKED_FIELD_MUTATION_ATTEMPT',
  ]);
});

test('decision memory accepts explicit policy and system-safety sources', () => {
  const item = reviewCase();
  const policy = createDecisionRecord({
    reviewCase: item, decision: 'APPROVE', decisionSource: DECISION_SOURCES.POLICY,
    decisionReference: 'policy-run', policyVersion: 'v1', resolverVersion: 'v1',
    evidenceVersion: 'v1', boundaryVersion: 'v1', approvedFields: ['address'],
  });
  const safety = createDecisionRecord({
    reviewCase: item, decision: 'REJECT', decisionSource: DECISION_SOURCES.SYSTEM_SAFETY,
    decisionReference: 'safety-run', policyVersion: 'v1', resolverVersion: 'v1',
    evidenceVersion: 'v1', boundaryVersion: 'v1',
  });
  assert.equal(policy.decisionSource, 'POLICY');
  assert.equal(safety.decisionSource, 'SYSTEM_SAFETY');
});

test('authoritative decision importer preserves human source and review reference', () => {
  const item = reviewCase();
  const memory = importHumanMemory([{
    case_id: item.caseId, reviewer_decision: 'APPROVE', reviewer_note: 'approved fixture',
  }], [item]);
  assert.equal(memory.length, 1);
  assert.equal(memory[0].decisionSource, 'HUMAN');
  assert.equal(memory[0].reviewEvidenceReference, 'stage4j-human-canary-decisions');
});

test('delta replay covers the approved A-F human-on-exception scenarios', () => {
  const approved = reviewCase();
  const rejected = reviewCase({
    caseId: 'MATCH:overture:rejected',
    source: { ...reviewCase().source, sourceId: 'overture:rejected' },
  });
  const memory = [memoryFor(approved), memoryFor(rejected, 'REJECT')];
  const replay = buildDeltaReplay([approved, rejected], memory, config);
  assert.deepEqual(replay.A_100_UNCHANGED.outcomeCounts, { SKIP_UNCHANGED: 100 });
  assert.deepEqual(replay.B_WEBSITE_ONLY.fields, ['website']);
  assert.equal(replay.C_LARGE_COORDINATE_MOVE.outcome, 'HUMAN_REVIEW');
  assert.equal(replay.D_UNCHANGED_REJECTION, 'REUSE_REJECTION');
  assert.equal(replay.E_NEW_TIER_A, 'HUMAN_REVIEW');
  assert.equal(replay.F_OUTSIDE_NEW, 'AUTO_REJECT');
});

test('citypack sync exposes guarded policy-auto mode without a source snapshot', () => {
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'urbanagent-stage4l-'));
  const casesPath = path.join(temp, 'cases.json');
  const configPath = path.join(temp, 'config.json');
  fs.writeFileSync(casesPath, JSON.stringify([reviewCase()]), 'utf8');
  fs.writeFileSync(configPath, JSON.stringify(config), 'utf8');
  try {
    const result = runCli([
      'sync', '--policy', 'auto', '--policy-cases', casesPath, '--policy-config', configPath,
    ]);
    assert.equal(result.status, 'CANDIDATE_NON_RUNTIME_NOT_CANONICAL_POLICY_PLAN');
    assert.equal(result.canonicalWriteAuthorized, false);
  } finally {
    fs.rmSync(temp, { recursive: true, force: true });
  }
});
