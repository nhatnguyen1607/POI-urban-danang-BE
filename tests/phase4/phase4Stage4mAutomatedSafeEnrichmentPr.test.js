const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const executorConfig = require('../../config/phase4_stage4m_auto_pr.json');
const policyConfig = require('../../config/phase4_stage4l_policy.json');
const {
  DEFAULT_VERSIONS,
  evaluateReviewCase,
} = require('../../src/modules/cityPackPreparation/automatedReviewPolicy');
const {
  DECISION_SOURCES,
  createDecisionRecord,
} = require('../../src/modules/cityPackPreparation/decisionMemory');
const {
  EXPECTED_CANONICAL_SHA,
  resolveExpectedCanonicalSha,
} = require('../../src/modules/cityPackPreparation/canonicalDataset');
const {
  BLOCKED_BY_SAFETY_GATE,
  NO_SAFE_CHANGES,
  applySafeEnrichment,
  buildSafeEnrichmentPlan,
  serializeSidecar,
  validateAppliedResult,
} = require('../../src/modules/cityPackPreparation/safeEnrichmentExecutor');
const {
  buildPrSummary,
  deterministicAutomationBranch,
  githubCompareUrl,
  parseArgs,
} = require('../../scripts/citypack_auto_pr');
const {
  inspectCanonicalText,
} = require('../../scripts/phase4_stage4k_controlled_canary_apply');

const ROOT = path.resolve(__dirname, '..', '..');
const CANONICAL_PATH = path.join(ROOT, 'data', 'canonical', 'urbanagent_poi_master_v1.csv');
const CANONICAL_TEXT = fs.readFileSync(CANONICAL_PATH, 'utf8');
const CANONICAL = inspectCanonicalText(CANONICAL_TEXT);
const EMPTY_ADDRESS_ROW = CANONICAL.rows.find((row) => !String(row.Address_Current || '').trim());
const NONEMPTY_ADDRESS_ROW = CANONICAL.rows.find((row) => String(row.Address_Current || '').trim());

const PROVENANCE = Object.freeze({
  source: 'overture',
  sourceId: 'overture:test-record',
  snapshotRef: 'stage4j-fixed-snapshot',
  license: 'CDLA-Permissive-2.0',
  policyClass: 'OPEN_PERMISSIVE_CANDIDATE',
  attribution: 'Overture Maps Foundation',
  licenseUrl: 'https://docs.overturemaps.org/attribution/',
});

function operation(field, newValue, overrides = {}) {
  return {
    caseId: overrides.caseId || 'MATCH:overture:test-record',
    canonicalId: overrides.canonicalId || EMPTY_ADDRESS_ROW.Global_ID,
    source: overrides.source || 'overture',
    sourceId: overrides.sourceId || 'overture:test-record',
    policyDecision: 'AUTO_ACCEPT_SAFE',
    policyVersion: DEFAULT_VERSIONS.policy,
    decisionSource: 'POLICY',
    reasonCodes: ['strict_existing_match_safe_fields_only'],
    operation: 'ENRICH_EXISTING',
    field,
    oldValue: overrides.oldValue ?? null,
    newValue,
    provenance: overrides.provenance || PROVENANCE,
    license: overrides.license || PROVENANCE.license,
    ...overrides,
  };
}

function memory(caseId = 'MATCH:overture:test-record') {
  return [{
    caseId,
    decisionReference: `human:${caseId}`,
    identityFingerprint: 'identity-fingerprint',
    fieldFingerprints: {
      address: 'address-fingerprint',
      phone: 'phone-fingerprint',
      website: 'website-fingerprint',
      openingHours: 'hours-fingerprint',
      media: 'media-fingerprint',
    },
    provenanceFingerprint: 'provenance-fingerprint',
  }];
}

function policyResult(operations, resultOverrides = {}) {
  const caseIds = [...new Set(operations.map((item) => item.caseId))];
  return {
    applyPlan: operations,
    results: caseIds.map((caseId) => ({
      caseId,
      outcome: 'AUTO_ACCEPT_SAFE',
      source: 'overture',
      sourceId: 'overture:test-record',
      canonicalId: EMPTY_ADDRESS_ROW.Global_ID,
      reasonCodes: [],
    })),
    circuitBreaker: { tripped: false, reasons: [] },
    anomalyGate: { tripped: false, reasons: [] },
    processingVersions: DEFAULT_VERSIONS,
    createNewAuthorized: false,
    ...resultOverrides,
  };
}

function plan(operations, options = {}) {
  const caseIds = [...new Set(operations.map((item) => item.caseId))];
  return buildSafeEnrichmentPlan({
    policyResult: policyResult(operations, options.policyOverrides),
    decisionMemory: caseIds.flatMap((caseId) => memory(caseId)),
    canonicalText: CANONICAL_TEXT,
    existingSidecar: options.existingSidecar || null,
    executorConfig,
    recoveredEnrichments: options.recoveredEnrichments || [],
    previousExceptionCaseIds: options.previousExceptionCaseIds || [],
    budgetOverrides: options.budgetOverrides || {},
  });
}

function reviewCase(overrides = {}) {
  return {
    caseId: 'MATCH:overture:policy-case',
    source: {
      source: 'overture', sourceId: 'overture:policy-case', normalizedName: 'cafe test',
      category: 'cafe', latitude: 16.06, longitude: 108.22, address: '1 Test Street',
    },
    canonical: {
      id: EMPTY_ADDRESS_ROW.Global_ID, normalizedName: 'cafe test', category: 'cafe',
      latitude: 16.06, longitude: 108.22, address: null,
    },
    classification: 'HIGH_CONFIDENCE_MATCH',
    resolver: { classification: 'HIGH_CONFIDENCE_MATCH', confidence: 0.99 },
    evidence: { independent: true, confidence: 'STRONG', checks: { resolverDistanceMeters: 2 } },
    provenanceComplete: true,
    provenance: PROVENANCE,
    license: { license: PROVENANCE.license },
    fieldChanges: [{
      field: 'address', oldValue: null, newValue: '1 Test Street',
      state: 'SAFE_ADDITION', provenance: PROVENANCE,
    }],
    ...overrides,
  };
}

test('Stage 4M CLI is dry-run by default and rejects ambiguous write modes', () => {
  const parsed = parseArgs(['--branch', 'phase4/test', '--artifact-dir', 'D:/artifacts']);
  assert.equal(parsed.applySafe, undefined);
  assert.throws(() => parseArgs(['--dry-run', '--apply-safe']), /mutually exclusive/);
});

test('automation branch, compare URL, and PR summary are deterministic', () => {
  assert.equal(deterministicAutomationBranch('Run 001'), 'auto/poi-safe-enrichment/run-001');
  assert.equal(
    githubCompareUrl('https://github.com/example/urbanagent.git', 'phase4/test'),
    'https://github.com/example/urbanagent/compare/main...phase4%2Ftest',
  );
  assert.match(buildPrSummary({
    sourceSnapshotReferences: ['fixed'], policyVersion: 'v1', canonicalBaselineSha: 'a',
    canonicalAfterSha: 'b', autoAcceptSafeCount: 1, canonicalOperationCount: 1,
    sidecarOperationCount: 0, existingValueSkips: 0, humanReviewDeltaCount: 0,
    safetyBudgetStatus: 'PASS', testStatus: 'PASS', rollbackStatus: 'PASS',
  }), /Canonical: a -> b/);
});

test('Stage 4L remains the policy authority and can materialize unchanged approved fields for Stage 4M', () => {
  const item = reviewCase();
  const record = createDecisionRecord({
    reviewCase: item,
    decision: 'APPROVE',
    decisionSource: DECISION_SOURCES.HUMAN,
    decisionReference: 'stage4k:human-canary',
    policyVersion: DEFAULT_VERSIONS.policy,
    resolverVersion: DEFAULT_VERSIONS.resolver,
    evidenceVersion: DEFAULT_VERSIONS.evidence,
    boundaryVersion: DEFAULT_VERSIONS.boundary,
    approvedFields: ['address'],
  });
  assert.equal(evaluateReviewCase(item, { memory: [record], config: policyConfig }).outcome, 'REUSE_APPROVAL');
  const materialized = evaluateReviewCase(item, {
    memory: [record], config: policyConfig, materializeReusableApprovals: true,
  });
  assert.equal(materialized.outcome, 'AUTO_ACCEPT_SAFE');
  assert.deepEqual(materialized.applyOperations.map((entry) => entry.field), ['address']);
});

test('only an empty canonical address is mutable while enrichment fields go to the sidecar', () => {
  const result = plan([
    operation('address', '1 Safe Street'),
    operation('phone', '+84 912 345 678'),
    operation('website', 'https://example.test'),
    operation('openingHours', 'Mo-Fr 08:00-18:00'),
    operation('media', { url: 'https://example.test/image.jpg' }),
  ]);
  assert.equal(result.status, 'SAFE_PLAN_READY');
  assert.equal(result.canonicalAddressOperations, 1);
  assert.equal(result.sidecarOperationCount, 4);
  assert.ok(result.sidecarOperations.every((item) => item.destination === 'ENRICHMENT_SIDECAR'));
});

test('an identical existing canonical value becomes a no-op and a conflicting value blocks', () => {
  const same = operation('address', NONEMPTY_ADDRESS_ROW.Address_Current, {
    canonicalId: NONEMPTY_ADDRESS_ROW.Global_ID,
  });
  const samePlan = plan([same]);
  assert.equal(samePlan.status, NO_SAFE_CHANGES);
  assert.equal(samePlan.existingValueSkips, 1);
  const conflict = plan([operation('address', 'Different verified address', {
    canonicalId: NONEMPTY_ADDRESS_ROW.Global_ID,
  })]);
  assert.equal(conflict.status, BLOCKED_BY_SAFETY_GATE);
  assert.ok(conflict.violations.some((item) => item.startsWith('NON_EMPTY_CANONICAL_OVERWRITE')));
});

test('locked fields, CREATE_NEW, DELETE, and MERGE trip the safety gate', () => {
  const locked = plan([operation('name', 'Renamed POI')]);
  assert.equal(locked.status, BLOCKED_BY_SAFETY_GATE);
  assert.ok(locked.violations.some((item) => item.startsWith('LOCKED_OR_UNSUPPORTED_FIELD')));
  const create = plan([operation('address', '1 Safe Street')], {
    policyOverrides: { createNewAuthorized: true },
  });
  assert.ok(create.violations.includes('CREATE_NEW_AUTHORIZATION'));
  const destructive = plan([operation('address', '1 Safe Street')], {
    policyOverrides: { deleteAuthorized: true, mergeAuthorized: true },
  });
  assert.ok(destructive.violations.includes('DELETE_AUTHORIZATION'));
  assert.ok(destructive.violations.includes('MERGE_AUTHORIZATION'));
  const createOperation = plan([operation('address', '1 Safe Street', { operation: 'CREATE_NEW' })]);
  assert.ok(createOperation.violations.some((item) => item.includes('UNAUTHORIZED_OPERATION')));
});

test('policy and boundary version mismatches stop before mutation', () => {
  const mismatched = plan([operation('address', '1 Safe Street')], {
    policyOverrides: { processingVersions: { ...DEFAULT_VERSIONS, policy: 'unexpected', boundary: 'unexpected' } },
  });
  assert.ok(mismatched.violations.includes('POLICY_VERSION_MISMATCH'));
  assert.ok(mismatched.violations.includes('BOUNDARY_VERSION_MISMATCH'));
});

test('a tripped Stage 4L circuit breaker suppresses all mutation', () => {
  const blocked = plan([operation('address', '1 Safe Street')], {
    policyOverrides: { circuitBreaker: { tripped: true, reasons: ['AUTO_ACCEPT_CASE_BUDGET'] } },
  });
  assert.equal(blocked.status, BLOCKED_BY_SAFETY_GATE);
  assert.ok(blocked.violations.includes('STAGE4L_CIRCUIT_BREAKER'));
});

test('missing provenance, Google provenance, and invalid field values are blocked', () => {
  const incomplete = plan([operation('address', '1 Safe Street', {
    provenance: { source: 'overture' },
  })]);
  assert.ok(incomplete.violations.some((item) => item.startsWith('PROVENANCE_LICENSE_FAILURE')));
  const google = plan([operation('website', 'https://example.test', {
    source: 'google', provenance: { ...PROVENANCE, source: 'google' },
  })]);
  assert.ok(google.violations.some((item) => item.startsWith('PROVENANCE_LICENSE_FAILURE')));
  const invalid = plan([operation('website', 'ftp://example.test')]);
  assert.ok(invalid.violations.some((item) => item.startsWith('INVALID_FIELD_VALUE')));
});

test('CLI budget overrides can tighten but never loosen configured safety budgets', () => {
  const operations = [operation('address', '1 Safe Street')];
  const tightened = plan(operations, { budgetOverrides: { maxAutoCases: 0 } });
  assert.ok(tightened.violations.includes('AUTO_CASE_BUDGET'));
  const notLoosened = plan(operations, { budgetOverrides: { maxAutoCases: 500 } });
  assert.equal(notLoosened.budgets.maxAutoCases, 50);
  const fieldTightened = plan(operations, { budgetOverrides: { maxFieldMutations: 0 } });
  assert.ok(fieldTightened.violations.includes('CANONICAL_ADDRESS_BUDGET'));
});

test('human-review cases never enter the apply plan and emit only new exception deltas', () => {
  const human = {
    caseId: 'AMBIGUOUS:overture:1', outcome: 'HUMAN_REVIEW', source: 'overture',
    sourceId: 'overture:1', canonicalId: EMPTY_ADDRESS_ROW.Global_ID,
    reasonCodes: ['identity_not_safe_for_automation'],
  };
  const result = buildSafeEnrichmentPlan({
    policyResult: policyResult([], { results: [human] }),
    decisionMemory: [], canonicalText: CANONICAL_TEXT, existingSidecar: null,
    executorConfig, recoveredEnrichments: [],
  });
  assert.equal(result.status, NO_SAFE_CHANGES);
  assert.equal(result.exceptions, 1);
  const repeated = buildSafeEnrichmentPlan({
    policyResult: policyResult([], { results: [human] }),
    decisionMemory: [], canonicalText: CANONICAL_TEXT, existingSidecar: null,
    executorConfig, recoveredEnrichments: [], previousExceptionCaseIds: [human.caseId],
  });
  assert.equal(repeated.exceptions, 0);
});

test('exact plan records policy, memory, provenance, fingerprints, and rollback trace', () => {
  const result = plan([operation('address', '1 Safe Street')]);
  const item = result.canonicalOperations[0];
  assert.equal(item.policyOutcome, 'AUTO_ACCEPT_SAFE');
  assert.equal(item.decisionMemoryReference, 'human:MATCH:overture:test-record');
  assert.equal(item.provenance.snapshotRef, PROVENANCE.snapshotRef);
  assert.equal(item.fingerprints.identity, 'identity-fingerprint');
  assert.equal(item.rollbackValue, null);
  assert.match(result.planHash, /^[a-f0-9]{64}$/);
});

test('apply is a minimal address-only diff with deterministic sidecar and byte-identical rollback', () => {
  const safePlan = plan([
    operation('address', '1 Safe Street'),
    operation('phone', '+84 912 345 678'),
  ]);
  const first = applySafeEnrichment({
    plan: safePlan, canonicalText: CANONICAL_TEXT, existingSidecar: null, executorConfig,
  });
  const validation = validateAppliedResult({
    plan: safePlan, beforeCanonicalText: CANONICAL_TEXT, result: first,
    existingSidecar: null, executorConfig,
  });
  assert.equal(validation.canonicalAddressChanges, 1);
  assert.equal(validation.sidecarChanges, 1);
  assert.equal(validation.nonTargetRowsUnchanged, 4172);
  assert.equal(validation.nameChanges + validation.coordinateChanges
    + validation.categoryChanges + validation.externalIdsChanges, 0);
  assert.equal(validation.rollbackCanonicalByteIdentical, true);
  assert.equal(validation.rollbackCanonicalSha256, CANONICAL.sha256);
  assert.equal(serializeSidecar(JSON.parse(first.sidecarText)), first.sidecarText);
});

test('a second identical plan skips canonical and sidecar values with NO_SAFE_CHANGES', () => {
  const operations = [
    operation('address', '1 Safe Street'),
    operation('website', 'https://example.test'),
  ];
  const firstPlan = plan(operations);
  const first = applySafeEnrichment({
    plan: firstPlan, canonicalText: CANONICAL_TEXT, existingSidecar: null, executorConfig,
  });
  const second = buildSafeEnrichmentPlan({
    policyResult: policyResult(operations),
    decisionMemory: memory(),
    canonicalText: first.canonicalText,
    existingSidecar: JSON.parse(first.sidecarText),
    executorConfig,
    recoveredEnrichments: [],
  });
  assert.equal(second.status, NO_SAFE_CHANGES);
  assert.equal(second.existingValueSkips, 2);
  assert.equal(second.canonicalAddressOperations + second.sidecarOperationCount, 0);
});

test('Stage 4K recovery gate requires every approved sidecar opportunity', () => {
  const recovered = [{
    caseId: 'MATCH:overture:test-record', field: 'phone', value: '+84 912 345 678',
  }];
  const missing = plan([operation('address', '1 Safe Street')], { recoveredEnrichments: recovered });
  assert.ok(missing.violations.includes('STAGE4K_RECOVERY_INCOMPLETE'));
  const complete = plan([operation('phone', '+84 912 345 678')], { recoveredEnrichments: recovered });
  assert.equal(complete.recoveredStage4kEnrichments, 1);
});

test('delta simulations A-E preserve unchanged, safe, ambiguous, duplicate, and new-entity policy', () => {
  const unchanged = evaluateReviewCase(reviewCase({ syncStatus: 'UNCHANGED' }), {
    memory: [], config: policyConfig,
  });
  const safe = evaluateReviewCase(reviewCase(), { memory: [], config: policyConfig });
  const ambiguous = evaluateReviewCase(reviewCase({
    classification: 'AMBIGUOUS', resolver: { classification: 'AMBIGUOUS', confidence: 0.8 },
  }), { memory: [], config: policyConfig });
  const duplicate = evaluateReviewCase(reviewCase({
    classification: 'SOURCE_DUPLICATE', resolver: { classification: 'SOURCE_DUPLICATE', confidence: 1 },
  }), { memory: [], config: policyConfig });
  const newEntity = evaluateReviewCase(reviewCase({
    classification: 'NEW_CANDIDATE', resolver: { classification: 'NEW_CANDIDATE', confidence: 0.99 },
    newCandidateTier: 'NEW_TIER_A_STRONG',
  }), { memory: [], config: policyConfig });
  assert.deepEqual(
    [unchanged.outcome, safe.outcome, ambiguous.outcome, duplicate.outcome, newEntity.outcome],
    ['SKIP_UNCHANGED', 'AUTO_ACCEPT_SAFE', 'HUMAN_REVIEW', 'AUTO_REJECT', 'HUMAN_REVIEW'],
  );
});

test('canonical expectation follows committed Stage 4M state and rejects malformed state', () => {
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'stage4m-state-'));
  const validPath = path.join(temp, 'valid.json');
  fs.writeFileSync(validPath, JSON.stringify({ canonicalShaAfter: 'a'.repeat(64) }));
  assert.equal(resolveExpectedCanonicalSha(validPath), 'a'.repeat(64));
  const invalidPath = path.join(temp, 'invalid.json');
  fs.writeFileSync(invalidPath, JSON.stringify({ canonicalShaAfter: 'invalid' }));
  assert.throws(() => resolveExpectedCanonicalSha(invalidPath), /Invalid canonical SHA in City Pack state/);
  fs.rmSync(temp, { recursive: true, force: true });
});

test('current canonical baseline remains unique at the state-approved hash', () => {
  assert.equal(CANONICAL.rows.length, 4173);
  assert.equal(new Set(CANONICAL.rows.map((row) => row.Global_ID)).size, 4173);
  assert.equal(CANONICAL.sha256, EXPECTED_CANONICAL_SHA);
});
