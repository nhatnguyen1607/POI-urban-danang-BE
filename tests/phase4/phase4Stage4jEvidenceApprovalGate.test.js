const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');

const {
  EXPECTED_CANONICAL_SHA,
  inspectCanonicalDataset,
} = require('../../src/modules/cityPackPreparation/canonicalDataset');
const {
  APPLY_OPERATIONS,
  EVIDENCE_CONFIDENCE,
  EVIDENCE_LABELS,
  FIELD_STATES,
  REVIEW_DECISIONS,
  buildApprovedApplyPlan,
  calculateEvidenceGate,
  classifyFieldEvidence,
  dryRunApprovedPlan,
  evaluateMatchEvidence,
  revalidateDuplicateEvidence,
  validateReviewDecisions,
} = require('../../src/modules/cityPackPreparation/evidenceApprovalGate');
const { runIncrementalSync } = require('../../src/modules/cityPackPreparation/incrementalSync');
const { boundaryPolicy } = require('../../scripts/phase4_stage4j_evidence_approval_gate');

const ROOT = path.resolve(__dirname, '..', '..');
const CANONICAL_PATH = path.join(ROOT, 'data', 'canonical', 'urbanagent_poi_master_v1.csv');
const BOUNDARY_PATH = path.join(
  ROOT, 'data', 'spikes', 'phase4', 'stage4i', 'boundary',
  'osm-relation-1891418-v72.geojson',
);
const EXPECTED_BOUNDARY_SHA = 'c0db1b84c8fcbc77b1b5cca14ba6d280627f59713db56965bd6198b72e78a84d';

function source(overrides = {}) {
  return {
    source: 'overture',
    sourceId: 'overture:test',
    name: 'Cafe Moc',
    category: 'cafe',
    latitude: 16.06,
    longitude: 108.22,
    address: '12 Bach Dang, Hai Chau',
    website: 'https://cafemoc.example',
    phone: '+84 912 345 678',
    externalIds: {},
    license: { license: 'CDLA-Permissive-2.0', attribution: 'Overture Maps Foundation' },
    provenance: { source: 'overture', sourceId: 'overture:test', snapshotRef: 'fixed' },
    ...overrides,
  };
}

function canonical(overrides = {}) {
  return {
    id: 'google_maps_1',
    name: 'Cafe Moc',
    category: 'cafe',
    latitude: 16.06001,
    longitude: 108.22001,
    address: '12 Bach Dang, Hai Chau',
    website: null,
    externalIds: {},
    evidenceMetadata: { restaurantIdIsSourceIdentifier: 'place-1' },
    ...overrides,
  };
}

function historical(overrides = {}) {
  return {
    placeId: 'place-1',
    name: 'Cafe Moc',
    category: 'cafe',
    latitude: 16.06001,
    longitude: 108.22001,
    address: '12 Bach Dang, Hai Chau',
    phone: '0912345678',
    ...overrides,
  };
}

function match(overrides = {}) {
  return {
    decision: 'HIGH_CONFIDENCE_MATCH',
    confidence: 0.99,
    reasonCodes: ['resolver_context_only'],
    bestCandidate: { distanceMeters: 2, canonicalPoiId: 'google_maps_1' },
    ...overrides,
  };
}

function caseRecord(overrides = {}) {
  return {
    caseId: 'case-1',
    stratum: 'HIGH_CONFIDENCE_MATCH',
    source: { sourceId: 'overture:test' },
    canonical: { id: 'google_maps_1' },
    sourceIds: ['overture:test'],
    evidence: { label: EVIDENCE_LABELS.MATCH, confidence: EVIDENCE_CONFIDENCE.STRONG },
    provenance: { source: 'overture' },
    license: { license: 'CDLA-Permissive-2.0' },
    provenanceComplete: true,
    fieldChanges: [{ field: 'website', oldValue: null, newValue: 'https://example.test', state: FIELD_STATES.SAFE_ADDITION }],
    ...overrides,
  };
}

test('Stage 4J boundary policy pins version/hash while requiring reviewed future updates', () => {
  const artifact = JSON.parse(fs.readFileSync(BOUNDARY_PATH, 'utf8'));
  const policy = boundaryPolicy(artifact);
  assert.equal(policy.status, 'PASS_VERSIONED_PRODUCT_SCOPE');
  assert.equal(policy.sourceIdentifier, 'relation/1891418');
  assert.equal(policy.sourceVersion, 72);
  assert.equal(policy.artifactSha256, EXPECTED_BOUNDARY_SHA);
  assert.match(policy.futureUpdateRule, /explicit scope review/);
});

test('Stage 4J resolver output alone remains UNCERTAIN', () => {
  const evidence = evaluateMatchEvidence({
    match: match(),
    sourceRecord: source({ phone: null, website: null, address: null }),
    canonicalRecord: canonical({ evidenceMetadata: {}, address: null }),
    historicalRecord: null,
  });
  assert.equal(evidence.label, EVIDENCE_LABELS.UNCERTAIN);
  assert.equal(evidence.independent, false);
});

test('Stage 4J accepts exact stable cross-source identifier evidence', () => {
  const evidence = evaluateMatchEvidence({
    match: match(),
    sourceRecord: source({ externalIds: { wikidata: 'Q123' } }),
    canonicalRecord: canonical({ externalIds: { wikidata: 'Q123' } }),
    historicalRecord: null,
  });
  assert.equal(evidence.label, EVIDENCE_LABELS.MATCH);
  assert.equal(evidence.confidence, EVIDENCE_CONFIDENCE.STRONG);
  assert.equal(evidence.checks.exactExternalId, true);
});

test('Stage 4J accepts historical phone and location evidence without exposing payload', () => {
  const evidence = evaluateMatchEvidence({
    match: match(), sourceRecord: source(), canonicalRecord: canonical(), historicalRecord: historical(),
  });
  assert.equal(evidence.label, EVIDENCE_LABELS.MATCH);
  assert.equal(evidence.checks.samePhone, true);
  assert.equal(evidence.checks.historicalLink, true);
  assert.ok(evidence.checks.historicalDistanceMeters < 5);
});

test('Stage 4J accepts moderate historical address evidence with close coordinates', () => {
  const evidence = evaluateMatchEvidence({
    match: match(),
    sourceRecord: source({ phone: null, website: null }),
    canonicalRecord: canonical(),
    historicalRecord: historical({ phone: null }),
  });
  assert.equal(evidence.label, EVIDENCE_LABELS.MATCH);
  assert.equal(evidence.confidence, EVIDENCE_CONFIDENCE.MODERATE);
});

test('Stage 4J accepts same official domain only with compatible location', () => {
  const evidence = evaluateMatchEvidence({
    match: match({ bestCandidate: { distanceMeters: 80, canonicalPoiId: 'google_maps_1' } }),
    sourceRecord: source({ phone: null, website: 'https://www.cafemoc.example/menu' }),
    canonicalRecord: canonical({ website: 'https://cafemoc.example/about', evidenceMetadata: {} }),
    historicalRecord: null,
  });
  assert.equal(evidence.label, EVIDENCE_LABELS.MATCH);
  assert.equal(evidence.checks.sameWebsite, true);
  assert.equal(evidence.confidence, EVIDENCE_CONFIDENCE.STRONG);
});

test('Stage 4J precision denominator excludes UNCERTAIN evidence', () => {
  const cases = [
    { evidence: { independent: true, label: EVIDENCE_LABELS.MATCH } },
    { evidence: { independent: false, label: EVIDENCE_LABELS.UNCERTAIN } },
  ];
  const gate = calculateEvidenceGate(cases, { minimum: 1, requiredPrecision: 0.98 });
  assert.equal(gate.usable, 1);
  assert.equal(gate.precision, 1);
  assert.equal(gate.status, 'PASS');
});

test('Stage 4J HIGH gate enforces minimum independent support', () => {
  const cases = Array.from({ length: 30 }, () => ({ evidence: { independent: true, label: EVIDENCE_LABELS.MATCH } }));
  assert.equal(calculateEvidenceGate(cases, { minimum: 30, requiredPrecision: 0.98 }).status, 'PASS');
});

test('Stage 4J PROBABLE gate reports insufficient evidence without lowering standards', () => {
  const cases = Array.from({ length: 19 }, () => ({ evidence: { independent: true, label: EVIDENCE_LABELS.MATCH } }));
  assert.equal(calculateEvidenceGate(cases, { minimum: 20, requiredPrecision: 0.99 }).status, 'INSUFFICIENT_EVIDENCE');
});

test('Stage 4J duplicate support requires independent linkage and compatible distance', () => {
  const valid = revalidateDuplicateEvidence({
    evidenceSupported: { status: 'EVIDENCE_SUPPORTED', label: 'DUPLICATE_SAME_PLACE', evidenceType: 'EXACT_PHONE_LINKAGE' },
    candidateEvidence: { distanceMeters: 20 },
  });
  const invalid = revalidateDuplicateEvidence({
    evidenceSupported: { status: 'EVIDENCE_SUPPORTED', label: 'DUPLICATE_SAME_PLACE', evidenceType: 'RESOLVER_SCORE' },
    candidateEvidence: { distanceMeters: 20 },
  });
  assert.equal(valid.independent, true);
  assert.equal(invalid.independent, false);
});

test('Stage 4J decision validation rejects unknown, duplicate and malformed decisions', () => {
  const cases = new Map([['case-1', caseRecord()]]);
  const result = validateReviewDecisions([
    { caseId: 'unknown', reviewerDecision: REVIEW_DECISIONS.DEFER },
    { caseId: 'case-1', reviewerDecision: 'YES' },
    { caseId: 'case-1', reviewerDecision: REVIEW_DECISIONS.DEFER },
  ], cases);
  assert.equal(result.valid, false);
  assert.ok(result.errors.some((error) => error.startsWith('unknown_case_id')));
  assert.ok(result.errors.some((error) => error.startsWith('duplicate_decision')));
  assert.ok(result.errors.some((error) => error.startsWith('malformed_decision')));
});

test('Stage 4J decision validation rejects incompatible approval and missing provenance', () => {
  const cases = new Map([['case-1', caseRecord({ provenanceComplete: false })]]);
  const result = validateReviewDecisions([{
    caseId: 'case-1',
    reviewerDecision: REVIEW_DECISIONS.APPROVE,
    approvedOperation: APPLY_OPERATIONS.CREATE_NEW,
  }], cases);
  assert.equal(result.valid, false);
  assert.ok(result.errors.some((error) => error.startsWith('incompatible_approval_operation')));
  assert.ok(result.errors.some((error) => error.startsWith('missing_required_provenance')));
});

test('Stage 4J excludes unapproved cases from apply plan', () => {
  const plan = buildApprovedApplyPlan({
    cases: [caseRecord()],
    decisions: [{ caseId: 'case-1', reviewerDecision: REVIEW_DECISIONS.DEFER }],
  });
  assert.equal(plan.operations.length, 0);
  assert.equal(plan.canonicalWriteAuthorized, false);
});

test('Stage 4J includes explicitly approved compatible case with rollback metadata', () => {
  const plan = buildApprovedApplyPlan({
    cases: [caseRecord()],
    decisions: [{
      caseId: 'case-1', reviewerDecision: REVIEW_DECISIONS.APPROVE,
      approvedOperation: APPLY_OPERATIONS.ENRICH_EXISTING,
    }],
  });
  assert.equal(plan.operations.length, 1);
  assert.equal(plan.operations[0].rollback.action, 'RESTORE_PREVIOUS_VALUES');
});

test('Stage 4J field policy separates safe additions, same values and conflicts', () => {
  const base = { field: 'website', provenance: { license: 'ODbL-1.0' }, evidenceConfidence: EVIDENCE_CONFIDENCE.STRONG };
  assert.equal(classifyFieldEvidence({ ...base, oldValue: null, newValue: 'https://a.test' }), FIELD_STATES.SAFE_ADDITION);
  assert.equal(classifyFieldEvidence({ ...base, oldValue: 'https://a.test', newValue: 'https://a.test' }), FIELD_STATES.SAME_VALUE);
  assert.equal(classifyFieldEvidence({ ...base, field: 'name', oldValue: 'A', newValue: 'B' }), FIELD_STATES.CONFLICT_REVIEW);
});

test('Stage 4J dry run never authorizes writes and zero approvals preserve count', () => {
  const dryRun = dryRunApprovedPlan({
    beforeCount: 4166,
    plan: { operations: [] },
    deferredCount: 90,
  });
  assert.equal(dryRun.dryRunOnly, true);
  assert.equal(dryRun.beforeCount, 4166);
  assert.equal(dryRun.proposedAfterCount, 4166);
  assert.equal(dryRun.mutationCount, 0);
});

test('Stage 4J boundary version change invalidates geography only', () => {
  const record = source();
  const first = runIncrementalSync({
    snapshotId: 'A', records: [record],
    versions: { boundaryVersion: 'v72', boundaryHash: 'hash72' },
    resolveRecord: () => ({ decision: 'NEW_CANDIDATE' }),
    classifyGeography: () => 'INSIDE_DANANG',
  });
  const second = runIncrementalSync({
    snapshotId: 'B', records: [record], previousState: first.state,
    previousMediaRegistry: first.mediaRegistry,
    versions: { boundaryVersion: 'v73', boundaryHash: 'hash73' },
    resolveRecord: () => { throw new Error('resolver must not run'); },
    classifyGeography: () => 'OUTSIDE_DANANG',
  });
  assert.equal(second.metrics.geographyProcessed, 1);
  assert.equal(second.metrics.resolutionProcessed, 0);
  assert.equal(second.metrics.mediaProcessed, 0);
  assert.equal(second.metrics.provenanceProcessed, 0);
  assert.equal(second.state[0].geographicEligibility, 'OUTSIDE_DANANG');
});

test('Stage 4J identical boundary rerun skips geography and remains deterministic', () => {
  const versions = { boundaryVersion: 'v72', boundaryHash: 'hash72' };
  const first = runIncrementalSync({
    snapshotId: 'A', records: [source()], versions,
    resolveRecord: () => ({ decision: 'NEW_CANDIDATE' }),
    classifyGeography: () => 'INSIDE_DANANG',
  });
  const repeat = runIncrementalSync({
    snapshotId: 'B', records: [source()], versions,
    previousState: first.state, previousMediaRegistry: first.mediaRegistry,
    classifyGeography: () => { throw new Error('geography must not run'); },
  });
  assert.equal(repeat.metrics.processed, 0);
  assert.equal(repeat.metrics.geographyProcessed, 0);
  assert.equal(repeat.state[0].geographicEligibility, 'INSIDE_DANANG');
});

test('Stage 4J canonical baseline remains exactly 4166 with approved SHA', () => {
  const result = inspectCanonicalDataset(CANONICAL_PATH);
  assert.equal(result.rows, 4166);
  assert.equal(result.sha256, EXPECTED_CANONICAL_SHA);
});
