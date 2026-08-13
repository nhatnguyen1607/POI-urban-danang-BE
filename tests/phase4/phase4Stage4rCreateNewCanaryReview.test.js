const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const config = require('../../config/phase4_stage4r_create_new_canary.json');
const stage4qConfig = require('../../config/phase4_stage4q_create_new_evidence.json');
const {
  RECOMMENDATIONS,
  SUPPORT_CLASSES,
  assertPreparationOnly,
  buildDryRunCreatePlan,
  candidateFingerprint,
  categoryBucket,
  categoryNameConsistent,
  deduplicateEligiblePool,
  duplicatePair,
  evaluateCandidateForReview,
  nameIsTravelerRelevant,
  selectHumanCanary,
  stableProposedCanonicalId,
  validateReviewerDecisions,
} = require('../../src/modules/cityPackPreparation/createNewCanaryReview');
const {
  EVIDENCE_CLASSES,
  buildExistenceIndex,
} = require('../../src/modules/cityPackPreparation/createNewEvidencePolicy');
const { prepareCandidateRecord } = require('../../src/modules/cityPackPreparation/createNewResearchPolicy');
const { inspectCanonicalDataset } = require('../../src/modules/cityPackPreparation/canonicalDataset');
const { stableHash } = require('../../src/modules/cityPackPreparation/incrementalSync');
const {
  decisionRows,
  reviewRow,
  writeCsv,
  writeEvidenceJsonl,
} = require('../../scripts/phase4_stage4r_create_new_canary_review');

const ROOT = path.resolve(__dirname, '..', '..');
const CANONICAL_SHA = '39647b29308813a7ec19e4695fd2b95ffa27743db8e8cb46800c45d3a3fe6ded';
const BOUNDARY = { type: 'Polygon', coordinates: [[
  [108, 15.9], [108.5, 15.9], [108.5, 16.4], [108, 16.4], [108, 15.9],
]] };

function record(overrides = {}) {
  const source = overrides.source || 'overture';
  const sourceId = overrides.sourceId || 'overture:one';
  return prepareCandidateRecord({
    source,
    sourceId,
    name: 'Bao tang du lich bien',
    category: 'museum',
    latitude: 16.06,
    longitude: 108.22,
    address: '12 Bach Dang, Hai Chau, Da Nang',
    phone: '+842361234567',
    website: 'https://baotangdulich.vn',
    openingHours: 'Mo-Fr 08:00-17:00',
    provenance: {
      source,
      sourceId,
      snapshotRef: 'overture-2026-07',
      policyClass: 'OPEN_PERMISSIVE_CANDIDATE',
      license: 'CDLA-Permissive-2.0',
      attribution: 'Overture Maps Foundation',
      licenseUrl: 'https://docs.overturemaps.org/attribution/',
      retrievedAt: '2026-08-10T00:00:00Z',
      upstreamSources: [{ dataset: 'Meta', updateTime: '2026-07-22T00:00:00Z' }],
    },
    ...overrides,
  });
}

function stage4q(overrides = {}) {
  return {
    caseId: 'CREATE_NEW:overture:overture:one',
    source: 'overture',
    sourceId: 'overture:one',
    evidenceClass: EVIDENCE_CLASSES.CANARY,
    existenceClass: 'CLEARLY_ABSENT',
    duplicateStatus: 'RESOLVED_OR_NONE',
    crossSourceSupport: ['osm'],
    supportIndependence: 'INDEPENDENT_SUPPORT',
    historicalEvidence: 'NONE',
    wikidataEvidence: 'NONE',
    identityFingerprint: 'stage4q-fingerprint',
    evidenceDetails: {
      graphEdges: [{ type: 'SAME_PHONE', source: 'osm', sourceId: 'node:1' }],
      independentEvidence: { strongCrossSource: true, pathSatisfied: true },
    },
    ...overrides,
  };
}

function canonical(overrides = {}) {
  return {
    id: 'canonical-far', source: 'canonical', sourceId: 'canonical-far',
    name: 'Cong vien xa', category: 'park', latitude: 16.3, longitude: 108.4,
    address: 'Lien Chieu', aliases: [], externalIds: {}, ...overrides,
  };
}

function existenceIndex(canonicalPois = [canonical()], historicalPois = []) {
  return {
    combined: buildExistenceIndex(canonicalPois, historicalPois),
    thresholds: stage4qConfig.thresholds,
  };
}

function evaluated(overrides = {}) {
  const sourceRecord = overrides.record || record();
  const q = overrides.stage4qResult || stage4q({
    caseId: `CREATE_NEW:${sourceRecord.source}:${sourceRecord.sourceId}`,
    source: sourceRecord.source,
    sourceId: sourceRecord.sourceId,
  });
  return evaluateCandidateForReview({
    record: sourceRecord,
    stage4qResult: q,
    existenceIndex: overrides.existenceIndex || existenceIndex(),
    boundary: overrides.boundary || BOUNDARY,
    config,
    now: config.evaluationReferenceTime,
  });
}

test('current canonical second pass removes a newly matching candidate', () => {
  const sourceRecord = record();
  const result = evaluated({ record: sourceRecord, existenceIndex: existenceIndex([
    canonical({ id: 'current-match', name: sourceRecord.name, category: sourceRecord.category,
      latitude: 16.0601, longitude: 108.2201 }),
  ]) });
  assert.notEqual(result.recommendation, RECOMMENDATIONS.APPROVE);
  assert.ok(result.reasonCodes.includes('current_canonical_or_historical_absence_not_clear'));
});

test('strong independent candidate passes all preparation gates without approval', () => {
  const result = evaluated();
  assert.equal(result.recommendation, RECOMMENDATIONS.APPROVE);
  assert.equal(result.supportClass, SUPPORT_CLASSES.STRONG);
  assert.equal(result.reviewerDecision, 'DEFER');
  assert.equal(result.proposedOperation, 'CREATE_NEW');
});

test('traveler name guard catches taxonomy false positives', () => {
  assert.equal(nameIsTravelerRelevant('Trường THCS Nguyễn Bỉnh Khiêm'), false);
  assert.equal(nameIsTravelerRelevant('Tòa Án Nhân Dân Thành Phố'), false);
  const result = evaluated({ record: record({ name: 'Học sinh THCS Nguyễn Huệ', category: 'market' }) });
  assert.equal(result.recommendation, RECOMMENDATIONS.REJECT);
  assert.equal(nameIsTravelerRelevant('Sông Hàn Montessori Kindergarten'), false);
  assert.equal(nameIsTravelerRelevant('Tân Sơn Container'), false);
  assert.equal(categoryNameConsistent(record({ name: 'Suối Lương - Hai Van Park',
    category: 'accommodation' })), false);
  assert.equal(categoryNameConsistent(record({ name: 'Dominika Apartment',
    category: 'attraction' })), false);
  assert.equal(categoryNameConsistent(record({ name: 'Căn Hộ Mường Thanh Đà Nẵng',
    category: 'attraction' })), false);
});

test('chain candidates require branch-specific evidence', () => {
  const result = evaluated({ record: record({ name: 'Highlands Coffee', category: 'cafe',
    address: null, phone: null, website: null, openingHours: null }) });
  assert.notEqual(result.recommendation, RECOMMENDATIONS.APPROVE);
  assert.ok(result.reasonCodes.includes('chain_branch_identity_insufficient'));
});

test('pairwise duplicate guard detects same phone, official domain and near identity', () => {
  const left = evaluated();
  const right = evaluated({ record: record({ sourceId: 'overture:two', latitude: 16.0601,
    longitude: 108.2201 }), stage4qResult: stage4q({
      caseId: 'CREATE_NEW:overture:overture:two', sourceId: 'overture:two',
    }) });
  assert.ok(duplicatePair(left, right, config));
  const deduped = deduplicateEligiblePool([left, right], config);
  assert.equal(deduped.retained.length, 1);
  assert.equal(deduped.removed.length, 1);
});

test('pairwise duplicate guard catches near alternate landmark names', () => {
  const left = evaluated({ record: record({ sourceId: 'overture:landmark-1',
    name: 'Di tích danh thắng Ngũ Hành Sơn', category: 'attraction', phone: null,
    website: null, openingHours: null }), stage4qResult: stage4q({
      caseId: 'CREATE_NEW:overture:overture:landmark-1', sourceId: 'overture:landmark-1',
    }) });
  const right = evaluated({ record: record({ sourceId: 'overture:landmark-2',
    name: 'Núi Ngũ Hành Sơn - Đà Nẵng', category: 'attraction', latitude: 16.0601,
    longitude: 108.2201, phone: null, website: null, openingHours: null }),
  stage4qResult: stage4q({
    caseId: 'CREATE_NEW:overture:overture:landmark-2', sourceId: 'overture:landmark-2',
  }) });
  assert.equal(duplicatePair(left, right, config).reason, 'near_same_landmark_identity');
});

test('candidate diversity fills documented category buckets deterministically', () => {
  const categories = ['restaurant', 'cafe', 'accommodation', 'resort', 'attraction',
    'museum', 'market', 'shopping', 'tours', 'beach'];
  const candidates = Array.from({ length: 30 }, (_, index) => {
    const category = categories[index % categories.length];
    const sourceRecord = record({ sourceId: `overture:${index}`, category,
      name: `Traveler place ${category} ${index}`, phone: `+84236${String(1000000 + index)}`,
      website: `https://place-${index}.example.vn`, latitude: 16.01 + index * 0.007,
      longitude: 108.11 + (index % 10) * 0.009 });
    return evaluated({ record: sourceRecord, stage4qResult: stage4q({
      caseId: `CREATE_NEW:overture:overture:${index}`, sourceId: `overture:${index}`,
      evidenceDetails: { graphEdges: [], independentEvidence: { strongCrossSource: index < 5 } },
      crossSourceSupport: index < 5 ? ['osm'] : [],
      supportIndependence: index < 5 ? 'INDEPENDENT_SUPPORT' : null,
    }) });
  });
  const first = selectHumanCanary(candidates, config);
  const second = selectHumanCanary([...candidates].reverse(), config);
  assert.ok(first.selected.length >= config.minimumPreferred);
  assert.ok(first.selected.length <= config.maximumSelected);
  assert.deepEqual(first.selected.map((item) => item.caseId), second.selected.map((item) => item.caseId));
  assert.ok(new Set(first.selected.map((item) => categoryBucket(item.record.category))).size >= 4);
  assert.equal(first.selectedDuplicatePairs.length, 0);
});

test('review rows and evidence JSONL are compact with DEFER defaults', () => {
  const selected = [evaluated()];
  const row = reviewRow(selected[0]);
  assert.equal(row.reviewer_decision, 'DEFER');
  assert.equal(row.proposed_operation, 'CREATE_NEW');
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'stage4r-artifact-'));
  const reviewPath = path.join(dir, 'review.csv');
  const evidencePath = path.join(dir, 'evidence.jsonl');
  writeCsv(reviewPath, Object.keys(row), [row]);
  writeEvidenceJsonl(evidencePath, selected);
  assert.match(fs.readFileSync(reviewPath, 'utf8'), /reviewer_decision/);
  assert.equal(JSON.parse(fs.readFileSync(evidencePath, 'utf8').trim()).caseId, selected[0].caseId);
});

test('reviewer decision validation permits only APPROVE, REJECT and DEFER', () => {
  const ids = ['case-1'];
  assert.doesNotThrow(() => validateReviewerDecisions([
    { caseId: 'case-1', reviewerDecision: 'DEFER' },
  ], ids));
  assert.throws(() => validateReviewerDecisions([
    { caseId: 'case-1', reviewerDecision: 'AUTO_APPROVE' },
  ], ids), /Invalid Stage 4R reviewer decision/);
  assert.throws(() => validateReviewerDecisions([], ids), /does not cover/);
});

test('stable proposed canonical ID reuses repository candidate ID policy', () => {
  const first = stableProposedCanonicalId(record(), 'da-nang');
  const second = stableProposedCanonicalId(record(), 'da-nang');
  assert.equal(first, second);
  assert.match(first, /^candidate:da-nang:[a-f0-9]{16}$/);
  assert.notEqual(first, stableProposedCanonicalId(record({ sourceId: 'overture:other' }), 'da-nang'));
});

test('dry-run plan has zero approved CREATE_NEW while decisions remain DEFER', () => {
  const selected = [evaluated()];
  const decisions = decisionRows(selected).map((row) => ({
    caseId: row.case_id, reviewerDecision: row.reviewer_decision, reviewerNote: row.reviewer_note,
  }));
  const plan = buildDryRunCreatePlan(selected, decisions, config);
  assert.equal(plan.approvedCreateNewCount, 0);
  assert.equal(plan.canonicalWritesExecuted, 0);
  assert.equal(plan.proposals[0].reviewDecision, 'DEFER');
  assert.equal(plan.runtimeInstalled, false);
});

test('decision-memory fingerprint covers current absence and duplicate evidence', () => {
  const candidate = evaluated();
  const first = candidateFingerprint(candidate, config);
  const changed = candidateFingerprint({
    ...candidate,
    duplicateStatus: 'UNRESOLVED_DUPLICATE_RISK',
  }, config);
  assert.notEqual(first, changed);
  assert.equal(first, candidateFingerprint(candidate, config));
});

test('same selection produces identical artifact inputs and IDs', () => {
  const items = [evaluated(), evaluated({ record: record({ sourceId: 'overture:two',
    name: 'Khach san bien xanh', category: 'accommodation', phone: '+842369999999',
    website: 'https://bienxanh.vn', latitude: 16.18, longitude: 108.31 }),
  stage4qResult: stage4q({ caseId: 'CREATE_NEW:overture:overture:two', sourceId: 'overture:two',
    crossSourceSupport: [], supportIndependence: null,
    evidenceDetails: { graphEdges: [], independentEvidence: { strongCrossSource: false } } }) })];
  const first = selectHumanCanary(items, config).selected;
  const second = selectHumanCanary([...items].reverse(), config).selected;
  assert.equal(stableHash(first.map(reviewRow)), stableHash(second.map(reviewRow)));
  assert.deepEqual(first.map((item) => item.proposedCanonicalId),
    second.map((item) => item.proposedCanonicalId));
});

test('AUTO_CREATE_NEW remains false and canonical integrity stays exact', () => {
  assert.doesNotThrow(() => assertPreparationOnly(config));
  assert.throws(() => assertPreparationOnly({ ...config, autoCreateNew: true }),
    /AUTO_CREATE_NEW disabled/);
  const canonicalData = inspectCanonicalDataset(path.join(ROOT, 'data', 'canonical',
    'urbanagent_poi_master_v1.csv'));
  assert.equal(canonicalData.rows, 4166);
  assert.equal(canonicalData.sha256, CANONICAL_SHA);
  assert.equal(config.canonicalWrites, 0);
  assert.equal(config.deleteOperations, 0);
  assert.equal(config.runtimeExposure, false);
});
