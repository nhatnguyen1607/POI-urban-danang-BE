const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');

const {
  BOUNDARY_CLASSIFICATIONS,
  classifyAdministrativeBoundary,
  loadBoundaryArtifact,
} = require('../../src/modules/cityPackPreparation/administrativeBoundary');
const { inspectCanonicalDataset, readCanonicalPois } = require('../../src/modules/cityPackPreparation/canonicalDataset');
const { DECISIONS } = require('../../src/modules/cityPackPreparation/entityResolution');
const { EXPECTED_CANONICAL_SHA } = require('../../src/modules/cityPackPreparation/canonicalDataset');
const { STATUSES } = require('../../src/modules/cityPackPreparation/incrementalSync');
const {
  APPLY_OPERATIONS,
  CATEGORY_ELIGIBILITY,
  FIELD_POLICIES,
  NEW_TIERS,
  REVIEW_PRIORITIES,
  VALIDATION_LABELS,
  buildFieldChanges,
  buildPrioritizedReviewQueue,
  calculatePrecisionGate,
  classifyFieldConflict,
  classifyTravelerCategory,
  dryRunApplyPlan,
  evidenceLabelForMatch,
  prepareIncrementalReviewDelta,
  selectCanaryReview,
  triageNewCandidate,
} = require('../../src/modules/cityPackPreparation/reviewApplyPreparation');

const ROOT = path.resolve(__dirname, '..', '..');
const BOUNDARY_PATH = path.join(
  ROOT,
  'data',
  'spikes',
  'phase4',
  'stage4i',
  'boundary',
  'osm-relation-1891418-v72.geojson',
);
const CANONICAL_PATH = path.join(ROOT, 'data', 'canonical', 'urbanagent_poi_master_v1.csv');

function source(overrides = {}) {
  return {
    source: 'overture',
    sourceId: 'overture:test',
    name: 'Cafe Moc',
    category: 'cafe',
    latitude: 16.06,
    longitude: 108.22,
    address: '12 Bach Dang, Hai Chau',
    website: 'https://example.test',
    phone: '+84123456789',
    openingHours: null,
    externalIds: {},
    boundaryClassification: BOUNDARY_CLASSIFICATIONS.INSIDE_DANANG,
    provenance: {
      source: 'overture',
      sourceId: 'overture:test',
      fields: {},
    },
    license: {
      license: 'CDLA-Permissive-2.0',
      policyClass: 'OPEN_PERMISSIVE_CANDIDATE',
      attribution: 'Overture Maps Foundation',
    },
    ...overrides,
  };
}

test('Stage 4I point-in-polygon classifies inside, outside, edge and invalid coordinates', () => {
  const square = {
    type: 'Polygon',
    coordinates: [[[108, 16], [109, 16], [109, 17], [108, 17], [108, 16]]],
  };
  assert.equal(classifyAdministrativeBoundary({ latitude: 16.5, longitude: 108.5 }, square), BOUNDARY_CLASSIFICATIONS.INSIDE_DANANG);
  assert.equal(classifyAdministrativeBoundary({ latitude: 15.5, longitude: 108.5 }, square), BOUNDARY_CLASSIFICATIONS.OUTSIDE_DANANG);
  assert.equal(classifyAdministrativeBoundary({ latitude: 16, longitude: 108.5 }, square), BOUNDARY_CLASSIFICATIONS.BOUNDARY_EDGE);
  assert.equal(classifyAdministrativeBoundary({ latitude: null, longitude: 108.5 }, square), BOUNDARY_CLASSIFICATIONS.INVALID_COORDINATE);
});

test('Stage 4I versioned product polygon keeps Da Nang cases and excludes Hoi An', () => {
  const boundary = loadBoundaryArtifact(BOUNDARY_PATH);
  assert.equal(boundary.properties.sourceIdentifier, 'relation/1891418');
  assert.equal(boundary.properties.sourceVersion, 72);
  assert.equal(boundary.properties.currentLegalBoundary, false);
  assert.equal(classifyAdministrativeBoundary({ latitude: 16.0678, longitude: 108.2208 }, boundary), BOUNDARY_CLASSIFICATIONS.INSIDE_DANANG);
  assert.equal(classifyAdministrativeBoundary({ latitude: 15.9966, longitude: 107.9948 }, boundary), BOUNDARY_CLASSIFICATIONS.INSIDE_DANANG);
  assert.equal(classifyAdministrativeBoundary({ latitude: 16.121, longitude: 108.278 }, boundary), BOUNDARY_CLASSIFICATIONS.INSIDE_DANANG);
  assert.equal(classifyAdministrativeBoundary({ latitude: 15.88, longitude: 108.328 }, boundary), BOUNDARY_CLASSIFICATIONS.OUTSIDE_DANANG);
});

test('Stage 4I never copies a resolver suggestion into an evidence-supported label', () => {
  const result = evidenceLabelForMatch({
    decision: DECISIONS.HIGH_CONFIDENCE_MATCH,
    bestCandidate: {
      fullNameExact: true,
      distanceMeters: 2,
      categoryCompatibility: 1,
      addressSimilarity: 0.1,
    },
  }, source(), { id: 'canonical:1', name: 'Cafe Moc', category: 'cafe', externalIds: {} });
  assert.equal(result.label, VALIDATION_LABELS.UNCERTAIN);
  assert.equal(result.status, 'MACHINE_SUGGESTED');
});

test('Stage 4I supports MATCH only when separate structured evidence is strong', () => {
  const result = evidenceLabelForMatch({
    decision: DECISIONS.HIGH_CONFIDENCE_MATCH,
    bestCandidate: {
      fullNameExact: true,
      distanceMeters: 2,
      categoryCompatibility: 1,
      addressSimilarity: 0.8,
    },
  }, source(), { id: 'canonical:1', name: 'Cafe Moc', category: 'cafe', externalIds: {} });
  assert.equal(result.label, VALIDATION_LABELS.MATCH);
  assert.equal(result.status, 'EVIDENCE_SUPPORTED');
});

test('Stage 4I precision gate excludes UNCERTAIN labels and enforces minimum support', () => {
  const cases = Array.from({ length: 20 }, (_, index) => ({
    stratum: DECISIONS.HIGH_CONFIDENCE_MATCH,
    evidenceSupported: index < 10
      ? { status: 'EVIDENCE_SUPPORTED', label: VALIDATION_LABELS.MATCH }
      : { status: 'MACHINE_SUGGESTED', label: VALIDATION_LABELS.UNCERTAIN },
  }));
  const gate = calculatePrecisionGate(cases, DECISIONS.HIGH_CONFIDENCE_MATCH, { minEvidence: 15 });
  assert.equal(gate.supportedLabels, 10);
  assert.equal(gate.precision, 1);
  assert.equal(gate.status, 'INSUFFICIENT_EVIDENCE');
});

test('Stage 4I traveler category policy reflects the current source and product taxonomy', () => {
  assert.equal(classifyTravelerCategory('restaurant'), CATEGORY_ELIGIBILITY.TRAVELER_RELEVANT);
  assert.equal(classifyTravelerCategory('transportation'), CATEGORY_ELIGIBILITY.CONTEXTUAL);
  assert.equal(classifyTravelerCategory('professional'), CATEGORY_ELIGIBILITY.LOW_VALUE);
  assert.equal(classifyTravelerCategory('unknown'), CATEGORY_ELIGIBILITY.EXCLUDED);
});

test('Stage 4I NEW tiers are conservative and never imply auto-create', () => {
  const strong = triageNewCandidate(source({ raw: { properties: { confidence: 0.95 } } }), {
    bestCandidate: { distanceMeters: 500 },
  }, {
    boundaryClassification: BOUNDARY_CLASSIFICATIONS.INSIDE_DANANG,
    duplicateSourceIds: new Set(),
  });
  assert.equal(strong.tier, NEW_TIERS.A);
  const duplicate = triageNewCandidate(source(), null, {
    boundaryClassification: BOUNDARY_CLASSIFICATIONS.INSIDE_DANANG,
    duplicateSourceIds: new Set(['overture:test']),
  });
  assert.equal(duplicate.tier, NEW_TIERS.C);
  const outside = triageNewCandidate(source(), null, {
    boundaryClassification: BOUNDARY_CLASSIFICATIONS.OUTSIDE_DANANG,
    duplicateSourceIds: new Set(),
  });
  assert.equal(outside.tier, NEW_TIERS.D);
});

test('Stage 4I review queue assigns safety and product-value priorities', () => {
  const records = [
    source({ sourceId: 'high' }),
    source({ sourceId: 'ambiguous' }),
    source({ sourceId: 'probable' }),
    source({ sourceId: 'new', raw: { properties: { confidence: 0.95 } } }),
    source({ sourceId: 'duplicate' }),
  ];
  const recordsById = new Map(records.map((record) => [record.sourceId, record]));
  const match = (sourceId, decision, bestCandidate = null) => ({
    source: 'overture', sourceId, sourceName: sourceId, decision,
    confidence: 0.9, reasonCodes: [], bestCandidate,
  });
  const duplicateRecord = {
    source: 'overture', sourceId: 'duplicate-2', name: 'Cafe Moc', category: 'cafe',
    latitude: 16.06001, longitude: 108.22001, provenance: {}, license: {},
  };
  recordsById.set(duplicateRecord.sourceId, duplicateRecord);
  const queue = buildPrioritizedReviewQueue({
    matches: [
      match('high', DECISIONS.HIGH_CONFIDENCE_MATCH, { canonicalPoiId: 'c1' }),
      match('ambiguous', DECISIONS.AMBIGUOUS, { canonicalPoiId: 'c1' }),
      match('probable', DECISIONS.PROBABLE_MATCH, { canonicalPoiId: 'c1' }),
      match('new', DECISIONS.NEW_CANDIDATE, { distanceMeters: 500 }),
    ],
    duplicates: [{
      source: 'overture', sourceIdA: 'duplicate', sourceIdB: 'duplicate-2',
      nameA: 'Cafe Moc', confidence: 1, distanceMeters: 1, reasonCodes: ['same_phone'],
      records: [recordsById.get('duplicate'), duplicateRecord],
    }],
    recordsById,
    highGate: { status: 'INSUFFICIENT_EVIDENCE' },
  });
  const priorities = Object.fromEntries(queue.map((item) => [item.queueId, item.priority]));
  assert.equal(priorities['MATCH:overture:high'], REVIEW_PRIORITIES.P0);
  assert.equal(priorities['MATCH:overture:ambiguous'], REVIEW_PRIORITIES.P0);
  assert.equal(priorities['MATCH:overture:probable'], REVIEW_PRIORITIES.P1);
  assert.equal(priorities['MATCH:overture:new'], REVIEW_PRIORITIES.P2);
  assert.equal(priorities['DUPLICATE:duplicate:duplicate-2'], REVIEW_PRIORITIES.P3);
});

test('Stage 4I canary is bounded and pins ambiguous, duplicate, and category-edge cases', () => {
  const queue = [];
  for (let index = 0; index < 20; index += 1) {
    queue.push({
      queueId: `high:${index}`, priority: 'P0', decision: DECISIONS.HIGH_CONFIDENCE_MATCH,
      categoryEligibility: CATEGORY_ELIGIBILITY.TRAVELER_RELEVANT, evidence: { reasonCodes: [] },
    });
  }
  for (let index = 0; index < 6; index += 1) {
    queue.push({
      queueId: `ambiguous:${index}`, priority: 'P0', decision: DECISIONS.AMBIGUOUS,
      categoryEligibility: CATEGORY_ELIGIBILITY.TRAVELER_RELEVANT,
      evidence: { reasonCodes: ['chain_branch_collision'] },
    });
  }
  for (const [priority, count] of [['P1', 20], ['P2', 30], ['P3', 20], ['P4', 20]]) {
    for (let index = 0; index < count; index += 1) {
      queue.push({
        queueId: `${priority}:${index}`,
        priority,
        decision: priority === 'P3' ? DECISIONS.SOURCE_DUPLICATE : DECISIONS.NEW_CANDIDATE,
        categoryEligibility: priority === 'P4' && index < 6
          ? CATEGORY_ELIGIBILITY.LOW_VALUE
          : CATEGORY_ELIGIBILITY.TRAVELER_RELEVANT,
        evidence: { reasonCodes: [] },
      });
    }
  }
  const canary = selectCanaryReview(queue, 75);
  assert.equal(canary.length, 75);
  assert.equal(canary.filter((item) => item.decision === DECISIONS.AMBIGUOUS).length, 5);
  assert.equal(canary.filter((item) => item.priority === 'P3').length, 15);
  assert.equal(canary.filter((item) => (
    item.priority === 'P4' && item.categoryEligibility === CATEGORY_ELIGIBILITY.LOW_VALUE
  )).length, 5);
});

test('Stage 4I field conflict policy never overwrites populated canonical values implicitly', () => {
  assert.equal(classifyFieldConflict('address', null, '12 Bach Dang', {}), FIELD_POLICIES.SAFE_ENRICHMENT);
  assert.equal(classifyFieldConflict('address', '12 Bach Dang', '12 Bach Dang', {}), FIELD_POLICIES.NO_CHANGE);
  assert.equal(classifyFieldConflict('address', '12 Bach Dang', '14 Bach Dang', {}), FIELD_POLICIES.CONFLICT_REVIEW);
  assert.equal(classifyFieldConflict('phone', null, null, {}), FIELD_POLICIES.REJECTED_SOURCE_VALUE);
  assert.equal(classifyFieldConflict('media', null, [{ url: 'https://example.test/a.jpg' }], {}), FIELD_POLICIES.REJECTED_SOURCE_VALUE);
});

test('Stage 4I dry-run defaults unapproved work to DEFER and preserves the baseline', () => {
  const canonical = [{ id: 'canonical:1', name: 'Existing', category: 'cafe' }];
  const candidate = source();
  const plan = {
    operations: [{
      operationId: 'one',
      operation: APPLY_OPERATIONS.CREATE_NEW,
      candidateId: candidate.sourceId,
      reviewerStatus: 'PENDING_HUMAN_REVIEW',
      exactFieldChanges: buildFieldChanges(null, candidate),
    }],
  };
  const result = dryRunApplyPlan({
    canonicalPois: canonical,
    applyPlan: plan,
    candidateRecordsById: new Map([[candidate.sourceId, candidate]]),
  });
  assert.equal(result.beforeCount, 1);
  assert.equal(result.proposedAfterCount, 1);
  assert.equal(result.additions, 0);
  assert.equal(result.deferred, 1);
});

test('Stage 4I approved test-only dry-run can enrich empty fields and create without writing canonical', () => {
  const canonical = [{ id: 'canonical:1', name: 'Existing', category: 'cafe', address: null }];
  const candidate = source();
  const enrichChanges = buildFieldChanges(canonical[0], candidate).filter((change) => change.field === 'address');
  const plan = {
    operations: [
      {
        operation: APPLY_OPERATIONS.ENRICH_EXISTING,
        canonicalId: 'canonical:1',
        candidateId: candidate.sourceId,
        reviewerStatus: 'APPROVED',
        exactFieldChanges: enrichChanges,
      },
      {
        operation: APPLY_OPERATIONS.CREATE_NEW,
        candidateId: 'overture:new',
        reviewerStatus: 'APPROVED',
        exactFieldChanges: [],
      },
    ],
  };
  const result = dryRunApplyPlan({
    canonicalPois: canonical,
    applyPlan: plan,
    candidateRecordsById: new Map([
      [candidate.sourceId, candidate],
      ['overture:new', source({ sourceId: 'overture:new', name: 'New Place' })],
    ]),
  });
  assert.equal(result.enrichments, 1);
  assert.equal(result.additions, 1);
  assert.equal(result.proposedAfterCount, 2);
  assert.equal(canonical[0].address, null);
});

test('Stage 4I proposed pack hash is deterministic', () => {
  const input = {
    canonicalPois: [{ id: 'canonical:1', name: 'Existing' }],
    applyPlan: { operations: [] },
    candidateRecordsById: new Map(),
  };
  const first = dryRunApplyPlan(input);
  const second = dryRunApplyPlan(input);
  assert.equal(first.deterministicProposedPackHash, second.deterministicProposedPackHash);
});

test('Stage 4I incremental integration skips unchanged and scopes new/changed/outside work', () => {
  const boundary = {
    type: 'Polygon',
    coordinates: [[[108, 16], [109, 16], [109, 17], [108, 17], [108, 16]]],
  };
  const records = [
    source({ sourceId: 'unchanged' }),
    source({ sourceId: 'new' }),
    source({ sourceId: 'changed' }),
    source({ sourceId: 'outside', latitude: 15 }),
  ];
  const recordsByKey = new Map(records.map((record) => [`${record.source}:${record.sourceId}`, record]));
  let resolutions = 0;
  const result = prepareIncrementalReviewDelta({
    syncState: [
      { source: 'overture', sourceId: 'unchanged', status: STATUSES.UNCHANGED, processing: { identity: false } },
      { source: 'overture', sourceId: 'new', status: STATUSES.NEW, processing: { identity: true } },
      { source: 'overture', sourceId: 'changed', status: STATUSES.CHANGED, processing: { identity: false, media: true } },
      { source: 'overture', sourceId: 'outside', status: STATUSES.NEW, processing: { identity: true } },
    ],
    recordsByKey,
    boundary,
    resolveRecord: () => { resolutions += 1; return { decision: DECISIONS.NEW_CANDIDATE }; },
  });
  assert.equal(result.metrics.unchangedSkipped, 1);
  assert.equal(result.metrics.identityResolved, 1);
  assert.equal(result.metrics.changedNonIdentity, 1);
  assert.equal(result.metrics.outsideExcluded, 1);
  assert.equal(result.metrics.reviewWork, 2);
  assert.equal(resolutions, 1);
});

test('Stage 4I leaves the canonical 4166-POI baseline immutable', () => {
  const before = inspectCanonicalDataset(CANONICAL_PATH);
  assert.equal(before.rows, 4166);
  assert.equal(before.sha256, EXPECTED_CANONICAL_SHA);
  assert.equal(readCanonicalPois(CANONICAL_PATH).length, 4166);
  assert.equal(fs.existsSync(BOUNDARY_PATH), true);
});
