const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');

const config = require('../../config/phase4_stage4s_controlled_create_new.json');
const {
  appendCanonicalRows,
  assertStage4SSafety,
  buildApplyPlan,
  buildCreateNewSidecar,
  importCreateNewDecisionMemory,
  mapCanonicalCategory,
  revalidateHumanApprovals,
  resolveHumanDecisionContract,
  safeWebsite,
  validateHumanDecisionRows,
} = require('../../src/modules/cityPackPreparation/controlledCreateNewCanary');
const { stableHash } = require('../../src/modules/cityPackPreparation/decisionMemory');
const { normalizeSidecar } = require('../../src/modules/cityPackPreparation/safeEnrichmentExecutor');
const { CanonicalCsvPoiRepository } = require('../../src/services/canonicalCsvPoiRepository');
const { getTravelerRecommendations } = require('../../src/modules/travelerApiV2/recommendations');
const { buildTripPreview } = require('../../src/modules/travelerApiV2/tripPreview');
const { validateTripPreviewRequest } = require('../../src/modules/travelerApiV2/tripPreviewValidation');

const ROOT = path.resolve(__dirname, '..', '..');
const CANONICAL = path.join(ROOT, 'data', 'canonical', 'urbanagent_poi_master_v1.csv');
const SIDECAR = path.join(ROOT, 'data', 'citypacks', 'enrichments', 'danang',
  'stage4l-enrichment-v1.json');
const STATE = path.join(ROOT, 'data', 'citypacks', 'enrichments', 'danang',
  'stage4s_create_new_state.json');

function record(overrides = {}) {
  return {
    source: 'overture',
    sourceId: 'overture:unit-one',
    name: 'Stage 4S Hotel',
    normalizedName: 'stage 4s hotel',
    category: 'accommodation',
    sourceCategory: 'hotel',
    latitude: 16.05,
    longitude: 108.24,
    address: 'Da Nang, VN',
    phone: '+842361234567',
    website: 'https://example.org/hotel',
    openingHours: null,
    externalIds: {},
    media: null,
    provenance: {
      source: 'overture', sourceId: 'overture:unit-one', snapshotRef: 'snapshot-v1',
      license: 'CDLA-Permissive-2.0', policyClass: 'OPEN_PERMISSIVE_CANDIDATE',
      attribution: 'Overture Maps Foundation', licenseUrl: 'https://cdla.dev/permissive-2-0/',
    },
    license: {
      license: 'CDLA-Permissive-2.0', policyClass: 'OPEN_PERMISSIVE_CANDIDATE',
      attribution: 'Overture Maps Foundation', licenseUrl: 'https://cdla.dev/permissive-2-0/',
    },
    ...overrides,
  };
}

function candidate(overrides = {}) {
  const value = {
    caseId: 'CREATE_NEW:overture:overture:unit-one',
    record: record(),
    stage4qResult: {
      crossSourceSupport: ['osm'], historicalEvidence: 'HISTORICAL_CANDIDATE_FOUND',
      resolverVersion: 'resolver-v1', evidenceVersion: 'evidence-v1',
    },
    currentExistence: { existenceClass: 'CLEARLY_ABSENT', nearestCanonical: null },
    duplicateStatus: 'RESOLVED_OR_NONE',
    travelerRelevance: 'TRAVELER_RELEVANT',
    provenanceStatus: 'PASS',
    licenseStatus: 'PASS',
    boundaryClass: 'INSIDE_DANANG',
    boundaryVersion: 'boundary-v1',
    supportClass: 'STRONG_INDEPENDENT',
    proposedCanonicalId: 'candidate:da-nang:unit-one',
    decisionFingerprint: 'review-fingerprint-one',
    canonicalCategory: 'accommodation',
    ...overrides,
  };
  return value;
}

function memoryAndDecisions(candidates, decisions = ['APPROVE']) {
  const rows = candidates.map((item, index) => ({
    case_id: item.caseId,
    reviewer_decision: decisions[index],
    reviewer_note: 'human review',
  }));
  const memory = importCreateNewDecisionMemory({
    candidates, decisionRows: rows, config, decisionReference: 'human:test',
  });
  return { rows, memory };
}

test('Stage 4S config keeps automatic CREATE_NEW and destructive operations disabled', () => {
  assert.equal(assertStage4SSafety(config), undefined);
  assert.equal(config.autoCreateNew, false);
  assert.equal(config.deleteOperations, 0);
  assert.equal(config.mergeOperations, 0);
  assert.throws(() => assertStage4SSafety({ ...config, autoCreateNew: true }));
});

test('authoritative prefix contract resolves exactly 26 full case IDs and expected names', () => {
  const reviewRows = config.authoritativeHumanDecisions.map((item) => ({
    case_id: item.caseId, name: `${item.nameContains} verified`,
  }));
  const resolved = resolveHumanDecisionContract(reviewRows, config.authoritativeHumanDecisions);
  assert.equal(resolved.length, 26);
  assert.equal(new Set(resolved.map((item) => item.caseId)).size, 26);
  assert.ok(resolved.every((item) => item.caseId.length > 8));
});

test('human decisions require exact 7 APPROVE, 4 REJECT and 15 DEFER allowlist', () => {
  const rows = config.authoritativeHumanDecisions.map((item) => ({
    case_id: item.caseId, reviewer_decision: item.decision,
  }));
  const counts = validateHumanDecisionRows(rows, rows.map((row) => row.case_id),
    new Map(config.authoritativeHumanDecisions.map((item) => [item.caseId, item.decision])));
  assert.deepEqual(counts, { APPROVE: 7, REJECT: 4, DEFER: 15 });
  assert.throws(() => validateHumanDecisionRows(
    rows.map((row, index) => index === 0 ? { ...row, reviewer_decision: 'DEFER' } : row),
    rows.map((row) => row.case_id),
    new Map(config.authoritativeHumanDecisions.map((item) => [item.caseId, item.decision])),
  ), /allowlist mismatch/);
});

test('decision memory imports HUMAN CREATE_NEW_CANARY scope only', () => {
  const item = candidate();
  const { memory } = memoryAndDecisions([item]);
  assert.equal(memory[0].decisionSource, 'HUMAN');
  assert.equal(memory[0].decisionScope, 'CREATE_NEW_CANARY');
  assert.equal(memory[0].applicationStatus, 'APPROVED_NOT_APPLIED');
});

test('stale human approval and non-clear current canonical result are dropped', () => {
  const item = candidate();
  const evidence = new Map([[item.caseId, { fingerprints: { stage4rDecision: 'old' } }]]);
  const result = revalidateHumanApprovals({
    candidates: [item], decisionRows: [{ case_id: item.caseId, reviewer_decision: 'APPROVE' }],
    evidenceByCaseId: evidence, config,
  });
  assert.equal(result.surviving.length, 0);
  assert.ok(result.dropped[0].reasons.includes('STALE_HUMAN_APPROVAL'));
  const possible = candidate({ decisionFingerprint: 'same',
    currentExistence: { existenceClass: 'POSSIBLE_EXISTING' } });
  const second = revalidateHumanApprovals({
    candidates: [possible], decisionRows: [{ case_id: possible.caseId, reviewer_decision: 'APPROVE' }],
    evidenceByCaseId: new Map([[possible.caseId, { fingerprints: { stage4rDecision: 'same' } }]]), config,
  });
  assert.ok(second.dropped[0].reasons.includes('CURRENT_CANONICAL_NOT_CLEARLY_ABSENT'));
});

test('source duplicate, polygon, traveler, category and provenance gates cannot be bypassed', () => {
  const item = candidate({ duplicateStatus: 'UNRESOLVED_DUPLICATE_RISK',
    travelerRelevance: 'NOT_TRAVELER_RELEVANT', provenanceStatus: 'REJECT',
    boundaryClass: 'OUTSIDE_DANANG', record: record({ category: 'electronics' }) });
  const result = revalidateHumanApprovals({
    candidates: [item], decisionRows: [{ case_id: item.caseId, reviewer_decision: 'APPROVE' }],
    evidenceByCaseId: new Map([[item.caseId,
      { fingerprints: { stage4rDecision: item.decisionFingerprint } }]]), config,
  });
  assert.equal(result.surviving.length, 0);
  assert.ok(result.dropped[0].reasons.includes('SOURCE_DUPLICATE_UNRESOLVED'));
  assert.ok(result.dropped[0].reasons.includes('UNSUPPORTED_CANONICAL_CATEGORY'));
});

test('pairwise approved duplicate guard removes both unresolved candidates', () => {
  const left = candidate();
  const right = candidate({ caseId: 'CREATE_NEW:overture:overture:unit-two',
    proposedCanonicalId: 'candidate:da-nang:unit-two',
    record: record({ sourceId: 'overture:unit-two', latitude: 16.05001 }) });
  const decisions = [left, right].map((item) => ({ case_id: item.caseId, reviewer_decision: 'APPROVE' }));
  const evidence = new Map([left, right].map((item) => [item.caseId,
    { fingerprints: { stage4rDecision: item.decisionFingerprint } }]));
  const result = revalidateHumanApprovals({ candidates: [left, right],
    decisionRows: decisions, evidenceByCaseId: evidence, config });
  assert.equal(result.surviving.length, 0);
  assert.equal(result.duplicatePairs.length, 1);
});

test('canonical category mapping and deterministic IDs stay bounded', () => {
  assert.equal(mapCanonicalCategory(candidate(), config), 'accommodation');
  assert.equal(mapCanonicalCategory(candidate({ record: record({ category: 'electronics' }) }), config), null);
  const ids = config.authoritativeHumanDecisions.filter((item) => item.decision === 'APPROVE')
    .map((item) => `candidate:da-nang:${stableHash(item.caseId).slice(0, 16)}`);
  assert.equal(new Set(ids).size, 7);
});

test('apply plan includes only exact HUMAN APPROVE and excludes REJECT, DEFER and policy-only cases', () => {
  const approved = candidate();
  const { memory } = memoryAndDecisions([approved]);
  const plan = buildApplyPlan({ surviving: [approved], decisionMemory: memory, config });
  assert.equal(plan.operations.length, 1);
  assert.equal(plan.operations[0].decisionSource, 'HUMAN');
  assert.deepEqual(plan.forbiddenOperations,
    { reject: 0, defer: 0, policyOnly: 0, delete: 0, merge: 0, existingRowMutation: 0 });
  assert.throws(() => buildApplyPlan({ surviving: [approved],
    decisionMemory: [{ ...memory[0], decisionSource: 'POLICY' }], config }), /allowlist/);
});

test('canonical append preserves every existing byte and appends only new rows', () => {
  const headers = ['Global_ID', 'City_ID', 'Entity_Type', 'Restaurant Name', 'Category_Normalized',
    'Lat', 'Lon', 'Source'];
  const before = `${headers.join(',')}\nold,da-nang,poi,Old,cafe,16,108,google_maps\n`;
  const after = appendCanonicalRows(before, [candidate()], headers);
  assert.ok(after.startsWith(before));
  assert.equal(after.split('\n').filter(Boolean).length, 3);
  assert.ok(after.includes('candidate:da-nang:unit-one'));
});

test('new canonical sidecar records are deterministic, non-runtime and omit Google URLs', () => {
  const item = candidate({ record: record({ website: 'https://maps.app.goo.gl/test' }) });
  const { memory } = memoryAndDecisions([item]);
  const input = { existingSidecar: null, candidates: [item],
    memoryByCaseId: new Map([[item.caseId, memory[0]]]), config };
  const first = buildCreateNewSidecar(input);
  const second = buildCreateNewSidecar(input);
  assert.equal(first.text, second.text);
  assert.equal(first.normalized.runtimeEnabled, false);
  assert.ok(first.created.some((row) => row.field === 'externalIds'));
  assert.ok(first.created.some((row) => row.field === 'phone'));
  assert.equal(first.created.some((row) => row.field === 'website'), false);
  assert.equal(safeWebsite('https://example.org'), 'https://example.org');
});

test('current Stage 4S branch has exact append, sidecar, provenance, rollback and deterministic state', () => {
  const state = JSON.parse(fs.readFileSync(STATE, 'utf8'));
  const sidecar = normalizeSidecar(JSON.parse(fs.readFileSync(SIDECAR, 'utf8')));
  assert.equal(state.canonicalRowsBefore, 4166);
  assert.equal(state.canonicalRowsAfter, 4166 + state.appliedCaseIds.length);
  assert.ok(state.appliedCaseIds.length <= 7);
  assert.equal(state.autoCreateNew, false);
  assert.equal(state.runtimeSidecarEnabled, false);
  assert.equal(state.rollback.canonicalByteIdentical, true);
  assert.equal(state.rollback.sidecarExact, true);
  assert.equal(state.deterministic, true);
  const newSidecar = sidecar.records.filter((row) => state.appliedCanonicalIds.includes(row.canonicalId));
  assert.equal(newSidecar.length, state.sidecarRecordsCreated);
  assert.ok(newSidecar.every((row) => row.decisionSource === 'HUMAN'
    && row.provenance && row.license && !/google/i.test(row.source)));
  assert.equal(new Set(newSidecar.map((row) => row.recordId)).size, newSidecar.length);
});

test('runtime loader sees all applied POIs with unique IDs and valid coordinates', async () => {
  const state = JSON.parse(fs.readFileSync(STATE, 'utf8'));
  const repository = new CanonicalCsvPoiRepository({ filePath: CANONICAL });
  const pois = await repository.loadAll();
  const quality = await repository.getQualityReport();
  assert.equal(pois.length, state.canonicalRowsAfter);
  assert.equal(quality.totals.invalidRows, 0);
  assert.equal(new Set(pois.map((poi) => poi.id)).size, pois.length);
  const applied = pois.filter((poi) => state.appliedCanonicalIds.includes(poi.id));
  assert.equal(applied.length, state.appliedCanonicalIds.length);
  assert.ok(applied.every((poi) => Number.isFinite(poi.lat) && Number.isFinite(poi.lon)
    && poi.categoryNormalized));
});

test('recommendation and trip preview smoke remain nonempty after controlled append', async () => {
  const recommendation = await getTravelerRecommendations({
    query: 'cafe và điểm tham quan ở Đà Nẵng', context: {}, limit: 5, cityId: 'da-nang',
  });
  assert.ok(recommendation.recommendations.length > 0);
  const validation = validateTripPreviewRequest({
    cityId: 'da-nang', query: 'cafe và điểm tham quan ở Đà Nẵng',
    trip: { dayCount: 1, date: '2026-08-20', transport: 'motorbike', pace: 'balanced',
      dailyWindow: { start: '08:00', end: '18:00' } },
    startLocation: { lat: 16.0544, lon: 108.2022 },
    constraints: { maxStopsPerDay: 3 }, recommendationOptions: { limit: 8 },
  });
  assert.equal(validation.errors, undefined);
  const preview = await buildTripPreview(validation.value);
  assert.equal(preview.error, undefined);
  assert.ok(preview.trip.stops.length > 0);
});
