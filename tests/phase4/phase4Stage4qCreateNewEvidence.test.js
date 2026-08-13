const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const config = require('../../config/phase4_stage4q_create_new_evidence.json');
const {
  CONSENSUS_TYPES,
  EVIDENCE_CLASSES,
  EXISTENCE_CLASSES,
  FRESHNESS_CLASSES,
  INDEPENDENCE,
  assertResearchOnly,
  buildExistenceIndex,
  buildSourceEntityGraph,
  buildStage4qFingerprint,
  checkIncrementalReuse,
  classifyFreshness,
  classifySupportIndependence,
  deterministicReviewSample,
  evaluateEvidenceCandidate,
  evaluateStrongPool,
  isOfficialDomain,
  normalizeAddress,
  normalizeDomain,
  normalizePhone,
  secondPassExistenceSearch,
} = require('../../src/modules/cityPackPreparation/createNewEvidencePolicy');
const { prepareCandidateRecord } = require('../../src/modules/cityPackPreparation/createNewResearchPolicy');
const { inspectCanonicalDataset } = require('../../src/modules/cityPackPreparation/canonicalDataset');
const { writeEvidenceJsonl, writeReviewCsv } = require('../../scripts/phase4_stage4q_create_new_evidence');

const ROOT = path.resolve(__dirname, '..', '..');
const NOW = '2026-08-13T08:00:00.000Z';
const CANONICAL_SHA = 'dcb404cc8b5c7a9b5fd70df63039ab8f828c504270e22b12a671fa4ed61583f4';

function record(overrides = {}) {
  const source = overrides.source || 'overture';
  const sourceId = overrides.sourceId || `${source}:one`;
  return prepareCandidateRecord({
    source,
    sourceId,
    name: 'Bao tang song Han',
    category: 'museum',
    latitude: 16.06,
    longitude: 108.22,
    address: '12 Duong Bach Dang, Quan Hai Chau, Da Nang',
    phone: '+84 (236) 123 4567',
    website: 'https://baotangsonghan.vn/visit',
    openingHours: 'Mo-Fr 08:00-17:00',
    externalIds: {},
    operatingStatus: null,
    provenance: {
      source,
      sourceId,
      snapshotRef: `${source}-snapshot`,
      policyClass: 'OPEN_PERMISSIVE_CANDIDATE',
      license: source === 'osm' ? 'ODbL-1.0' : 'CDLA-Permissive-2.0',
      attribution: source === 'osm' ? 'OpenStreetMap contributors' : 'Overture Maps Foundation',
      retrievedAt: '2026-08-10T00:00:00Z',
      upstreamSources: source === 'overture'
        ? [{ dataset: 'Meta business listing', updateTime: '2026-07-22T00:00:00Z' }]
        : [],
    },
    ...overrides,
  });
}

function canonical(overrides = {}) {
  return {
    id: 'canonical-far', globalId: 'canonical-far', source: 'canonical', sourceId: 'canonical-far',
    name: 'Cong vien bien', category: 'park', latitude: 16.2, longitude: 108.3,
    address: 'Son Tra', aliases: [], externalIds: {}, ...overrides,
  };
}

function clearExistence() {
  return {
    existenceClass: EXISTENCE_CLASSES.CLEAR,
    bestCandidate: null,
    candidates: [],
    reasonCodes: ['no_supported_canonical_or_historical_match'],
  };
}

function evaluate(overrides = {}) {
  return evaluateEvidenceCandidate({
    record: overrides.record || record(),
    existence: overrides.existence || clearExistence(),
    graphEntries: overrides.graphEntries || [],
    config,
    now: NOW,
  });
}

test('second-pass canonical search uses spatial/name blocking conservatively', () => {
  const candidate = record();
  const index = buildExistenceIndex([canonical({ id: 'near', name: candidate.name,
    category: candidate.category, latitude: 16.0602, longitude: 108.2202 })]);
  const result = secondPassExistenceSearch(candidate, index, config.thresholds);
  assert.equal(result.existenceClass, EXISTENCE_CLASSES.LIKELY);
  assert.equal(result.bestCandidate.id, 'near');
});

test('historical Global_ID alias prevents a false-new result', () => {
  const candidate = record({ externalIds: { global: 'legacy-42' } });
  const historical = canonical({ evidenceKind: 'historical', id: 'legacy-42', globalId: 'legacy-42',
    name: 'Old name', latitude: 16.061, longitude: 108.221, aliases: ['legacy-42'] });
  const result = secondPassExistenceSearch(candidate, buildExistenceIndex([], [historical]), config.thresholds);
  assert.equal(result.existenceClass, EXISTENCE_CLASSES.LIKELY);
  assert.ok(result.reasonCodes.includes('existing_exact_historical_or_canonical_identifier'));
  const evaluated = evaluate({ record: candidate, existence: result });
  assert.ok(evaluated.evidenceDetails.graphEdges.some((edge) => edge.type === 'HISTORICAL_LINK'));
});

test('canonical Global_ID participates in exact identity search', () => {
  const candidate = record({ externalIds: { global: 'canonical-global-42' } });
  const existing = canonical({ id: 'canonical-global-42' });
  const result = secondPassExistenceSearch(candidate, buildExistenceIndex([existing]), config.thresholds);
  assert.equal(result.existenceClass, EXISTENCE_CLASSES.LIKELY);
  assert.ok(result.bestCandidate.sharedIdentityKeys.includes('global:canonical-global-42'));
});

test('cross-source graph exposes independent support and same-upstream handling', () => {
  const overture = record();
  const independentOsm = record({ source: 'osm', sourceId: 'osm:node:2', latitude: 16.0601,
    longitude: 108.2201, provenance: {
      source: 'osm', sourceId: 'osm:node:2', snapshotRef: 'osm', policyClass: 'OPEN_SHAREALIKE_ISOLATED',
      license: 'ODbL-1.0', attribution: 'OpenStreetMap contributors', retrievedAt: '2026-08-10T00:00:00Z',
      upstreamSources: [],
    } });
  const sharedOsm = record({ source: 'osm', sourceId: 'osm:node:3', provenance: {
    source: 'osm', sourceId: 'osm:node:3', snapshotRef: 'osm', policyClass: 'OPEN_SHAREALIKE_ISOLATED',
    license: 'ODbL-1.0', attribution: 'OpenStreetMap contributors',
  } });
  const overtureFromOsm = record({ provenance: {
    ...overture.provenance,
    upstreamSources: [{ dataset: 'OpenStreetMap' }],
  } });
  assert.equal(classifySupportIndependence(overture, independentOsm), INDEPENDENCE.INDEPENDENT);
  assert.equal(classifySupportIndependence(overtureFromOsm, sharedOsm), INDEPENDENCE.SHARED);
  const graph = buildSourceEntityGraph([overture], [overture, independentOsm], config.thresholds);
  const result = evaluate({ record: overture, graphEntries: graph.get(`overture:${overture.sourceId}`) });
  assert.ok(result.crossSourceSupport.includes('osm'));
  assert.equal(result.consensusType, CONSENSUS_TYPES.STRONG);
});

test('phone, official domain and Vietnamese address normalization preserve evidence', () => {
  assert.equal(normalizePhone('0236 123 4567'), '842361234567');
  assert.equal(normalizePhone('+84 (236) 123-4567'), '842361234567');
  assert.equal(normalizeDomain('www.Example.VN/path'), 'example.vn');
  assert.equal(isOfficialDomain('https://facebook.com/place'), false);
  assert.equal(isOfficialDomain('https://example.vn'), true);
  assert.equal(normalizeAddress('12 Đường Bạch Đằng, Q. Hải Châu, TP Đà Nẵng'),
    normalizeAddress('12 Bach Dang, Quan Hai Chau, Da Nang'));
});

test('spatial same-source duplicate is an unresolved duplicate hard gate', () => {
  const one = record();
  const duplicate = record({ sourceId: 'overture:duplicate', latitude: 16.06005, longitude: 108.22005 });
  const graph = buildSourceEntityGraph([one], [one, duplicate], config.thresholds);
  const result = evaluate({ record: one, graphEntries: graph.get(`overture:${one.sourceId}`) });
  assert.equal(result.evidenceClass, EVIDENCE_CLASSES.DUPLICATE);
  assert.equal(result.duplicateStatus, 'UNRESOLVED_DUPLICATE_RISK');
});

test('chain branch conflict does not become independent canary evidence', () => {
  const left = record({ name: 'Highlands Coffee Chi nhanh Bach Dang', category: 'cafe' });
  const right = record({ source: 'osm', sourceId: 'osm:highlands', name: 'Highlands Coffee Chi nhanh Le Duan',
    category: 'cafe', latitude: 16.0601, longitude: 108.2201, address: '12 Le Duan', provenance: {
      source: 'osm', sourceId: 'osm:highlands', snapshotRef: 'osm', policyClass: 'OPEN_SHAREALIKE_ISOLATED',
      license: 'ODbL-1.0', attribution: 'OpenStreetMap contributors',
    } });
  const graph = buildSourceEntityGraph([left], [left, right], config.thresholds);
  const result = evaluate({ record: left, graphEntries: graph.get(`overture:${left.sourceId}`) });
  assert.notEqual(result.evidenceClass, EVIDENCE_CLASSES.CANARY);
});

test('freshness classes distinguish good, unknown, stale and closed evidence', () => {
  assert.equal(classifyFreshness(record(), NOW, config.thresholds).freshnessClass, FRESHNESS_CLASSES.GOOD);
  assert.equal(classifyFreshness(record({ provenance: { ...record().provenance, retrievedAt: null,
    upstreamSources: [] } }), NOW, config.thresholds).freshnessClass, FRESHNESS_CLASSES.UNKNOWN);
  assert.equal(classifyFreshness(record({ provenance: { ...record().provenance,
    retrievedAt: '2020-01-01T00:00:00Z', upstreamSources: [] } }), NOW,
  config.thresholds).freshnessClass, FRESHNESS_CLASSES.STALE);
  assert.equal(classifyFreshness(record({ operatingStatus: 'permanently closed' }), NOW,
  config.thresholds).freshnessClass, FRESHNESS_CLASSES.CLOSED);
});

test('traveler, provenance and duplicate gates cannot be bypassed', () => {
  assert.equal(evaluate({ record: record({ category: 'dentist' }) }).evidenceClass, EVIDENCE_CLASSES.REJECT);
  assert.equal(evaluate({ record: record({ provenance: {} }) }).evidenceClass, EVIDENCE_CLASSES.REJECT);
  assert.equal(evaluate({ existence: { ...clearExistence(), existenceClass: EXISTENCE_CLASSES.POSSIBLE } })
    .evidenceClass, EVIDENCE_CLASSES.POSSIBLE_EXISTING);
});

test('official identity can qualify only a clear, complete research canary', () => {
  const result = evaluate();
  assert.equal(result.evidenceClass, EVIDENCE_CLASSES.CANARY);
  assert.equal(result.reviewerDecision, 'DEFER');
  assert.ok(result.reasonCodes.includes('research_only_no_canonical_write'));
});

test('single-source name, coordinates and category alone remain strong human review', () => {
  const result = evaluate({ record: record({ phone: null, website: null, openingHours: null }) });
  assert.notEqual(result.evidenceClass, EVIDENCE_CLASSES.CANARY);
  assert.ok([EVIDENCE_CLASSES.STRONG, EVIDENCE_CLASSES.NORMAL].includes(result.evidenceClass));
});

test('zero-canary outcome is a valid deterministic full-pool result', () => {
  const weak = record({ phone: null, website: null, openingHours: null });
  const first = evaluateStrongPool({ strongRecords: [weak], allRecords: [weak], canonicalPois: [canonical()],
    historicalPois: [], config, now: NOW });
  const second = evaluateStrongPool({ strongRecords: [weak], allRecords: [weak], canonicalPois: [canonical()],
    historicalPois: [], config, now: NOW });
  assert.equal(first.summary.classifications.CREATE_CANARY_ELIGIBLE, 0);
  assert.equal(first.summary.deterministicHash, second.summary.deterministicHash);
});

test('review artifacts stay compact and every decision defaults to DEFER', () => {
  const canaryRows = Array.from({ length: 90 }, (_, index) => ({
    ...evaluate({ record: record({ sourceId: `overture:${index}` }) }),
    caseId: `CREATE_NEW:overture:${String(index).padStart(3, '0')}`,
    category: index % 2 ? 'museum' : 'cafe',
  }));
  const negativeRows = Array.from({ length: 60 }, (_, index) => ({
    ...evaluate({ record: record({ sourceId: `overture:negative:${index}` }) }),
    caseId: `CREATE_NEW:overture:negative:${String(index).padStart(3, '0')}`,
    evidenceClass: index % 2 ? EVIDENCE_CLASSES.DUPLICATE : EVIDENCE_CLASSES.NORMAL,
  }));
  const rows = [...canaryRows, ...negativeRows];
  const review = deterministicReviewSample(rows, config);
  assert.ok(review.length <= 100);
  assert.ok(review.filter((item) => item.evidenceClass === EVIDENCE_CLASSES.CANARY).length <= 30);
  assert.ok(new Set(review.filter((item) => item.evidenceClass === EVIDENCE_CLASSES.CANARY)
    .map((item) => item.category)).size >= 2);
  assert.ok(review.some((item) => item.evidenceClass === EVIDENCE_CLASSES.DUPLICATE));
  assert.ok(review.some((item) => item.evidenceClass === EVIDENCE_CLASSES.NORMAL));
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'stage4q-review-'));
  const csv = path.join(dir, 'review.csv');
  const jsonl = path.join(dir, 'details.jsonl');
  writeReviewCsv(csv, review);
  writeEvidenceJsonl(jsonl, review);
  assert.ok(fs.readFileSync(csv, 'utf8').split('\n').slice(1).filter(Boolean)
    .every((line) => line.includes('DEFER')));
  assert.equal(fs.readFileSync(jsonl, 'utf8').trim().split('\n').length, review.length);
});

test('decision fingerprint covers policy evidence and incremental state reuse', () => {
  const result = evaluate();
  const fingerprint = buildStage4qFingerprint(record(), {
    existence: clearExistence(), sourceSupport: result.evidenceDetails.independentEvidence,
  }, config);
  const changed = buildStage4qFingerprint(record({ phone: '+842369999999' }), {
    existence: clearExistence(), sourceSupport: result.evidenceDetails.independentEvidence,
  }, config);
  assert.notEqual(fingerprint, changed);
  const state = { policyVersion: config.policyVersion, evidenceVersion: config.evidenceVersion,
    inputFingerprint: 'same', results: [result] };
  assert.equal(checkIncrementalReuse(state, 'same', config).reusable, true);
  assert.equal(checkIncrementalReuse(state, 'changed', config).reusable, false);
});

test('AUTO_CREATE_NEW is disabled and canonical integrity is unchanged', () => {
  assert.doesNotThrow(() => assertResearchOnly(config));
  assert.throws(() => assertResearchOnly({ ...config, autoCreateNew: true }), /AUTO_CREATE_NEW disabled/);
  const canonicalData = inspectCanonicalDataset(path.join(ROOT, 'data', 'canonical',
    'urbanagent_poi_master_v1.csv'));
  assert.equal(canonicalData.rows, 4173);
  assert.equal(canonicalData.sha256, CANONICAL_SHA);
  assert.equal(config.canonicalWrites, 0);
  assert.equal(config.deleteOperations, 0);
  assert.equal(config.runtimeExposure, false);
});
