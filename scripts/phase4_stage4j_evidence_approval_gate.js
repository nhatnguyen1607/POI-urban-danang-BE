const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const { performance } = require('node:perf_hooks');

const {
  BOUNDARY_CLASSIFICATIONS,
  loadBoundaryArtifact,
  summarizeBoundaryClassifications,
} = require('../src/modules/cityPackPreparation/administrativeBoundary');
const {
  inspectCanonicalDataset,
  parseCsv,
  sha256,
} = require('../src/modules/cityPackPreparation/canonicalDataset');
const { normalizeWithAdapters } = require('../src/modules/cityPackPreparation/dryRun');
const { DECISIONS, resolveSourceRecords } = require('../src/modules/cityPackPreparation/entityResolution');
const {
  APPLY_OPERATIONS,
  EVIDENCE_CONFIDENCE,
  FIELD_STATES,
  REVIEW_DECISIONS,
  buildApprovedApplyPlan,
  calculateEvidenceGate,
  classifyFieldEvidence,
  deterministicSelect,
  dryRunApprovedPlan,
  evaluateMatchEvidence,
  evaluateNewCandidate,
  revalidateDuplicateEvidence,
} = require('../src/modules/cityPackPreparation/evidenceApprovalGate');
const { stableHash } = require('../src/modules/cityPackPreparation/incrementalSync');
const { loadRealSnapshotRecords } = require('../src/modules/cityPackPreparation/realSnapshot');
const {
  NEW_TIERS,
  buildFieldChanges,
  buildValidationDataset,
  triageNewCandidate,
} = require('../src/modules/cityPackPreparation/reviewApplyPreparation');
const { normalizeName } = require('../src/modules/cityPackPreparation/sourceRecord');
const { verifySnapshots } = require('./phase4_stage4h_full_city_comparison');
const {
  BOUNDARY_PATH,
  CANONICAL_PATH,
  readCanonicalEvidencePois,
} = require('./phase4_stage4i_review_apply_preparation');

const ROOT = path.resolve(__dirname, '..');
const OUTPUT_DIR = path.join(ROOT, 'data', 'spikes', 'phase4', 'stage4j');
const STAGE4I_DIR = path.join(ROOT, 'data', 'spikes', 'phase4', 'stage4i');
const MANIFEST_PATH = path.join(ROOT, 'data', 'spikes', 'phase4', 'stage4g', 'snapshot_manifest.json');
const STAGE4G_SUMMARY_PATH = path.join(ROOT, 'data', 'spikes', 'phase4', 'stage4g', 'stage4g_summary.json');
const EXPECTED_BOUNDARY_SHA = 'c0db1b84c8fcbc77b1b5cca14ba6d280627f59713db56965bd6198b72e78a84d';
const EXPECTED_CANONICAL_SHA = '5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae';
const PROCESSING_VERSIONS = Object.freeze({
  evidence: 'stage4j-independent-evidence-v1',
  approval: 'stage4j-human-gate-v1',
  boundary: 'osm-relation-1891418-v72-product-scope-v1',
});

function normalizedTextSha256(filePath) {
  const normalized = fs.readFileSync(filePath, 'utf8').replace(/\r\n/g, '\n');
  return crypto.createHash('sha256').update(normalized, 'utf8').digest('hex');
}

function snapshotPaths(cacheDir) {
  return {
    overture: path.join(cacheDir, 'overture_full_danang.json'),
    osm: path.join(cacheDir, 'osm_full_danang.json'),
    wikidata: path.join(cacheDir, 'wikidata_full_danang.json'),
    commons: path.join(cacheDir, 'wikimedia_commons_full_danang_metadata.json'),
  };
}

function insideBbox(record, boundary) {
  return Number.isFinite(record.latitude)
    && Number.isFinite(record.longitude)
    && record.longitude >= boundary.west
    && record.longitude <= boundary.east
    && record.latitude >= boundary.south
    && record.latitude <= boundary.north;
}

function readCsvObjects(filePath) {
  const rows = parseCsv(fs.readFileSync(filePath, 'utf8'));
  const headers = rows[0].map((header, index) => (
    index === 0 ? header.replace(/^\uFEFF/, '').trim() : header.trim()
  ));
  return rows.slice(1).map((cells) => Object.fromEntries(
    headers.map((header, index) => [header, cells[index] ?? '']),
  ));
}

function numberOrNull(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function loadHistoricalIndex(filePath) {
  const records = readCsvObjects(filePath);
  const byPlaceId = new Map();
  for (const row of records) {
    if (!row.place_id) continue;
    byPlaceId.set(row.place_id, {
      placeId: row.place_id,
      name: row.name || null,
      category: row.category || null,
      latitude: numberOrNull(row.lat),
      longitude: numberOrNull(row.lng),
      address: row.address || null,
      website: null,
      phone: row.phone || null,
    });
  }
  return byPlaceId;
}

function compactSource(record) {
  return {
    source: record?.source || null,
    sourceId: record?.sourceId || null,
    originalName: record?.name || null,
    normalizedName: normalizeName(record?.name),
    category: record?.category || null,
    latitude: record?.latitude ?? null,
    longitude: record?.longitude ?? null,
    address: record?.address || null,
    website: record?.website || null,
    phone: record?.phone || null,
    externalIds: record?.externalIds || {},
  };
}

function compactCanonical(record) {
  return {
    id: record?.id || null,
    name: record?.name || null,
    normalizedName: normalizeName(record?.name),
    category: record?.category || null,
    latitude: record?.latitude ?? null,
    longitude: record?.longitude ?? null,
    address: record?.address || null,
    website: record?.website || null,
    phone: null,
    externalIds: record?.externalIds || {},
  };
}

function provenanceFor(record) {
  return {
    source: record?.provenance?.source || record?.source || null,
    sourceId: record?.provenance?.sourceId || record?.sourceId || null,
    snapshotRef: record?.provenance?.snapshotRef || null,
    license: record?.license?.license || null,
    policyClass: record?.license?.policyClass || null,
    attribution: record?.license?.attribution || null,
    licenseUrl: record?.license?.licenseUrl || null,
  };
}

function fieldChanges(sourceRecord, canonicalRecord, evidenceConfidence) {
  return buildFieldChanges(canonicalRecord, sourceRecord).map((change) => ({
    field: change.field,
    oldValue: change.oldValue,
    newValue: change.proposedValue,
    state: classifyFieldEvidence({
      field: change.field,
      oldValue: change.oldValue,
      newValue: change.proposedValue,
      provenance: change.fieldProvenance,
      evidenceConfidence,
    }),
    provenance: change.fieldProvenance,
  }));
}

function makeMatchCase(match, sourceRecord, canonicalRecord, historicalRecord) {
  const evidence = evaluateMatchEvidence({ match, sourceRecord, canonicalRecord, historicalRecord });
  const provenance = provenanceFor(sourceRecord);
  return {
    caseId: `MATCH:${match.source}:${match.sourceId}`,
    priority: match.decision === DECISIONS.HIGH_CONFIDENCE_MATCH ? 'P0' : 'P1',
    stratum: match.decision,
    resolver: {
      classification: match.decision,
      confidence: match.confidence,
      reasonCodes: match.reasonCodes,
    },
    source: compactSource(sourceRecord),
    canonical: compactCanonical(canonicalRecord),
    sourceIds: [sourceRecord.sourceId],
    evidence,
    historicalEvidence: {
      concreteRecordLink: evidence.checks.historicalLink,
      sourceRepository: 'poi_urban',
      linkageField: evidence.checks.historicalLink ? 'RestaurantID=place_id' : null,
      restrictedPayloadCommitted: false,
    },
    provenance,
    license: sourceRecord?.license || null,
    provenanceComplete: Boolean(provenance.source && provenance.sourceId
      && provenance.license && provenance.attribution),
    fieldChanges: fieldChanges(sourceRecord, canonicalRecord, evidence.confidence),
    reviewerDecision: REVIEW_DECISIONS.DEFER,
  };
}

function diverseEvidenceSelection(cases, { count, minimumUsable }) {
  const usable = deterministicSelect(cases.filter((item) => item.evidence.independent), minimumUsable,
    (item) => `${item.source.source}:${item.source.category}:${Math.floor(item.source.latitude * 100)}:${item.caseId}`);
  const selectedIds = new Set(usable.map((item) => item.caseId));
  const remainder = deterministicSelect(cases.filter((item) => !selectedIds.has(item.caseId)), count - usable.length,
    (item) => `${item.source.source}:${item.source.category}:${Math.floor(item.source.longitude * 100)}:${item.caseId}`);
  return [...usable, ...remainder].sort((left, right) => left.caseId.localeCompare(right.caseId));
}

function makeNewCase(match, record, duplicateIds) {
  const evidence = evaluateNewCandidate({
    match,
    record,
    insideBoundary: [BOUNDARY_CLASSIFICATIONS.INSIDE_DANANG, BOUNDARY_CLASSIFICATIONS.BOUNDARY_EDGE]
      .includes(record.boundaryClassification),
    sourceDuplicate: duplicateIds.has(record.sourceId),
  });
  const provenance = provenanceFor(record);
  return {
    caseId: `NEW:${match.source}:${match.sourceId}`,
    priority: 'P2',
    stratum: 'NEW_TIER_A',
    resolver: {
      classification: match.decision,
      confidence: match.confidence,
      reasonCodes: match.reasonCodes,
    },
    source: compactSource(record),
    canonical: null,
    sourceIds: [record.sourceId],
    evidence,
    provenance,
    license: record.license || null,
    provenanceComplete: Boolean(provenance.source && provenance.sourceId
      && provenance.license && provenance.attribution),
    fieldChanges: fieldChanges(record, null, evidence.confidence),
    reviewerDecision: REVIEW_DECISIONS.DEFER,
  };
}

function countValues(items, valueFunction, initial = []) {
  const counts = Object.fromEntries(initial.map((value) => [value, 0]));
  for (const item of items) {
    const value = valueFunction(item);
    counts[value] = (counts[value] || 0) + 1;
  }
  return counts;
}

function csvEscape(value) {
  if (value === null || value === undefined) return '';
  const text = Array.isArray(value) ? value.join('|') : String(value);
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function writeCsv(filePath, rows, fields) {
  const body = [fields.join(','), ...rows.map((row) => fields
    .map((field) => csvEscape(row[field])).join(','))].join('\n');
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${body}\n`, 'utf8');
}

function writeJson(filePath, value) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${JSON.stringify(value)}\n`, 'utf8');
}

function writeJsonl(filePath, records) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${records.map((item) => JSON.stringify(item)).join('\n')}\n`, 'utf8');
}

function reviewRows(cases) {
  return cases.map((item) => ({
    case_id: item.caseId,
    priority: item.priority,
    resolver_class: item.resolver.classification,
    source: item.source.source,
    source_id: item.source.sourceId,
    source_name: item.source.originalName,
    canonical_id: item.canonical?.id || '',
    canonical_name: item.canonical?.name || '',
    distance_m: item.evidence.checks?.resolverDistanceMeters
      ?? item.evidence.checks?.nearestCanonicalDistanceMeters ?? '',
    source_address: item.source.address || '',
    canonical_address: item.canonical?.address || '',
    same_phone: item.evidence.checks?.samePhone || false,
    same_website: item.evidence.checks?.sameWebsite || false,
    exact_external_id: item.evidence.checks?.exactExternalId || false,
    historical_link: item.evidence.checks?.historicalLink || false,
    evidence_label: item.evidence.label,
    evidence_confidence: item.evidence.confidence,
    reason_codes: item.evidence.reasonCodes,
    reviewer_decision: REVIEW_DECISIONS.DEFER,
    approved_operation: '',
    reviewer_note: '',
  }));
}

function decisionRows(cases) {
  return cases.map((item) => ({
    case_id: item.caseId,
    reviewer_decision: REVIEW_DECISIONS.DEFER,
    approved_operation: '',
    reviewer_note: '',
  }));
}

function readDecisions(filePath) {
  if (!fs.existsSync(filePath)) return null;
  return readCsvObjects(filePath).map((row) => ({
    caseId: row.case_id,
    reviewerDecision: row.reviewer_decision,
    approvedOperation: row.approved_operation || null,
    reviewerNote: row.reviewer_note || '',
  }));
}

function boundaryPolicy(boundary) {
  return {
    status: 'PASS_VERSIONED_PRODUCT_SCOPE',
    cityId: 'da-nang',
    scope: 'PRE_2025_MERGER_DA_NANG_CITY_PACK',
    currentLegalBoundary: false,
    source: boundary.properties.source,
    sourceIdentifier: boundary.properties.sourceIdentifier,
    relationId: 1891418,
    sourceVersion: 72,
    crs: 'EPSG:4326',
    artifactPath: 'data/spikes/phase4/stage4i/boundary/osm-relation-1891418-v72.geojson',
    artifactSha256: EXPECTED_BOUNDARY_SHA,
    hashNormalization: 'UTF-8 text with LF line endings',
    inclusionRule: 'Include INSIDE_DANANG and BOUNDARY_EDGE records after point-in-polygon classification.',
    boundaryEdgeRule: 'Retain edge points for review; do not reject them solely for numerical boundary contact.',
    coastalRule: 'Polygon membership is authoritative; islands, peninsulas, bridges, and coastal POIs are not removed by a simplistic land mask.',
    acquisitionVsFinal: 'The acquisition bounding box limits retrieval only; the versioned polygon controls final geographic eligibility.',
    futureUpdateRule: 'A new boundary version/hash requires explicit scope review and geography-only reclassification before use.',
    intendedScopeEvidence: 'Project context explicitly keeps Hoi An as a separate future City Pack.',
    approvalMeaning: 'This policy approves evaluation scope only and does not authorize canonical writes.',
  };
}

function expectedOutputPaths() {
  return [
    'boundary_policy.json',
    'evidence_pack.jsonl',
    'human_review.csv',
    'review_decisions.csv',
    'approved_apply_plan.json',
    'stage4j_summary.json',
  ].map((name) => path.join(OUTPUT_DIR, name));
}

function inputFingerprint({ historicalPath }) {
  const decisionsPath = path.join(OUTPUT_DIR, 'review_decisions.csv');
  return stableHash({
    canonicalSha: sha256(CANONICAL_PATH),
    boundarySha: normalizedTextSha256(BOUNDARY_PATH),
    snapshotManifestSha: sha256(MANIFEST_PATH),
    historicalEvidenceSha: sha256(historicalPath),
    decisionFileSha: fs.existsSync(decisionsPath) ? sha256(decisionsPath) : null,
    processingVersions: PROCESSING_VERSIONS,
  });
}

function existingCacheResult(fingerprint) {
  const summaryPath = path.join(OUTPUT_DIR, 'stage4j_summary.json');
  if (!expectedOutputPaths().every((filePath) => fs.existsSync(filePath))) return null;
  const summary = JSON.parse(fs.readFileSync(summaryPath, 'utf8'));
  if (summary.inputFingerprint !== fingerprint) return null;
  return {
    ...summary,
    noChange: {
      cacheHit: true,
      poiReresolution: 0,
      mediaProcessing: 0,
      evidenceRegeneration: 0,
      applyPlanChanges: 0,
      deterministicHash: summary.deterministicHash,
    },
  };
}

function runStage4j({ cacheDir, historicalPath }) {
  const beforeFingerprint = inputFingerprint({ historicalPath });
  const cached = existingCacheResult(beforeFingerprint);
  if (cached) return cached;

  const started = performance.now();
  const canonicalBefore = inspectCanonicalDataset(CANONICAL_PATH);
  if (canonicalBefore.rows !== 4166 || canonicalBefore.sha256 !== EXPECTED_CANONICAL_SHA) {
    throw new Error('Canonical baseline mismatch before Stage 4J.');
  }
  if (normalizedTextSha256(BOUNDARY_PATH) !== EXPECTED_BOUNDARY_SHA) {
    throw new Error('Boundary artifact normalized content hash mismatch.');
  }

  const manifest = JSON.parse(fs.readFileSync(MANIFEST_PATH, 'utf8'));
  const stage4gSummary = JSON.parse(fs.readFileSync(STAGE4G_SUMMARY_PATH, 'utf8'));
  const paths = snapshotPaths(cacheDir);
  verifySnapshots(manifest, paths);
  const canonicalPois = readCanonicalEvidencePois(CANONICAL_PATH);
  const canonicalById = new Map(canonicalPois.map((item) => [item.id, item]));
  const historicalByPlaceId = loadHistoricalIndex(historicalPath);
  const boundary = loadBoundaryArtifact(BOUNDARY_PATH);
  const normalized = normalizeWithAdapters(loadRealSnapshotRecords(paths)
    .filter((record) => insideBbox(record, stage4gSummary.boundary)));
  const boundaryResult = summarizeBoundaryClassifications(normalized, boundary);
  const eligibleRecords = boundaryResult.classified.filter((record) => (
    [BOUNDARY_CLASSIFICATIONS.INSIDE_DANANG, BOUNDARY_CLASSIFICATIONS.BOUNDARY_EDGE]
      .includes(record.boundaryClassification)
  ));
  const recordsById = new Map(eligibleRecords.map((item) => [item.sourceId, item]));
  const resolution = resolveSourceRecords(eligibleRecords, canonicalPois, undefined, {
    useSpatialIndex: true,
    useSpatialDuplicateIndex: true,
    includeCrossSourceDuplicates: true,
    maxCandidateDistanceMeters: 800,
    hardened: true,
    includeMetrics: true,
    resolutionVersion: 'stage4i-v1',
  });

  const matchCases = resolution.matches
    .filter((match) => [DECISIONS.HIGH_CONFIDENCE_MATCH, DECISIONS.PROBABLE_MATCH]
      .includes(match.decision))
    .map((match) => {
      const sourceRecord = recordsById.get(match.sourceId);
      const canonicalRecord = canonicalById.get(match.bestCandidate?.canonicalPoiId);
      const placeId = canonicalRecord?.evidenceMetadata?.restaurantIdIsSourceIdentifier;
      return makeMatchCase(match, sourceRecord, canonicalRecord, historicalByPlaceId.get(placeId));
    });
  const highCases = diverseEvidenceSelection(
    matchCases.filter((item) => item.stratum === DECISIONS.HIGH_CONFIDENCE_MATCH),
    { count: 40, minimumUsable: 30 },
  );
  const probableCases = diverseEvidenceSelection(
    matchCases.filter((item) => item.stratum === DECISIONS.PROBABLE_MATCH),
    { count: 30, minimumUsable: 20 },
  );

  const stage4iValidation = fs.readFileSync(path.join(STAGE4I_DIR, 'validation_labels.jsonl'), 'utf8')
    .split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));
  const duplicateEvidence = stage4iValidation
    .filter((item) => item.stratum === DECISIONS.SOURCE_DUPLICATE)
    .map((item) => ({ ...item, stage4jEvidence: revalidateDuplicateEvidence(item) }));
  const duplicateUsable = duplicateEvidence.filter((item) => item.stage4jEvidence.independent);

  const duplicateIds = new Set(resolution.duplicates.flatMap((item) => [item.sourceIdA, item.sourceIdB]));
  const newTierA = resolution.matches
    .filter((match) => match.decision === DECISIONS.NEW_CANDIDATE)
    .map((match) => ({ match, record: recordsById.get(match.sourceId) }))
    .filter(({ match, record }) => triageNewCandidate(record, match, { duplicateSourceIds: duplicateIds }).tier === NEW_TIERS.A)
    .map(({ match, record }) => makeNewCase(match, record, duplicateIds));
  const newCases = deterministicSelect(newTierA, 20,
    (item) => `${item.source.source}:${item.source.category}:${Math.floor(item.source.latitude * 100)}:${item.caseId}`);
  const evidenceCases = [...highCases, ...probableCases, ...newCases]
    .sort((left, right) => left.caseId.localeCompare(right.caseId));

  const highGate = calculateEvidenceGate(highCases, { minimum: 30, requiredPrecision: 0.98 });
  const probableGate = calculateEvidenceGate(probableCases, { minimum: 20, requiredPrecision: 0.99 });
  const duplicateCorrect = duplicateUsable.filter((item) => item.stage4jEvidence.label
    === 'DUPLICATE_SAME_PLACE').length;
  const duplicateGate = {
    usable: duplicateUsable.length,
    correct: duplicateCorrect,
    precision: duplicateUsable.length ? Number((duplicateCorrect / duplicateUsable.length).toFixed(4)) : null,
    status: duplicateUsable.length >= 20 && duplicateCorrect === duplicateUsable.length ? 'PASS' : 'FAIL',
  };

  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  const decisionsPath = path.join(OUTPUT_DIR, 'review_decisions.csv');
  if (!fs.existsSync(decisionsPath)) {
    writeCsv(decisionsPath, decisionRows(evidenceCases), [
      'case_id', 'reviewer_decision', 'approved_operation', 'reviewer_note',
    ]);
  }
  const decisions = readDecisions(decisionsPath);
  const plan = buildApprovedApplyPlan({ cases: evidenceCases, decisions });
  const decisionCounts = countValues(decisions, (item) => item.reviewerDecision,
    Object.values(REVIEW_DECISIONS));
  const dryRun = dryRunApprovedPlan({
    beforeCount: canonicalBefore.rows,
    plan,
    deferredCount: decisionCounts.DEFER,
    rejectedCount: decisionCounts.REJECT,
  });
  const allFieldChanges = evidenceCases.flatMap((item) => item.fieldChanges || []);
  const fieldCounts = countValues(allFieldChanges, (item) => item.state, Object.values(FIELD_STATES));
  const confidenceCounts = countValues(evidenceCases, (item) => item.evidence.confidence,
    Object.values(EVIDENCE_CONFIDENCE));
  const policy = boundaryPolicy(boundary);

  writeJson(path.join(OUTPUT_DIR, 'boundary_policy.json'), policy);
  writeJsonl(path.join(OUTPUT_DIR, 'evidence_pack.jsonl'), evidenceCases);
  writeCsv(path.join(OUTPUT_DIR, 'human_review.csv'), reviewRows(evidenceCases), [
    'case_id', 'priority', 'resolver_class', 'source', 'source_id', 'source_name',
    'canonical_id', 'canonical_name', 'distance_m', 'source_address',
    'canonical_address', 'same_phone', 'same_website', 'exact_external_id',
    'historical_link', 'evidence_label', 'evidence_confidence', 'reason_codes',
    'reviewer_decision', 'approved_operation', 'reviewer_note',
  ]);
  writeJson(path.join(OUTPUT_DIR, 'approved_apply_plan.json'), plan);

  const canonicalAfter = inspectCanonicalDataset(CANONICAL_PATH);
  if (canonicalAfter.rows !== canonicalBefore.rows || canonicalAfter.sha256 !== canonicalBefore.sha256) {
    throw new Error('Canonical changed during Stage 4J.');
  }
  const fingerprint = inputFingerprint({ historicalPath });
  const summaryCore = {
    status: 'CANDIDATE_NON_RUNTIME_NOT_CANONICAL_DRY_RUN_ONLY',
    inputFingerprint: fingerprint,
    processingVersions: PROCESSING_VERSIONS,
    boundaryPolicy: policy,
    evidencePack: {
      cases: evidenceCases.length,
      highCases: highCases.length,
      probableCases: probableCases.length,
      newTierACases: newCases.length,
      confidenceCounts,
      historicalRestrictedPayloadCommitted: false,
    },
    gates: { high: highGate, probable: probableGate, duplicate: duplicateGate },
    duplicateEvidence: {
      reviewed: duplicateEvidence.length,
      usable: duplicateUsable.length,
      correctedUnsupported: duplicateEvidence.length - duplicateUsable.length,
    },
    humanDecisions: decisionCounts,
    fieldEvidence: fieldCounts,
    applyPlan: {
      approvedOperations: plan.operations.length,
      rollbackMetadataComplete: plan.operations.every((item) => item.rollback),
      deterministicHash: plan.deterministicHash,
    },
    dryRun,
    incrementalBoundaryPolicy: {
      boundaryVersion: PROCESSING_VERSIONS.boundary,
      boundaryHash: EXPECTED_BOUNDARY_SHA,
      boundaryChangeInvalidates: ['geographicEligibility'],
      boundaryChangeDoesNotInvalidate: ['normalization', 'entityResolution', 'media', 'vision', 'provenance'],
    },
    canonical: {
      ...canonicalAfter,
      path: 'data/canonical/urbanagent_poi_master_v1.csv',
    },
    safety: {
      canonicalModified: false,
      runtimeModified: false,
      frontendModified: false,
      productionDatabaseOrFirebaseTouched: false,
      googleBulkIncluded: false,
      secondCityCreated: false,
      canonicalApplyExecuted: false,
    },
    readyForControlledCanaryApply: policy.status === 'PASS_VERSIONED_PRODUCT_SCOPE'
      && highGate.status === 'PASS'
      && probableGate.status === 'PASS'
      && duplicateGate.status === 'PASS'
      && plan.operations.length > 0,
    blocker: plan.operations.length === 0
      ? 'No explicit human APPROVE decisions exist; Stage 4J must not authorize canonical mutation.'
      : null,
    proposedStage4k: 'Controlled canary apply of only a very small explicitly approved subset, after a separate approval.',
  };
  const summary = {
    ...summaryCore,
    processingSeconds: Number(((performance.now() - started) / 1000).toFixed(3)),
    deterministicHash: stableHash(summaryCore),
  };
  writeJson(path.join(OUTPUT_DIR, 'stage4j_summary.json'), summary);
  return summary;
}

if (require.main === module) {
  const [cacheDir, historicalPath] = process.argv.slice(2);
  if (!cacheDir || !historicalPath) {
    console.error('Usage: node scripts/phase4_stage4j_evidence_approval_gate.js <stage4g-cache-dir> <historical-google-csv>');
    process.exitCode = 1;
  } else {
    console.log(JSON.stringify(runStage4j({
      cacheDir: path.resolve(cacheDir),
      historicalPath: path.resolve(historicalPath),
    }), null, 2));
  }
}

module.exports = {
  PROCESSING_VERSIONS,
  boundaryPolicy,
  diverseEvidenceSelection,
  inputFingerprint,
  loadHistoricalIndex,
  makeMatchCase,
  normalizedTextSha256,
  reviewRows,
  runStage4j,
};
