const fs = require('node:fs');
const path = require('node:path');

const config = require('../config/phase4_stage4r_create_new_canary.json');
const stage4qConfig = require('../config/phase4_stage4q_create_new_evidence.json');
const { loadBoundaryArtifact } = require('../src/modules/cityPackPreparation/administrativeBoundary');
const { inspectCanonicalDataset } = require('../src/modules/cityPackPreparation/canonicalDataset');
const {
  RECOMMENDATIONS,
  assertPreparationOnly,
  buildDryRunCreatePlan,
  evaluateCandidateForReview,
  selectHumanCanary,
  summarizeSelection,
} = require('../src/modules/cityPackPreparation/createNewCanaryReview');
const {
  EVIDENCE_CLASSES,
  buildExistenceIndex,
} = require('../src/modules/cityPackPreparation/createNewEvidencePolicy');
const {
  aggregateOperationalSoak,
  readOperationalManifests,
} = require('../src/modules/cityPackPreparation/operationalSoakEvaluator');
const { stableHash } = require('../src/modules/cityPackPreparation/incrementalSync');
const {
  candidateForStage4q,
  readHistoricalPois,
} = require('./phase4_stage4q_create_new_evidence');
const {
  readCanonicalEvidencePois,
  readJsonArrayObjects,
} = require('./phase4_stage4p_create_new_policy_research');

const ROOT = path.resolve(__dirname, '..');
const DEFAULTS = {
  stage4qState: 'D:\\UrbanAgent-artifacts\\stage4q-create-new\\stage4q-935d4629660a\\stage4q_state.json',
  candidatePack: 'D:\\UrbanAgent-spikes\\phase4-stage4g-20260810\\candidate\\candidate_city_pack.json',
  artifactRoot: 'D:\\UrbanAgent-artifacts\\stage4r-create-new',
  operationalRuns: 'D:\\UrbanAgent-artifacts\\stage4o-runs',
  operationalLogs: 'D:\\UrbanAgent-artifacts\\stage4o-logs',
  canonical: path.join(ROOT, 'data', 'canonical', 'urbanagent_poi_master_v1.csv'),
  boundary: path.join(ROOT, 'data', 'spikes', 'phase4', 'stage4i', 'boundary',
    'osm-relation-1891418-v72.geojson'),
  historical: [
    'D:\\POI-urban-danang-BE\\data\\raw\\legacy\\poi_urban_ggmap.csv',
    'D:\\POI-urban-danang-BE\\data\\raw\\legacy\\poi_urban_foody.csv',
  ],
};

function parseArgs(argv) {
  const parsed = { ...DEFAULTS, historical: [...DEFAULTS.historical] };
  for (let index = 0; index < argv.length; index += 1) {
    const key = argv[index];
    const value = argv[index + 1];
    if (key === '--stage4q-state') parsed.stage4qState = value;
    else if (key === '--candidate-pack') parsed.candidatePack = value;
    else if (key === '--artifact-root') parsed.artifactRoot = value;
    else if (key === '--operational-runs') parsed.operationalRuns = value;
    else if (key === '--operational-logs') parsed.operationalLogs = value;
    else if (key === '--canonical') parsed.canonical = value;
    else if (key === '--boundary') parsed.boundary = value;
    else if (key === '--historical') parsed.historical = value.split(';').filter(Boolean);
    else continue;
    index += 1;
  }
  return parsed;
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8').replace(/^\uFEFF/, ''));
}

function csvEscape(value) {
  if (value === null || value === undefined) return '';
  const text = Array.isArray(value) ? value.join('|') : String(value);
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function writeCsv(filePath, fields, rows) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${[
    fields.join(','),
    ...rows.map((row) => fields.map((field) => csvEscape(row[field])).join(',')),
  ].join('\n')}\n`, 'utf8');
}

function writeJson(filePath, value) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

function reviewRow(item) {
  const record = item.record;
  const stage4q = item.stage4qResult;
  const nearest = item.currentExistence.nearestCanonical;
  return {
    case_id: item.caseId,
    source: record.source,
    source_id: record.sourceId,
    name: record.name,
    category: record.category,
    lat: record.latitude,
    lon: record.longitude,
    address: record.address || null,
    phone: record.phone || null,
    website: record.website || null,
    opening_hours: record.openingHours || null,
    nearest_canonical_id: nearest?.id || null,
    nearest_canonical_name: nearest?.name || null,
    nearest_canonical_distance_m: nearest?.distanceMeters ?? null,
    canonical_absence_class: item.currentExistence.existenceClass,
    cross_source_support: stage4q.crossSourceSupport || [],
    support_independence: item.supportClass,
    historical_evidence: stage4q.historicalEvidence,
    wikidata_evidence: stage4q.wikidataEvidence,
    freshness_class: item.freshnessClass,
    traveler_relevance: item.travelerRelevance,
    duplicate_status: item.duplicateStatus,
    provenance_status: item.provenanceStatus,
    license_status: item.licenseStatus,
    evidence_confidence: item.evidenceConfidence,
    reason_codes: item.reasonCodes,
    recommendation: item.recommendation,
    proposed_operation: 'CREATE_NEW',
    reviewer_decision: 'DEFER',
    reviewer_note: '',
  };
}

function decisionRows(selected) {
  return selected.map((item) => ({
    case_id: item.caseId,
    reviewer_decision: 'DEFER',
    reviewer_note: '',
  }));
}

function writeEvidenceJsonl(filePath, selected) {
  const rows = selected.map((item) => JSON.stringify({
    schemaVersion: 'stage4r-create-new-canary-evidence-v1',
    caseId: item.caseId,
    proposedCanonicalId: item.proposedCanonicalId,
    currentCanonicalSecondPassCandidates: item.currentExistence.candidates,
    sourceEntityGraphEdges: item.stage4qResult.evidenceDetails?.graphEdges || [],
    duplicateChecks: {
      stage4qStatus: item.duplicateStatus,
      selectedPairwiseStatus: 'NO_UNRESOLVED_SELECTED_PAIR',
    },
    independentEvidence: {
      class: item.supportClass,
      details: item.stage4qResult.evidenceDetails?.independentEvidence || null,
    },
    provenance: item.record.provenance,
    license: item.record.license,
    reasonCodes: item.reasonCodes,
    policyVersions: {
      stage4r: config.policyVersion,
      stage4q: config.stage4qPolicyVersion,
      stage4qEvidence: config.stage4qEvidenceVersion,
    },
    fingerprints: {
      stage4q: item.stage4qResult.identityFingerprint,
      stage4rDecision: item.decisionFingerprint,
    },
  }));
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${rows.join('\n')}\n`, 'utf8');
}

async function loadTargetRecords(candidatePack, targetCaseIds) {
  const byCaseId = new Map();
  await readJsonArrayObjects(candidatePack, 'proposedNewCandidates', (candidate) => {
    const caseId = `CREATE_NEW:${candidate.source}:${candidate.sourceId}`;
    if (targetCaseIds.has(caseId)) byCaseId.set(caseId, candidateForStage4q(candidate));
  });
  return byCaseId;
}

function decisionsForPlan(selected) {
  return selected.map((item) => ({
    caseId: item.caseId,
    reviewerDecision: 'DEFER',
    reviewerNote: '',
  }));
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  assertPreparationOnly(config);
  for (const required of [args.stage4qState, args.candidatePack, args.canonical, args.boundary]) {
    if (!fs.existsSync(required)) throw new Error(`Required Stage 4R input missing: ${required}`);
  }
  const canonicalBefore = inspectCanonicalDataset(args.canonical);
  if (canonicalBefore.rows !== 4166 || canonicalBefore.sha256
    !== '39647b29308813a7ec19e4695fd2b95ffa27743db8e8cb46800c45d3a3fe6ded') {
    throw new Error('Stage 4R current canonical baseline mismatch.');
  }
  const stage4qState = readJson(args.stage4qState);
  if (stage4qState.policyVersion !== config.stage4qPolicyVersion
    || stage4qState.evidenceVersion !== config.stage4qEvidenceVersion) {
    throw new Error('Stage 4Q state policy/evidence version mismatch.');
  }
  const eligibleResults = stage4qState.results.filter((item) => (
    item.evidenceClass === EVIDENCE_CLASSES.CANARY
  ));
  if (eligibleResults.length !== 974) throw new Error(`Expected 974 Stage 4Q canary inputs, found ${eligibleResults.length}.`);
  const resultByCaseId = new Map(eligibleResults.map((item) => [item.caseId, item]));
  const recordByCaseId = await loadTargetRecords(args.candidatePack, new Set(resultByCaseId.keys()));
  if (recordByCaseId.size !== eligibleResults.length) throw new Error('Stage 4R could not recover all eligible source records.');

  const canonicalPois = readCanonicalEvidencePois(args.canonical);
  const historicalFiles = args.historical.filter((filePath) => fs.existsSync(filePath));
  const historicalPois = readHistoricalPois(historicalFiles);
  const existenceIndex = {
    combined: buildExistenceIndex(canonicalPois, historicalPois),
    thresholds: stage4qConfig.thresholds,
  };
  const boundary = loadBoundaryArtifact(args.boundary);
  const evaluated = eligibleResults.map((stage4qResult) => evaluateCandidateForReview({
    record: recordByCaseId.get(stage4qResult.caseId),
    stage4qResult,
    existenceIndex,
    boundary,
    config,
    now: config.evaluationReferenceTime,
  })).sort((left, right) => left.caseId.localeCompare(right.caseId));
  const selection = selectHumanCanary(evaluated, config);
  const decisions = decisionsForPlan(selection.selected);
  const plan = buildDryRunCreatePlan(selection.selected, decisions, config);
  if (plan.approvedCreateNewCount !== 0) throw new Error('Stage 4R dry-run unexpectedly authorized CREATE_NEW.');
  const independentInspected = stage4qState.results.filter((item) => (
    item.evidenceDetails?.independentEvidence?.strongCrossSource
  )).length;
  const summary = summarizeSelection({
    inputCount: eligibleResults.length,
    independentInspected,
    evaluated,
    selection,
    plan,
  });
  const runFingerprint = stableHash({
    canonicalSha: canonicalBefore.sha256,
    stage4qInput: stage4qState.inputFingerprint,
    stage4qResults: stage4qState.results.map((item) => item.identityFingerprint),
    config,
  });
  const runId = `stage4r-${runFingerprint.slice(0, 12)}`;
  const artifactDir = path.join(args.artifactRoot, runId);
  const reviewPath = path.join(artifactDir, 'create_new_canary_review.csv');
  const decisionPath = path.join(artifactDir, 'create_new_canary_decisions.csv');
  const evidencePath = path.join(artifactDir, 'create_new_canary_evidence.jsonl');
  const planPath = path.join(artifactDir, 'create_new_dry_run_plan.json');
  const reviewFields = [
    'case_id', 'source', 'source_id', 'name', 'category', 'lat', 'lon', 'address',
    'phone', 'website', 'opening_hours', 'nearest_canonical_id', 'nearest_canonical_name',
    'nearest_canonical_distance_m', 'canonical_absence_class', 'cross_source_support',
    'support_independence', 'historical_evidence', 'wikidata_evidence', 'freshness_class',
    'traveler_relevance', 'duplicate_status', 'provenance_status', 'license_status',
    'evidence_confidence', 'reason_codes', 'recommendation', 'proposed_operation',
    'reviewer_decision', 'reviewer_note',
  ];
  writeCsv(reviewPath, reviewFields, selection.selected.map(reviewRow));
  writeCsv(decisionPath, ['case_id', 'reviewer_decision', 'reviewer_note'],
    decisionRows(selection.selected));
  writeEvidenceJsonl(evidencePath, selection.selected);
  writeJson(planPath, plan);
  const soak = aggregateOperationalSoak({
    manifests: readOperationalManifests(args.operationalRuns),
    gate: stage4qConfig.soakGate,
    artifactRoot: args.operationalRuns,
    logRoot: args.operationalLogs,
  });
  const artifactHashes = {
    review: stableHash(selection.selected.map(reviewRow)),
    decisions: stableHash(decisionRows(selection.selected)),
    evidence: stableHash(selection.selected.map((item) => ({
      caseId: item.caseId, fingerprint: item.decisionFingerprint,
      candidates: item.currentExistence.candidates,
    }))),
    plan: plan.deterministicHash,
  };
  const output = {
    schemaVersion: 'stage4r-create-new-canary-run-v1',
    status: 'HUMAN_REVIEW_PREPARATION_ONLY',
    runId,
    runFingerprint,
    summary,
    selected: selection.selected.map((item) => ({
      caseId: item.caseId,
      proposedCanonicalId: item.proposedCanonicalId,
      name: item.record.name,
      category: item.record.category,
      source: item.record.source,
      supportClass: item.supportClass,
      nearestCanonical: item.currentExistence.nearestCanonical,
      duplicateStatus: item.duplicateStatus,
      recommendation: item.recommendation,
      reasonCodes: item.reasonCodes,
    })),
    duplicateRemoved: selection.duplicateRemoved.map((item) => ({
      caseId: item.candidate.caseId,
      retainedCaseId: item.retainedCaseId,
      duplicate: item.duplicate,
    })),
    artifacts: { reviewPath, decisionPath, evidencePath, planPath, hashes: artifactHashes },
    soak,
    autoCreateNew: false,
    canonicalWrites: 0,
    deletes: 0,
    runtimeChanged: false,
  };
  writeJson(path.join(artifactDir, 'stage4r_summary.json'), output);
  const canonicalAfter = inspectCanonicalDataset(args.canonical);
  if (canonicalAfter.rows !== canonicalBefore.rows || canonicalAfter.sha256 !== canonicalBefore.sha256) {
    throw new Error('Canonical changed during Stage 4R preparation.');
  }
  process.stdout.write(`${JSON.stringify({
    verdict: 'PASS_HUMAN_REVIEW_PREPARATION_ONLY',
    runId,
    summary,
    artifacts: output.artifacts,
    soak: {
      status: soak.status,
      runs: soak.scheduledInvocationCount,
      successful: soak.successfulCount,
      failed: soak.failedRuns,
      dueChecks: soak.sourceDueCount,
      acquisitions: soak.realAcquisitionCount,
      deltas: soak.actualSourceDeltaCount,
      safePrRuns: soak.prProducingRuns,
    },
    canonicalRows: canonicalAfter.rows,
    canonicalSha: canonicalAfter.sha256,
    autoCreateNew: false,
  }, null, 2)}\n`);
}

if (require.main === module) main().catch((error) => {
  process.stderr.write(`${error.stack || error.message}\n`);
  process.exitCode = 1;
});

module.exports = {
  decisionRows,
  loadTargetRecords,
  parseArgs,
  reviewRow,
  writeCsv,
  writeEvidenceJsonl,
};
