const fs = require('node:fs');
const path = require('node:path');

const config = require('../config/phase4_stage4q_create_new_evidence.json');
const stage4pConfig = require('../config/phase4_stage4p_create_new_policy.json');
const { loadBoundaryArtifact } = require('../src/modules/cityPackPreparation/administrativeBoundary');
const {
  inspectCanonicalDataset,
  parseCsv,
} = require('../src/modules/cityPackPreparation/canonicalDataset');
const {
  CREATE_NEW_STATES,
  evaluateCreateNewResearch,
  prepareCandidateRecord,
} = require('../src/modules/cityPackPreparation/createNewResearchPolicy');
const {
  checkIncrementalReuse,
  evaluateStrongPool,
} = require('../src/modules/cityPackPreparation/createNewEvidencePolicy');
const {
  aggregateOperationalSoak,
  readOperationalManifests,
} = require('../src/modules/cityPackPreparation/operationalSoakEvaluator');
const { stableHash } = require('../src/modules/cityPackPreparation/incrementalSync');
const {
  compactCandidate,
  readCanonicalEvidencePois,
  readJsonArrayObjects,
} = require('./phase4_stage4p_create_new_policy_research');

const ROOT = path.resolve(__dirname, '..');
const DEFAULTS = {
  candidatePack: 'D:\\UrbanAgent-spikes\\phase4-stage4g-20260810\\candidate\\candidate_city_pack.json',
  artifactRoot: 'D:\\UrbanAgent-artifacts\\stage4q-create-new',
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
  const parsed = { ...DEFAULTS, historical: [...DEFAULTS.historical], now: new Date().toISOString() };
  for (let index = 0; index < argv.length; index += 1) {
    const key = argv[index];
    const value = argv[index + 1];
    if (key === '--candidate-pack') parsed.candidatePack = value;
    else if (key === '--artifact-root') parsed.artifactRoot = value;
    else if (key === '--operational-runs') parsed.operationalRuns = value;
    else if (key === '--operational-logs') parsed.operationalLogs = value;
    else if (key === '--canonical') parsed.canonical = value;
    else if (key === '--boundary') parsed.boundary = value;
    else if (key === '--historical') parsed.historical = value.split(';').filter(Boolean);
    else if (key === '--now') parsed.now = value;
    else continue;
    index += 1;
  }
  return parsed;
}

function candidateForStage4q(candidate) {
  const record = compactCandidate(candidate);
  return prepareCandidateRecord({
    ...record,
    operatingStatus: candidate.operatingStatus || candidate.status
      || candidate.raw?.properties?.operating_status || null,
  });
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

function readHistoricalPois(filePaths) {
  const results = [];
  for (const filePath of filePaths) {
    if (!fs.existsSync(filePath)) continue;
    const source = /foody/i.test(path.basename(filePath)) ? 'historical_foody' : 'historical_google_maps';
    for (const row of readCsvObjects(filePath)) {
      const latitude = Number(row.Lat || row.Latitude);
      const longitude = Number(row.Lon || row.Longitude);
      if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) continue;
      const globalId = String(row.Global_ID || '').trim() || null;
      const sourceRecordId = String(row.RestaurantID || '').trim() || null;
      results.push({
        id: globalId || `${source}:${sourceRecordId}`,
        globalId,
        source,
        sourceId: sourceRecordId,
        name: String(row['Restaurant Name'] || row.Name || '').trim(),
        category: row.Category || null,
        latitude,
        longitude,
        address: row.Address || row.District || null,
        phone: row.Phone || null,
        website: row.Website || null,
        aliases: globalId ? [globalId] : [],
        externalIds: sourceRecordId ? { source_record_id: sourceRecordId } : {},
      });
    }
  }
  return results.sort((left, right) => String(left.id).localeCompare(String(right.id)));
}

function fileIdentity(filePath) {
  const stat = fs.statSync(filePath);
  return { path: filePath, bytes: stat.size, modifiedMs: Math.trunc(stat.mtimeMs) };
}

function csvEscape(value) {
  if (value === null || value === undefined) return '';
  const text = Array.isArray(value) ? value.join('|') : String(value);
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function writeReviewCsv(filePath, rows) {
  const fields = [
    'case_id', 'source', 'source_id', 'name', 'category', 'lat', 'lon', 'address',
    'nearest_canonical_id', 'nearest_canonical_name', 'nearest_distance_m',
    'existence_class', 'cross_source_support', 'support_independence', 'phone_evidence',
    'website_evidence', 'address_evidence', 'historical_evidence', 'wikidata_evidence',
    'freshness_class', 'traveler_relevance', 'provenance_status', 'duplicate_status',
    'evidence_class', 'reason_codes', 'recommendation', 'reviewer_decision', 'reviewer_note',
  ];
  const map = (item) => ({
    case_id: item.caseId, source: item.source, source_id: item.sourceId, name: item.name,
    category: item.category, lat: item.latitude, lon: item.longitude, address: item.address,
    nearest_canonical_id: item.nearestCanonicalId,
    nearest_canonical_name: item.nearestCanonicalName,
    nearest_distance_m: item.nearestDistanceMeters, existence_class: item.existenceClass,
    cross_source_support: item.crossSourceSupport, support_independence: item.supportIndependence,
    phone_evidence: item.phoneEvidence, website_evidence: item.websiteEvidence,
    address_evidence: item.addressEvidence, historical_evidence: item.historicalEvidence,
    wikidata_evidence: item.wikidataEvidence, freshness_class: item.freshnessClass,
    traveler_relevance: item.travelerRelevance, provenance_status: item.provenanceStatus,
    duplicate_status: item.duplicateStatus, evidence_class: item.evidenceClass,
    reason_codes: item.reasonCodes, recommendation: item.recommendation,
    reviewer_decision: 'DEFER', reviewer_note: '',
  });
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${[
    fields.join(','),
    ...rows.map(map).map((row) => fields.map((field) => csvEscape(row[field])).join(',')),
  ].join('\n')}\n`, 'utf8');
}

function writeEvidenceJsonl(filePath, rows) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${rows.map((item) => JSON.stringify({
    caseId: item.caseId,
    evidenceClass: item.evidenceClass,
    identityFingerprint: item.identityFingerprint,
    sourceGraphEdges: item.evidenceDetails.graphEdges,
    canonicalSecondPassCandidates: item.evidenceDetails.canonicalSecondPassCandidates,
    independentEvidence: item.evidenceDetails.independentEvidence,
    provenanceReferences: item.evidenceDetails.provenanceReferences,
    reasonCodes: item.reasonCodes,
  })).join('\n')}\n`, 'utf8');
}

function readState(filePath) {
  if (!fs.existsSync(filePath)) return null;
  return JSON.parse(fs.readFileSync(filePath, 'utf8').replace(/^\uFEFF/, ''));
}

function writeJson(filePath, value) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  const tempPath = `${filePath}.${process.pid}.tmp`;
  fs.writeFileSync(tempPath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
  fs.renameSync(tempPath, filePath);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  for (const required of [args.candidatePack, args.canonical, args.boundary]) {
    if (!fs.existsSync(required)) throw new Error(`Required Stage 4Q input missing: ${required}`);
  }
  const canonicalBefore = inspectCanonicalDataset(args.canonical);
  if (canonicalBefore.rows !== 4166 || !canonicalBefore.shaMatchesExpected) {
    throw new Error('Stage 4Q canonical baseline mismatch.');
  }
  const historicalFiles = args.historical.filter((filePath) => fs.existsSync(filePath));
  const inputFingerprint = stableHash({
    candidatePack: fileIdentity(args.candidatePack),
    canonicalSha: canonicalBefore.sha256,
    historical: historicalFiles.map(fileIdentity),
    stage4pPolicy: stage4pConfig.policyVersion,
    policy: config.policyVersion,
    evidence: config.evidenceVersion,
  });
  const runId = `stage4q-${inputFingerprint.slice(0, 12)}`;
  const artifactDir = path.join(args.artifactRoot, runId);
  const statePath = path.join(artifactDir, 'stage4q_state.json');
  const state = readState(statePath);
  const reuse = checkIncrementalReuse(state, inputFingerprint, config);
  let results;
  let strongPoolCount;
  let incrementalReuse = false;

  if (reuse.reusable) {
    results = reuse.results;
    strongPoolCount = results.length;
    incrementalReuse = true;
  } else {
    const candidates = [];
    const loaded = await readJsonArrayObjects(args.candidatePack, 'proposedNewCandidates', (candidate) => {
      candidates.push(candidateForStage4q(candidate));
    });
    const stage4p = evaluateCreateNewResearch({
      candidates,
      canonicalPois: readCanonicalEvidencePois(args.canonical),
      boundary: loadBoundaryArtifact(args.boundary),
      config: stage4pConfig,
      now: args.now,
    });
    if (loaded !== stage4p.summary.totalInputCandidates) throw new Error('Stage 4Q candidate stream mismatch.');
    const strongIds = new Set(stage4p.results.filter((item) => [
      CREATE_NEW_STATES.STRONG, CREATE_NEW_STATES.CANARY,
    ].includes(item.recommendation)).map((item) => item.caseId));
    const strongRecords = candidates.filter((record) => strongIds.has(
      `CREATE_NEW:${record.source}:${record.sourceId}`,
    ));
    if (strongRecords.length !== 4217) {
      throw new Error(`Stage 4Q expected Stage 4P strong pool 4217, found ${strongRecords.length}.`);
    }
    const evaluated = evaluateStrongPool({
      strongRecords,
      allRecords: candidates,
      canonicalPois: readCanonicalEvidencePois(args.canonical),
      historicalPois: readHistoricalPois(historicalFiles),
      config,
      now: args.now,
    });
    results = evaluated.results;
    strongPoolCount = strongRecords.length;
    writeJson(statePath, {
      schemaVersion: 'stage4q-incremental-state-v1',
      policyVersion: config.policyVersion,
      evidenceVersion: config.evidenceVersion,
      inputFingerprint,
      canonicalWriteAuthorized: false,
      results,
    });
  }

  const evaluated = evaluateStrongPoolSummary(results);
  const review = selectReview(results);
  const reviewPath = path.join(artifactDir, 'create_new_evidence_review.csv');
  const evidencePath = path.join(artifactDir, 'create_new_evidence_details.jsonl');
  writeReviewCsv(reviewPath, review);
  writeEvidenceJsonl(evidencePath, review);
  const soak = aggregateOperationalSoak({
    manifests: readOperationalManifests(args.operationalRuns),
    gate: config.soakGate,
    artifactRoot: args.operationalRuns,
    logRoot: args.operationalLogs,
  });
  const output = {
    schemaVersion: 'stage4q-create-new-evidence-run-v1',
    status: 'RESEARCH_ONLY_NON_RUNTIME_NOT_CANONICAL',
    runId,
    inputFingerprint,
    incrementalReuse,
    strongPoolCount,
    historicalFiles: historicalFiles.map(fileIdentity),
    historicalLineageCompleteness: 'PARTIAL_AVAILABLE_WORKSPACE_ONLY',
    summary: evaluated,
    reviewCount: review.length,
    reviewPath,
    evidencePath,
    soak,
    autoCreateNew: false,
    canonicalWrites: 0,
    deletes: 0,
  };
  writeJson(path.join(artifactDir, 'stage4q_summary.json'), output);
  const canonicalAfter = inspectCanonicalDataset(args.canonical);
  if (canonicalAfter.rows !== canonicalBefore.rows || canonicalAfter.sha256 !== canonicalBefore.sha256) {
    throw new Error('Canonical changed during Stage 4Q research.');
  }
  process.stdout.write(`${JSON.stringify({
    verdict: 'PASS_RESEARCH_ONLY', runId, incrementalReuse, strongPoolCount,
    summary: evaluated, reviewCount: review.length, reviewPath, evidencePath,
    soak: { status: soak.status, scheduledRuns: soak.scheduledInvocationCount,
      successful: soak.successfulCount, failures: soak.failedRuns, dueSourceChecks: soak.sourceDueCount,
      acquisitions: soak.realAcquisitionCount, deltas: soak.actualSourceDeltaCount },
    canonicalRows: canonicalAfter.rows, canonicalSha: canonicalAfter.sha256,
    autoCreateNew: false,
  }, null, 2)}\n`);
}

function evaluateStrongPoolSummary(results) {
  const { summarizeEvidence } = require('../src/modules/cityPackPreparation/createNewEvidencePolicy');
  return summarizeEvidence(results);
}

function selectReview(results) {
  const { deterministicReviewSample } = require('../src/modules/cityPackPreparation/createNewEvidencePolicy');
  return deterministicReviewSample(results, config);
}

if (require.main === module) main().catch((error) => {
  process.stderr.write(`${error.stack || error.message}\n`);
  process.exitCode = 1;
});

module.exports = {
  candidateForStage4q,
  parseArgs,
  readHistoricalPois,
  writeEvidenceJsonl,
  writeReviewCsv,
};
