const fs = require('node:fs');
const path = require('node:path');

const policyConfig = require('../config/phase4_stage4p_create_new_policy.json');
const { loadBoundaryArtifact } = require('../src/modules/cityPackPreparation/administrativeBoundary');
const {
  inspectCanonicalDataset,
  parseCsv,
  readCanonicalPois,
} = require('../src/modules/cityPackPreparation/canonicalDataset');
const {
  evaluateCreateNewResearch,
} = require('../src/modules/cityPackPreparation/createNewResearchPolicy');
const {
  aggregateOperationalSoak,
  readOperationalManifests,
} = require('../src/modules/cityPackPreparation/operationalSoakEvaluator');
const { stableHash } = require('../src/modules/cityPackPreparation/incrementalSync');
const { writeJsonAtomic } = require('../src/modules/cityPackPreparation/scheduledSyncOrchestrator');

const ROOT = path.resolve(__dirname, '..');
const DEFAULTS = {
  candidatePack: 'D:\\UrbanAgent-spikes\\phase4-stage4g-20260810\\candidate\\candidate_city_pack.json',
  artifactRoot: 'D:\\UrbanAgent-artifacts\\stage4p-create-new',
  operationalRuns: 'D:\\UrbanAgent-artifacts\\stage4o-runs',
  operationalLogs: 'D:\\UrbanAgent-artifacts\\stage4o-logs',
  statusPath: 'D:\\UrbanAgent-artifacts\\stage4o-status\\status.json',
  canonical: path.join(ROOT, 'data', 'canonical', 'urbanagent_poi_master_v1.csv'),
  boundary: path.join(ROOT, 'data', 'spikes', 'phase4', 'stage4i', 'boundary',
    'osm-relation-1891418-v72.geojson'),
};

function parseArgs(argv) {
  const parsed = { ...DEFAULTS, now: new Date().toISOString() };
  for (let index = 0; index < argv.length; index += 1) {
    const key = argv[index];
    const value = argv[index + 1];
    if (key === '--candidate-pack') parsed.candidatePack = value;
    else if (key === '--artifact-root') parsed.artifactRoot = value;
    else if (key === '--operational-runs') parsed.operationalRuns = value;
    else if (key === '--operational-logs') parsed.operationalLogs = value;
    else if (key === '--status-path') parsed.statusPath = value;
    else if (key === '--canonical') parsed.canonical = value;
    else if (key === '--boundary') parsed.boundary = value;
    else if (key === '--now') parsed.now = value;
    else continue;
    index += 1;
  }
  return parsed;
}

function compactCandidate(candidate) {
  const provenance = candidate.provenance || {};
  return {
    candidateId: candidate.candidateId,
    source: candidate.source,
    sourceId: candidate.sourceId,
    name: candidate.name,
    category: candidate.category,
    latitude: candidate.latitude,
    longitude: candidate.longitude,
    address: candidate.address || null,
    website: candidate.website || null,
    phone: candidate.phone || null,
    openingHours: candidate.openingHours || null,
    externalIds: candidate.externalIds || {},
    provenance: {
      source: provenance.source,
      sourceId: provenance.sourceId,
      snapshotRef: provenance.snapshotRef,
      policyClass: provenance.policyClass,
      license: provenance.license,
      attribution: provenance.attribution,
      licenseUrl: provenance.licenseUrl,
      retrievedAt: provenance.retrievedAt || null,
      upstreamSources: (provenance.upstreamSources || []).map((item) => ({
        dataset: item.dataset || null,
        recordId: item.recordId || null,
        updateTime: item.updateTime || null,
        confidence: item.confidence ?? null,
      })),
    },
  };
}

async function readJsonArrayObjects(filePath, arrayKey, onObject) {
  const marker = `\"${arrayKey}\":[`;
  let markerBuffer = '';
  let found = false;
  let collecting = false;
  let objectBuffer = '';
  let objectDepth = 0;
  let inString = false;
  let escaped = false;
  let count = 0;
  let finished = false;
  const stream = fs.createReadStream(filePath, { encoding: 'utf8', highWaterMark: 1024 * 1024 });
  for await (const chunk of stream) {
    if (finished) break;
    let text = chunk;
    if (!found) {
      const searchable = markerBuffer + text;
      const markerIndex = searchable.indexOf(marker);
      if (markerIndex < 0) {
        markerBuffer = searchable.slice(-(marker.length - 1));
        continue;
      }
      found = true;
      text = searchable.slice(markerIndex + marker.length);
      markerBuffer = '';
    }
    for (const character of text) {
      if (!collecting) {
        if (character === '{') {
          collecting = true;
          objectDepth = 1;
          objectBuffer = '{';
          inString = false;
          escaped = false;
        } else if (character === ']') {
          finished = true;
          break;
        }
        continue;
      }
      objectBuffer += character;
      if (escaped) { escaped = false; continue; }
      if (character === '\\' && inString) { escaped = true; continue; }
      if (character === '"') { inString = !inString; continue; }
      if (inString) continue;
      if (character === '{') objectDepth += 1;
      else if (character === '}') objectDepth -= 1;
      if (objectDepth === 0) {
        onObject(JSON.parse(objectBuffer));
        count += 1;
        collecting = false;
        objectBuffer = '';
      }
    }
  }
  if (!found || collecting) throw new Error(`Could not read complete JSON array: ${arrayKey}`);
  return count;
}

function parseSourceIds(value) {
  const ids = {};
  for (const item of String(value || '').split('|').map((part) => part.trim()).filter(Boolean)) {
    const separator = item.indexOf(':');
    if (separator > 0) ids[item.slice(0, separator)] = item.slice(separator + 1);
  }
  return ids;
}

function readCanonicalEvidencePois(canonicalPath) {
  const canonical = readCanonicalPois(canonicalPath);
  const rows = parseCsv(fs.readFileSync(canonicalPath, 'utf8'));
  const headers = rows[0].map((header, index) => (
    index === 0 ? header.replace(/^\uFEFF/, '').trim() : header.trim()
  ));
  const rawById = new Map(rows.slice(1).map((cells) => {
    const raw = Object.fromEntries(headers.map((header, index) => [header, cells[index] ?? '']));
    return [raw.Global_ID, raw];
  }));
  return canonical.map((poi) => {
    const raw = rawById.get(poi.id) || {};
    return {
      ...poi,
      sourceId: poi.id,
      externalIds: parseSourceIds(raw.Source_IDs),
      aliases: String(raw.Alias_Global_IDs || '').split('|').filter(Boolean),
    };
  });
}

function csvEscape(value) {
  if (value === null || value === undefined) return '';
  const text = Array.isArray(value) ? value.join('|') : typeof value === 'object'
    ? JSON.stringify(value) : String(value);
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function writeCanaryCsv(filePath, rows) {
  const fields = [
    'case_id', 'source', 'source_id', 'name', 'category', 'lat', 'lon', 'address',
    'phone_present', 'website_present', 'nearest_canonical_id', 'nearest_canonical_name',
    'distance_m', 'cross_source_support', 'traveler_relevance', 'freshness_signal',
    'provenance_status', 'duplicate_risk', 'evidence_label', 'evidence_confidence',
    'reason_codes', 'recommendation', 'reviewer_decision', 'reviewer_note',
  ];
  const mapped = rows.map((item) => ({
    case_id: item.caseId,
    source: item.source,
    source_id: item.sourceId,
    name: item.name,
    category: item.category,
    lat: item.latitude,
    lon: item.longitude,
    address: item.address,
    phone_present: item.phonePresent,
    website_present: item.websitePresent,
    nearest_canonical_id: item.nearestCanonicalId,
    nearest_canonical_name: item.nearestCanonicalName,
    distance_m: item.distanceMeters,
    cross_source_support: item.crossSourceSupport,
    traveler_relevance: item.travelerRelevance,
    freshness_signal: item.freshnessSignal,
    provenance_status: item.provenanceStatus,
    duplicate_risk: item.duplicateRisk,
    evidence_label: item.evidenceLabel,
    evidence_confidence: item.evidenceConfidence,
    reason_codes: item.reasonCodes,
    recommendation: item.recommendation,
    reviewer_decision: item.reviewerDecision,
    reviewer_note: item.reviewerNote,
  }));
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${[
    fields.join(','),
    ...mapped.map((row) => fields.map((field) => csvEscape(row[field])).join(',')),
  ].join('\n')}\n`, 'utf8');
}

function updateOperationalStatus(statusPath, soak, research, canaryCount) {
  if (!fs.existsSync(statusPath)) return false;
  const status = JSON.parse(fs.readFileSync(statusPath, 'utf8').replace(/^\uFEFF/, ''));
  const next = {
    ...status,
    schedulerSoakStatus: soak.status,
    scheduledRunCount: soak.scheduledInvocationCount,
    lastDueSource: soak.lastDueSource,
    lastActualSourceDelta: soak.lastActualSourceDelta,
    lastSafePr: status.lastPrUrl || null,
    pendingEnrichmentExceptions: status.lastHumanReviewDeltaCount || 0,
    pendingCreateNewResearchReviewCount: canaryCount,
    stage4pResearchHash: research.deterministicHash,
  };
  writeJsonAtomic(statusPath, next);
  return true;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  for (const [name, filePath] of [
    ['candidate pack', args.candidatePack], ['canonical', args.canonical], ['boundary', args.boundary],
  ]) if (!fs.existsSync(filePath)) throw new Error(`${name} not found: ${filePath}`);

  const canonicalBefore = inspectCanonicalDataset(args.canonical);
  if (canonicalBefore.rows !== 4166 || !canonicalBefore.shaMatchesExpected) {
    throw new Error('Stage 4P canonical baseline mismatch.');
  }
  const candidates = [];
  const loaded = await readJsonArrayObjects(args.candidatePack, 'proposedNewCandidates', (candidate) => {
    candidates.push(compactCandidate(candidate));
  });
  const soak = aggregateOperationalSoak({
    manifests: readOperationalManifests(args.operationalRuns),
    gate: policyConfig.soakGate,
    artifactRoot: args.operationalRuns,
    logRoot: args.operationalLogs,
  });
  const research = evaluateCreateNewResearch({
    candidates,
    canonicalPois: readCanonicalEvidencePois(args.canonical),
    boundary: loadBoundaryArtifact(args.boundary),
    config: policyConfig,
    now: args.now,
  });
  if (loaded !== research.summary.totalInputCandidates) throw new Error('Candidate stream count mismatch.');
  const runId = `stage4p-${research.summary.deterministicHash.slice(0, 12)}`;
  const artifactDir = path.join(args.artifactRoot, runId);
  const canaryPath = path.join(artifactDir, 'create_new_canary_review.csv');
  writeCanaryCsv(canaryPath, research.canary);
  const output = {
    schemaVersion: 'stage4p-soak-create-new-run-v1',
    status: 'RESEARCH_ONLY_NON_RUNTIME_NOT_CANONICAL',
    runId,
    soak,
    research: research.summary,
    resolutionSummary: research.resolutionSummary,
    duplicatePairCount: research.duplicatePairCount,
    canaryReviewCount: research.canary.length,
    canaryReviewPath: canaryPath,
    decisionMemory: {
      schemaVersion: 'stage4p-create-new-decision-memory-v1',
      unchangedDecisionReusable: true,
      changedFingerprintInvalidates: true,
      defaultReviewerDecision: 'DEFER',
      canonicalWriteAuthorized: false,
    },
    incrementalFlow: [
      'polygon', 'normalization', 'existing_entity_second_pass', 'cross_source_dedupe_support',
      'traveler_relevance', 'quality_provenance', 'research_policy', 'decision_memory',
      'worthwhile_human_review',
    ],
    outputHash: stableHash({ soak, research: research.summary, canary: research.canary }),
  };
  fs.mkdirSync(artifactDir, { recursive: true });
  writeJsonAtomic(path.join(artifactDir, 'stage4p_summary.json'), output);
  updateOperationalStatus(args.statusPath, soak, research.summary, research.canary.length);
  const canonicalAfter = inspectCanonicalDataset(args.canonical);
  if (canonicalAfter.rows !== canonicalBefore.rows || canonicalAfter.sha256 !== canonicalBefore.sha256) {
    throw new Error('Canonical changed during Stage 4P research.');
  }
  process.stdout.write(`${JSON.stringify({
    verdict: 'PASS_WITH_SOAK_IN_PROGRESS',
    runId,
    soakStatus: soak.status,
    scheduledRuns: soak.scheduledInvocationCount,
    totalCandidates: research.summary.totalNewCandidatesEvaluated,
    strongReviewCandidates: research.summary.strongReviewCandidates,
    strongestCanaryCandidates: research.summary.strongestCanaryCandidates,
    canaryReviewPath: canaryPath,
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
  compactCandidate,
  parseArgs,
  readCanonicalEvidencePois,
  readJsonArrayObjects,
  updateOperationalStatus,
  writeCanaryCsv,
};
