const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');

const config = require('../config/phase4_stage4s_controlled_create_new.json');
const stage4rConfig = require('../config/phase4_stage4r_create_new_canary.json');
const stage4qConfig = require('../config/phase4_stage4q_create_new_evidence.json');
const { loadBoundaryArtifact } = require('../src/modules/cityPackPreparation/administrativeBoundary');
const { inspectCanonicalDataset, parseCsv } = require('../src/modules/cityPackPreparation/canonicalDataset');
const {
  appendCanonicalRows,
  assertStage4SSafety,
  buildApplyPlan,
  buildCreateNewSidecar,
  importCreateNewDecisionMemory,
  revalidateHumanApprovals,
  resolveHumanDecisionContract,
  validateHumanDecisionRows,
} = require('../src/modules/cityPackPreparation/controlledCreateNewCanary');
const { evaluateCandidateForReview } = require('../src/modules/cityPackPreparation/createNewCanaryReview');
const { buildExistenceIndex } = require('../src/modules/cityPackPreparation/createNewEvidencePolicy');
const { stableHash } = require('../src/modules/cityPackPreparation/decisionMemory');
const { normalizeSidecar, serializeSidecar } = require('../src/modules/cityPackPreparation/safeEnrichmentExecutor');
const { readCsvObjects } = require('./phase4_stage4k_controlled_canary_apply');
const { readHistoricalPois } = require('./phase4_stage4q_create_new_evidence');
const { readCanonicalEvidencePois } = require('./phase4_stage4p_create_new_policy_research');
const { loadTargetRecords } = require('./phase4_stage4r_create_new_canary_review');

const ROOT = path.resolve(__dirname, '..');
const DEFAULTS = {
  canonical: path.join(ROOT, 'data', 'canonical', 'urbanagent_poi_master_v1.csv'),
  manifest: path.join(ROOT, 'data', 'canonical', 'urbanagent_poi_master_v1_manifest.json'),
  sidecar: path.join(ROOT, 'data', 'citypacks', 'enrichments', 'danang', 'stage4l-enrichment-v1.json'),
  state: path.join(ROOT, 'data', 'citypacks', 'enrichments', 'danang', 'stage4s_create_new_state.json'),
  decisions: 'D:\\UrbanAgent-artifacts\\stage4r-create-new\\stage4r-fd19a4080107\\create_new_canary_decisions.csv',
  review: 'D:\\UrbanAgent-artifacts\\stage4r-create-new\\stage4r-fd19a4080107\\create_new_canary_review.csv',
  evidence: 'D:\\UrbanAgent-artifacts\\stage4r-create-new\\stage4r-fd19a4080107\\create_new_canary_evidence.jsonl',
  stage4qState: 'D:\\UrbanAgent-artifacts\\stage4q-create-new\\stage4q-935d4629660a\\stage4q_state.json',
  candidatePack: 'D:\\UrbanAgent-spikes\\phase4-stage4g-20260810\\candidate\\candidate_city_pack.json',
  boundary: path.join(ROOT, 'data', 'spikes', 'phase4', 'stage4i', 'boundary',
    'osm-relation-1891418-v72.geojson'),
  historical: [
    'D:\\POI-urban-danang-BE\\data\\raw\\legacy\\poi_urban_ggmap.csv',
    'D:\\POI-urban-danang-BE\\data\\raw\\legacy\\poi_urban_foody.csv',
  ],
};

function parseArgs(argv) {
  const args = { ...DEFAULTS, historical: [...DEFAULTS.historical], apply: false, artifactDir: null };
  for (let index = 0; index < argv.length; index += 1) {
    const key = argv[index];
    if (key === '--apply') { args.apply = true; continue; }
    const value = argv[index + 1];
    if (!value || value.startsWith('--')) throw new Error(`Missing value for ${key}`);
    if (key === '--artifact-dir') args.artifactDir = value;
    else if (key === '--canonical') args.canonical = value;
    else if (key === '--manifest') args.manifest = value;
    else if (key === '--sidecar') args.sidecar = value;
    else if (key === '--state') args.state = value;
    else if (key === '--decisions') args.decisions = value;
    else if (key === '--review') args.review = value;
    else if (key === '--evidence') args.evidence = value;
    else if (key === '--stage4q-state') args.stage4qState = value;
    else if (key === '--candidate-pack') args.candidatePack = value;
    else if (key === '--boundary') args.boundary = value;
    else if (key === '--historical') args.historical = value.split(';').filter(Boolean);
    else throw new Error(`Unknown argument: ${key}`);
    index += 1;
  }
  if (!args.artifactDir) throw new Error('--artifact-dir is required.');
  return args;
}

function sha256(value) {
  return crypto.createHash('sha256').update(value).digest('hex');
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8').replace(/^\uFEFF/, ''));
}

function readJsonl(filePath) {
  return fs.readFileSync(filePath, 'utf8').split(/\r?\n/).filter(Boolean).map(JSON.parse);
}

function writeJson(filePath, value) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

function copyBackup(source, destination) {
  fs.mkdirSync(path.dirname(destination), { recursive: true });
  if (!fs.existsSync(destination)) fs.copyFileSync(source, destination);
}

function updateManifest(manifest, canonicalRows, canonicalSha, applied) {
  const categories = { ...(manifest.output.category_distribution || {}) };
  for (const candidate of applied) {
    categories[candidate.canonicalCategory] = (categories[candidate.canonicalCategory] || 0) + 1;
  }
  return {
    ...manifest,
    output: {
      ...manifest.output,
      rows: canonicalRows,
      unique_global_ids: canonicalRows,
      missing_images: Number(manifest.output.missing_images || 0) + applied.length,
      missing_primary_rating: Number(manifest.output.missing_primary_rating || 0) + applied.length,
      controlled_create_new_rows: applied.length,
      sha256: canonicalSha,
      category_distribution: categories,
    },
  };
}

function validateAppend(beforeText, afterText, beforeRows, afterRows, expectedNewRows) {
  if (!afterText.startsWith(beforeText)) throw new Error('Existing canonical bytes changed.');
  if (afterRows.length !== beforeRows.length + expectedNewRows) throw new Error('Canonical row delta mismatch.');
  const beforeIds = new Set(beforeRows.map((row) => row[0]));
  const afterIds = new Set(afterRows.map((row) => row[0]));
  if (afterIds.size !== afterRows.length) throw new Error('Duplicate canonical ID after apply.');
  for (const id of beforeIds) if (!afterIds.has(id)) throw new Error(`Existing canonical ID missing: ${id}`);
  return true;
}

async function prepare(args) {
  assertStage4SSafety(config);
  const required = [args.canonical, args.manifest, args.sidecar, args.decisions, args.review,
    args.evidence, args.stage4qState, args.candidatePack, args.boundary];
  for (const filePath of required) if (!fs.existsSync(filePath)) throw new Error(`Missing Stage 4S input: ${filePath}`);

  const canonicalBefore = inspectCanonicalDataset(args.canonical);
  if (canonicalBefore.rows !== config.expectedBaselineRows
    || canonicalBefore.sha256 !== config.expectedBaselineSha256) {
    throw new Error(`Unexpected Stage 4S canonical baseline: ${canonicalBefore.rows}/${canonicalBefore.sha256}`);
  }
  const reviewRows = readCsvObjects(args.review);
  const decisionRows = readCsvObjects(args.decisions);
  const resolvedContract = resolveHumanDecisionContract(reviewRows, config.authoritativeHumanDecisions);
  const decisionCounts = validateHumanDecisionRows(
    decisionRows,
    reviewRows.map((row) => row.case_id),
    new Map(config.authoritativeHumanDecisions.map((item) => [item.caseId, item.decision])),
  );
  const reviewedCaseIds = new Set(reviewRows.map((row) => row.case_id));
  const stage4qState = readJson(args.stage4qState);
  const resultByCaseId = new Map(stage4qState.results
    .filter((result) => reviewedCaseIds.has(result.caseId)).map((result) => [result.caseId, result]));
  if (resultByCaseId.size !== 26) throw new Error('Stage 4Q evidence does not cover all Stage 4R review cases.');
  const recordByCaseId = await loadTargetRecords(args.candidatePack, reviewedCaseIds);
  if (recordByCaseId.size !== 26) throw new Error('Source pack does not cover all Stage 4R review cases.');
  const canonicalPois = readCanonicalEvidencePois(args.canonical);
  const historicalPois = readHistoricalPois(args.historical.filter((filePath) => fs.existsSync(filePath)));
  const existenceIndex = {
    combined: buildExistenceIndex(canonicalPois, historicalPois),
    thresholds: stage4qConfig.thresholds,
  };
  const boundary = loadBoundaryArtifact(args.boundary);
  const candidates = [...reviewedCaseIds].map((caseId) => evaluateCandidateForReview({
    record: recordByCaseId.get(caseId),
    stage4qResult: resultByCaseId.get(caseId),
    existenceIndex,
    boundary,
    config: stage4rConfig,
    now: stage4rConfig.evaluationReferenceTime,
  })).sort((left, right) => left.caseId.localeCompare(right.caseId));
  const evidenceByCaseId = new Map(readJsonl(args.evidence).map((row) => [row.caseId, row]));
  const decisionSha = sha256(fs.readFileSync(args.decisions));
  const decisionReference = `stage4r-create-new-canary-decisions:${decisionSha}`;
  const memory = importCreateNewDecisionMemory({ candidates, decisionRows, config, decisionReference });
  const revalidation = revalidateHumanApprovals({
    candidates, decisionRows, evidenceByCaseId, config,
  });
  const canonicalIds = new Set(canonicalPois.map((poi) => poi.id));
  const proposedIds = new Set();
  for (const candidate of revalidation.surviving) {
    if (canonicalIds.has(candidate.proposedCanonicalId) || proposedIds.has(candidate.proposedCanonicalId)) {
      throw new Error(`Canonical ID collision: ${candidate.proposedCanonicalId}`);
    }
    proposedIds.add(candidate.proposedCanonicalId);
  }
  const plan = buildApplyPlan({ surviving: revalidation.surviving, decisionMemory: memory, config });
  return { canonicalBefore, reviewRows, decisionRows, decisionCounts, resolvedContract, candidates, memory,
    revalidation, plan, decisionSha, decisionReference };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const prepared = await prepare(args);
  const canonicalText = fs.readFileSync(args.canonical, 'utf8');
  const sidecarText = fs.readFileSync(args.sidecar, 'utf8');
  const existingSidecar = JSON.parse(sidecarText);
  const manifest = readJson(args.manifest);
  const headers = parseCsv(canonicalText)[0].map((header, index) => (
    index === 0 ? header.replace(/^\uFEFF/, '').trim() : header.trim()
  ));
  const memoryByCaseId = new Map(prepared.memory.map((record) => [record.caseId, record]));
  const nextCanonicalText = appendCanonicalRows(canonicalText, prepared.revalidation.surviving, headers);
  const sidecarResult = buildCreateNewSidecar({ existingSidecar,
    candidates: prepared.revalidation.surviving, memoryByCaseId, config });
  const beforeRows = parseCsv(canonicalText).slice(1);
  const afterRows = parseCsv(nextCanonicalText).slice(1);
  validateAppend(canonicalText, nextCanonicalText, beforeRows, afterRows,
    prepared.revalidation.surviving.length);
  const canonicalShaAfter = sha256(Buffer.from(nextCanonicalText, 'utf8'));
  const sidecarHashBefore = stableHash(normalizeSidecar(existingSidecar).records);
  const parsedNextSidecar = JSON.parse(sidecarResult.text);
  const sidecarHashAfter = parsedNextSidecar.contentHash;
  if (serializeSidecar(parsedNextSidecar) !== sidecarResult.text) throw new Error('Sidecar serialization is not deterministic.');
  const planRepeat = buildApplyPlan({ surviving: prepared.revalidation.surviving,
    decisionMemory: prepared.memory, config });
  const canonicalRepeat = appendCanonicalRows(canonicalText, prepared.revalidation.surviving, headers);
  const sidecarRepeat = buildCreateNewSidecar({ existingSidecar,
    candidates: prepared.revalidation.surviving, memoryByCaseId, config });
  if (planRepeat.planHash !== prepared.plan.planHash || canonicalRepeat !== nextCanonicalText
    || sidecarRepeat.text !== sidecarResult.text) throw new Error('Stage 4S deterministic replay failed.');

  const inputBackup = path.join(args.artifactDir, 'backup');
  copyBackup(args.canonical, path.join(inputBackup, 'urbanagent_poi_master_v1.before_stage4s.csv'));
  copyBackup(args.sidecar, path.join(inputBackup, 'stage4l-enrichment-v1.before_stage4s.json'));
  copyBackup(args.decisions, path.join(inputBackup, 'create_new_canary_decisions.stage4s.csv'));
  copyBackup(args.manifest, path.join(inputBackup, 'urbanagent_poi_master_v1_manifest.before_stage4s.json'));
  writeJson(path.join(args.artifactDir, 'create_new_apply_plan.json'), prepared.plan);
  writeJson(path.join(inputBackup, 'create_new_apply_plan.pre_apply.json'), prepared.plan);

  const rollback = {
    canonicalSha256: sha256(Buffer.from(canonicalText, 'utf8')),
    canonicalByteIdentical: nextCanonicalText.slice(0, canonicalText.length) === canonicalText,
    sidecarSha256: sha256(Buffer.from(sidecarText, 'utf8')),
    sidecarExact: fs.readFileSync(path.join(inputBackup, 'stage4l-enrichment-v1.before_stage4s.json'), 'utf8')
      === sidecarText,
  };
  if (!rollback.canonicalByteIdentical || !rollback.sidecarExact
    || rollback.canonicalSha256 !== prepared.canonicalBefore.sha256) throw new Error('Rollback verification failed.');

  const appliedMemory = prepared.memory.map((record) => ({
    ...record,
    applicationStatus: prepared.revalidation.surviving.some((candidate) => candidate.caseId === record.caseId)
      ? 'APPLIED' : prepared.revalidation.dropped.some((item) => item.caseId === record.caseId)
        ? 'STALE_OR_DROPPED' : record.applicationStatus,
  }));
  const state = {
    schemaVersion: 'stage4s-controlled-create-new-state-v1',
    status: args.apply ? 'APPLIED_TO_FEATURE_BRANCH' : 'DRY_RUN_ONLY',
    policyVersion: config.policyVersion,
    decisionScope: config.decisionScope,
    decisionReference: prepared.decisionReference,
    decisionSha256: prepared.decisionSha,
    planHash: prepared.plan.planHash,
    canonicalRowsBefore: prepared.canonicalBefore.rows,
    canonicalRowsAfter: afterRows.length,
    canonicalShaBefore: prepared.canonicalBefore.sha256,
    canonicalShaAfter,
    sidecarHashBefore,
    sidecarHashAfter,
    appliedCaseIds: prepared.revalidation.surviving.map((candidate) => candidate.caseId),
    appliedCanonicalIds: prepared.revalidation.surviving.map((candidate) => candidate.proposedCanonicalId),
    sidecarRecordsCreated: sidecarResult.created.length,
    dropped: prepared.revalidation.dropped,
    autoCreateNew: false,
    runtimeSidecarEnabled: false,
    runtimeCodeChanged: false,
    deletes: 0,
    merges: 0,
    existingRowMutations: 0,
    rollback,
    deterministic: true,
  };

  if (args.apply) {
    if (process.env.URBANAGENT_ALLOW_STAGE4S_CANONICAL_WRITE !== 'true') {
      throw new Error('URBANAGENT_ALLOW_STAGE4S_CANONICAL_WRITE=true is required.');
    }
    fs.writeFileSync(args.canonical, nextCanonicalText, 'utf8');
    fs.writeFileSync(args.sidecar, sidecarResult.text, 'utf8');
    writeJson(args.manifest, updateManifest(manifest, afterRows.length, canonicalShaAfter,
      prepared.revalidation.surviving));
    writeJson(args.state, state);
  }
  writeJson(path.join(args.artifactDir, 'decision_memory.json'), appliedMemory);
  writeJson(path.join(args.artifactDir, 'rollback_verification.json'), rollback);
  writeJson(path.join(args.artifactDir, 'stage4s_summary.json'), state);
  process.stdout.write(`${JSON.stringify({
    verdict: args.apply ? 'PASS_APPLIED_TO_FEATURE_BRANCH' : 'PASS_DRY_RUN_ONLY',
    decisions: prepared.decisionCounts,
    prefixResolution: { resolved: prepared.resolvedContract.length, unique: true },
    originalApproved: 7,
    surviving: prepared.revalidation.surviving.map((candidate) => ({
      caseId: candidate.caseId, name: candidate.record.name, canonicalId: candidate.proposedCanonicalId,
    })),
    dropped: prepared.revalidation.dropped,
    canonical: { before: prepared.canonicalBefore.rows, after: afterRows.length,
      shaBefore: prepared.canonicalBefore.sha256, shaAfter: canonicalShaAfter },
    sidecar: { before: sidecarHashBefore, after: sidecarHashAfter, created: sidecarResult.created.length },
    planHash: prepared.plan.planHash,
    rollback,
    deterministic: true,
    artifactDir: args.artifactDir,
  }, null, 2)}\n`);
}

if (require.main === module) main().catch((error) => {
  process.stderr.write(`${error.stack || error.message}\n`);
  process.exitCode = 1;
});

module.exports = { DEFAULTS, main, parseArgs, prepare, updateManifest, validateAppend };
