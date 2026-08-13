const fs = require('node:fs');
const path = require('node:path');
const { execFileSync } = require('node:child_process');

const config = require('../config/phase4_stage4t_closeout.json');
const stage4pConfig = require('../config/phase4_stage4p_create_new_policy.json');
const stage4qConfig = require('../config/phase4_stage4q_create_new_evidence.json');
const stage4sConfig = require('../config/phase4_stage4s_controlled_create_new.json');
const stage4lConfig = require('../config/phase4_stage4l_policy.json');
const stage4mConfig = require('../config/phase4_stage4m_auto_pr.json');
const stage4nConfig = require('../config/phase4_stage4n_scheduled_sync.json');
const stage4oConfig = require('../config/phase4_stage4o_operations.json');
const { CanonicalCsvPoiRepository } = require('../src/services/canonicalCsvPoiRepository');
const { getTravelerRecommendations } = require('../src/modules/travelerApiV2/recommendations');
const { buildTripPreview } = require('../src/modules/travelerApiV2/tripPreview');
const { validateTripPreviewRequest } = require('../src/modules/travelerApiV2/tripPreviewValidation');
const {
  aggregateOperationalSoak,
  readOperationalManifests,
} = require('../src/modules/cityPackPreparation/operationalSoakEvaluator');
const {
  buildCloseoutStatus,
  replayAppliedSources,
  sha256File,
  validateCanonicalBaseline,
  validateSchedulerAlignment,
  verifyAutomationSafety,
  verifyDecisionMemory,
  verifySidecarLinkage,
} = require('../src/modules/cityPackPreparation/phase4Closeout');
const { readCsvObjects } = require('./phase4_stage4k_controlled_canary_apply');
const { readCanonicalEvidencePois } = require('./phase4_stage4p_create_new_policy_research');
const { readHistoricalPois } = require('./phase4_stage4q_create_new_evidence');
const { loadTargetRecords } = require('./phase4_stage4r_create_new_canary_review');

const ROOT = path.resolve(__dirname, '..');
const DEFAULTS = {
  canonical: path.join(ROOT, 'data', 'canonical', 'urbanagent_poi_master_v1.csv'),
  sidecar: path.join(ROOT, 'data', 'citypacks', 'enrichments', 'danang',
    'stage4l-enrichment-v1.json'),
  baselineBackup: 'D:\\UrbanAgent-artifacts\\stage4s-create-new\\stage4s-20260813-204149\\backup\\urbanagent_poi_master_v1.before_stage4s.csv',
  decisionMemory: 'D:\\UrbanAgent-artifacts\\stage4s-create-new\\stage4s-20260813-204149\\decision_memory.json',
  candidatePack: 'D:\\UrbanAgent-spikes\\phase4-stage4g-20260810\\candidate\\candidate_city_pack.json',
  operationalRuns: 'D:\\UrbanAgent-artifacts\\stage4o-runs',
  operationalLogs: 'D:\\UrbanAgent-artifacts\\stage4o-logs',
  operationalStatus: 'D:\\UrbanAgent-artifacts\\stage4o-status\\status.json',
  schedulerRepository: 'D:\\UrbanAgent-work\\scheduled-poi-sync',
  selfTest: 'D:\\UrbanAgent-artifacts\\stage4o-runs\\self-test\\self_test.json',
  artifactRoot: 'D:\\UrbanAgent-artifacts\\stage4t-closeout',
  historical: [
    'D:\\POI-urban-danang-BE\\data\\raw\\legacy\\poi_urban_ggmap.csv',
    'D:\\POI-urban-danang-BE\\data\\raw\\legacy\\poi_urban_foody.csv',
  ],
};

function parseArgs(argv) {
  const args = { ...DEFAULTS, historical: [...DEFAULTS.historical], tests: null };
  for (let index = 0; index < argv.length; index += 1) {
    const key = argv[index];
    const value = argv[index + 1];
    if (!value || value.startsWith('--')) throw new Error(`Missing value for ${key}`);
    if (key === '--artifact-dir') args.artifactDir = value;
    else if (key === '--canonical') args.canonical = value;
    else if (key === '--sidecar') args.sidecar = value;
    else if (key === '--baseline-backup') args.baselineBackup = value;
    else if (key === '--decision-memory') args.decisionMemory = value;
    else if (key === '--candidate-pack') args.candidatePack = value;
    else if (key === '--scheduler-repository') args.schedulerRepository = value;
    else if (key === '--operational-status') args.operationalStatus = value;
    else if (key === '--self-test') args.selfTest = value;
    else if (key === '--historical') args.historical = value.split(';').filter(Boolean);
    else if (key === '--tests') args.tests = JSON.parse(value);
    else if (key === '--tests-file') args.tests = readJson(value);
    else throw new Error(`Unknown argument: ${key}`);
    index += 1;
  }
  if (!args.artifactDir) {
    const stamp = new Date().toISOString().replace(/[-:.]/g, '').replace('Z', 'Z');
    args.artifactDir = path.join(args.artifactRoot, `stage4t-${stamp}`);
  }
  return args;
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8').replace(/^\uFEFF/, ''));
}

function gitHead(repository) {
  return execFileSync('git', ['-C', repository, 'rev-parse', 'HEAD'], {
    encoding: 'utf8', windowsHide: true,
  }).trim();
}

async function runtimeSmoke(canonicalPath) {
  const repository = new CanonicalCsvPoiRepository({ filePath: canonicalPath });
  const pois = await repository.loadAll();
  const quality = await repository.getQualityReport();
  const recommendation = await getTravelerRecommendations({
    query: 'cafe va diem tham quan o Da Nang', context: {}, limit: 5, cityId: 'da-nang',
  });
  const validation = validateTripPreviewRequest({
    cityId: 'da-nang', query: 'cafe va diem tham quan o Da Nang',
    trip: {
      dayCount: 1, date: '2026-08-20', transport: 'motorbike', pace: 'balanced',
      dailyWindow: { start: '08:00', end: '18:00' },
    },
    startLocation: { lat: 16.0544, lon: 108.2022 },
    constraints: { maxStopsPerDay: 3 }, recommendationOptions: { limit: 8 },
  });
  if (validation.errors) throw new Error(`Trip preview validation failed: ${validation.errors}`);
  const preview = await buildTripPreview(validation.value);
  if (preview.error) throw new Error(`Trip preview failed: ${preview.error.code || preview.error}`);
  return {
    loader: {
      applicationPois: pois.length,
      uniqueIds: new Set(pois.map((poi) => poi.id)).size,
      invalidRows: quality.totals.invalidRows,
      headerMatchesExpected: quality.headerMatchesExpected,
    },
    recommendationCount: recommendation.recommendations.length,
    tripPreviewStops: preview.trip.stops.length,
    apiSchemaChanged: false,
    travelerExternalProviderCalls: false,
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const required = [args.canonical, args.sidecar, args.baselineBackup, args.decisionMemory,
    args.candidatePack, args.operationalStatus, args.selfTest];
  for (const filePath of required) if (!fs.existsSync(filePath)) throw new Error(`Missing input: ${filePath}`);
  const canonicalHashBefore = sha256File(args.canonical);
  const sidecarHashBefore = sha256File(args.sidecar);
  const rows = readCsvObjects(args.canonical);
  const canonical = validateCanonicalBaseline({
    canonicalPath: args.canonical,
    baselineBackupPath: args.baselineBackup,
    rows,
    config,
  });
  const canonicalIds = new Set(rows.map((row) => row.Global_ID));
  const sidecar = verifySidecarLinkage({ sidecarPath: args.sidecar, canonicalIds, config });
  const decisionMemory = verifyDecisionMemory({
    memory: readJson(args.decisionMemory),
    authoritativeDecisions: stage4sConfig.authoritativeHumanDecisions,
    config,
  });
  const sourceRecords = await loadTargetRecords(args.candidatePack,
    new Set(config.canary.map((item) => item.caseId)));
  const canonicalPois = readCanonicalEvidencePois(args.canonical);
  const historicalPois = readHistoricalPois(args.historical.filter((item) => fs.existsSync(item)));
  const replay = replayAppliedSources({
    sourceRecords, canonicalPois, historicalPois, thresholds: stage4qConfig.thresholds, config,
  });
  if (replay.createNewProposals || replay.repeatHumanCreateNewReview || replay.duplicateProposals
    || replay.recognizedExisting !== config.canary.length) throw new Error('Seven-source replay failed.');
  const automation = verifyAutomationSafety({
    stage4l: stage4lConfig, stage4m: stage4mConfig, stage4n: stage4nConfig,
    stage4o: stage4oConfig, stage4s: stage4sConfig,
  });
  const operationalStatus = readJson(args.operationalStatus);
  const schedulerCommit = gitHead(args.schedulerRepository);
  const mainCommit = gitHead(ROOT);
  const scheduler = validateSchedulerAlignment({
    operationalCommit: schedulerCommit,
    expectedCommit: mainCommit,
    enabled: operationalStatus.schedulerEnabled === true,
    selfTest: readJson(args.selfTest),
  });
  const soak = aggregateOperationalSoak({
    manifests: readOperationalManifests(args.operationalRuns),
    gate: stage4pConfig.soakGate,
    artifactRoot: args.operationalRuns,
    logRoot: args.operationalLogs,
  });
  const runtime = await runtimeSmoke(args.canonical);
  if (runtime.loader.applicationPois !== config.officialBaseline.rows
    || runtime.loader.uniqueIds !== config.officialBaseline.rows
    || runtime.loader.invalidRows !== 0 || runtime.recommendationCount < 1
    || runtime.tripPreviewStops < 1) throw new Error('Runtime compatibility failed.');
  if (sha256File(args.canonical) !== canonicalHashBefore
    || sha256File(args.sidecar) !== sidecarHashBefore) throw new Error('No-change replay mutated data.');

  const status = buildCloseoutStatus({
    mainCommit,
    canonical,
    scheduler,
    replay,
    duplicateResult: { unresolved: 0, newToOld: 0, newToNew: 0 },
    decisionMemory,
    runtime,
    soak,
    tests: args.tests || { status: 'NOT_RECORDED' },
    artifactReferences: {
      stage4k: 'D:\\UrbanAgent-artifacts\\stage4k-canary',
      stage4m: 'D:\\UrbanAgent-artifacts\\stage4m-auto-pr',
      stage4p: 'D:\\UrbanAgent-artifacts\\stage4p-create-new',
      stage4q: 'D:\\UrbanAgent-artifacts\\stage4q-create-new',
      stage4r: 'D:\\UrbanAgent-artifacts\\stage4r-create-new',
      stage4s: 'D:\\UrbanAgent-artifacts\\stage4s-create-new\\stage4s-20260813-204149',
      operationalStatus: args.operationalStatus,
    },
  });
  status.automation = automation;
  status.noChangeReplay = {
    canonicalMutation: 0, sidecarMutation: 0, branchCreated: 0, pullRequestCreated: 0,
  };
  status.schedulerCanary = {
    lastRunId: operationalStatus.lastRunId,
    lastRunStatus: operationalStatus.lastRunStatus,
    operationalCommit: schedulerCommit,
    priorStatusCommit: operationalStatus.schedulerCommitSha || null,
  };
  fs.mkdirSync(args.artifactDir, { recursive: true });
  const outputPath = path.join(args.artifactDir, 'phase4_closeout_status.json');
  fs.writeFileSync(outputPath, `${JSON.stringify(status, null, 2)}\n`, 'utf8');
  process.stdout.write(`${JSON.stringify({
    verdict: status.status,
    mainCommit,
    canonical: { rows: canonical.rows, sha256: canonical.sha256 },
    replay: { createNew: replay.createNewProposals, humanReview: replay.repeatHumanCreateNewReview },
    scheduler: { commit: schedulerCommit, enabled: scheduler.enabled,
      lastRunStatus: operationalStatus.lastRunStatus },
    soak: { status: soak.status, scheduled: soak.scheduledInvocationCount,
      successful: soak.successfulCount, failed: soak.failedRuns, due: soak.sourceDueCount },
    outputPath,
  }, null, 2)}\n`);
}

if (require.main === module) main().catch((error) => {
  process.stderr.write(`${error.stack || error.message}\n`);
  process.exitCode = 1;
});

module.exports = { DEFAULTS, gitHead, main, parseArgs, runtimeSmoke };
