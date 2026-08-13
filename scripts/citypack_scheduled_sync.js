const { spawnSync } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');

const { runAutomatedReviewPolicy } = require('../src/modules/cityPackPreparation/automatedReviewPolicy');
const { EXPECTED_CANONICAL_SHA, inspectCanonicalDataset } = require('../src/modules/cityPackPreparation/canonicalDataset');
const { stableValue } = require('../src/modules/cityPackPreparation/decisionMemory');
const { normalizeWithAdapters } = require('../src/modules/cityPackPreparation/dryRun');
const { readJsonl, writeJsonl } = require('../src/modules/cityPackPreparation/incrementalSync');
const { loadRealSnapshotRecords } = require('../src/modules/cityPackPreparation/realSnapshot');
const {
  RUN_STATES,
  orchestrateScheduledSync,
  readJson,
  selectSources,
  writeJsonAtomic,
} = require('../src/modules/cityPackPreparation/scheduledSyncOrchestrator');
const { enrichReviewCases } = require('./phase4_stage4l_automated_review_policy');

const ROOT = path.resolve(__dirname, '..');
const DEFAULT_CONFIG = path.join(ROOT, 'config', 'phase4_stage4n_scheduled_sync.json');
const DEFAULT_POLICY_CONFIG = path.join(ROOT, 'config', 'phase4_stage4l_policy.json');
const DEFAULT_CANONICAL = path.join(ROOT, 'data', 'canonical', 'urbanagent_poi_master_v1.csv');

function parseArgs(argv) {
  const args = {};
  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index];
    if (!value.startsWith('--')) throw new Error(`Unexpected argument: ${value}`);
    const key = value.slice(2).replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
    const next = argv[index + 1];
    if (!next || next.startsWith('--')) args[key] = true;
    else { args[key] = next; index += 1; }
  }
  if (args.source && args.allSources) throw new Error('--source and --all-sources are mutually exclusive.');
  return args;
}

function writeJson(filePath, value) {
  writeJsonAtomic(filePath, stableValue(value));
}

function snapshotMetadata(manifest, snapshotDir) {
  const metadataBySource = {};
  const snapshotsBySource = {};
  const pathsBySource = {};
  for (const item of manifest?.snapshots || []) {
    const source = item.source;
    const filePath = path.resolve(snapshotDir, item.file || item.cacheFile);
    pathsBySource[source] = filePath;
    metadataBySource[source] = {
      version: item.snapshotRef || item.sha256,
      sha256: item.sha256,
      retrievedAt: item.retrievedAt || manifest.retrievedAt || null,
      filePath,
    };
  }
  const isRealBoundedManifest = manifest?.status === 'NON_CANONICAL_BOUNDED_REAL_SOURCE_SNAPSHOT_MANIFEST';
  if (isRealBoundedManifest) {
    const adapterRecords = loadRealSnapshotRecords({
      overture: pathsBySource.overture,
      osm: pathsBySource.osm,
      wikidata: pathsBySource.wikidata,
      commons: pathsBySource.wikimedia_commons,
    });
    const normalized = normalizeWithAdapters(adapterRecords);
    for (const source of Object.keys(metadataBySource)) {
      const records = source === 'wikimedia_commons'
        ? normalized.filter((record) => record.source === 'wikidata' && record.media)
        : normalized.filter((record) => record.source === source);
      snapshotsBySource[source] = { records };
    }
  } else {
    for (const [source, filePath] of Object.entries(pathsBySource)) {
      const snapshot = readJson(filePath, null);
      if (!snapshot) throw new Error(`Missing configured local snapshot: ${filePath}`);
      snapshotsBySource[source] = {
        ...snapshot,
        records: normalizeWithAdapters(snapshot.records || []),
      };
    }
  }
  return { metadataBySource, snapshotsBySource };
}

function affectedPolicyCases({ cases, incremental }) {
  const affected = new Set((incremental.affectedState || incremental.state)
    .filter((item) => ['NEW', 'CHANGED', 'MISSING'].includes(item.status))
    .map((item) => `${item.source}:${item.sourceId}`));
  return cases.filter((item) => {
    const source = item.source?.source || item.source;
    const sourceId = item.source?.sourceId || item.sourceId;
    return affected.has(`${source}:${sourceId}`);
  });
}

function invokeStage4m({ args, runId, artifactDir, preparePr = false }) {
  const required = ['policyCases', 'decisionMemory', 'recoveredEnrichments'];
  const missing = required.filter((key) => !args[key]);
  if (missing.length) throw new Error(`Stage 4M preparation requires: ${missing.join(', ')}`);
  const stage4mArgs = [
    'run', 'citypack:auto-pr', '--',
    preparePr ? '--apply-safe' : '--dry-run',
    '--create-branch', '--run-id', runId,
    '--artifact-dir', path.join(artifactDir, 'safe-pr'),
    '--policy-cases', path.resolve(args.policyCases),
    '--decision-memory', path.resolve(args.decisionMemory),
    '--recovered-enrichments', path.resolve(args.recoveredEnrichments),
    '--max-auto-cases', String(args.maxAutoCases || 50),
    '--max-field-mutations', String(args.maxFieldMutations || 100),
  ];
  if (args.noNetwork) stage4mArgs.push('--no-network');
  if (preparePr && !args.noNetwork) stage4mArgs.push('--run-tests', '--commit', '--push', '--create-pr');
  const result = spawnSync('npm.cmd', stage4mArgs, { cwd: ROOT, encoding: 'utf8', shell: false });
  if (result.status !== 0) throw new Error(`Stage 4M executor failed: ${String(result.stderr || '').trim()}`);
  const output = String(result.stdout || '');
  const start = output.lastIndexOf('\n{');
  return JSON.parse(output.slice(start >= 0 ? start + 1 : output.indexOf('{')));
}

function promoteOperationalState({ dataDir, result, sourceState, metadataBySource, now }) {
  if (!result.statePromoted) return;
  const stateDir = path.join(dataDir, 'state');
  if (result.proposedState) writeJsonl(path.join(stateDir, 'incremental_state.jsonl'), result.proposedState);
  if (result.proposedMediaRegistry) {
    writeJsonl(path.join(stateDir, 'media_registry.jsonl'), result.proposedMediaRegistry);
  }
  if (result.exceptionLedger) writeJson(path.join(stateDir, 'exception_ledger.json'), result.exceptionLedger);
  const nextSources = { ...(sourceState.sources || {}) };
  for (const [source, metadata] of Object.entries(metadataBySource)) {
    nextSources[source] = {
      ...nextSources[source], version: metadata.version, sha256: metadata.sha256 || null,
      lastCheckedAt: new Date(now).toISOString(), lastSuccessfulAt: new Date(now).toISOString(),
    };
  }
  writeJson(path.join(stateDir, 'source_state.json'), {
    schemaVersion: 'stage4n-source-state-v1', cityId: 'da-nang', sources: nextSources,
  });
  if (['PR_CREATED', 'EXISTING_PR'].includes(result.safeResult?.prCreation)) {
    writeJson(path.join(stateDir, 'open_pr.json'), {
      schemaVersion: 'stage4n-open-pr-v1',
      runId: result.runId,
      planHash: result.planHash,
      inputStateHash: result.inputStateHash,
      url: result.safeResult.compareUrl || result.safeResult.prUrl || null,
      branch: result.safeResult.branch || null,
      status: 'OPEN',
      recordedAt: new Date(now).toISOString(),
    });
  }
}

function cleanupSuccessfulTemp(tempDir) {
  if (!fs.existsSync(tempDir)) return 0;
  const entries = fs.readdirSync(tempDir, { withFileTypes: true });
  for (const entry of entries) fs.rmSync(path.join(tempDir, entry.name), { recursive: true, force: true });
  return entries.length;
}

async function runCli(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  const required = ['artifactDir', 'dataDir', 'tempDir', 'snapshotManifest'];
  const missing = required.filter((key) => !args[key]);
  if (missing.length) throw new Error(`Missing required arguments: ${missing.join(', ')}`);
  const config = readJson(path.resolve(args.config || DEFAULT_CONFIG));
  const sources = selectSources({ source: args.source, allSources: args.allSources, config });
  const noNetwork = args.noNetwork === true || config.networkEnabledByDefault === false;
  if (!noNetwork && !args.networkProvider) {
    throw new Error('No approved network source provider is configured; use --no-network with prepared bounded snapshots.');
  }
  const canonical = inspectCanonicalDataset(path.resolve(args.canonical || DEFAULT_CANONICAL));
  if (canonical.rows !== config.expectedCanonicalRows || canonical.sha256 !== EXPECTED_CANONICAL_SHA) {
    throw new Error(`BLOCKED_BY_SAFETY_GATE:UNEXPECTED_CANONICAL_BASELINE:${canonical.sha256}`);
  }
  const artifactDir = path.resolve(args.artifactDir);
  const dataDir = path.resolve(args.dataDir);
  const tempDir = path.resolve(args.tempDir);
  const manifestPath = path.resolve(args.snapshotManifest);
  const localManifest = readJson(manifestPath);
  const { metadataBySource: allMetadata, snapshotsBySource: allSnapshots } = snapshotMetadata(
    localManifest, path.dirname(manifestPath),
  );
  const metadataBySource = Object.fromEntries(sources.map((source) => [source, allMetadata[source]]));
  if (Object.values(metadataBySource).some((item) => !item)) throw new Error('Snapshot manifest lacks a selected source.');
  const snapshotsBySource = Object.fromEntries(sources.map((source) => [source, allSnapshots[source]]));
  const stateDir = path.join(dataDir, 'state');
  const sourceState = readJson(path.join(stateDir, 'source_state.json'), { sources: {} });
  const previousState = readJsonl(path.join(stateDir, 'incremental_state.jsonl'));
  const previousMediaRegistry = readJsonl(path.join(stateDir, 'media_registry.jsonl'));
  const checkpointPath = path.join(stateDir, 'checkpoint.json');
  const checkpoint = args.resume ? readJson(checkpointPath, null) : null;
  const policyCases = args.policyCases
    ? enrichReviewCases(readJsonl(path.resolve(args.policyCases))) : [];
  const decisionMemory = args.decisionMemory ? readJsonl(path.resolve(args.decisionMemory)) : [];
  const policyConfig = readJson(path.resolve(args.policyConfig || DEFAULT_POLICY_CONFIG));
  policyConfig.anomalyThresholds.expectedCanonicalSha256 = canonical.sha256;
  const now = args.now || new Date().toISOString();
  const result = await orchestrateScheduledSync({
    config, sources, now, dataDir, artifactDir, tempDir, sourceState,
    metadataBySource, snapshotsBySource, previousState, previousMediaRegistry,
    checkpoint, resume: args.resume === true,
    dryRun: args.dryRun !== false && args.preparePr !== true && args.applyState !== true,
    preparePr: args.preparePr === true,
    openPr: readJson(path.join(stateDir, 'open_pr.json'), null),
    exceptionLedger: readJson(path.join(stateDir, 'exception_ledger.json'), []),
    policyRunner: ({ incremental }) => runAutomatedReviewPolicy(
      affectedPolicyCases({ cases: policyCases, incremental }),
      { memory: decisionMemory, config: policyConfig, materializeReusableApprovals: true,
        runMetrics: { canonicalSha256: canonical.sha256 } },
    ),
    safePrRunner: ({ runId, artifactDir: runArtifacts, preparePr }) => invokeStage4m({
      args, runId, artifactDir: runArtifacts, preparePr,
    }),
  });
  if (result.resumable && result.checkpoint) writeJson(checkpointPath, result.checkpoint);
  else if (result.status !== RUN_STATES.FAILED) fs.rmSync(checkpointPath, { force: true });
  promoteOperationalState({ dataDir, result, sourceState, metadataBySource, now });
  const tempCleaned = [RUN_STATES.NO_SOURCE_CHANGE, RUN_STATES.NO_RELEVANT_POI_DELTA,
    RUN_STATES.NO_SAFE_CHANGES, RUN_STATES.EXCEPTIONS_ONLY, RUN_STATES.COMPLETED].includes(result.status)
    ? cleanupSuccessfulTemp(tempDir) : 0;
  const { proposedState, proposedMediaRegistry, exceptionLedger, policyOutcomes, ...summary } = result;
  return { ...summary, proposedStateRecords: proposedState?.length || 0,
    proposedMediaRecords: proposedMediaRegistry?.length || 0,
    exceptionLedgerRecords: exceptionLedger?.length || 0,
    policyOutcomeCounts: (policyOutcomes || []).reduce((counts, item) => ({
      ...counts, [item.outcome]: (counts[item.outcome] || 0) + 1,
    }), {}), canonical, noNetwork, tempCleaned, schedulerActivated: false,
    ciScheduleEnabled: false };
}

if (require.main === module) {
  runCli().then((result) => console.log(JSON.stringify(result, null, 2))).catch((error) => {
    console.error(error.message);
    process.exitCode = 1;
  });
}

module.exports = {
  affectedPolicyCases,
  cleanupSuccessfulTemp,
  invokeStage4m,
  parseArgs,
  promoteOperationalState,
  runCli,
  snapshotMetadata,
};
