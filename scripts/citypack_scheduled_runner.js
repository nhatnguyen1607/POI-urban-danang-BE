const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

const { readJson, writeJsonAtomic } = require('../src/modules/cityPackPreparation/scheduledSyncOrchestrator');
const {
  FAILURE_CLASSES,
  classifyOperationalError,
  cleanupOperationalRetention,
  inspectStaleLock,
  runOperationalPreflight,
  summarizeRunStatus,
  writeOperationalResult,
} = require('../src/modules/cityPackPreparation/scheduledSyncOperations');

const ROOT = path.resolve(__dirname, '..');
const DEFAULT_OPERATIONS_CONFIG = path.join(ROOT, 'config', 'phase4_stage4o_operations.json');
const DEFAULT_STAGE4N_CONFIG = path.join(ROOT, 'config', 'phase4_stage4n_scheduled_sync.json');
const DEFAULT_POLICY_CONFIG = path.join(ROOT, 'config', 'phase4_stage4l_policy.json');
const DEFAULT_BOUNDARY_POLICY = path.join(ROOT, 'data', 'spikes', 'phase4', 'stage4j', 'boundary_policy.json');
const DEFAULT_BOUNDARY_GEOJSON = path.join(ROOT, 'data', 'spikes', 'phase4', 'stage4i', 'boundary',
  'osm-relation-1891418-v72.geojson');
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
  return args;
}

function resolvePaths(args, config) {
  return {
    repository: path.resolve(args.repository || ROOT),
    dataDir: path.resolve(args.dataDir || config.paths.dataRoot),
    artifactRoot: path.resolve(args.artifactRoot || config.paths.artifactRoot),
    tempRoot: path.resolve(args.tempDir || config.paths.tempRoot),
    statusPath: path.resolve(args.statusPath || config.paths.statusPath),
    snapshotManifest: args.snapshotManifest ? path.resolve(args.snapshotManifest) : null,
  };
}

function compactError(error) {
  return {
    name: error.name || 'Error',
    code: error.code || null,
    message: String(error.message || error).slice(0, 500),
  };
}

function parseChildJson(stdout) {
  const text = String(stdout || '').trim();
  const candidates = [];
  for (let index = text.lastIndexOf('\n{'); index >= 0; index = text.lastIndexOf('\n{', index - 1)) {
    candidates.push(text.slice(index + 1));
  }
  if (text.startsWith('{')) candidates.push(text);
  for (const candidate of candidates) {
    try { return JSON.parse(candidate); } catch (_) { /* inspect next candidate */ }
  }
  throw Object.assign(new Error('Scheduled sync child did not produce a valid JSON result.'), {
    code: 'CHILD_OUTPUT_SCHEMA',
  });
}

function runStage4nChild({ args, paths, config, artifactDir, spawnSyncImpl = spawnSync }) {
  if (!paths.snapshotManifest) throw new Error('--snapshot-manifest is required for an operational run.');
  const childArgs = [
    path.join(paths.repository, 'scripts', 'citypack_scheduled_sync.js'),
    args.dryRun ? '--dry-run' : '--apply-state',
    '--no-network',
    '--artifact-dir', path.join(artifactDir, 'stage4n'),
    '--data-dir', paths.dataDir,
    '--temp-dir', path.join(paths.tempRoot, path.basename(artifactDir)),
    '--snapshot-manifest', paths.snapshotManifest,
  ];
  if (args.source && args.source !== 'all') childArgs.push('--source', args.source);
  else childArgs.push('--all-sources');
  if (args.resume) childArgs.push('--resume');
  if (args.preparePr) {
    const applyIndex = childArgs.indexOf('--apply-state');
    if (applyIndex >= 0) childArgs.splice(applyIndex, 1, '--prepare-pr');
    for (const [flag, key] of [
      ['--policy-cases', 'policyCases'],
      ['--decision-memory', 'decisionMemory'],
      ['--recovered-enrichments', 'recoveredEnrichments'],
    ]) {
      if (!args[key]) throw new Error(`${flag} is required with --prepare-pr.`);
      childArgs.push(flag, path.resolve(args[key]));
    }
  }
  const stdoutPath = path.join(artifactDir, 'scheduled-sync.stdout.log');
  const stderrPath = path.join(artifactDir, 'scheduled-sync.stderr.log');
  const child = spawnSyncImpl(process.execPath, childArgs, {
    cwd: paths.repository,
    encoding: 'utf8',
    shell: false,
    timeout: config.timeoutsSeconds.scheduledRun * 1000,
    maxBuffer: 8 * 1024 * 1024,
  });
  fs.writeFileSync(stdoutPath, String(child.stdout || ''), 'utf8');
  fs.writeFileSync(stderrPath, String(child.stderr || ''), 'utf8');
  if (child.error) throw child.error;
  if (child.status !== 0) {
    const message = String(child.stderr || child.stdout || `Child exited ${child.status}`).trim().slice(-1000);
    throw Object.assign(new Error(message), { code: `CHILD_EXIT_${child.status}` });
  }
  return { result: parseChildJson(child.stdout), stdoutPath, stderrPath };
}

function runSelfTest({ args, config, paths, spawnSyncImpl }) {
  const artifactDir = path.join(paths.artifactRoot, 'self-test');
  const preflight = runOperationalPreflight({
    repository: paths.repository,
    dataDir: paths.dataDir,
    artifactDir,
    tempDir: paths.tempRoot,
    operationalConfig: config,
    canonicalPath: path.resolve(args.canonical || DEFAULT_CANONICAL),
    stage4nConfigPath: path.resolve(args.stage4nConfig || DEFAULT_STAGE4N_CONFIG),
    policyConfigPath: path.resolve(args.policyConfig || DEFAULT_POLICY_CONFIG),
    boundaryPolicyPath: path.resolve(args.boundaryPolicy || DEFAULT_BOUNDARY_POLICY),
    boundaryGeoJsonPath: path.resolve(args.boundaryGeojson || DEFAULT_BOUNDARY_GEOJSON),
    spawnSyncImpl,
  });
  writeJsonAtomic(path.join(artifactDir, 'self_test.json'), preflight);
  return preflight;
}

function readHealth(statusPath, lastFailure = false) {
  const status = readJson(statusPath, null);
  if (!status) return { status: 'NO_STATUS', statusPath };
  if (lastFailure) {
    return {
      failureClass: status.lastFailureClass,
      failureMessage: status.lastFailureMessage,
      lastFailure: status.lastFailure,
      manifestPath: status.lastManifestPath,
      resumeAvailable: status.resumeAvailable,
    };
  }
  return {
    scheduler: status.schedulerEnabled ? 'ENABLED' : 'DISABLED_OR_UNKNOWN',
    taskName: status.schedulerTaskName,
    lastRun: status.lastRunEnd,
    lastRunStatus: status.lastRunStatus,
    lastSuccess: status.lastSuccessfulRun,
    nextRun: status.nextScheduledRun,
    sources: status.sourceHealth || {},
    openSafePr: status.lastPrUrl,
    pendingHumanExceptions: status.lastHumanReviewDeltaCount || 0,
    canonicalSha: status.currentCanonicalSha,
    lock: status.currentLock,
    statusPath,
  };
}

function handleExistingLock(paths, config, now) {
  const lockPath = path.join(paths.dataDir, 'locks', 'da-nang-stage4n.lock.json');
  const lock = readJson(lockPath, null);
  if (!lock) return { action: 'NO_LOCK' };
  const inspection = inspectStaleLock({
    lock,
    now,
    staleAfterMinutes: readJson(DEFAULT_STAGE4N_CONFIG).lock.staleAfterMinutes,
  });
  if (inspection.action === 'CLEAR_STALE_LOCK') {
    const stalePath = `${lockPath}.stage4o-cleared-${lock.runId || 'unknown'}`;
    fs.renameSync(lockPath, stalePath);
    return { ...inspection, stalePath };
  }
  return { ...inspection, lock };
}

async function runOperational(argv = process.argv.slice(2), dependencies = {}) {
  const args = parseArgs(argv);
  const configPath = path.resolve(args.operationsConfig || DEFAULT_OPERATIONS_CONFIG);
  const config = readJson(configPath);
  const paths = resolvePaths(args, config);
  if (args.health || args.lastFailure) return readHealth(paths.statusPath, args.lastFailure === true);
  if (args.selfTest) return runSelfTest({ args, config, paths, spawnSyncImpl: dependencies.spawnSyncImpl });

  const startedAt = args.now || new Date().toISOString();
  const runDirectoryName = startedAt.replace(/[-:.]/g, '').replace('Z', 'Z');
  const artifactDir = path.join(paths.artifactRoot, runDirectoryName);
  fs.mkdirSync(artifactDir, { recursive: true });
  const previous = readJson(paths.statusPath, {});
  let result;
  try {
    const preflight = runSelfTest({ args, config, paths, spawnSyncImpl: dependencies.spawnSyncImpl });
    if (!preflight.ok) {
      const failed = preflight.checks.filter((item) => !item.ok).map((item) => item.name).join(',');
      throw Object.assign(new Error(`Runner preflight failed: ${failed}`), { code: 'PREFLIGHT_FAILED' });
    }
    const lockInspection = handleExistingLock(paths, config, startedAt);
    if (lockInspection.action === 'KEEP_LOCK') {
      result = { runId: `stage4o-${runDirectoryName}`, status: 'SKIPPED_ALREADY_RUNNING',
        triggerType: args.triggerType || 'MANUAL', lock: lockInspection.lock,
        failureClass: null, lockInspection };
    } else {
      const child = runStage4nChild({ args, paths, config, artifactDir,
        spawnSyncImpl: dependencies.spawnSyncImpl || spawnSync });
      result = { ...child.result, triggerType: args.triggerType || 'MANUAL',
        runnerStdout: child.stdoutPath, runnerStderr: child.stderrPath, lockInspection };
      const exceptionArtifactPath = path.join(artifactDir, 'stage4n', 'human_review_delta.json');
      if (fs.existsSync(exceptionArtifactPath)) result.exceptionArtifactPath = exceptionArtifactPath;
    }
  } catch (error) {
    result = {
      runId: `stage4o-${runDirectoryName}`,
      status: 'FAILED',
      triggerType: args.triggerType || 'MANUAL',
      failureClass: classifyOperationalError(error),
      error: compactError(error),
      statePromoted: false,
    };
  }
  const endedAt = new Date().toISOString();
  const written = writeOperationalResult({
    statusPath: paths.statusPath,
    artifactDir,
    previous,
    result,
    config,
    startedAt,
    endedAt,
  });
  const retention = summarizeRunStatus(result) === 'FAILED'
    ? { removedRunDirectories: 0, removedLogs: 0 }
    : cleanupOperationalRetention({
      artifactRoot: paths.artifactRoot,
      logRoot: path.resolve(config.paths.logRoot),
      now: endedAt,
      retention: config.retention,
    });
  return {
    runId: result.runId,
    finalStatus: summarizeRunStatus(result),
    stage4nStatus: result.status,
    failureClass: result.failureClass || null,
    sourceChecks: result.discovery?.checks || [],
    deltaCounts: result.deltaCounts || {},
    safeMutations: Number(result.canonicalMutations || 0) + Number(result.sidecarMutations || 0),
    humanExceptions: result.exceptions || 0,
    prUrl: result.safeResult?.compareUrl || result.openPr || null,
    manifestPath: written.manifestPath,
    statusPath: paths.statusPath,
    retention,
  };
}

if (require.main === module) {
  runOperational().then((result) => {
    console.log(JSON.stringify(result, null, 2));
    if (result.ok === false || result.finalStatus === 'FAILED' || result.finalStatus === 'BLOCKED') {
      process.exitCode = 1;
    }
  }).catch((error) => {
    console.error(error.message);
    process.exitCode = 1;
  });
}

module.exports = {
  compactError,
  handleExistingLock,
  parseArgs,
  parseChildJson,
  readHealth,
  resolvePaths,
  runOperational,
  runSelfTest,
  runStage4nChild,
};
