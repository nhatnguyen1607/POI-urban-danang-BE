const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

const { inspectCanonicalDataset } = require('./canonicalDataset');
const {
  acquireExecutionLock,
  readJson,
  releaseExecutionLock,
  writeJsonAtomic,
} = require('./scheduledSyncOrchestrator');

const FAILURE_CLASSES = Object.freeze({
  RUNNER_FAILURE: 'RUNNER_FAILURE',
  APPLICATION_FAILURE: 'APPLICATION_FAILURE',
  SOURCE_NETWORK_FAILURE: 'SOURCE_NETWORK_FAILURE',
  SOURCE_RATE_LIMITED: 'SOURCE_RATE_LIMITED',
  SOURCE_SCHEMA_FAILURE: 'SOURCE_SCHEMA_FAILURE',
  STATE_FAILURE: 'STATE_FAILURE',
  LOCK_FAILURE: 'LOCK_FAILURE',
  GIT_FAILURE: 'GIT_FAILURE',
  PR_FAILURE: 'PR_FAILURE',
  SAFETY_GATE_BLOCK: 'SAFETY_GATE_BLOCK',
});

const SUCCESSFUL_RUN_STATUSES = new Set([
  'NO_SOURCE_CHANGE',
  'NO_RELEVANT_POI_DELTA',
  'NO_SAFE_CHANGES',
  'EXCEPTIONS_ONLY',
  'COMPLETED',
]);

function classifyOperationalError(error = {}) {
  const message = String(error.message || error).toLowerCase();
  const code = String(error.code || '').toUpperCase();
  const status = Number(error.statusCode || error.status || 0);
  if (message.includes('blocked_by_safety_gate') || message.includes('safety gate')) {
    return FAILURE_CLASSES.SAFETY_GATE_BLOCK;
  }
  if (status === 429 || code === 'RATE_LIMITED' || message.includes('rate limit')) {
    return FAILURE_CLASSES.SOURCE_RATE_LIMITED;
  }
  if (message.includes('spawn') || message.includes('executable') || message.includes('child process')) {
    return FAILURE_CLASSES.RUNNER_FAILURE;
  }
  if (['ECONNRESET', 'ENETUNREACH', 'ETIMEDOUT', 'EAI_AGAIN'].includes(code)
    || message.includes('network')) return FAILURE_CLASSES.SOURCE_NETWORK_FAILURE;
  if (message.includes('schema') || message.includes('invalid source')) {
    return FAILURE_CLASSES.SOURCE_SCHEMA_FAILURE;
  }
  if (message.includes('lock')) return FAILURE_CLASSES.LOCK_FAILURE;
  if (message.includes('git') || message.includes('dirty repository')) return FAILURE_CLASSES.GIT_FAILURE;
  if (message.includes('pull request') || message.includes('github') || message.includes('pr ')) {
    return FAILURE_CLASSES.PR_FAILURE;
  }
  if (message.includes('state') || message.includes('checkpoint')) return FAILURE_CLASSES.STATE_FAILURE;
  if (message.includes('timeout')) {
    return FAILURE_CLASSES.RUNNER_FAILURE;
  }
  return FAILURE_CLASSES.APPLICATION_FAILURE;
}

function ensureWritableDirectory(directory) {
  fs.mkdirSync(directory, { recursive: true });
  const probe = path.join(directory, `.stage4o-write-probe-${process.pid}`);
  fs.writeFileSync(probe, 'ok\n', 'utf8');
  fs.rmSync(probe, { force: true });
  return true;
}

function commandCheck(command, args, options = {}) {
  const result = (options.spawnSyncImpl || spawnSync)(command, args, {
    cwd: options.cwd,
    encoding: 'utf8',
    shell: false,
    timeout: options.timeout || 30000,
  });
  return {
    ok: result.status === 0,
    status: result.status,
    output: String(result.stdout || '').trim().split(/\r?\n/)[0] || null,
    error: result.error?.message || String(result.stderr || '').trim().slice(0, 300) || null,
  };
}

function resolvePrApiMode({ env = process.env, spawnSyncImpl } = {}) {
  const gh = commandCheck('gh', ['--version'], { spawnSyncImpl, timeout: 10000 });
  if (gh.ok) return 'GH';
  if (env.GH_TOKEN || env.GITHUB_TOKEN) return 'TOKEN_API';
  return 'COMPARE_URL_ONLY';
}

function validateStateSchemas(dataDir) {
  const stateDir = path.join(dataDir, 'state');
  const checks = [];
  const schemas = {
    'source_state.json': 'stage4n-source-state-v1',
    'open_pr.json': 'stage4n-open-pr-v1',
  };
  for (const [name, expected] of Object.entries(schemas)) {
    const filePath = path.join(stateDir, name);
    if (!fs.existsSync(filePath)) {
      checks.push({ name, ok: true, status: 'ABSENT_ALLOWED' });
      continue;
    }
    const parsed = readJson(filePath);
    checks.push({ name, ok: parsed?.schemaVersion === expected,
      status: parsed?.schemaVersion || 'MISSING_SCHEMA' });
  }
  return checks;
}

function runOperationalPreflight({
  repository, dataDir, artifactDir, tempDir, operationalConfig,
  canonicalPath, stage4nConfigPath, policyConfigPath, boundaryPolicyPath,
  boundaryGeoJsonPath, spawnSyncImpl,
}) {
  const checks = [];
  const add = (name, ok, detail = null) => checks.push({ name, ok: Boolean(ok), detail });
  add('node_executable', fs.existsSync(process.execPath), process.execPath);
  add('repository_package', fs.existsSync(path.join(repository, 'package.json')), repository);
  add('dependencies', fs.existsSync(path.join(repository, 'node_modules')), 'node_modules');
  const git = commandCheck('git', ['--version'], { cwd: repository, spawnSyncImpl });
  add('git_executable', git.ok, git.output || git.error);
  const childNode = commandCheck(process.execPath, ['-e', "process.stdout.write('STAGE4O_CHILD_OK')"], {
    cwd: repository, spawnSyncImpl, timeout: 10000,
  });
  add('child_node_invocation', childNode.ok && childNode.output === 'STAGE4O_CHILD_OK',
    childNode.output || childNode.error);
  const gitStatus = commandCheck('git', ['status', '--porcelain'], { cwd: repository, spawnSyncImpl });
  add('git_clean', gitStatus.ok && !gitStatus.output, gitStatus.output || gitStatus.error || 'clean');
  add('operations_config', operationalConfig?.schemaVersion === 'stage4o-operations-v1');
  for (const [name, filePath] of [
    ['stage4n_config', stage4nConfigPath],
    ['policy_config', policyConfigPath],
    ['boundary_policy', boundaryPolicyPath],
    ['boundary_geometry', boundaryGeoJsonPath],
  ]) {
    try {
      const value = readJson(filePath);
      add(name, Boolean(value), filePath);
    } catch (error) {
      add(name, false, error.message);
    }
  }
  for (const [name, directory] of [
    ['data_path_writable', dataDir],
    ['artifact_path_writable', artifactDir],
    ['temp_path_writable', tempDir],
  ]) {
    try { add(name, ensureWritableDirectory(directory), directory); }
    catch (error) { add(name, false, error.message); }
  }
  for (const state of validateStateSchemas(dataDir)) add(`state_schema:${state.name}`, state.ok, state.status);
  let canonical = null;
  try {
    canonical = inspectCanonicalDataset(canonicalPath);
    add('canonical_rows', canonical.rows === operationalConfig.expectedCanonicalRows, canonical.rows);
    add('canonical_sha', canonical.shaMatchesExpected, canonical.sha256);
  } catch (error) {
    add('canonical_readable', false, error.message);
  }
  const selfTestLock = path.join(dataDir, 'locks', 'stage4o-self-test.lock.json');
  try {
    const lock = acquireExecutionLock({
      lockPath: selfTestLock,
      runId: `self-test-${process.pid}`,
      now: new Date().toISOString(),
      config: { lock: { staleAfterMinutes: 5 } },
    });
    add('lock_create_release', lock.acquired === true);
    if (lock.acquired) releaseExecutionLock(selfTestLock, `self-test-${process.pid}`);
  } catch (error) {
    add('lock_create_release', false, error.message);
  }
  const prApiMode = resolvePrApiMode({ spawnSyncImpl });
  add('pr_api_mode', true, prApiMode);
  return {
    schemaVersion: 'stage4o-self-test-v1',
    ok: checks.every((item) => item.ok),
    checkedAt: new Date().toISOString(),
    checks,
    prApiMode,
    canonical,
  };
}

function processIsRunning(pid) {
  if (!Number.isInteger(Number(pid)) || Number(pid) <= 0) return false;
  try { process.kill(Number(pid), 0); return true; }
  catch (error) { return error.code === 'EPERM'; }
}

function inspectStaleLock({ lock, now, staleAfterMinutes, host = os.hostname(), pidCheck = processIsRunning }) {
  if (!lock?.startedAt) return { action: 'KEEP_LOCK', reason: 'LOCK_START_UNKNOWN' };
  const ageMs = new Date(now).getTime() - new Date(lock.startedAt).getTime();
  if (ageMs <= staleAfterMinutes * 60000) return { action: 'KEEP_LOCK', reason: 'LOCK_NOT_STALE' };
  if (!lock.host || lock.host.toLowerCase() !== host.toLowerCase()) {
    return { action: 'KEEP_LOCK', reason: 'FOREIGN_HOST_NOT_PROVEN_DEAD' };
  }
  if (pidCheck(lock.processId)) return { action: 'KEEP_LOCK', reason: 'PROCESS_STILL_RUNNING' };
  return { action: 'CLEAR_STALE_LOCK', reason: 'SAME_HOST_PROCESS_CONFIRMED_ABSENT' };
}

function summarizeRunStatus(result = {}) {
  if (result.status === 'NO_SOURCE_CHANGE' || result.status === 'SKIPPED_ALREADY_RUNNING') return 'NO_WORK';
  if (result.status === 'EXCEPTIONS_ONLY') return 'EXCEPTIONS_ONLY';
  if (result.status === 'BLOCKED_BY_SAFETY_GATE' || result.status === 'OPEN_PR_REQUIRES_RELEASE') return 'BLOCKED';
  if (result.status === 'FAILED') return 'FAILED';
  if (result.prCreated > 0) return 'SAFE_PR_CREATED';
  if (SUCCESSFUL_RUN_STATUSES.has(result.status)) return 'COMPLETED_NO_PR';
  return 'FAILED';
}

function updateSourceHealth(previous = {}, discovery = {}, config = {}, now, finalStatus) {
  const next = { ...previous };
  for (const check of discovery.checks || []) {
    const old = previous[check.source] || {};
    const success = finalStatus !== 'FAILED';
    next[check.source] = {
      source: check.source,
      capability: check.capability,
      configuredCadenceMinutes: config.sourceCadenceMinutes?.[check.source] || null,
      lastCheck: check.due ? now : old.lastCheck || null,
      lastSuccessfulCheck: check.due && success ? now : old.lastSuccessfulCheck || null,
      lastChangeDetected: check.changed ? now : old.lastChangeDetected || null,
      currentSnapshot: check.version || old.currentSnapshot || null,
      consecutiveFailures: success ? 0 : (old.consecutiveFailures || 0) + 1,
      lastFailureClass: success ? null : old.lastFailureClass || FAILURE_CLASSES.APPLICATION_FAILURE,
      nextEligibleCheck: check.due && success && config.sourceCadenceMinutes?.[check.source]
        ? new Date(new Date(now).getTime() + config.sourceCadenceMinutes[check.source] * 60000).toISOString()
        : old.nextEligibleCheck || null,
    };
  }
  return next;
}

function buildHealthStatus({ previous = {}, result = {}, config, manifestPath, startedAt, endedAt }) {
  const summary = summarizeRunStatus(result);
  const success = summary !== 'FAILED' && summary !== 'BLOCKED';
  const failureClass = result.failureClass || (summary === 'BLOCKED'
    ? FAILURE_CLASSES.SAFETY_GATE_BLOCK : null);
  return {
    schemaVersion: 'stage4o-health-v1',
    updatedAt: endedAt,
    schedulerTaskName: config.taskName,
    schedulerEnabled: previous.schedulerEnabled ?? false,
    schedulerCommitSha: previous.schedulerCommitSha || null,
    nextScheduledRun: previous.nextScheduledRun || null,
    lastRunId: result.runId || null,
    lastRunStart: startedAt,
    lastRunEnd: endedAt,
    lastRunStatus: summary,
    lastSuccessfulRun: success ? endedAt : previous.lastSuccessfulRun || null,
    lastFailure: failureClass ? endedAt : previous.lastFailure || null,
    lastFailureClass: failureClass || previous.lastFailureClass || null,
    lastFailureMessage: result.error?.message || result.failureMessage || previous.lastFailureMessage || null,
    currentLock: result.lock || null,
    currentCanonicalSha: result.canonical?.sha256 || previous.currentCanonicalSha || null,
    policyVersion: 'stage4l-review-policy-v1',
    boundaryVersion: 'stage4j-boundary-policy-v1',
    sourceHealth: updateSourceHealth(previous.sourceHealth, result.discovery, config, endedAt, summary),
    lastSourceChecks: result.discovery?.checks || [],
    lastSourceSuccessfulSync: success ? endedAt : previous.lastSourceSuccessfulSync || null,
    lastDeltaCounts: result.deltaCounts || {},
    lastPolicyCounts: result.policyOutcomeCounts || {},
    lastSafeMutationCount: Number(result.canonicalMutations || 0) + Number(result.sidecarMutations || 0),
    lastHumanReviewDeltaCount: result.exceptions || 0,
    exceptionArtifactPath: result.exceptionArtifactPath || null,
    lastBranch: result.safeResult?.branch || previous.lastBranch || null,
    lastPrUrl: result.safeResult?.compareUrl || result.openPr || previous.lastPrUrl || null,
    lastManifestPath: manifestPath,
    resumeAvailable: result.resumable === true,
    prApiMode: previous.prApiMode || 'COMPARE_URL_ONLY',
  };
}

function writeOperationalResult({ statusPath, artifactDir, previous, result, config, startedAt, endedAt }) {
  const manifestPath = path.join(artifactDir, 'operational_run_manifest.json');
  const manifest = {
    schemaVersion: 'stage4o-run-manifest-v1',
    runId: result.runId || null,
    triggerType: result.triggerType || 'MANUAL',
    taskName: config.taskName,
    startedAt,
    endedAt,
    host: os.hostname(),
    processId: process.pid,
    sourceChecks: result.discovery?.checks || [],
    lockInspection: result.lockInspection || null,
    deltaCounts: result.deltaCounts || {},
    policyOutcomeCounts: result.policyOutcomeCounts || {},
    safeMutations: Number(result.canonicalMutations || 0) + Number(result.sidecarMutations || 0),
    humanExceptions: result.exceptions || 0,
    safetyStatus: result.status,
    branch: result.safeResult?.branch || null,
    prUrl: result.safeResult?.compareUrl || result.openPr || null,
    pr: result.safeResult || result.openPr ? {
      url: result.safeResult?.compareUrl || result.openPr || null,
      branch: result.safeResult?.branch || null,
      planHash: result.planHash || null,
      canonicalBeforeSha: result.canonical?.sha256 || null,
      candidateAfterSha: result.safeResult?.canonicalAfterSha || null,
      safeCaseCount: result.safeResult?.safeCaseCount || 0,
      exceptionCount: result.exceptions || 0,
    } : null,
    rollbackStatus: result.safeResult?.rollbackStatus || null,
    finalStatus: summarizeRunStatus(result),
    failureClass: result.failureClass || null,
    resumeCheckpoint: result.resumable ? 'AVAILABLE' : null,
  };
  writeJsonAtomic(manifestPath, manifest);
  const status = buildHealthStatus({ previous, result, config, manifestPath, startedAt, endedAt });
  writeJsonAtomic(statusPath, status);
  return { manifest, status, manifestPath };
}

function selectOperationalLogDeletions({ entries, now, retention }) {
  const successCutoff = new Date(now).getTime() - retention.successfulRunDays * 86400000;
  const failureCutoff = new Date(now).getTime() - retention.failedRunDays * 86400000;
  const successes = entries.filter((item) => item.status !== 'FAILED')
    .sort((a, b) => new Date(b.completedAt) - new Date(a.completedAt));
  const protectedSuccess = new Set(successes.slice(0, retention.minimumSuccessfulRuns).map((item) => item.path));
  return entries.filter((item) => {
    if (!item.disposable || item.provenanceEvidence || item.activeState || item.openPrRollback) return false;
    const completed = new Date(item.completedAt || 0).getTime();
    if (item.status === 'FAILED') return completed < failureCutoff;
    return completed < successCutoff && !protectedSuccess.has(item.path);
  }).map((item) => item.path);
}

function cleanupOperationalRetention({ artifactRoot, logRoot, now, retention }) {
  const entries = [];
  if (fs.existsSync(artifactRoot)) {
    for (const item of fs.readdirSync(artifactRoot, { withFileTypes: true })) {
      if (!item.isDirectory() || item.name === 'self-test') continue;
      const directory = path.join(artifactRoot, item.name);
      const manifest = readJson(path.join(directory, 'operational_run_manifest.json'), null);
      if (!manifest) continue;
      entries.push({
        path: directory,
        status: manifest.finalStatus === 'FAILED' ? 'FAILED' : 'COMPLETED',
        completedAt: manifest.endedAt,
        disposable: true,
        openPrRollback: Boolean(manifest.prUrl),
      });
    }
  }
  const deletions = selectOperationalLogDeletions({ entries, now, retention });
  for (const target of deletions) {
    const resolved = path.resolve(target);
    if (resolved.startsWith(`${path.resolve(artifactRoot)}${path.sep}`)) {
      fs.rmSync(resolved, { recursive: true, force: true });
    }
  }
  let removedLogs = 0;
  if (fs.existsSync(logRoot)) {
    const logCutoff = new Date(now).getTime() - retention.failedRunDays * 86400000;
    for (const item of fs.readdirSync(logRoot, { withFileTypes: true })) {
      if (!item.isFile()) continue;
      const filePath = path.join(logRoot, item.name);
      if (fs.statSync(filePath).mtimeMs < logCutoff) {
        fs.rmSync(filePath, { force: true });
        removedLogs += 1;
      }
    }
  }
  return { removedRunDirectories: deletions.length, removedLogs };
}

function interpretTaskSchedulerResult({ lastTaskResult, taskState, healthStatus }) {
  if (lastTaskResult === null || lastTaskResult === undefined) {
    return { state: 'NEVER_RUN', successful: false };
  }
  const resultCode = Number(lastTaskResult);
  if (resultCode === 267009 || String(taskState || '').toLowerCase() === 'running') {
    return { state: 'RUNNING', successful: false, resultCode };
  }
  if (resultCode !== 0) {
    return { state: 'FAILED', successful: false, resultCode };
  }
  const runStatus = healthStatus?.lastRunStatus || 'UNKNOWN';
  return {
    state: runStatus === 'FAILED' || runStatus === 'BLOCKED' ? 'FAILED' : 'COMPLETED',
    successful: runStatus !== 'FAILED' && runStatus !== 'BLOCKED',
    resultCode,
    runStatus,
  };
}

module.exports = {
  FAILURE_CLASSES,
  SUCCESSFUL_RUN_STATUSES,
  buildHealthStatus,
  classifyOperationalError,
  cleanupOperationalRetention,
  commandCheck,
  ensureWritableDirectory,
  inspectStaleLock,
  interpretTaskSchedulerResult,
  processIsRunning,
  resolvePrApiMode,
  runOperationalPreflight,
  selectOperationalLogDeletions,
  summarizeRunStatus,
  updateSourceHealth,
  validateStateSchemas,
  writeOperationalResult,
};
