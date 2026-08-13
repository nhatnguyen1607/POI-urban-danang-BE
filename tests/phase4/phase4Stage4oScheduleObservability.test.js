const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const operationsConfig = require('../../config/phase4_stage4o_operations.json');
const stage4nConfig = require('../../config/phase4_stage4n_scheduled_sync.json');
const { inspectCanonicalDataset } = require('../../src/modules/cityPackPreparation/canonicalDataset');
const { discoverSourceChanges, withRetry } = require('../../src/modules/cityPackPreparation/scheduledSyncOrchestrator');
const {
  FAILURE_CLASSES,
  buildHealthStatus,
  classifyOperationalError,
  cleanupOperationalRetention,
  inspectStaleLock,
  interpretTaskSchedulerResult,
  runOperationalPreflight,
  selectOperationalLogDeletions,
  summarizeRunStatus,
  writeOperationalResult,
} = require('../../src/modules/cityPackPreparation/scheduledSyncOperations');
const {
  parseChildJson,
  readHealth,
  runOperational,
  runStage4nChild,
} = require('../../scripts/citypack_scheduled_runner');

const ROOT = path.resolve(__dirname, '..', '..');
const NOW = '2026-08-13T06:00:00.000Z';
const EXPECTED_SHA = '39647b29308813a7ec19e4695fd2b95ffa27743db8e8cb46800c45d3a3fe6ded';

function temp(name) {
  return fs.mkdtempSync(path.join(os.tmpdir(), `stage4o-${name}-`));
}

function successfulSpawn(command, args) {
  if (command === 'gh') return { status: 1, stdout: '', stderr: 'not installed' };
  if (command === 'git' && args[0] === 'status') return { status: 0, stdout: '', stderr: '' };
  if (command === process.execPath && args[0] === '-e') {
    return { status: 0, stdout: 'STAGE4O_CHILD_OK', stderr: '' };
  }
  return { status: 0, stdout: command === 'git' ? 'git version test\n' : 'v-test\n', stderr: '' };
}

test('Stage 4O operations config separates six-hour scheduler ticks from source cadence', () => {
  assert.equal(operationsConfig.schedulerTickHours, 6);
  assert.deepEqual(operationsConfig.sourceCadenceMinutes, {
    overture: 10080, osm: 1440, wikidata: 4320, wikimedia_commons: 10080,
  });
  assert.equal(operationsConfig.autoMerge, false);
  assert.equal(operationsConfig.autoCreateCanonicalPoi, false);
  assert.equal(operationsConfig.ciScheduleEnabled, false);
});

test('failure classification preserves operational cause instead of generic FAILED', () => {
  assert.equal(classifyOperationalError({ statusCode: 429 }), FAILURE_CLASSES.SOURCE_RATE_LIMITED);
  assert.equal(classifyOperationalError({ code: 'ETIMEDOUT', message: 'network timeout' }),
    FAILURE_CLASSES.SOURCE_NETWORK_FAILURE);
  assert.equal(classifyOperationalError(new Error('source schema mismatch')),
    FAILURE_CLASSES.SOURCE_SCHEMA_FAILURE);
  assert.equal(classifyOperationalError(new Error('BLOCKED_BY_SAFETY_GATE')),
    FAILURE_CLASSES.SAFETY_GATE_BLOCK);
  assert.equal(classifyOperationalError(new Error('dirty repository prevents git branch')),
    FAILURE_CLASSES.GIT_FAILURE);
  assert.equal(classifyOperationalError(Object.assign(new Error('spawn timeout'), { code: 'ETIMEDOUT' })),
    FAILURE_CLASSES.RUNNER_FAILURE);
});

test('runner passes Windows paths with spaces as argument-array elements', () => {
  const root = temp('path with spaces');
  const artifactDir = path.join(root, 'artifact output');
  fs.mkdirSync(artifactDir, { recursive: true });
  let invocation = null;
  const result = runStage4nChild({
    args: { source: 'all' },
    paths: {
      repository: path.join(root, 'repository with spaces'),
      dataDir: path.join(root, 'data state'),
      tempRoot: path.join(root, 'temporary files'),
      snapshotManifest: path.join(root, 'snapshot manifest.json'),
    },
    config: operationsConfig,
    artifactDir,
    spawnSyncImpl: (command, args, options) => {
      invocation = { command, args, options };
      return { status: 0, stdout: '{"status":"NO_SOURCE_CHANGE","runId":"path-test"}', stderr: '' };
    },
  });
  assert.equal(invocation.command, process.execPath);
  assert.ok(invocation.args.includes(path.join(root, 'data state')));
  assert.ok(invocation.args.includes(path.join(root, 'snapshot manifest.json')));
  assert.equal(invocation.options.shell, false);
  assert.equal(result.result.status, 'NO_SOURCE_CHANGE');
});

test('runner timeout is bounded and remains a runner failure', () => {
  const root = temp('timeout');
  fs.mkdirSync(path.join(root, 'artifacts'));
  assert.throws(() => runStage4nChild({
    args: {},
    paths: { repository: root, dataDir: root, tempRoot: root, snapshotManifest: path.join(root, 'm.json') },
    config: operationsConfig,
    artifactDir: path.join(root, 'artifacts'),
    spawnSyncImpl: () => ({ status: null, stdout: '', stderr: '',
      error: Object.assign(new Error('spawn timeout'), { code: 'ETIMEDOUT' }) }),
  }), /spawn timeout/);
  assert.equal(classifyOperationalError(Object.assign(new Error('spawn timeout'), { code: 'ETIMEDOUT' })),
    FAILURE_CLASSES.RUNNER_FAILURE);
});

test('child result parser tolerates concise wrapper output before JSON', () => {
  assert.deepEqual(parseChildJson('runner ready\n{"status":"NO_SAFE_CHANGES","runId":"one"}\n'),
    { status: 'NO_SAFE_CHANGES', runId: 'one' });
  assert.throws(() => parseChildJson('not json'), /valid JSON result/);
});

test('preflight validates dependencies, D paths, schemas, boundary, canonical, child and lock', () => {
  const root = temp('preflight');
  const repository = path.join(root, 'repo');
  fs.mkdirSync(path.join(repository, 'node_modules'), { recursive: true });
  fs.writeFileSync(path.join(repository, 'package.json'), '{}\n');
  const result = runOperationalPreflight({
    repository,
    dataDir: path.join(root, 'data'),
    artifactDir: path.join(root, 'artifacts'),
    tempDir: path.join(root, 'temp'),
    operationalConfig: operationsConfig,
    canonicalPath: path.join(ROOT, 'data', 'canonical', 'urbanagent_poi_master_v1.csv'),
    stage4nConfigPath: path.join(ROOT, 'config', 'phase4_stage4n_scheduled_sync.json'),
    policyConfigPath: path.join(ROOT, 'config', 'phase4_stage4l_policy.json'),
    boundaryPolicyPath: path.join(ROOT, 'data', 'spikes', 'phase4', 'stage4j', 'boundary_policy.json'),
    boundaryGeoJsonPath: path.join(ROOT, 'data', 'spikes', 'phase4', 'stage4i', 'boundary',
      'osm-relation-1891418-v72.geojson'),
    spawnSyncImpl: successfulSpawn,
  });
  assert.equal(result.ok, true);
  assert.equal(result.prApiMode, 'COMPARE_URL_ONLY');
  assert.equal(result.canonical.rows, 4166);
  assert.equal(result.canonical.sha256, EXPECTED_SHA);
  assert.equal(result.checks.find((item) => item.name === 'lock_create_release').ok, true);
});

test('preflight blocks a dirty operational repository without resetting it', () => {
  const root = temp('dirty');
  fs.mkdirSync(path.join(root, 'node_modules'));
  fs.writeFileSync(path.join(root, 'package.json'), '{}');
  const result = runOperationalPreflight({
    repository: root,
    dataDir: path.join(root, 'data'), artifactDir: path.join(root, 'artifacts'), tempDir: path.join(root, 'temp'),
    operationalConfig: operationsConfig,
    canonicalPath: path.join(ROOT, 'data', 'canonical', 'urbanagent_poi_master_v1.csv'),
    stage4nConfigPath: path.join(ROOT, 'config', 'phase4_stage4n_scheduled_sync.json'),
    policyConfigPath: path.join(ROOT, 'config', 'phase4_stage4l_policy.json'),
    boundaryPolicyPath: path.join(ROOT, 'data', 'spikes', 'phase4', 'stage4j', 'boundary_policy.json'),
    boundaryGeoJsonPath: path.join(ROOT, 'data', 'spikes', 'phase4', 'stage4i', 'boundary',
      'osm-relation-1891418-v72.geojson'),
    spawnSyncImpl: (command, args) => command === 'git' && args[0] === 'status'
      ? { status: 0, stdout: ' M unknown-user-file\n', stderr: '' }
      : successfulSpawn(command, args),
  });
  assert.equal(result.ok, false);
  assert.equal(result.checks.find((item) => item.name === 'git_clean').ok, false);
  assert.equal(fs.existsSync(path.join(root, 'unknown-user-file')), false);
});

test('stale lock clears only after same-host process absence is proven', () => {
  const old = { startedAt: '2026-01-01T00:00:00Z', host: os.hostname(), processId: 999999 };
  assert.equal(inspectStaleLock({ lock: old, now: NOW, staleAfterMinutes: 5, pidCheck: () => false }).action,
    'CLEAR_STALE_LOCK');
  assert.equal(inspectStaleLock({ lock: old, now: NOW, staleAfterMinutes: 5, pidCheck: () => true }).reason,
    'PROCESS_STILL_RUNNING');
  assert.equal(inspectStaleLock({ lock: { ...old, host: 'other-host' }, now: NOW,
    staleAfterMinutes: 5, pidCheck: () => false }).reason, 'FOREIGN_HOST_NOT_PROVEN_DEAD');
});

test('health status preserves last-known-good state after a failed run', () => {
  const previous = { lastSuccessfulRun: '2026-08-12T00:00:00Z', currentCanonicalSha: EXPECTED_SHA };
  const status = buildHealthStatus({
    previous,
    result: { status: 'FAILED', failureClass: FAILURE_CLASSES.SOURCE_NETWORK_FAILURE,
      error: { message: 'bounded source unavailable' } },
    config: operationsConfig,
    manifestPath: 'D:/manifest.json',
    startedAt: NOW,
    endedAt: '2026-08-13T06:01:00Z',
  });
  assert.equal(status.lastSuccessfulRun, previous.lastSuccessfulRun);
  assert.equal(status.currentCanonicalSha, EXPECTED_SHA);
  assert.equal(status.lastFailureClass, FAILURE_CLASSES.SOURCE_NETWORK_FAILURE);
});

test('operational manifest and status are concise atomic artifacts', () => {
  const root = temp('manifest');
  const statusPath = path.join(root, 'status', 'status.json');
  const artifactDir = path.join(root, 'run');
  const written = writeOperationalResult({
    statusPath, artifactDir, previous: {}, config: operationsConfig,
    result: { runId: 'run-1', status: 'NO_SOURCE_CHANGE', triggerType: 'SCHEDULED',
      discovery: { checks: [] }, canonical: { sha256: EXPECTED_SHA },
      lockInspection: { action: 'NO_LOCK' } },
    startedAt: NOW, endedAt: '2026-08-13T06:00:01Z',
  });
  assert.equal(readHealth(statusPath).lastRunStatus, 'NO_WORK');
  assert.equal(JSON.parse(fs.readFileSync(written.manifestPath)).finalStatus, 'NO_WORK');
  assert.equal(JSON.parse(fs.readFileSync(written.manifestPath)).lockInspection.action, 'NO_LOCK');
  assert.ok(fs.statSync(written.manifestPath).size < 10000);
});

test('source health and cadence skip non-due sources even when scheduler ticks', () => {
  const sourceState = { sources: Object.fromEntries(Object.keys(stage4nConfig.sources).map((source) => [source, {
    version: 'v1', lastCheckedAt: '2026-08-13T05:00:00Z',
  }])) };
  const discovery = discoverSourceChanges({
    sources: Object.keys(stage4nConfig.sources), sourceState, config: stage4nConfig, now: NOW,
    metadataBySource: Object.fromEntries(Object.keys(stage4nConfig.sources).map((source) => [source, { version: 'v2' }])),
  });
  assert.deepEqual(discovery.dueSources, []);
  assert.deepEqual(discovery.changedSources, []);
});

test('retry/backoff integration remains bounded for scheduler sources', async () => {
  let attempts = 0;
  const waits = [];
  assert.equal(await withRetry(async () => {
    attempts += 1;
    if (attempts < 3) throw Object.assign(new Error('temporary'), { statusCode: 503 });
    return 'ok';
  }, { maxRetries: 3, baseBackoffMs: 2, sleep: async (value) => waits.push(value) }), 'ok');
  assert.deepEqual(waits, [2, 4]);
});

test('log retention protects recent successes and retains failures longer', () => {
  const entries = [
    { path: 'old-success', status: 'COMPLETED', completedAt: '2025-01-01T00:00:00Z', disposable: true },
    { path: 'old-failure', status: 'FAILED', completedAt: '2025-01-01T00:00:00Z', disposable: true },
    { path: 'evidence', status: 'COMPLETED', completedAt: '2025-01-01T00:00:00Z', disposable: true,
      provenanceEvidence: true },
    ...Array.from({ length: 11 }, (_, index) => ({ path: `recent-${index}`, status: 'COMPLETED',
      completedAt: new Date(Date.UTC(2026, 0, index + 1)).toISOString(), disposable: true })),
  ];
  const deletions = selectOperationalLogDeletions({ entries, now: NOW, retention: operationsConfig.retention });
  assert.ok(deletions.includes('old-success'));
  assert.ok(deletions.includes('old-failure'));
  assert.ok(!deletions.includes('evidence'));
  assert.equal(deletions.filter((item) => item.startsWith('recent-')).length, 1);
});

test('retention cleanup stays inside configured run and log roots', () => {
  const root = temp('retention-cleanup');
  const artifactRoot = path.join(root, 'runs');
  const logRoot = path.join(root, 'logs');
  const oldRun = path.join(artifactRoot, 'old-run');
  const protectedFile = path.join(root, 'do-not-delete.txt');
  fs.mkdirSync(oldRun, { recursive: true });
  fs.mkdirSync(logRoot, { recursive: true });
  fs.writeFileSync(path.join(oldRun, 'operational_run_manifest.json'), JSON.stringify({
    finalStatus: 'FAILED', endedAt: '2025-01-01T00:00:00Z', prUrl: null,
  }));
  fs.writeFileSync(protectedFile, 'protected');
  const oldLog = path.join(logRoot, 'old.log');
  fs.writeFileSync(oldLog, 'old');
  fs.utimesSync(oldLog, new Date('2025-01-01T00:00:00Z'), new Date('2025-01-01T00:00:00Z'));
  const result = cleanupOperationalRetention({ artifactRoot, logRoot, now: NOW,
    retention: operationsConfig.retention });
  assert.deepEqual(result, { removedRunDirectories: 1, removedLogs: 1 });
  assert.equal(fs.existsSync(protectedFile), true);
});

test('status taxonomy maps no-work, PR, exception, block and failure states', () => {
  assert.equal(summarizeRunStatus({ status: 'NO_SOURCE_CHANGE' }), 'NO_WORK');
  assert.equal(summarizeRunStatus({ status: 'COMPLETED', prCreated: 1 }), 'SAFE_PR_CREATED');
  assert.equal(summarizeRunStatus({ status: 'EXCEPTIONS_ONLY' }), 'EXCEPTIONS_ONLY');
  assert.equal(summarizeRunStatus({ status: 'BLOCKED_BY_SAFETY_GATE' }), 'BLOCKED');
  assert.equal(summarizeRunStatus({ status: 'FAILED' }), 'FAILED');
});

test('no-work runner updates observability without safe mutation or PR', async () => {
  const root = temp('runner');
  const config = JSON.parse(JSON.stringify(operationsConfig));
  config.paths = {
    stableRepository: ROOT,
    dataRoot: path.join(root, 'data'), artifactRoot: path.join(root, 'runs'),
    statusPath: path.join(root, 'status', 'status.json'), logRoot: path.join(root, 'logs'),
    tempRoot: path.join(root, 'temp'),
  };
  const configPath = path.join(root, 'operations.json');
  fs.writeFileSync(configPath, JSON.stringify(config));
  const fakeSpawn = (command, args) => {
    if (command === 'git') return successfulSpawn(command, args);
    if (command === 'gh') return { status: 1, stdout: '', stderr: 'not installed' };
    if (command === process.execPath && args[0] === '-e') {
      return { status: 0, stdout: 'STAGE4O_CHILD_OK', stderr: '' };
    }
    return { status: 0, stdout: JSON.stringify({ status: 'NO_SOURCE_CHANGE', runId: 'no-work',
      discovery: { checks: [] }, canonical: { rows: 4166, sha256: EXPECTED_SHA } }), stderr: '' };
  };
  const result = await runOperational([
    '--operations-config', configPath, '--repository', ROOT,
    '--data-dir', config.paths.dataRoot, '--artifact-root', config.paths.artifactRoot,
    '--temp-dir', config.paths.tempRoot, '--status-path', config.paths.statusPath,
    '--snapshot-manifest', path.join(ROOT, 'data', 'spikes', 'phase4', 'stage4e', 'snapshots',
      'snapshot_manifest.json'), '--source', 'all', '--trigger-type', 'CANARY',
  ], { spawnSyncImpl: fakeSpawn });
  assert.equal(result.finalStatus, 'NO_WORK');
  assert.equal(result.safeMutations, 0);
  assert.equal(result.prUrl, null);
  assert.equal(readHealth(config.paths.statusPath).lastRunStatus, 'NO_WORK');
});

test('Task Scheduler helpers are idempotent and never delete project state', () => {
  const install = fs.readFileSync(path.join(ROOT, 'scripts', 'phase4', 'installScheduledPoiSyncTask.ps1'), 'utf8');
  const disable = fs.readFileSync(path.join(ROOT, 'scripts', 'phase4', 'disableScheduledPoiSyncTask.ps1'), 'utf8');
  const uninstall = fs.readFileSync(path.join(ROOT, 'scripts', 'phase4', 'uninstallScheduledPoiSyncTask.ps1'), 'utf8');
  const status = fs.readFileSync(path.join(ROOT, 'scripts', 'phase4', 'getScheduledPoiSyncStatus.ps1'), 'utf8');
  assert.match(install, /Register-ScheduledTask[\s\S]*-Force/);
  assert.match(install, /MultipleInstances IgnoreNew/);
  assert.match(disable, /Disable-ScheduledTask/);
  assert.match(uninstall, /Unregister-ScheduledTask/);
  assert.doesNotMatch(uninstall, /Remove-Item|rm\s|rmdir/i);
  assert.match(status, /LastTaskResult/);
});

test('Task Scheduler canary result parsing distinguishes success, running and failure', () => {
  assert.deepEqual(interpretTaskSchedulerResult({
    lastTaskResult: 0,
    taskState: 'Ready',
    healthStatus: { lastRunStatus: 'NO_WORK' },
  }), {
    state: 'COMPLETED', successful: true, resultCode: 0, runStatus: 'NO_WORK',
  });
  assert.equal(interpretTaskSchedulerResult({
    lastTaskResult: 267009, taskState: 'Running', healthStatus: {},
  }).state, 'RUNNING');
  assert.equal(interpretTaskSchedulerResult({
    lastTaskResult: 1, taskState: 'Ready', healthStatus: { lastRunStatus: 'FAILED' },
  }).state, 'FAILED');
});

test('canonical integrity and runtime-sidecar boundary remain unchanged', () => {
  const canonical = inspectCanonicalDataset(path.join(ROOT, 'data', 'canonical', 'urbanagent_poi_master_v1.csv'));
  assert.equal(canonical.rows, 4166);
  assert.equal(canonical.sha256, EXPECTED_SHA);
  assert.equal(operationsConfig.runtimeSidecarEnabled, false);
  assert.equal(operationsConfig.networkMode, 'PREPARED_BOUNDED_SNAPSHOTS');
});
