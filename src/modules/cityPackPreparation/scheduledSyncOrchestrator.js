const crypto = require('node:crypto');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const {
  DEFAULT_SOURCE_CAPABILITIES,
  STATUSES,
  runIncrementalSync,
  stableHash,
} = require('./incrementalSync');

const RUN_STATES = Object.freeze({
  STARTED: 'STARTED',
  SOURCE_CHECK: 'SOURCE_CHECK',
  NO_SOURCE_CHANGE: 'NO_SOURCE_CHANGE',
  NO_RELEVANT_POI_DELTA: 'NO_RELEVANT_POI_DELTA',
  PROCESSING_DELTA: 'PROCESSING_DELTA',
  POLICY: 'POLICY',
  NO_SAFE_CHANGES: 'NO_SAFE_CHANGES',
  PREPARING_PR: 'PREPARING_PR',
  TESTING: 'TESTING',
  PUSHED: 'PUSHED',
  PR_CREATED: 'PR_CREATED',
  EXCEPTIONS_ONLY: 'EXCEPTIONS_ONLY',
  SKIPPED_ALREADY_RUNNING: 'SKIPPED_ALREADY_RUNNING',
  OPEN_PR_REQUIRES_RELEASE: 'OPEN_PR_REQUIRES_RELEASE',
  BLOCKED_BY_SAFETY_GATE: 'BLOCKED_BY_SAFETY_GATE',
  FAILED: 'FAILED',
  COMPLETED: 'COMPLETED',
});

const TRANSIENT_STATUS_CODES = new Set([429, 500, 502, 503, 504]);

function stableJson(value) {
  if (Array.isArray(value)) return value.map(stableJson);
  if (!value || typeof value !== 'object') return value;
  return Object.fromEntries(Object.keys(value).sort().map((key) => [key, stableJson(value[key])]));
}

function writeJsonAtomic(filePath, value) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  const temporary = `${filePath}.${process.pid}.tmp`;
  fs.writeFileSync(temporary, `${JSON.stringify(stableJson(value), null, 2)}\n`, 'utf8');
  fs.renameSync(temporary, filePath);
}

function readJson(filePath, fallback = null) {
  if (!filePath || !fs.existsSync(filePath)) return fallback;
  return JSON.parse(fs.readFileSync(filePath, 'utf8').replace(/^\uFEFF/, ''));
}

function deterministicRunId({ now, sources, sourceVersions }) {
  const date = new Date(now).toISOString().replace(/[-:]/g, '').replace(/\.\d{3}Z$/, 'Z');
  const digest = stableHash({ sources: [...sources].sort(), sourceVersions }).slice(0, 10);
  return `${date}-${digest}`;
}

function selectSources({ source, allSources, config }) {
  const available = Object.keys(config.sources).sort();
  if (allSources) return available;
  if (!source || !config.sources[source]) throw new Error('A configured --source or --all-sources is required.');
  return [source];
}

function isRefreshDue({ sourceState, sourceConfig, now }) {
  if (!sourceState?.lastCheckedAt) return true;
  const elapsed = new Date(now).getTime() - new Date(sourceState.lastCheckedAt).getTime();
  return elapsed >= sourceConfig.refreshMinutes * 60 * 1000;
}

function discoverSourceChanges({ sources, sourceState = {}, config, now, metadataBySource = {} }) {
  const checks = sources.map((source) => {
    const capability = config.sources[source]?.capability || DEFAULT_SOURCE_CAPABILITIES[source];
    const previous = sourceState.sources?.[source] || null;
    const due = isRefreshDue({ sourceState: previous, sourceConfig: config.sources[source], now });
    const metadata = metadataBySource[source] || null;
    const version = metadata?.version || metadata?.snapshotId || previous?.version || null;
    const changed = due && Boolean(metadata) && version !== previous?.version;
    return { source, capability, due, changed, version, previousVersion: previous?.version || null };
  });
  return {
    checks,
    dueSources: checks.filter((item) => item.due).map((item) => item.source),
    changedSources: checks.filter((item) => item.changed).map((item) => item.source),
  };
}

function lockIsStale(lock, now, staleAfterMinutes) {
  if (!lock?.startedAt) return true;
  return new Date(now).getTime() - new Date(lock.startedAt).getTime() > staleAfterMinutes * 60 * 1000;
}

function acquireExecutionLock({ lockPath, runId, now, config, snapshotReferences = [] }) {
  fs.mkdirSync(path.dirname(lockPath), { recursive: true });
  if (fs.existsSync(lockPath)) {
    const current = readJson(lockPath, {});
    if (!lockIsStale(current, now, config.lock.staleAfterMinutes)) {
      return { acquired: false, status: RUN_STATES.SKIPPED_ALREADY_RUNNING, current };
    }
    const stalePath = `${lockPath}.stale-${String(current.runId || 'unknown').replace(/[^a-z0-9._-]/gi, '-')}`;
    fs.renameSync(lockPath, stalePath);
  }
  const lock = {
    schemaVersion: 'stage4n-lock-v1', runId, startedAt: new Date(now).toISOString(),
    processId: process.pid, host: os.hostname(), policyVersion: 'stage4l-review-policy-v1',
    snapshotReferences: [...snapshotReferences].sort(),
  };
  try {
    fs.writeFileSync(lockPath, `${JSON.stringify(lock, null, 2)}\n`, { encoding: 'utf8', flag: 'wx' });
  } catch (error) {
    if (error.code === 'EEXIST') return { acquired: false, status: RUN_STATES.SKIPPED_ALREADY_RUNNING };
    throw error;
  }
  return { acquired: true, lock };
}

function releaseExecutionLock(lockPath, runId) {
  const current = readJson(lockPath, null);
  if (current?.runId === runId) fs.rmSync(lockPath, { force: true });
}

async function withRetry(operation, {
  maxRetries = 3, baseBackoffMs = 1000, sleep = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds)),
} = {}) {
  let attempt = 0;
  while (true) {
    try {
      return await operation(attempt + 1);
    } catch (error) {
      const status = Number(error.statusCode || error.status || 0);
      const retryable = TRANSIENT_STATUS_CODES.has(status) || error.transient === true;
      if (!retryable || attempt >= maxRetries - 1) throw error;
      await sleep(baseBackoffMs * (2 ** attempt));
      attempt += 1;
    }
  }
}

function deltaCounts(state = []) {
  return state.reduce((counts, item) => {
    counts[item.status] = (counts[item.status] || 0) + 1;
    return counts;
  }, { NEW: 0, CHANGED: 0, UNCHANGED: 0, MISSING: 0, INVALID: 0 });
}

function withOperationalFieldFingerprint(record) {
  const enrichmentHash = stableHash({
    phone: record.phone || record.phones || null,
    website: record.website || record.websites || null,
    openingHours: record.openingHours || null,
    status: record.operatingStatus || record.status || null,
  });
  return {
    ...record,
    provenance: { ...(record.provenance || {}), operationalEnrichmentHash: enrichmentHash },
  };
}

function buildExceptionDelta({ policyResults = [], ledger = [], now }) {
  const ledgerById = new Map(ledger.map((item) => [item.caseId, item]));
  const surfaced = [];
  const nextLedger = [...ledger];
  for (const item of policyResults.filter((entry) => entry.outcome === 'HUMAN_REVIEW')) {
    const fingerprint = stableHash({
      caseId: item.caseId, canonicalId: item.canonicalId, reasonCodes: item.reasonCodes,
      sourceId: item.sourceId,
    });
    const previous = ledgerById.get(item.caseId);
    if (previous?.fingerprint === fingerprint && previous.reviewStatus !== 'REVISIT') continue;
    const record = {
      caseId: item.caseId,
      priority: (item.reasonCodes || []).includes('IDENTITY_FINGERPRINT_CHANGED') ? 'P0' : 'P1',
      whatChanged: item.reasonCodes || [], source: item.source, sourceId: item.sourceId,
      name: item.name || null, canonicalCandidate: item.canonicalId || null,
      policyOutcome: item.outcome, reasonCodes: item.reasonCodes || [],
      suggestedReviewerAction: 'HUMAN_REVIEW', fingerprint,
      firstSurfaced: previous?.firstSurfaced || new Date(now).toISOString(),
      lastRelevantChange: new Date(now).toISOString(), reviewStatus: previous?.reviewStatus || 'OPEN',
    };
    surfaced.push(record);
    const index = nextLedger.findIndex((entry) => entry.caseId === item.caseId);
    if (index >= 0) nextLedger[index] = record; else nextLedger.push(record);
  }
  return { surfaced: surfaced.sort((a, b) => a.caseId.localeCompare(b.caseId)), ledger: nextLedger.sort((a, b) => a.caseId.localeCompare(b.caseId)) };
}

function resolveOpenPr({ openPr, planHash, inputStateHash }) {
  if (!openPr) return { action: 'CREATE_NEW_PR' };
  if (openPr.planHash === planHash && openPr.inputStateHash === inputStateHash) {
    return { action: 'REUSE_EXISTING_PR', url: openPr.url };
  }
  return { action: RUN_STATES.OPEN_PR_REQUIRES_RELEASE, url: openPr.url };
}

function retentionCandidates({ entries, now, retention, protectedPaths = [] }) {
  const protectedSet = new Set(protectedPaths.map((item) => path.resolve(item)));
  const cutoff = new Date(now).getTime() - retention.runArtifactsDays * 24 * 60 * 60 * 1000;
  const successful = entries.filter((item) => item.status === RUN_STATES.COMPLETED)
    .sort((a, b) => new Date(b.completedAt) - new Date(a.completedAt));
  const keepSuccess = new Set(successful.slice(0, retention.minimumSuccessfulRuns).map((item) => path.resolve(item.path)));
  return entries.filter((item) => (
    item.disposable === true
      && new Date(item.completedAt || item.updatedAt || 0).getTime() < cutoff
      && !protectedSet.has(path.resolve(item.path))
      && !keepSuccess.has(path.resolve(item.path))
      && item.activeState !== true
      && item.provenanceEvidence !== true
      && item.openPrRollback !== true
  )).map((item) => item.path);
}

function writeManifest(artifactDir, manifest) {
  const { proposedState, proposedMediaRegistry, exceptionLedger, policyOutcomes, ...summary } = manifest;
  writeJsonAtomic(path.join(artifactDir, 'run_manifest.json'), {
    ...summary,
    proposedStateRecords: proposedState?.length || 0,
    proposedMediaRecords: proposedMediaRegistry?.length || 0,
    exceptionLedgerRecords: exceptionLedger?.length || 0,
    policyOutcomeCounts: (policyOutcomes || []).reduce((counts, item) => ({
      ...counts,
      [item.outcome]: (counts[item.outcome] || 0) + 1,
    }), {},),
  });
}

async function orchestrateScheduledSync(options) {
  const {
    config, sources, now = new Date().toISOString(), dataDir, artifactDir, tempDir,
    sourceState = {}, metadataBySource = {}, snapshotsBySource = {}, previousState = [],
    previousMediaRegistry = [], checkpoint = null, resume = false, dryRun = true,
    preparePr = false, openPr = null, exceptionLedger = [],
    resolveRecord = () => null, classifyGeography = () => null,
    policyRunner = () => ({ results: [], applyPlan: [], deterministicHash: stableHash([]) }),
    safePrRunner = () => ({ status: 'NO_SAFE_CHANGES', plan: { planHash: stableHash([]) } }),
    interruptionAfter = null,
  } = options;
  const discovery = discoverSourceChanges({ sources, sourceState, config, now, metadataBySource });
  const runId = options.runId || deterministicRunId({
    now, sources, sourceVersions: Object.fromEntries(discovery.checks.map((item) => [item.source, item.version])),
  });
  const lockPath = path.join(dataDir, 'locks', 'da-nang-stage4n.lock.json');
  const lock = acquireExecutionLock({ lockPath, runId, now, config,
    snapshotReferences: discovery.checks.map((item) => item.version).filter(Boolean) });
  if (!lock.acquired) return { status: RUN_STATES.SKIPPED_ALREADY_RUNNING, runId, lock: lock.current || null };
  const states = [RUN_STATES.STARTED, RUN_STATES.SOURCE_CHECK];
  const manifestBase = { schemaVersion: 'stage4n-run-v1', runId, startedAt: new Date(now).toISOString(), sources, discovery, states };
  try {
    if (discovery.changedSources.length === 0) {
      states.push(RUN_STATES.NO_SOURCE_CHANGE, RUN_STATES.COMPLETED);
      const result = { ...manifestBase, status: RUN_STATES.NO_SOURCE_CHANGE,
        heavyAcquisition: 0, normalization: 0, entityResolution: 0, policyReevaluation: 0,
        canonicalMutations: 0, sidecarMutations: 0, branchCreated: 0, prCreated: 0,
        statePromoted: !dryRun };
      writeManifest(artifactDir, result);
      return result;
    }
    const discoveredRecords = discovery.changedSources.flatMap((source) => (
      snapshotsBySource[source]?.records || []
    )).map(withOperationalFieldFingerprint);
    const recordsByKey = new Map();
    for (const record of discoveredRecords) recordsByKey.set(`${record.source}:${record.sourceId}`, record);
    const records = [...recordsByKey.values()];
    const snapshotId = discovery.changedSources.map((source) => metadataBySource[source].version).sort().join('+');
    states.push(RUN_STATES.PROCESSING_DELTA);
    const selectedSources = new Set([
      ...discovery.changedSources,
      ...records.map((record) => record.source),
    ]);
    const selectedPreviousState = previousState.filter((item) => selectedSources.has(item.source));
    const retainedPreviousState = previousState.filter((item) => !selectedSources.has(item.source));
    const selectedPreviousMedia = previousMediaRegistry.filter((item) => selectedSources.has(item.source));
    const retainedPreviousMedia = previousMediaRegistry.filter((item) => !selectedSources.has(item.source));
    const incremental = runIncrementalSync({
      snapshotId, records, previousState: selectedPreviousState,
      previousMediaRegistry: selectedPreviousMedia,
      resolveRecord, classifyGeography, checkpoint: resume ? checkpoint : null,
      stopAfter: interruptionAfter,
    });
    if (!incremental.complete) {
      const result = { ...manifestBase, status: RUN_STATES.FAILED, resumable: true,
        checkpoint: incremental.checkpoint, states, statePromoted: false };
      writeManifest(artifactDir, result);
      return result;
    }
    const affectedState = incremental.state;
    incremental.state = [...retainedPreviousState, ...affectedState]
      .sort((left, right) => `${left.source}:${left.sourceId}`.localeCompare(`${right.source}:${right.sourceId}`));
    incremental.mediaRegistry = [...retainedPreviousMedia, ...incremental.mediaRegistry]
      .sort((left, right) => left.key.localeCompare(right.key));
    incremental.deterministicHash = stableHash({
      state: incremental.state, mediaRegistry: incremental.mediaRegistry, metrics: incremental.metrics,
    });
    incremental.affectedState = affectedState;
    const counts = deltaCounts(affectedState);
    const relevantDelta = counts.NEW + counts.CHANGED + counts.MISSING;
    if (relevantDelta === 0) {
      states.push(RUN_STATES.NO_RELEVANT_POI_DELTA, RUN_STATES.COMPLETED);
      const result = { ...manifestBase, status: RUN_STATES.NO_RELEVANT_POI_DELTA,
        deltaCounts: counts, incrementalHash: incremental.deterministicHash,
        entityResolution: incremental.metrics.resolutionProcessed, policyReevaluation: 0,
        canonicalMutations: 0, sidecarMutations: 0, branchCreated: 0, prCreated: 0,
        proposedState: incremental.state, proposedMediaRegistry: incremental.mediaRegistry,
        statePromoted: !dryRun };
      writeManifest(artifactDir, result);
      return result;
    }
    states.push(RUN_STATES.POLICY);
    const affected = incremental.affectedState
      .filter((item) => [STATUSES.NEW, STATUSES.CHANGED, STATUSES.MISSING].includes(item.status));
    const policy = policyRunner({ affected, incremental, runId });
    const exceptionDelta = buildExceptionDelta({ policyResults: policy.results, ledger: exceptionLedger, now });
    writeJsonAtomic(path.join(artifactDir, 'human_review_delta.json'), exceptionDelta.surfaced);
    const safeCount = (policy.results || []).filter((item) => item.outcome === 'AUTO_ACCEPT_SAFE').length;
    if (safeCount === 0) {
      states.push(exceptionDelta.surfaced.length ? RUN_STATES.EXCEPTIONS_ONLY : RUN_STATES.NO_SAFE_CHANGES, RUN_STATES.COMPLETED);
      const result = { ...manifestBase, status: exceptionDelta.surfaced.length
        ? RUN_STATES.EXCEPTIONS_ONLY : RUN_STATES.NO_SAFE_CHANGES, deltaCounts: counts,
      incrementalHash: incremental.deterministicHash, policyHash: policy.deterministicHash,
      policyOutcomes: policy.results, exceptions: exceptionDelta.surfaced.length,
      exceptionLedger: exceptionDelta.ledger, canonicalMutations: 0, sidecarMutations: 0,
      branchCreated: 0, prCreated: 0, statePromoted: !dryRun,
      proposedState: incremental.state, proposedMediaRegistry: incremental.mediaRegistry };
      writeManifest(artifactDir, result);
      return result;
    }
    states.push(RUN_STATES.PREPARING_PR);
    const previewSafe = safePrRunner({
      policy, runId, dryRun: true, preparePr: false, artifactDir, tempDir,
    });
    const planHash = previewSafe.plan?.planHash || previewSafe.planHash;
    const inputStateHash = stableHash({ incremental: incremental.deterministicHash, policy: policy.deterministicHash });
    if (previewSafe.status === RUN_STATES.BLOCKED_BY_SAFETY_GATE) {
      states.push(RUN_STATES.BLOCKED_BY_SAFETY_GATE);
      const result = { ...manifestBase, status: RUN_STATES.BLOCKED_BY_SAFETY_GATE,
        planHash, inputStateHash, safeResult: previewSafe, states, statePromoted: false,
        branchCreated: 0, prCreated: 0 };
      writeManifest(artifactDir, result);
      return result;
    }
    if (previewSafe.status === 'NO_SAFE_CHANGES') {
      states.push(RUN_STATES.NO_SAFE_CHANGES, RUN_STATES.COMPLETED);
      const result = { ...manifestBase, status: RUN_STATES.NO_SAFE_CHANGES,
        deltaCounts: counts, incrementalHash: incremental.deterministicHash,
        policyHash: policy.deterministicHash, safeResult: previewSafe,
        exceptions: exceptionDelta.surfaced.length, exceptionLedger: exceptionDelta.ledger,
        planHash, inputStateHash, statePromoted: !dryRun, branchCreated: 0, prCreated: 0,
        proposedState: incremental.state, proposedMediaRegistry: incremental.mediaRegistry };
      writeManifest(artifactDir, result);
      return result;
    }
    const prDecision = resolveOpenPr({ openPr, planHash, inputStateHash });
    if (prDecision.action === RUN_STATES.OPEN_PR_REQUIRES_RELEASE) {
      states.push(RUN_STATES.OPEN_PR_REQUIRES_RELEASE);
      const result = { ...manifestBase, status: RUN_STATES.OPEN_PR_REQUIRES_RELEASE,
        planHash, inputStateHash, openPr: prDecision.url, states, statePromoted: false };
      writeManifest(artifactDir, result);
      return result;
    }
    const safe = preparePr && prDecision.action === 'CREATE_NEW_PR'
      ? safePrRunner({ policy, runId, dryRun: false, preparePr: true, artifactDir, tempDir })
      : previewSafe;
    states.push(RUN_STATES.COMPLETED);
    const result = { ...manifestBase, status: RUN_STATES.COMPLETED, deltaCounts: counts,
      incrementalHash: incremental.deterministicHash, policyHash: policy.deterministicHash,
      safeResult: safe, exceptions: exceptionDelta.surfaced.length,
      exceptionLedger: exceptionDelta.ledger, prDecision, planHash, inputStateHash,
      statePromoted: !dryRun && !String(safe.status).includes('BLOCKED'),
      proposedState: incremental.state, proposedMediaRegistry: incremental.mediaRegistry,
      branchCreated: preparePr && prDecision.action === 'CREATE_NEW_PR' ? 1 : 0,
      prCreated: preparePr && prDecision.action === 'CREATE_NEW_PR'
        && ['PR_CREATED', 'EXISTING_PR'].includes(safe.prCreation) ? 1 : 0 };
    writeManifest(artifactDir, result);
    return result;
  } catch (error) {
    states.push(RUN_STATES.FAILED);
    const failure = { ...manifestBase, status: RUN_STATES.FAILED, states,
      error: { name: error.name, code: error.code || null, message: error.message }, statePromoted: false };
    writeManifest(artifactDir, failure);
    return failure;
  } finally {
    releaseExecutionLock(lockPath, runId);
  }
}

module.exports = {
  RUN_STATES,
  acquireExecutionLock,
  buildExceptionDelta,
  deltaCounts,
  deterministicRunId,
  discoverSourceChanges,
  isRefreshDue,
  lockIsStale,
  orchestrateScheduledSync,
  readJson,
  releaseExecutionLock,
  resolveOpenPr,
  retentionCandidates,
  selectSources,
  withOperationalFieldFingerprint,
  withRetry,
  writeJsonAtomic,
};
