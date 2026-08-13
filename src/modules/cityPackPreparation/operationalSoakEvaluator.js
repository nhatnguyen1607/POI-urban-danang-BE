const fs = require('node:fs');
const path = require('node:path');

const SUCCESS_STATUSES = new Set([
  'NO_WORK',
  'COMPLETED_NO_PR',
  'SAFE_PR_CREATED',
  'EXCEPTIONS_ONLY',
]);

function durationMilliseconds(manifest) {
  const start = new Date(manifest.startedAt).getTime();
  const end = new Date(manifest.endedAt).getTime();
  return Number.isFinite(start) && Number.isFinite(end) && end >= start ? end - start : null;
}

function readOperationalManifests(root) {
  if (!fs.existsSync(root)) return [];
  const manifests = [];
  const visit = (directory) => {
    for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
      const filePath = path.join(directory, entry.name);
      if (entry.isDirectory()) visit(filePath);
      else if (entry.name === 'operational_run_manifest.json') {
        manifests.push(JSON.parse(fs.readFileSync(filePath, 'utf8').replace(/^\uFEFF/, '')));
      }
    }
  };
  visit(root);
  return manifests.sort((left, right) => (
    String(left.startedAt || '').localeCompare(String(right.startedAt || ''))
    || String(left.runId || '').localeCompare(String(right.runId || ''))
  ));
}

function directoryBytes(root) {
  if (!fs.existsSync(root)) return 0;
  let total = 0;
  const visit = (directory) => {
    for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
      const filePath = path.join(directory, entry.name);
      if (entry.isDirectory()) visit(filePath);
      else total += fs.statSync(filePath).size;
    }
  };
  visit(root);
  return total;
}

function aggregateOperationalSoak({ manifests, gate, artifactRoot, logRoot }) {
  const scheduled = manifests.filter((item) => item.triggerType === 'SCHEDULED');
  const noWork = scheduled.filter((item) => item.finalStatus === 'NO_WORK');
  const sourceChecks = scheduled.flatMap((item) => item.sourceChecks || []);
  const dueChecks = sourceChecks.filter((item) => item.due === true);
  const actualDeltas = scheduled.filter((item) => (
    Object.values(item.deltaCounts || {}).some((value) => Number(value) > 0)
    || (item.sourceChecks || []).some((check) => check.changed === true)
  ));
  const failures = scheduled.filter((item) => item.finalStatus === 'FAILED');
  const blocked = scheduled.filter((item) => item.finalStatus === 'BLOCKED');
  const successful = scheduled.filter((item) => SUCCESS_STATUSES.has(item.finalStatus));
  const noWorkDurations = noWork.map(durationMilliseconds).filter(Number.isFinite);
  const staleLocks = manifests.filter((item) => item.lockInspection?.action === 'CLEAR_STALE_LOCK');
  const duplicatePrSuppressions = scheduled.filter((item) => (
    item.safetyStatus === 'EXISTING_EQUIVALENT_PR'
    || item.pr?.reused === true
  ));
  const stateCorruption = scheduled.filter((item) => item.failureClass === 'STATE_FAILURE');
  const activeLocks = scheduled.filter((item) => item.currentLock || item.lockReleased === false);
  const criteria = {
    scheduledRuns: scheduled.length >= gate.minimumScheduledRuns,
    dueSourceCheck: !gate.requireDueSourceCheck || dueChecks.length > 0,
    noFailures: !gate.requireNoFailures || (failures.length === 0 && blocked.length === 0),
    noStateCorruption: !gate.requireNoStateCorruption || stateCorruption.length === 0,
    noDuplicatePullRequests: !gate.requireNoDuplicatePullRequests || duplicatePrSuppressions.length === 0,
    lockReleaseHealthy: !gate.requireHealthyLockRelease || activeLocks.length === 0,
  };
  return {
    schemaVersion: 'stage4p-operational-soak-v1',
    status: Object.values(criteria).every(Boolean) ? 'COMPLETE' : 'IN_PROGRESS',
    criteria,
    scheduledInvocationCount: scheduled.length,
    successfulCount: successful.length,
    noWorkCount: noWork.length,
    sourceCheckCount: sourceChecks.length,
    sourceDueCount: dueChecks.length,
    realAcquisitionCount: scheduled.reduce((sum, item) => sum + Number(item.acquisitionCount || 0), 0),
    actualSourceDeltaCount: actualDeltas.length,
    autoAcceptSafeRuns: scheduled.filter((item) => Number(item.policyOutcomeCounts?.AUTO_ACCEPT_SAFE || 0) > 0).length,
    prProducingRuns: scheduled.filter((item) => Boolean(item.prUrl)).length,
    exceptionsOnlyRuns: scheduled.filter((item) => item.finalStatus === 'EXCEPTIONS_ONLY').length,
    blockedRuns: blocked.length,
    failedRuns: failures.length,
    staleLockActions: staleLocks.length,
    retries: scheduled.reduce((sum, item) => sum + Number(item.retryCount || 0), 0),
    resumeEvents: scheduled.filter((item) => Boolean(item.resumeCheckpoint)).length,
    duplicatePrSuppressions: duplicatePrSuppressions.length,
    averageNoWorkDurationMs: noWorkDurations.length
      ? Number((noWorkDurations.reduce((sum, value) => sum + value, 0) / noWorkDurations.length).toFixed(2))
      : null,
    artifactBytes: directoryBytes(artifactRoot),
    logBytes: directoryBytes(logRoot),
    lastDueSource: dueChecks.at(-1)?.source || null,
    lastActualSourceDelta: actualDeltas.at(-1)?.endedAt || null,
  };
}

module.exports = {
  SUCCESS_STATUSES,
  aggregateOperationalSoak,
  directoryBytes,
  readOperationalManifests,
};
