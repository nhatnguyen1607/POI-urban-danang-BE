const crypto = require('node:crypto');
const fs = require('node:fs');

const { inspectCanonicalDataset } = require('./canonicalDataset');
const { duplicatePair } = require('./createNewCanaryReview');
const {
  buildExistenceIndex,
  secondPassExistenceSearch,
} = require('./createNewEvidencePolicy');
const { serializeSidecar } = require('./safeEnrichmentExecutor');

function sha256Buffer(value) {
  return crypto.createHash('sha256').update(value).digest('hex');
}

function sha256File(filePath) {
  return sha256Buffer(fs.readFileSync(filePath));
}

function assertEqual(actual, expected, label) {
  if (actual !== expected) throw new Error(`${label}: expected ${expected}, got ${actual}`);
}

function validateCanonicalBaseline({ canonicalPath, baselineBackupPath, rows, config }) {
  const inspection = inspectCanonicalDataset(canonicalPath);
  assertEqual(inspection.rows, config.officialBaseline.rows, 'canonical rows');
  assertEqual(inspection.sha256, config.officialBaseline.sha256, 'canonical SHA');
  assertEqual(rows.length, config.officialBaseline.rows, 'parsed canonical rows');

  const ids = new Set(rows.map((row) => row.Global_ID));
  assertEqual(ids.size, rows.length, 'unique canonical IDs');
  const byId = new Map(rows.map((row) => [row.Global_ID, row]));
  const entities = config.canary.map((expected) => {
    const row = byId.get(expected.canonicalId);
    if (!row) throw new Error(`Missing Stage 4S canary: ${expected.canonicalId}`);
    assertEqual(row['Restaurant Name'], expected.name, `name ${expected.canonicalId}`);
    assertEqual(row.Category_Normalized, expected.category, `category ${expected.canonicalId}`);
    if (!config.allowedCategories.includes(row.Category_Normalized)) {
      throw new Error(`Unsupported category: ${row.Category_Normalized}`);
    }
    const latitude = Number(row.Lat);
    const longitude = Number(row.Lon);
    if (!Number.isFinite(latitude) || !Number.isFinite(longitude)
      || latitude < -90 || latitude > 90 || longitude < -180 || longitude > 180) {
      throw new Error(`Invalid coordinates: ${expected.canonicalId}`);
    }
    if (row.Source !== 'overture' || row.RestaurantID !== expected.sourceId
      || !String(row.Source_IDs || '').includes(expected.sourceId.replace(/^overture:/, ''))) {
      throw new Error(`Source lineage mismatch: ${expected.canonicalId}`);
    }
    return {
      canonicalId: row.Global_ID,
      name: row['Restaurant Name'],
      category: row.Category_Normalized,
      latitude,
      longitude,
      address: row.Address_Raw || null,
      sourceLineageReference: `${row.Source}:${row.RestaurantID}`,
    };
  });

  let oldRowsPreserved = null;
  if (baselineBackupPath) {
    const currentBytes = fs.readFileSync(canonicalPath);
    const previousBytes = fs.readFileSync(baselineBackupPath);
    assertEqual(sha256Buffer(previousBytes), config.previousBaseline.sha256, 'previous baseline SHA');
    if (!currentBytes.subarray(0, previousBytes.length).equals(previousBytes)) {
      throw new Error('The original 4166 canonical bytes were changed.');
    }
    oldRowsPreserved = true;
  }
  return {
    rows: inspection.rows,
    sha256: inspection.sha256,
    uniqueIds: ids.size,
    entities,
    oldRowsPreserved,
    oldRowsDeleted: oldRowsPreserved ? 0 : null,
    oldRowsMutated: oldRowsPreserved ? 0 : null,
  };
}

function verifyDecisionMemory({ memory, authoritativeDecisions, config }) {
  const expected = new Map(authoritativeDecisions.map((item) => [item.caseId, item.decision]));
  assertEqual(memory.length, expected.size, 'decision memory rows');
  const counts = { APPROVE: 0, REJECT: 0, DEFER: 0 };
  for (const record of memory) {
    const decision = expected.get(record.caseId);
    assertEqual(record.decision, decision, `decision ${record.caseId}`);
    assertEqual(record.decisionSource, 'HUMAN', `decision source ${record.caseId}`);
    assertEqual(record.decisionScope, 'CREATE_NEW_CANARY', `decision scope ${record.caseId}`);
    if (!record.identityFingerprint || !record.provenanceFingerprint) {
      throw new Error(`Missing decision fingerprint: ${record.caseId}`);
    }
    if (decision === 'APPROVE') assertEqual(record.applicationStatus, 'APPLIED', record.caseId);
    counts[decision] += 1;
  }
  for (const [decision, expectedCount] of Object.entries(config.decisionCounts)) {
    assertEqual(counts[decision], expectedCount, `${decision} decisions`);
  }
  return {
    appliedHumanApprovals: counts.APPROVE,
    reusableRejects: counts.REJECT,
    suppressedUnchangedDefers: counts.DEFER,
    fingerprintsValid: true,
    generalAutoCreateAuthorization: false,
  };
}

function verifySidecarLinkage({ sidecarPath, canonicalIds, config }) {
  const text = fs.readFileSync(sidecarPath, 'utf8');
  const sidecar = JSON.parse(text);
  const expectedIds = new Set(config.canary.map((item) => item.canonicalId));
  const applied = sidecar.records.filter((record) => expectedIds.has(record.canonicalId));
  assertEqual(applied.length, config.expectedSidecarRecords, 'Stage 4S sidecar records');
  const recordIds = new Set(applied.map((record) => record.recordId));
  assertEqual(recordIds.size, applied.length, 'unique sidecar identities');
  const orphans = applied.filter((record) => !canonicalIds.has(record.canonicalId));
  assertEqual(orphans.length, 0, 'sidecar orphans');
  if (applied.some((record) => record.decisionSource !== 'HUMAN'
    || !record.provenance?.source || !record.provenance?.sourceId
    || !record.license?.license || !record.license?.policyClass
    || /google/i.test(record.source))) {
    throw new Error('Stage 4S sidecar provenance or license is incomplete.');
  }
  if (sidecar.runtimeEnabled !== false) throw new Error('Runtime sidecar must remain disabled.');
  const normalizeLineEndings = (value) => value.replace(/\r\n/g, '\n');
  assertEqual(
    normalizeLineEndings(serializeSidecar(sidecar)),
    normalizeLineEndings(text),
    'deterministic sidecar serialization',
  );
  return {
    records: applied.length,
    orphanReferences: 0,
    duplicateIdentities: 0,
    provenance: 'PASS',
    license: 'PASS',
    deterministic: true,
    runtimeEnabled: false,
    hash: sidecar.contentHash,
  };
}

function replayAppliedSources({ sourceRecords, canonicalPois, historicalPois, thresholds, config }) {
  const index = buildExistenceIndex(canonicalPois, historicalPois);
  const canonicalById = new Map(canonicalPois.map((item) => [item.id, item]));
  const results = config.canary.map((expected) => {
    const record = sourceRecords.get(expected.caseId);
    if (!record) throw new Error(`Missing replay source record: ${expected.caseId}`);
    const existence = secondPassExistenceSearch(record, index, thresholds);
    const exactMatch = existence.existenceClass !== 'CLEARLY_ABSENT'
      && existence.bestCandidate?.kind === 'canonical'
      && existence.bestCandidate?.id === expected.canonicalId
      && canonicalById.has(expected.canonicalId);
    return {
      caseId: expected.caseId,
      canonicalId: expected.canonicalId,
      name: expected.name,
      existenceClass: existence.existenceClass,
      matchedCanonicalId: existence.bestCandidate?.id || null,
      exactMatch,
    };
  });
  const pairwiseDuplicates = [];
  for (let left = 0; left < results.length; left += 1) {
    for (let right = left + 1; right < results.length; right += 1) {
      const pair = duplicatePair(
        { caseId: results[left].caseId, record: sourceRecords.get(results[left].caseId) },
        { caseId: results[right].caseId, record: sourceRecords.get(results[right].caseId) },
        { duplicateDistanceMeters: 75, duplicateNameSimilarity: 0.86 },
      );
      if (pair) pairwiseDuplicates.push(pair);
    }
  }
  const createNew = results.filter((item) => item.existenceClass === 'CLEARLY_ABSENT').length;
  const repeatHumanReview = results.filter((item) => !item.exactMatch).length;
  return {
    results,
    createNewProposals: createNew,
    repeatHumanCreateNewReview: repeatHumanReview,
    duplicateProposals: pairwiseDuplicates.length,
    pairwiseDuplicates,
    recognizedExisting: results.filter((item) => item.exactMatch).length,
  };
}

function verifyAutomationSafety({ stage4l, stage4m, stage4n, stage4o, stage4s }) {
  const safe = stage4l.allowedAutoAcceptFields.length > 0
    && stage4m.budgets.maxCreateNew === 0
    && stage4m.budgets.maxDelete === 0
    && stage4m.budgets.maxMerge === 0
    && stage4m.budgets.maxLockedFieldMutations === 0
    && stage4n.autoCreateCanonicalPoi === false
    && stage4n.autoDeleteCanonicalPoi === false
    && stage4n.autoMerge === false
    && stage4o.autoCreateCanonicalPoi === false
    && stage4o.autoDeleteCanonicalPoi === false
    && stage4o.autoMerge === false
    && stage4s.autoCreateNew === false;
  if (!safe) throw new Error('Phase 4 automation safety boundary changed.');
  return {
    autoAcceptSafeExistingEnrichment: true,
    autoCreateNew: false,
    automaticDelete: false,
    automaticMerge: false,
    automaticLockedFieldOverwrite: false,
    prAutoMerge: false,
    googleBulk: false,
    secondCity: false,
  };
}

function validateSchedulerAlignment({ operationalCommit, expectedCommit, enabled, selfTest }) {
  assertEqual(operationalCommit, expectedCommit, 'scheduler operational commit');
  assertEqual(enabled, true, 'scheduler enabled');
  if (!selfTest?.ok) throw new Error('Scheduler self-test did not pass.');
  return { aligned: true, operationalCommit, enabled: true, selfTest: 'PASS' };
}

function buildCloseoutStatus(input) {
  return {
    schemaVersion: 'phase4-stage4t-closeout-status-v1',
    phase: 'PHASE_4',
    status: 'PHASE_4_IMPLEMENTATION_CLOSED',
    mainCommit: input.mainCommit,
    canonicalCount: input.canonical.rows,
    canonicalSha: input.canonical.sha256,
    schedulerOperationalCommit: input.scheduler.operationalCommit,
    schedulerEnabled: input.scheduler.enabled,
    autoCreateNew: false,
    sevenCanaryIds: input.canonical.entities.map((item) => item.canonicalId),
    sevenReplayResult: input.replay,
    duplicateResult: input.duplicateResult,
    decisionMemoryResult: input.decisionMemory,
    runtimeResult: input.runtime,
    soakStatus: input.soak.status,
    tests: input.tests,
    artifactReferences: input.artifactReferences,
  };
}

module.exports = {
  buildCloseoutStatus,
  replayAppliedSources,
  sha256Buffer,
  sha256File,
  validateCanonicalBaseline,
  validateSchedulerAlignment,
  verifyAutomationSafety,
  verifyDecisionMemory,
  verifySidecarLinkage,
};
