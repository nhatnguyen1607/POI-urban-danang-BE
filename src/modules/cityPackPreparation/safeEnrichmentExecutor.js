const crypto = require('node:crypto');

const {
  applyMutationsToText,
  inspectCanonicalText,
} = require('../../../scripts/phase4_stage4k_controlled_canary_apply');
const { stableHash, stableValue } = require('./decisionMemory');

const SIDECAR_STATUS = 'NON_RUNTIME_CITYPACK_ENRICHMENT';
const NO_SAFE_CHANGES = 'NO_SAFE_CHANGES';
const BLOCKED_BY_SAFETY_GATE = 'BLOCKED_BY_SAFETY_GATE';

function sha256Text(text) {
  return crypto.createHash('sha256').update(Buffer.from(text, 'utf8')).digest('hex');
}

function canonicalRowMap(canonicalText) {
  const inspected = inspectCanonicalText(canonicalText);
  return { inspected, rows: new Map(inspected.rows.map((row) => [row.Global_ID, row])) };
}

function validFieldValue(field, value) {
  const text = String(value || '').trim();
  if (!text) return false;
  if (field === 'website') return /^https?:\/\/\S+$/i.test(text);
  if (field === 'phone') return /^[+()\d][+()\d .-]{4,30}$/.test(text);
  if (field === 'address') return text.length >= 3 && text.length <= 500;
  if (field === 'openingHours') return text.length <= 500;
  if (field === 'media') return value && typeof value === 'object';
  return false;
}

function provenanceComplete(operation) {
  const provenance = operation.provenance || {};
  const required = ['source', 'sourceId', 'snapshotRef', 'license', 'policyClass', 'attribution', 'licenseUrl'];
  return required.every((field) => provenance[field])
    && !/google/i.test(String(operation.source || provenance.source || ''));
}

function normalizeSidecar(sidecar) {
  if (!sidecar) {
    return {
      schemaVersion: 'stage4l-enrichment-v1',
      status: SIDECAR_STATUS,
      cityId: 'da-nang',
      runtimeEnabled: false,
      records: [],
    };
  }
  if (sidecar.schemaVersion !== 'stage4l-enrichment-v1'
    || sidecar.status !== SIDECAR_STATUS
    || sidecar.runtimeEnabled !== false
    || !Array.isArray(sidecar.records)) {
    throw new Error('invalid_enrichment_sidecar_schema');
  }
  return sidecar;
}

function sidecarRecordId(operation) {
  return stableHash({
    canonicalId: operation.canonicalId,
    source: operation.source,
    sourceId: operation.sourceId,
    field: operation.field,
  });
}

function buildSidecarRecord(operation, executorVersion) {
  return {
    recordId: sidecarRecordId(operation),
    schemaVersion: 'stage4l-enrichment-v1',
    status: 'CANDIDATE_NON_RUNTIME_NOT_CANONICAL',
    cityId: 'da-nang',
    canonicalId: operation.canonicalId,
    canonicalName: operation.canonicalName,
    caseId: operation.caseId,
    field: operation.field,
    value: operation.newValue,
    source: operation.source,
    sourceId: operation.sourceId,
    snapshotReference: operation.provenance.snapshotRef,
    policyOutcome: operation.policyOutcome,
    policyVersion: operation.policyVersion,
    decisionSource: operation.decisionSource,
    decisionMemoryReference: operation.decisionMemoryReference,
    fingerprints: operation.fingerprints,
    provenance: operation.provenance,
    license: operation.license,
    firstSeenReference: operation.provenance.snapshotRef,
    lastSeenReference: operation.provenance.snapshotRef,
    processingVersion: executorVersion,
    rollback: { action: 'REMOVE_OR_RESTORE_SIDECAR_RECORD' },
  };
}

function serializeSidecar(sidecar) {
  const normalized = {
    ...sidecar,
    records: [...sidecar.records].sort((left, right) => left.recordId.localeCompare(right.recordId)),
  };
  normalized.contentHash = stableHash(normalized.records);
  return `${JSON.stringify(stableValue(normalized), null, 2)}\n`;
}

function memoryForCase(memory, caseId) {
  return [...memory].reverse().find((record) => record.caseId === caseId) || null;
}

function exactOperation(item, canonicalName, memoryRecord, destination) {
  return {
    policyVersion: item.policyVersion,
    policyOutcome: item.policyDecision,
    decisionSource: item.decisionSource,
    decisionMemoryReference: item.decisionMemoryReference
      || memoryRecord?.decisionReference || null,
    caseId: item.caseId,
    canonicalId: item.canonicalId,
    canonicalName,
    source: item.source,
    sourceId: item.sourceId,
    operation: item.operation,
    destination,
    field: item.field,
    oldValue: item.oldValue ?? null,
    newValue: item.newValue,
    provenance: item.provenance,
    license: item.license || item.provenance?.license || null,
    reasonCodes: item.reasonCodes || [],
    rollbackValue: item.oldValue ?? null,
    fingerprints: memoryRecord ? {
      identity: memoryRecord.identityFingerprint,
      field: memoryRecord.fieldFingerprints?.[item.field] || null,
      provenance: memoryRecord.provenanceFingerprint,
    } : null,
  };
}

function applyBudgetOverrides(config, overrides = {}) {
  const budgets = { ...config.budgets };
  if (overrides.maxAutoCases !== undefined) {
    budgets.maxAutoCases = Math.min(budgets.maxAutoCases, overrides.maxAutoCases);
  }
  if (overrides.maxFieldMutations !== undefined) {
    budgets.maxCanonicalAddressMutations = Math.min(
      budgets.maxCanonicalAddressMutations, overrides.maxFieldMutations,
    );
    budgets.maxSidecarFieldMutations = Math.min(
      budgets.maxSidecarFieldMutations, overrides.maxFieldMutations,
    );
  }
  return { ...config, budgets };
}

function buildSafeEnrichmentPlan({
  policyResult,
  decisionMemory,
  canonicalText,
  existingSidecar,
  executorConfig,
  recoveredEnrichments = [],
  previousExceptionCaseIds = [],
  budgetOverrides = {},
}) {
  const config = applyBudgetOverrides(executorConfig, budgetOverrides);
  const { inspected, rows } = canonicalRowMap(canonicalText);
  const sidecar = normalizeSidecar(existingSidecar);
  const sidecarById = new Map(sidecar.records.map((record) => [record.recordId, record]));
  const canonicalOperations = [];
  const sidecarOperations = [];
  const skipped = [];
  const violations = [];
  const plannedAddressValues = new Map();
  const allowedSidecar = new Set(config.sidecarFields);
  const allowedCore = new Set(config.canonicalCoreFields);

  if (inspected.rows.length !== config.expectedCanonicalRows) violations.push('CANONICAL_ROW_COUNT');
  if (policyResult.processingVersions?.policy !== config.expectedPolicyVersion) {
    violations.push('POLICY_VERSION_MISMATCH');
  }
  if (policyResult.processingVersions?.boundary !== config.expectedBoundaryVersion) {
    violations.push('BOUNDARY_VERSION_MISMATCH');
  }
  if (policyResult.circuitBreaker?.tripped) violations.push('STAGE4L_CIRCUIT_BREAKER');
  if (policyResult.anomalyGate?.tripped) violations.push('STAGE4L_ANOMALY_GATE');
  if (policyResult.createNewAuthorized !== false) violations.push('CREATE_NEW_AUTHORIZATION');
  if (policyResult.deleteAuthorized === true) violations.push('DELETE_AUTHORIZATION');
  if (policyResult.mergeAuthorized === true) violations.push('MERGE_AUTHORIZATION');

  for (const item of policyResult.applyPlan || []) {
    if (item.policyDecision !== 'AUTO_ACCEPT_SAFE') {
      violations.push(`NON_AUTO_ACCEPT_OPERATION:${item.caseId}:${item.field}`);
      continue;
    }
    if (item.operation !== 'ENRICH_EXISTING') {
      violations.push(`UNAUTHORIZED_OPERATION:${item.caseId}:${item.operation}`);
      continue;
    }
    if (!item.canonicalId || !rows.has(item.canonicalId)) {
      violations.push(`MISSING_CANONICAL_TARGET:${item.caseId}`);
      continue;
    }
    const destination = allowedCore.has(item.field)
      ? 'CANONICAL_CORE' : allowedSidecar.has(item.field) ? 'ENRICHMENT_SIDECAR' : null;
    if (!destination) {
      violations.push(`LOCKED_OR_UNSUPPORTED_FIELD:${item.caseId}:${item.field}`);
      continue;
    }
    if (!provenanceComplete(item)) {
      violations.push(`PROVENANCE_LICENSE_FAILURE:${item.caseId}:${item.field}`);
      continue;
    }
    if (!validFieldValue(item.field, item.newValue)) {
      violations.push(`INVALID_FIELD_VALUE:${item.caseId}:${item.field}`);
      continue;
    }
    const memoryRecord = memoryForCase(decisionMemory, item.caseId);
    if (!memoryRecord) {
      violations.push(`MISSING_DECISION_MEMORY:${item.caseId}`);
      continue;
    }
    const row = rows.get(item.canonicalId);
    const operation = exactOperation(item, row['Restaurant Name'], memoryRecord, destination);

    if (destination === 'CANONICAL_CORE') {
      if (item.field !== 'address') {
        violations.push(`NON_WHITELISTED_CANONICAL_FIELD:${item.field}`);
        continue;
      }
      const current = String(row.Address_Current || '').trim();
      if (current) {
        skipped.push({ ...operation, status: current === String(item.newValue).trim()
          ? 'SKIPPED_EXISTING_VALUE' : 'CONFLICT_NON_EMPTY_CANONICAL_VALUE' });
        if (current !== String(item.newValue).trim()) {
          violations.push(`NON_EMPTY_CANONICAL_OVERWRITE:${item.canonicalId}`);
        }
        continue;
      }
      const previous = plannedAddressValues.get(item.canonicalId);
      if (previous && previous !== item.newValue) {
        violations.push(`INCOMPATIBLE_ADDRESS_OPERATIONS:${item.canonicalId}`);
        continue;
      }
      plannedAddressValues.set(item.canonicalId, item.newValue);
      canonicalOperations.push(operation);
      continue;
    }

    const record = buildSidecarRecord(operation, config.executorVersion);
    const existing = sidecarById.get(record.recordId);
    if (existing) {
      skipped.push({ ...operation, status: stableHash(existing) === stableHash(record)
        ? 'SKIPPED_EXISTING_VALUE' : 'CONFLICT_EXISTING_SIDECAR_VALUE' });
      if (stableHash(existing) !== stableHash(record)) {
        violations.push(`SIDECAR_CONFLICT:${item.canonicalId}:${item.field}`);
      }
      continue;
    }
    sidecarOperations.push({ ...operation, record });
  }

  const autoCases = new Set((policyResult.results || [])
    .filter((item) => item.outcome === 'AUTO_ACCEPT_SAFE').map((item) => item.caseId)).size;
  const targetPercent = new Set(canonicalOperations.map((item) => item.canonicalId)).size
    / config.expectedCanonicalRows;
  if (autoCases > config.budgets.maxAutoCases) violations.push('AUTO_CASE_BUDGET');
  if (canonicalOperations.length > config.budgets.maxCanonicalAddressMutations) {
    violations.push('CANONICAL_ADDRESS_BUDGET');
  }
  if (sidecarOperations.length > config.budgets.maxSidecarFieldMutations) {
    violations.push('SIDECAR_FIELD_BUDGET');
  }
  if (targetPercent > config.budgets.maxCanonicalPercentTargets) {
    violations.push('CANONICAL_PERCENT_TARGET_BUDGET');
  }

  const priorExceptions = new Set(previousExceptionCaseIds);
  const exceptions = (policyResult.results || [])
    .filter((item) => item.outcome === 'HUMAN_REVIEW' && !priorExceptions.has(item.caseId))
    .map((item) => ({
      caseId: item.caseId,
      priority: item.reasonCodes.includes('IDENTITY_FINGERPRINT_CHANGED') ? 'P0' : 'P1',
      source: item.source,
      sourceId: item.sourceId,
      name: rows.get(item.canonicalId)?.['Restaurant Name'] || null,
      canonicalCandidate: item.canonicalId,
      reasonCodes: item.reasonCodes,
      policyOutcome: item.outcome,
      whatChanged: item.reasonCodes,
      recommendedNextAction: 'HUMAN_REVIEW',
    }));

  const plannedSidecarKeys = new Set(sidecarOperations.map((item) => (
    `${item.caseId}:${item.field}:${String(item.newValue)}`
  )));
  const existingSidecarKeys = new Set(sidecar.records.map((item) => (
    `${item.caseId}:${item.field}:${String(item.value)}`
  )));
  const recoveredCount = recoveredEnrichments.filter((item) => (
    plannedSidecarKeys.has(`${item.caseId}:${item.field}:${String(item.value)}`)
      || existingSidecarKeys.has(`${item.caseId}:${item.field}:${String(item.value)}`)
  )).length;
  if (recoveredCount !== recoveredEnrichments.length) violations.push('STAGE4K_RECOVERY_INCOMPLETE');

  const status = violations.length > 0
    ? BLOCKED_BY_SAFETY_GATE
    : canonicalOperations.length + sidecarOperations.length === 0 ? NO_SAFE_CHANGES : 'SAFE_PLAN_READY';
  const summary = {
    status,
    autoAcceptCases: autoCases,
    canonicalAddressOperations: canonicalOperations.length,
    sidecarOperationCount: sidecarOperations.length,
    existingValueSkips: skipped.filter((item) => item.status === 'SKIPPED_EXISTING_VALUE').length,
    recoveredStage4kEnrichments: recoveredCount,
    exceptions: exceptions.length,
    violations,
    budgets: config.budgets,
    targetPercent,
    createNew: 0,
    delete: 0,
    merge: 0,
  };
  return {
    ...summary,
    canonicalOperations,
    sidecarOperations,
    skipped,
    exceptionRecords: exceptions,
    planHash: stableHash({ canonicalOperations, sidecarOperations, summary }),
  };
}

function canonicalMutations(plan) {
  return plan.canonicalOperations.map((item) => ({
    case_id: item.caseId,
    canonical_id: item.canonicalId,
    canonical_column: 'Address_Current',
    field: 'address',
    old_value: item.oldValue,
    new_value: item.newValue,
  }));
}

function applySafeEnrichment({ plan, canonicalText, existingSidecar, executorConfig }) {
  if (plan.status === BLOCKED_BY_SAFETY_GATE) throw new Error(BLOCKED_BY_SAFETY_GATE);
  if (plan.status === NO_SAFE_CHANGES) {
    return {
      status: NO_SAFE_CHANGES,
      canonicalText,
      sidecarText: existingSidecar ? serializeSidecar(normalizeSidecar(existingSidecar)) : null,
    };
  }
  const mutations = canonicalMutations(plan);
  const afterCanonicalText = applyMutationsToText(canonicalText, mutations);
  const currentSidecar = normalizeSidecar(existingSidecar);
  const afterSidecar = {
    ...currentSidecar,
    records: [...currentSidecar.records, ...plan.sidecarOperations.map((item) => item.record)],
  };
  return {
    status: 'SAFE_CHANGES_APPLIED_TO_FEATURE_BRANCH',
    canonicalText: afterCanonicalText,
    sidecarText: serializeSidecar(afterSidecar),
    canonicalSha256: sha256Text(afterCanonicalText),
    sidecarHash: stableHash(afterSidecar.records.sort((a, b) => a.recordId.localeCompare(b.recordId))),
    rollback: {
      canonicalText,
      sidecarText: existingSidecar ? serializeSidecar(normalizeSidecar(existingSidecar)) : null,
    },
    executorVersion: executorConfig.executorVersion,
  };
}

function validateAppliedResult({ plan, beforeCanonicalText, result, existingSidecar, executorConfig }) {
  const before = canonicalRowMap(beforeCanonicalText).inspected;
  const after = canonicalRowMap(result.canonicalText).inspected;
  const expected = applyMutationsToText(beforeCanonicalText, canonicalMutations(plan));
  if (result.canonicalText !== expected) throw new Error('NON_MINIMAL_CANONICAL_DIFF');
  if (after.rows.length !== executorConfig.expectedCanonicalRows) throw new Error('POST_APPLY_ROW_COUNT');
  const beforeById = new Map(before.rows.map((row) => [row.Global_ID, row]));
  const targets = new Set(plan.canonicalOperations.map((item) => item.canonicalId));
  let nonTargetRowsUnchanged = 0;
  let nameChanges = 0;
  let coordinateChanges = 0;
  let categoryChanges = 0;
  let externalIdsChanges = 0;
  for (const row of after.rows) {
    const old = beforeById.get(row.Global_ID);
    if (!old) throw new Error(`UNEXPECTED_NEW_ROW:${row.Global_ID}`);
    const changed = before.headers.filter((header) => old[header] !== row[header]);
    if (!targets.has(row.Global_ID)) {
      if (changed.length) throw new Error(`NON_TARGET_ROW_CHANGED:${row.Global_ID}`);
      nonTargetRowsUnchanged += 1;
    } else if (changed.some((header) => header !== 'Address_Current')) {
      throw new Error(`LOCKED_CANONICAL_COLUMN_CHANGED:${row.Global_ID}`);
    }
    nameChanges += Number(old['Restaurant Name'] !== row['Restaurant Name']);
    coordinateChanges += Number(old.Lat !== row.Lat || old.Lon !== row.Lon);
    categoryChanges += Number(old.Category !== row.Category);
    externalIdsChanges += Number(old.Source_IDs !== row.Source_IDs);
  }
  const rolledBack = applyMutationsToText(result.canonicalText, canonicalMutations(plan), { rollback: true });
  if (rolledBack !== beforeCanonicalText) throw new Error('CANONICAL_ROLLBACK_NOT_BYTE_IDENTICAL');
  const sidecar = JSON.parse(result.sidecarText);
  if (serializeSidecar(sidecar) !== result.sidecarText) throw new Error('NON_DETERMINISTIC_SIDECAR');
  const expectedPreviousSidecar = existingSidecar
    ? serializeSidecar(normalizeSidecar(existingSidecar)) : null;
  if (result.rollback.sidecarText !== expectedPreviousSidecar) throw new Error('SIDECAR_ROLLBACK_MISMATCH');
  return {
    beforeRows: before.rows.length,
    afterRows: after.rows.length,
    beforeSha256: before.sha256,
    afterSha256: after.sha256,
    canonicalAddressChanges: plan.canonicalOperations.length,
    sidecarChanges: plan.sidecarOperations.length,
    nonTargetRowsUnchanged,
    newRows: 0,
    deletedRows: 0,
    nameChanges,
    coordinateChanges,
    categoryChanges,
    externalIdsChanges,
    rollbackCanonicalSha256: sha256Text(rolledBack),
    rollbackCanonicalByteIdentical: true,
    rollbackSidecarExact: true,
    sidecarSchemaPass: true,
    provenanceLicensePass: true,
    deterministic: true,
  };
}

module.exports = {
  BLOCKED_BY_SAFETY_GATE,
  NO_SAFE_CHANGES,
  SIDECAR_STATUS,
  applySafeEnrichment,
  buildSafeEnrichmentPlan,
  buildSidecarRecord,
  canonicalMutations,
  normalizeSidecar,
  provenanceComplete,
  serializeSidecar,
  sha256Text,
  sidecarRecordId,
  validateAppliedResult,
  validFieldValue,
};
