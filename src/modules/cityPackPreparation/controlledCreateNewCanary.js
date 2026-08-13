const {
  DECISION_SOURCES,
  createDecisionRecord,
  stableHash,
  stableValue,
} = require('./decisionMemory');
const { normalizeSidecar, serializeSidecar } = require('./safeEnrichmentExecutor');
const { normalizeCategory } = require('./sourceRecord');
const { duplicatePair } = require('./createNewCanaryReview');

const HUMAN_DECISIONS = new Set(['APPROVE', 'REJECT', 'DEFER']);

function assertStage4SSafety(config) {
  if (config.autoCreateNew !== false || config.maximumHumanApprovedCreateNew !== 7
    || config.deleteOperations !== 0 || config.mergeOperations !== 0
    || config.existingRowMutations !== 0 || config.runtimeSidecarEnabled !== false) {
    throw new Error('Stage 4S safety boundary mismatch.');
  }
}

function validateHumanDecisionRows(rows, reviewedCaseIds, authoritativeDecisions = null) {
  const expected = new Set(reviewedCaseIds);
  const seen = new Set();
  const counts = { APPROVE: 0, REJECT: 0, DEFER: 0 };
  for (const row of rows) {
    const caseId = row.case_id || row.caseId;
    const decision = row.reviewer_decision || row.reviewerDecision;
    if (!expected.has(caseId)) throw new Error(`Unknown Stage 4S decision case: ${caseId}`);
    if (seen.has(caseId)) throw new Error(`Duplicate Stage 4S decision case: ${caseId}`);
    if (!HUMAN_DECISIONS.has(decision)) throw new Error(`Invalid human decision: ${decision}`);
    if (authoritativeDecisions && authoritativeDecisions.get(caseId) !== decision) {
      throw new Error(`Human decision allowlist mismatch: ${caseId}`);
    }
    seen.add(caseId);
    counts[decision] += 1;
  }
  if (seen.size !== expected.size) throw new Error('Stage 4S decision coverage is incomplete.');
  if (counts.APPROVE !== 7 || counts.REJECT !== 4 || counts.DEFER !== 15) {
    throw new Error(`Stage 4S decision counts invalid: ${JSON.stringify(counts)}`);
  }
  return counts;
}

function resolveHumanDecisionContract(reviewRows, contract) {
  const reviewById = new Map(reviewRows.map((row) => [row.case_id || row.caseId, row]));
  const seen = new Set();
  return contract.map((expected) => {
    const prefix = expected.caseId.split(':').pop().slice(0, 8);
    const matches = reviewRows.filter((row) => (row.case_id || row.caseId).includes(prefix));
    if (matches.length !== 1) throw new Error(`Prefix ${prefix} resolved ${matches.length} cases.`);
    const row = matches[0];
    const caseId = row.case_id || row.caseId;
    const name = row.name;
    if (caseId !== expected.caseId || !name.includes(expected.nameContains)
      || !reviewById.has(expected.caseId) || seen.has(caseId)) {
      throw new Error(`Human decision contract mismatch: ${prefix}`);
    }
    seen.add(caseId);
    return { prefix, caseId, name, decision: expected.decision };
  });
}

function reviewCaseForMemory(candidate) {
  const record = candidate.record;
  return {
    caseId: candidate.caseId,
    source: {
      source: record.source,
      sourceId: record.sourceId,
      originalName: record.name,
      normalizedName: record.normalizedName,
      category: record.category,
      address: record.address,
      latitude: record.latitude,
      longitude: record.longitude,
      externalIds: record.externalIds,
    },
    canonical: null,
    evidence: {
      supportClass: candidate.supportClass,
      crossSourceSupport: candidate.stage4qResult.crossSourceSupport || [],
      canonicalAbsence: candidate.currentExistence.existenceClass,
      duplicateStatus: candidate.duplicateStatus,
    },
    historicalEvidence: candidate.stage4qResult.historicalEvidence,
    provenanceComplete: candidate.provenanceStatus === 'PASS' && candidate.licenseStatus === 'PASS',
    provenance: record.provenance,
    license: record.license,
  };
}

function importCreateNewDecisionMemory({ candidates, decisionRows, config, decisionReference }) {
  const byCaseId = new Map(candidates.map((candidate) => [candidate.caseId, candidate]));
  return decisionRows.map((row) => {
    const caseId = row.case_id || row.caseId;
    const decision = row.reviewer_decision || row.reviewerDecision;
    const candidate = byCaseId.get(caseId);
    if (!candidate) throw new Error(`Stage 4S evidence missing for ${caseId}`);
    return createDecisionRecord({
      reviewCase: reviewCaseForMemory(candidate),
      decision,
      decisionSource: DECISION_SOURCES.HUMAN,
      decisionReference,
      decisionScope: config.decisionScope,
      applicationStatus: decision === 'APPROVE' ? 'APPROVED_NOT_APPLIED' : 'REVIEWED',
      policyVersion: config.stage4rPolicyVersion,
      resolverVersion: candidate.stage4qResult.resolverVersion || 'stage4q-resolver-evidence-v1',
      evidenceVersion: candidate.stage4qResult.evidenceVersion || 'stage4q-source-graph-v2',
      boundaryVersion: candidate.boundaryVersion || 'osm-relation-1891418-v72-product-scope-v1',
      note: row.reviewer_note || row.reviewerNote || null,
    });
  }).sort((left, right) => left.caseId.localeCompare(right.caseId));
}

function mapCanonicalCategory(candidate, config) {
  const category = normalizeCategory(candidate.record.category);
  if (!config.allowedCanonicalCategories.includes(category)) return null;
  return category;
}

function revalidateHumanApprovals({ candidates, decisionRows, evidenceByCaseId, config }) {
  const decisions = new Map(decisionRows.map((row) => [row.case_id || row.caseId,
    row.reviewer_decision || row.reviewerDecision]));
  const approved = candidates.filter((candidate) => decisions.get(candidate.caseId) === 'APPROVE');
  const dropped = [];
  const surviving = [];
  for (const candidate of approved) {
    const evidence = evidenceByCaseId.get(candidate.caseId);
    const reasons = [];
    if (!evidence || evidence.fingerprints?.stage4rDecision !== candidate.decisionFingerprint) {
      reasons.push('STALE_HUMAN_APPROVAL');
    }
    if (candidate.currentExistence.existenceClass !== 'CLEARLY_ABSENT') {
      reasons.push('CURRENT_CANONICAL_NOT_CLEARLY_ABSENT');
    }
    if (candidate.duplicateStatus !== 'RESOLVED_OR_NONE') reasons.push('SOURCE_DUPLICATE_UNRESOLVED');
    if (candidate.travelerRelevance !== 'TRAVELER_RELEVANT') reasons.push('NOT_TRAVELER_RELEVANT');
    if (candidate.provenanceStatus !== 'PASS' || candidate.licenseStatus !== 'PASS') {
      reasons.push('PROVENANCE_OR_LICENSE_FAILURE');
    }
    if (!['INSIDE_DANANG', 'BOUNDARY_EDGE'].includes(candidate.boundaryClass)) {
      reasons.push('OUTSIDE_APPROVED_POLYGON');
    }
    const category = mapCanonicalCategory(candidate, config);
    if (!category) reasons.push('UNSUPPORTED_CANONICAL_CATEGORY');
    if (reasons.length) dropped.push({ caseId: candidate.caseId, reasons });
    else surviving.push({ ...candidate, canonicalCategory: category });
  }
  const duplicatePairs = [];
  for (let left = 0; left < surviving.length; left += 1) {
    for (let right = left + 1; right < surviving.length; right += 1) {
      const pair = duplicatePair(surviving[left], surviving[right], config);
      if (pair) duplicatePairs.push(pair);
    }
  }
  if (duplicatePairs.length) {
    const affected = new Set(duplicatePairs.flatMap((pair) => [pair.leftCaseId, pair.rightCaseId]));
    for (const caseId of affected) dropped.push({ caseId, reasons: ['APPROVED_PAIR_DUPLICATE_UNRESOLVED'] });
    return { surviving: surviving.filter((item) => !affected.has(item.caseId)), dropped, duplicatePairs };
  }
  return { surviving, dropped, duplicatePairs };
}

function csvEscape(value) {
  const text = value === null || value === undefined ? '' : String(value);
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function canonicalRow(candidate, headers) {
  const record = candidate.record;
  const values = {
    Global_ID: candidate.proposedCanonicalId,
    Alias_Global_IDs: '',
    City_ID: 'da-nang',
    Entity_Type: 'poi',
    RestaurantID: record.sourceId,
    Source_IDs: `${record.source}:${record.sourceId.replace(/^overture:/, '')}`,
    'Restaurant Name': record.name,
    District: '',
    District_Raw: '',
    Admin_Normalization_Status: 'pending_spatial_join',
    Address_Raw: record.address || '',
    Address_Current: '',
    Category: record.sourceCategory || record.category,
    Category_Normalized: candidate.canonicalCategory,
    Lat: record.latitude,
    Lon: record.longitude,
    Source: record.source,
    Merge_Status: 'human_approved_create_new_canary',
    Data_Quality_Flags: 'stage4s_controlled_create_new|admin_pending_spatial_join|ratings_unknown',
  };
  return headers.map((header) => csvEscape(values[header] ?? '')).join(',');
}

function appendCanonicalRows(canonicalText, candidates, headers) {
  const ordered = [...candidates].sort((left, right) => left.caseId.localeCompare(right.caseId));
  const suffix = `${ordered.map((candidate) => canonicalRow(candidate, headers)).join('\n')}\n`;
  const separator = canonicalText.endsWith('\n') ? '' : '\n';
  return `${canonicalText}${separator}${suffix}`;
}

function safeWebsite(value) {
  if (!value) return null;
  try {
    const url = new URL(value);
    if (!['http:', 'https:'].includes(url.protocol) || /(^|\.)google\./i.test(url.hostname)
      || /(^|\.)goo\.gl$/i.test(url.hostname)) return null;
    return value;
  } catch (_) {
    return null;
  }
}

function sidecarFields(candidate) {
  const record = candidate.record;
  const fields = [{ field: 'externalIds', value: { overture: record.sourceId } }];
  if (record.phone) fields.push({ field: 'phone', value: record.phone });
  const website = safeWebsite(record.website);
  if (website) fields.push({ field: 'website', value: website });
  if (record.openingHours) fields.push({ field: 'openingHours', value: record.openingHours });
  if (record.media) fields.push({ field: 'media', value: record.media });
  return fields;
}

function buildCreateNewSidecar({ existingSidecar, candidates, memoryByCaseId, config }) {
  const sidecar = normalizeSidecar(existingSidecar);
  const existingIds = new Set(sidecar.records.map((record) => record.recordId));
  const created = [];
  for (const candidate of candidates) {
    const memory = memoryByCaseId.get(candidate.caseId);
    for (const item of sidecarFields(candidate)) {
      const recordId = stableHash({
        canonicalId: candidate.proposedCanonicalId,
        source: candidate.record.source,
        sourceId: candidate.record.sourceId,
        field: item.field,
      });
      if (existingIds.has(recordId)) throw new Error(`Duplicate sidecar identity: ${recordId}`);
      existingIds.add(recordId);
      created.push({
        recordId,
        schemaVersion: 'stage4l-enrichment-v1',
        status: 'CANONICAL_POI_NON_RUNTIME_ENRICHMENT',
        cityId: 'da-nang',
        canonicalId: candidate.proposedCanonicalId,
        canonicalName: candidate.record.name,
        caseId: candidate.caseId,
        field: item.field,
        value: item.value,
        source: candidate.record.source,
        sourceId: candidate.record.sourceId,
        snapshotReference: candidate.record.provenance.snapshotRef,
        policyOutcome: 'HUMAN_APPROVED_CREATE_NEW_CANARY',
        policyVersion: config.policyVersion,
        decisionSource: 'HUMAN',
        decisionMemoryReference: memory.decisionReference,
        fingerprints: {
          identity: memory.identityFingerprint,
          provenance: memory.provenanceFingerprint,
          review: candidate.decisionFingerprint,
        },
        provenance: candidate.record.provenance,
        license: candidate.record.license,
        firstSeenReference: candidate.record.provenance.snapshotRef,
        lastSeenReference: candidate.record.provenance.snapshotRef,
        processingVersion: config.policyVersion,
        rollback: { action: 'REMOVE_CREATED_SIDECAR_RECORD' },
      });
    }
  }
  const next = { ...sidecar, runtimeEnabled: false, records: [...sidecar.records, ...created] };
  return { text: serializeSidecar(next), created, normalized: stableValue(next) };
}

function buildApplyPlan({ surviving, decisionMemory, config }) {
  assertStage4SSafety(config);
  if (surviving.length > config.maximumHumanApprovedCreateNew) throw new Error('Stage 4S budget exceeded.');
  const memoryByCaseId = new Map(decisionMemory.map((record) => [record.caseId, record]));
  const operations = surviving.map((candidate) => {
    const memory = memoryByCaseId.get(candidate.caseId);
    if (!memory || memory.decision !== 'APPROVE' || memory.decisionSource !== 'HUMAN'
      || memory.decisionScope !== config.decisionScope) throw new Error(`Human allowlist failure: ${candidate.caseId}`);
    return {
      caseId: candidate.caseId,
      humanDecision: 'APPROVE',
      decisionSource: 'HUMAN',
      decisionMemoryReference: memory.decisionReference,
      reviewFingerprint: candidate.decisionFingerprint,
      currentFingerprint: candidate.decisionFingerprint,
      source: candidate.record.source,
      sourceId: candidate.record.sourceId,
      canonicalId: candidate.proposedCanonicalId,
      name: candidate.record.name,
      category: candidate.canonicalCategory,
      latitude: candidate.record.latitude,
      longitude: candidate.record.longitude,
      address: candidate.record.address || null,
      canonicalAbsence: candidate.currentExistence.existenceClass,
      nearestCanonical: candidate.currentExistence.nearestCanonical,
      duplicateResult: candidate.duplicateStatus,
      polygonResult: candidate.boundaryClass,
      travelerRelevance: candidate.travelerRelevance,
      provenance: candidate.record.provenance,
      license: candidate.record.license,
      sidecarEnrichments: sidecarFields(candidate),
      reasonCodes: ['human_approved_exact_fingerprint', 'all_stage4s_revalidation_gates_passed'],
      rollback: { action: 'REMOVE_APPENDED_CANONICAL_ROW_AND_SIDECAR_RECORDS' },
    };
  }).sort((left, right) => left.caseId.localeCompare(right.caseId));
  return {
    schemaVersion: 'stage4s-controlled-create-new-plan-v1',
    status: 'READY_FOR_FEATURE_BRANCH_APPLY',
    policyVersion: config.policyVersion,
    authorization: { originalHumanApprove: 7, maximumCreateNew: 7, autoCreateNew: false },
    operations,
    forbiddenOperations: { reject: 0, defer: 0, policyOnly: 0, delete: 0, merge: 0, existingRowMutation: 0 },
    planHash: stableHash(operations),
  };
}

module.exports = {
  HUMAN_DECISIONS,
  appendCanonicalRows,
  assertStage4SSafety,
  buildApplyPlan,
  buildCreateNewSidecar,
  canonicalRow,
  importCreateNewDecisionMemory,
  mapCanonicalCategory,
  revalidateHumanApprovals,
  resolveHumanDecisionContract,
  safeWebsite,
  sidecarFields,
  validateHumanDecisionRows,
};
