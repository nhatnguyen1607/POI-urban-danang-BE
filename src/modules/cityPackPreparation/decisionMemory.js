const crypto = require('node:crypto');

const DECISION_SOURCES = Object.freeze({
  HUMAN: 'HUMAN',
  POLICY: 'POLICY',
  SYSTEM_SAFETY: 'SYSTEM_SAFETY',
});

const HUMAN_DECISIONS = new Set(['APPROVE', 'REJECT', 'DEFER']);

function stableValue(value) {
  if (Array.isArray(value)) return value.map(stableValue);
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.keys(value).sort().map((key) => [key, stableValue(value[key])]),
    );
  }
  return value === undefined ? null : value;
}

function stableHash(value) {
  return crypto.createHash('sha256').update(JSON.stringify(stableValue(value))).digest('hex');
}

function normalizedIdentity(reviewCase) {
  return {
    caseId: reviewCase.caseId,
    source: reviewCase.source?.source || reviewCase.source,
    sourceId: reviewCase.source?.sourceId || reviewCase.sourceId,
    canonicalId: reviewCase.canonical?.id || reviewCase.canonicalId || null,
    sourceName: reviewCase.source?.normalizedName || reviewCase.source?.originalName || null,
    canonicalName: reviewCase.canonical?.normalizedName || reviewCase.canonical?.name || null,
    sourceCategory: reviewCase.source?.category || null,
    canonicalCategory: reviewCase.canonical?.category || null,
    sourceAddress: reviewCase.source?.address || null,
    canonicalAddress: reviewCase.canonical?.address || null,
    sourceExternalIds: reviewCase.source?.externalIds || null,
    canonicalExternalIds: reviewCase.canonical?.externalIds || null,
    sourceCoordinates: reviewCase.source ? {
      latitude: reviewCase.source.latitude ?? null,
      longitude: reviewCase.source.longitude ?? null,
    } : null,
    canonicalCoordinates: reviewCase.canonical ? {
      latitude: reviewCase.canonical.latitude ?? null,
      longitude: reviewCase.canonical.longitude ?? null,
    } : null,
  };
}

function buildIdentityFingerprint(reviewCase) {
  return stableHash(normalizedIdentity(reviewCase));
}

function buildFieldFingerprints(reviewCase) {
  const changes = new Map((reviewCase.fieldChanges || []).map((change) => [change.field, change]));
  return Object.fromEntries([
    'address', 'phone', 'website', 'openingHours', 'media', 'externalIds', 'provenance',
  ].map((field) => {
    const change = changes.get(field);
    return [field, stableHash(change ? {
      oldValue: change.oldValue ?? null,
      newValue: change.newValue ?? null,
      state: change.state || null,
      provenance: change.provenance || null,
    } : field === 'provenance' ? {
      provenance: reviewCase.provenance || null,
      license: reviewCase.license || null,
    } : null)];
  }));
}

function buildEvidenceFingerprint(reviewCase) {
  return stableHash({
    evidence: reviewCase.evidence || null,
    historicalEvidence: reviewCase.historicalEvidence || null,
    provenanceComplete: reviewCase.provenanceComplete === true,
  });
}

function createDecisionRecord({
  reviewCase,
  decision,
  decisionSource,
  decisionReference,
  policyVersion,
  resolverVersion,
  evidenceVersion,
  boundaryVersion,
  approvedFields = [],
  note = null,
}) {
  if (!Object.values(DECISION_SOURCES).includes(decisionSource)) {
    throw new Error(`Unsupported decision source: ${decisionSource}`);
  }
  if (!HUMAN_DECISIONS.has(decision)) throw new Error(`Unsupported decision: ${decision}`);
  if (decisionSource === DECISION_SOURCES.HUMAN && !HUMAN_DECISIONS.has(decision)) {
    throw new Error(`Unsupported human decision: ${decision}`);
  }

  return {
    schemaVersion: 'stage4l-decision-memory-v1',
    caseId: reviewCase.caseId,
    source: reviewCase.source?.source || reviewCase.source || null,
    sourceId: reviewCase.source?.sourceId || reviewCase.sourceId || null,
    canonicalId: reviewCase.canonical?.id || reviewCase.canonicalId || null,
    decision,
    decisionSource,
    decisionScope: decision === 'APPROVE' ? 'FIELD_SCOPED_ENRICHMENT' : 'CASE_REVIEW',
    decisionReference,
    identityFingerprint: buildIdentityFingerprint(reviewCase),
    fieldFingerprints: buildFieldFingerprints(reviewCase),
    evidenceFingerprint: buildEvidenceFingerprint(reviewCase),
    provenanceFingerprint: stableHash({
      provenance: reviewCase.provenance || null,
      license: reviewCase.license || null,
    }),
    approvedFields: [...new Set(approvedFields)].sort(),
    rejectedFields: decision === 'REJECT'
      ? Object.keys(buildFieldFingerprints(reviewCase)).sort() : [],
    versions: {
      policy: policyVersion,
      resolver: resolverVersion,
      evidence: evidenceVersion,
      boundary: boundaryVersion,
    },
    snapshotReference: reviewCase.provenance?.snapshotRef || null,
    lastValidatedSnapshot: reviewCase.provenance?.snapshotRef || null,
    reviewEvidenceReference: decisionReference,
    note,
  };
}

function findReusableDecision(memory, reviewCase, versions) {
  const record = [...memory].reverse().find((item) => item.caseId === reviewCase.caseId);
  if (!record) return { reusable: false, reason: 'NO_PRIOR_DECISION' };
  if (record.identityFingerprint !== buildIdentityFingerprint(reviewCase)) {
    return { reusable: false, reason: 'IDENTITY_FINGERPRINT_CHANGED', invalidatedFields: ['identity'] };
  }
  if (record.versions.boundary !== versions.boundary) {
    return { reusable: false, reason: 'BOUNDARY_VERSION_CHANGED', invalidatedFields: ['geographicEligibility'] };
  }
  if (record.versions.policy !== versions.policy) {
    return { reusable: false, reason: 'POLICY_VERSION_CHANGED', invalidatedFields: ['policy'] };
  }
  if (record.versions.resolver !== versions.resolver) {
    return { reusable: false, reason: 'RESOLVER_VERSION_CHANGED', invalidatedFields: ['identityResolution'] };
  }
  if (record.versions.evidence !== versions.evidence) {
    return { reusable: false, reason: 'EVIDENCE_VERSION_CHANGED', invalidatedFields: ['evidence'] };
  }

  const currentFields = buildFieldFingerprints(reviewCase);
  const changedFields = Object.keys(record.fieldFingerprints).filter(
    (field) => record.fieldFingerprints[field] !== currentFields[field],
  );
  const decisionFieldsChanged = record.decision === 'APPROVE'
    ? changedFields.filter((field) => record.approvedFields.includes(field))
    : changedFields;
  if (decisionFieldsChanged.length > 0) {
    return { reusable: false, reason: 'FIELD_FINGERPRINT_CHANGED', invalidatedFields: decisionFieldsChanged };
  }

  if (record.decision === 'DEFER' && record.evidenceFingerprint !== buildEvidenceFingerprint(reviewCase)) {
    return { reusable: false, reason: 'EVIDENCE_CHANGED', invalidatedFields: ['evidence'] };
  }

  const outcome = record.decision === 'APPROVE'
    ? 'REUSE_APPROVAL'
    : record.decision === 'REJECT' ? 'REUSE_REJECTION' : 'DEFER';
  return {
    reusable: true,
    outcome,
    record,
    changedFields,
    suppressedReview: record.decision === 'DEFER',
  };
}

module.exports = {
  DECISION_SOURCES,
  buildEvidenceFingerprint,
  buildFieldFingerprints,
  buildIdentityFingerprint,
  createDecisionRecord,
  findReusableDecision,
  stableHash,
  stableValue,
};
