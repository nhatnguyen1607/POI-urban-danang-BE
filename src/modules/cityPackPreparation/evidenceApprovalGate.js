const { stableHash } = require('./incrementalSync');
const { normalizeCategory, normalizeName } = require('./sourceRecord');

const EVIDENCE_LABELS = Object.freeze({
  MATCH: 'MATCH',
  NOT_MATCH: 'NOT_MATCH',
  UNCERTAIN: 'UNCERTAIN',
  DUPLICATE_SAME_PLACE: 'DUPLICATE_SAME_PLACE',
  SEPARATE_PLACE: 'SEPARATE_PLACE',
  STRONG_CREATE_CANDIDATE: 'STRONG_CREATE_CANDIDATE',
  REVIEW_REQUIRED: 'REVIEW_REQUIRED',
  REJECT: 'REJECT',
});

const EVIDENCE_CONFIDENCE = Object.freeze({
  STRONG: 'STRONG',
  MODERATE: 'MODERATE',
  WEAK: 'WEAK',
});

const REVIEW_DECISIONS = Object.freeze({
  APPROVE: 'APPROVE',
  REJECT: 'REJECT',
  DEFER: 'DEFER',
});

const APPLY_OPERATIONS = Object.freeze({
  ENRICH_EXISTING: 'ENRICH_EXISTING',
  CREATE_NEW: 'CREATE_NEW',
  KEEP_SEPARATE: 'KEEP_SEPARATE',
  MERGE_SOURCE_DUPLICATE: 'MERGE_SOURCE_DUPLICATE',
});

const FIELD_STATES = Object.freeze({
  SAFE_ADDITION: 'SAFE_ADDITION',
  SAME_VALUE: 'SAME_VALUE',
  CONFLICT_REVIEW: 'CONFLICT_REVIEW',
  LOW_CONFIDENCE: 'LOW_CONFIDENCE',
  LICENSE_RESTRICTED: 'LICENSE_RESTRICTED',
  REJECT: 'REJECT',
});

function normalizePhone(value) {
  const digits = String(value || '').replace(/\D/g, '');
  if (digits.length < 8) return null;
  if (digits.startsWith('84')) return `0${digits.slice(2)}`;
  return digits;
}

function domainOf(value) {
  if (!value) return null;
  try {
    return new URL(/^https?:\/\//i.test(value) ? value : `https://${value}`)
      .hostname.toLowerCase().replace(/^www\./, '');
  } catch {
    return null;
  }
}

function addressTokens(value) {
  return new Set(normalizeName(value).split(/\s+/).filter((token) => token.length > 1));
}

function tokenOverlap(left, right) {
  const leftTokens = addressTokens(left);
  const rightTokens = addressTokens(right);
  if (leftTokens.size === 0 || rightTokens.size === 0) return 0;
  const shared = [...leftTokens].filter((token) => rightTokens.has(token)).length;
  return Number((shared / Math.min(leftTokens.size, rightTokens.size)).toFixed(4));
}

function haversineMeters(left, right) {
  const lat1 = Number(left?.latitude ?? left?.lat);
  const lon1 = Number(left?.longitude ?? left?.lng);
  const lat2 = Number(right?.latitude ?? right?.lat);
  const lon2 = Number(right?.longitude ?? right?.lng);
  if (![lat1, lon1, lat2, lon2].every(Number.isFinite)) return null;
  const radians = (value) => value * Math.PI / 180;
  const deltaLat = radians(lat2 - lat1);
  const deltaLon = radians(lon2 - lon1);
  const a = Math.sin(deltaLat / 2) ** 2
    + Math.cos(radians(lat1)) * Math.cos(radians(lat2)) * Math.sin(deltaLon / 2) ** 2;
  return Number((6371000 * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))).toFixed(2));
}

function sharedExternalIds(left, right) {
  const ignored = new Set(['dataset', 'osm_type', 'source_type']);
  const leftValues = Object.entries(left?.externalIds || {})
    .filter(([key, value]) => !ignored.has(key) && value !== null && value !== '')
    .map(([key, value]) => [key, String(value)]);
  const rightValues = Object.entries(right?.externalIds || {})
    .filter(([key, value]) => !ignored.has(key) && value !== null && value !== '')
    .map(([key, value]) => [key, String(value)]);
  return leftValues.flatMap(([leftKey, leftValue]) => rightValues
    .filter(([, rightValue]) => rightValue === leftValue)
    .map(([rightKey]) => `${leftKey}:${rightKey}`));
}

function evaluateMatchEvidence({ match, sourceRecord, canonicalRecord, historicalRecord }) {
  const resolverDistance = Number(match?.bestCandidate?.distanceMeters);
  const historicalDistance = historicalRecord ? haversineMeters(sourceRecord, historicalRecord) : null;
  const exactIds = sharedExternalIds(sourceRecord, canonicalRecord);
  const sourcePhone = normalizePhone(sourceRecord?.phone);
  const historicalPhone = normalizePhone(historicalRecord?.phone);
  const samePhone = Boolean(sourcePhone && historicalPhone && sourcePhone === historicalPhone);
  const sourceDomain = domainOf(sourceRecord?.website);
  const canonicalDomain = domainOf(canonicalRecord?.website);
  const sameWebsite = Boolean(sourceDomain && canonicalDomain && sourceDomain === canonicalDomain);
  const historicalAddressOverlap = historicalRecord
    ? tokenOverlap(sourceRecord?.address, historicalRecord?.address)
    : 0;
  const canonicalAddressOverlap = tokenOverlap(sourceRecord?.address, canonicalRecord?.address);
  const historicalLink = Boolean(historicalRecord && canonicalRecord?.evidenceMetadata
    ?.restaurantIdIsSourceIdentifier === historicalRecord.placeId);
  const compatibleLocation = historicalDistance !== null && historicalDistance <= 150;
  const categoryCompatible = normalizeCategory(sourceRecord?.category)
    === normalizeCategory(canonicalRecord?.category);
  const reasons = [];

  if (exactIds.length > 0) reasons.push('exact_stable_cross_source_identifier');
  if (samePhone && compatibleLocation && historicalLink) {
    reasons.push('historical_lineage_exact_phone_compatible_location');
  }
  if (sameWebsite && Number.isFinite(resolverDistance) && resolverDistance <= 100) {
    reasons.push('same_official_domain_compatible_location');
  }
  if (
    historicalLink
    && compatibleLocation
    && historicalAddressOverlap >= 0.4
    && categoryCompatible
  ) {
    reasons.push('historical_lineage_address_location_category');
  }

  const strong = exactIds.length > 0
    || (samePhone && compatibleLocation && historicalLink)
    || (sameWebsite && Number.isFinite(resolverDistance) && resolverDistance <= 100);
  const moderate = !strong && reasons.includes('historical_lineage_address_location_category');
  return {
    label: strong || moderate ? EVIDENCE_LABELS.MATCH : EVIDENCE_LABELS.UNCERTAIN,
    confidence: strong
      ? EVIDENCE_CONFIDENCE.STRONG
      : moderate
        ? EVIDENCE_CONFIDENCE.MODERATE
        : EVIDENCE_CONFIDENCE.WEAK,
    independent: strong || moderate,
    reasonCodes: reasons.length > 0 ? reasons : ['insufficient_independent_evidence'],
    checks: {
      exactExternalId: exactIds.length > 0,
      samePhone,
      sameWebsite,
      historicalLink,
      historicalAddressOverlap,
      canonicalAddressOverlap,
      historicalDistanceMeters: historicalDistance,
      resolverDistanceMeters: Number.isFinite(resolverDistance) ? resolverDistance : null,
      categoryCompatible,
    },
  };
}

function calculateEvidenceGate(cases, { minimum, requiredPrecision }) {
  const usable = cases.filter((item) => item.evidence.independent
    && [EVIDENCE_LABELS.MATCH, EVIDENCE_LABELS.NOT_MATCH].includes(item.evidence.label));
  const correct = usable.filter((item) => item.evidence.label === EVIDENCE_LABELS.MATCH).length;
  const precision = usable.length ? Number((correct / usable.length).toFixed(4)) : null;
  return {
    usable: usable.length,
    correct,
    precision,
    uncertain: cases.length - usable.length,
    minimum,
    requiredPrecision,
    status: usable.length < minimum
      ? 'INSUFFICIENT_EVIDENCE'
      : precision >= requiredPrecision ? 'PASS' : 'FAIL',
  };
}

function revalidateDuplicateEvidence(item) {
  const evidence = item?.evidenceSupported || {};
  const allowedTypes = new Set([
    'EXACT_STABLE_EXTERNAL_IDENTIFIER',
    'EXACT_PHONE_LINKAGE',
    'EXACT_WEBSITE_NAME_LOCATION',
  ]);
  const distance = Number(item?.candidateEvidence?.distanceMeters);
  const independent = evidence.status === 'EVIDENCE_SUPPORTED'
    && evidence.label === 'DUPLICATE_SAME_PLACE'
    && allowedTypes.has(evidence.evidenceType)
    && Number.isFinite(distance)
    && distance <= 150;
  return {
    label: independent ? EVIDENCE_LABELS.DUPLICATE_SAME_PLACE : EVIDENCE_LABELS.UNCERTAIN,
    confidence: independent ? EVIDENCE_CONFIDENCE.STRONG : EVIDENCE_CONFIDENCE.WEAK,
    independent,
    reasonCodes: independent
      ? [`independent_${evidence.evidenceType.toLowerCase()}`, 'compatible_location']
      : ['duplicate_evidence_not_independent_or_location_incompatible'],
  };
}

function provenanceComplete(record) {
  return Boolean(record?.provenance?.source
    && record?.provenance?.sourceId
    && record?.license?.license
    && record?.license?.attribution);
}

function evaluateNewCandidate({ match, record, insideBoundary, sourceDuplicate }) {
  const nearestDistance = Number(match?.bestCandidate?.distanceMeters);
  const absentFromCanonical = !match?.bestCandidate || nearestDistance > 250;
  const appearsReal = normalizeName(record?.name).length >= 3
    && Number.isFinite(Number(record?.latitude))
    && Number.isFinite(Number(record?.longitude));
  const travelerUseful = new Set([
    'accommodation', 'attraction', 'bakery', 'bar', 'beach', 'cafe', 'market',
    'museum', 'park', 'restaurant', 'resort', 'shopping', 'tours', 'travel',
  ]).has(normalizeCategory(record?.category));
  const complete = provenanceComplete(record);
  const strong = appearsReal && insideBoundary && absentFromCanonical
    && complete && travelerUseful && !sourceDuplicate;
  const reject = !appearsReal || !insideBoundary || sourceDuplicate;
  return {
    label: strong
      ? EVIDENCE_LABELS.STRONG_CREATE_CANDIDATE
      : reject ? EVIDENCE_LABELS.REJECT : EVIDENCE_LABELS.REVIEW_REQUIRED,
    confidence: strong ? EVIDENCE_CONFIDENCE.STRONG : EVIDENCE_CONFIDENCE.WEAK,
    independent: strong,
    reasonCodes: [
      appearsReal ? 'appears_real_poi' : 'invalid_poi_signal',
      insideBoundary ? 'inside_approved_product_boundary' : 'outside_product_boundary',
      absentFromCanonical ? 'no_nearby_canonical_candidate' : 'canonical_overlap_review',
      complete ? 'provenance_license_complete' : 'provenance_license_incomplete',
      travelerUseful ? 'traveler_relevant' : 'low_traveler_value',
      sourceDuplicate ? 'source_duplicate' : 'no_source_duplicate',
    ],
    checks: {
      appearsReal,
      insideBoundary,
      absentFromCanonical,
      provenanceLicenseComplete: complete,
      travelerUseful,
      sourceDuplicate,
      nearestCanonicalDistanceMeters: Number.isFinite(nearestDistance) ? nearestDistance : null,
    },
  };
}

function normalizedComparable(value) {
  if (value === null || value === undefined || value === '') return null;
  if (typeof value === 'object') return stableHash(value);
  return normalizeName(value);
}

function classifyFieldEvidence({ field, oldValue, newValue, provenance, evidenceConfidence }) {
  if (newValue === null || newValue === undefined || newValue === '') return FIELD_STATES.REJECT;
  if (!provenance?.license && !provenance?.licenseName) return FIELD_STATES.LICENSE_RESTRICTED;
  if (evidenceConfidence === EVIDENCE_CONFIDENCE.WEAK) return FIELD_STATES.LOW_CONFIDENCE;
  if (oldValue === null || oldValue === undefined || oldValue === '') return FIELD_STATES.SAFE_ADDITION;
  if (normalizedComparable(oldValue) === normalizedComparable(newValue)) return FIELD_STATES.SAME_VALUE;
  if (['name', 'coordinates', 'category', 'externalIds'].includes(field)) {
    return FIELD_STATES.CONFLICT_REVIEW;
  }
  return FIELD_STATES.CONFLICT_REVIEW;
}

function allowedOperationForCase(item) {
  if (item.stratum === 'NEW_TIER_A') return APPLY_OPERATIONS.CREATE_NEW;
  if (item.stratum === 'SOURCE_DUPLICATE') return APPLY_OPERATIONS.MERGE_SOURCE_DUPLICATE;
  if (item.evidence?.label === EVIDENCE_LABELS.NOT_MATCH) return APPLY_OPERATIONS.KEEP_SEPARATE;
  return APPLY_OPERATIONS.ENRICH_EXISTING;
}

function validateReviewDecisions(decisions, casesById) {
  const errors = [];
  const seen = new Set();
  const validDecisions = new Set(Object.values(REVIEW_DECISIONS));
  const validOperations = new Set(Object.values(APPLY_OPERATIONS));
  for (const decision of decisions) {
    if (!casesById.has(decision.caseId)) errors.push(`unknown_case_id:${decision.caseId}`);
    if (seen.has(decision.caseId)) errors.push(`duplicate_decision:${decision.caseId}`);
    seen.add(decision.caseId);
    if (!validDecisions.has(decision.reviewerDecision)) {
      errors.push(`malformed_decision:${decision.caseId}`);
    }
    if (decision.approvedOperation && !validOperations.has(decision.approvedOperation)) {
      errors.push(`malformed_action:${decision.caseId}`);
    }
    const item = casesById.get(decision.caseId);
    if (decision.reviewerDecision === REVIEW_DECISIONS.APPROVE && item) {
      if (!decision.approvedOperation || decision.approvedOperation !== allowedOperationForCase(item)) {
        errors.push(`incompatible_approval_operation:${decision.caseId}`);
      }
      if (!item.provenanceComplete) errors.push(`missing_required_provenance:${decision.caseId}`);
    }
  }
  return { valid: errors.length === 0, errors };
}

function buildApprovedApplyPlan({ cases, decisions }) {
  const casesById = new Map(cases.map((item) => [item.caseId, item]));
  const validation = validateReviewDecisions(decisions, casesById);
  if (!validation.valid) throw new Error(validation.errors.join(','));
  const decisionById = new Map(decisions.map((item) => [item.caseId, item]));
  const operations = cases
    .filter((item) => decisionById.get(item.caseId)?.reviewerDecision === REVIEW_DECISIONS.APPROVE)
    .map((item) => {
      const decision = decisionById.get(item.caseId);
      const operation = decision.approvedOperation;
      return {
        operationId: stableHash({ caseId: item.caseId, operation }).slice(0, 20),
        caseId: item.caseId,
        operation,
        canonicalId: item.canonical?.id || null,
        sourceIds: item.sourceIds,
        fieldChanges: item.fieldChanges || [],
        provenance: item.provenance,
        license: item.license,
        reviewDecision: REVIEW_DECISIONS.APPROVE,
        rollback: operation === APPLY_OPERATIONS.CREATE_NEW
          ? { action: 'REMOVE_CREATED_RECORD', candidateId: item.source?.sourceId, sourceLineage: item.sourceIds }
          : operation === APPLY_OPERATIONS.MERGE_SOURCE_DUPLICATE
            ? { action: 'RESTORE_SOURCE_IDENTITIES', sourceIds: item.sourceIds }
            : { action: 'RESTORE_PREVIOUS_VALUES', previousValues: (item.fieldChanges || []).map((change) => ({ field: change.field, value: change.oldValue })) },
      };
    })
    .sort((left, right) => left.caseId.localeCompare(right.caseId));
  return {
    status: 'DRY_RUN_ONLY_NOT_APPLIED',
    canonicalWriteAuthorized: false,
    operations,
    deterministicHash: stableHash(operations),
  };
}

function dryRunApprovedPlan({ beforeCount, plan, deferredCount = 0, rejectedCount = 0 }) {
  const enrichments = plan.operations.filter((item) => item.operation === APPLY_OPERATIONS.ENRICH_EXISTING).length;
  const newRecords = plan.operations.filter((item) => item.operation === APPLY_OPERATIONS.CREATE_NEW).length;
  const conflicts = plan.operations.reduce((total, item) => total + (item.fieldChanges || [])
    .filter((change) => change.state === FIELD_STATES.CONFLICT_REVIEW).length, 0);
  const result = {
    dryRunOnly: true,
    beforeCount,
    proposedAfterCount: beforeCount + newRecords,
    enrichments,
    newRecords,
    conflicts,
    rejected: rejectedCount,
    deferred: deferredCount,
    mutationCount: enrichments + newRecords + plan.operations
      .filter((item) => item.operation === APPLY_OPERATIONS.MERGE_SOURCE_DUPLICATE).length,
  };
  return { ...result, deterministicHash: stableHash(result) };
}

function deterministicSelect(items, count, key = (item) => item.caseId) {
  return [...items].sort((left, right) => {
    const leftHash = stableHash(key(left));
    const rightHash = stableHash(key(right));
    return leftHash === rightHash
      ? String(key(left)).localeCompare(String(key(right)))
      : leftHash.localeCompare(rightHash);
  }).slice(0, count);
}

module.exports = {
  APPLY_OPERATIONS,
  EVIDENCE_CONFIDENCE,
  EVIDENCE_LABELS,
  FIELD_STATES,
  REVIEW_DECISIONS,
  buildApprovedApplyPlan,
  calculateEvidenceGate,
  classifyFieldEvidence,
  deterministicSelect,
  domainOf,
  dryRunApprovedPlan,
  evaluateMatchEvidence,
  evaluateNewCandidate,
  haversineMeters,
  normalizePhone,
  revalidateDuplicateEvidence,
  tokenOverlap,
  validateReviewDecisions,
};
