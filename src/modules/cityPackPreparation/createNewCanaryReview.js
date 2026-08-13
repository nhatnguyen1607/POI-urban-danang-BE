const {
  BOUNDARY_CLASSIFICATIONS,
  classifyAdministrativeBoundary,
} = require('./administrativeBoundary');
const {
  categoryCompatibility,
  haversineMeters,
  nameSimilarity,
} = require('./entityResolution');
const {
  buildIdentityProfile,
  hasBranchConflict,
  normalizeVietnameseText,
  tokenOverlap,
} = require('./historicalNormalization');
const { stableHash } = require('./incrementalSync');
const {
  CATEGORY_ELIGIBILITY,
  classifyTravelerCategory,
} = require('./reviewApplyPreparation');
const { branchAssessment, provenanceAssessment } = require('./createNewResearchPolicy');
const {
  EVIDENCE_CLASSES,
  EXISTENCE_CLASSES,
  classifyFreshness,
  isOfficialDomain,
  normalizeDomain,
  normalizePhone,
  secondPassExistenceSearch,
} = require('./createNewEvidencePolicy');
const { candidateId } = require('./cityPackBuild/cityPackBuilder');
const { hasValidCoordinates, normalizeCategory } = require('./sourceRecord');

const REVIEW_DECISIONS = new Set(['APPROVE', 'REJECT', 'DEFER']);
const RECOMMENDATIONS = Object.freeze({
  APPROVE: 'RECOMMEND_APPROVE',
  REVIEW: 'RECOMMEND_REVIEW',
  REJECT: 'RECOMMEND_REJECT',
});
const SUPPORT_CLASSES = Object.freeze({
  STRONG: 'STRONG_INDEPENDENT',
  MODERATE: 'MODERATE_INDEPENDENT',
  SINGLE: 'SINGLE_SOURCE_STRONG',
  SHARED: 'SAME_UPSTREAM_ONLY',
});
const CATEGORY_BUCKETS = Object.freeze({
  FOOD: 'foodCafe',
  ACCOMMODATION: 'accommodation',
  CULTURAL: 'attractionCultural',
  SHOPPING: 'shoppingMarket',
  OTHER: 'otherTraveler',
});

const NON_TRAVELER_NAME_PATTERNS = [
  /\b(hoc sinh|truong th|truong thcs|truong thpt|truong dai hoc|school|university|kindergarten|montessori)\b/,
  /\b(toa an|court|uy ban|cong an|police|government office|benh vien|hospital|clinic|dentist)\b/,
  /\b(kho hang|warehouse|tram bien ap|substation|nha may|factory)\b/,
  /\b(trung tam anh ngu|english center|language center|research centre|research center)\b/,
  /\b(media and education|media education|real estate|bat dong san|tati land|container)\b/,
  /\b(gioi tre|youth group)\b/,
];

function assertPreparationOnly(config) {
  if (config.autoCreateNew !== false || config.canonicalWrites !== 0
    || config.deleteOperations !== 0 || config.runtimeExposure !== false) {
    throw new Error('Stage 4R must remain preparation-only with AUTO_CREATE_NEW disabled.');
  }
}

function sourceIdentity(record) {
  return `${record.source}:${record.sourceId}`;
}

function categoryBucket(category) {
  const normalized = normalizeCategory(category);
  if (['restaurant', 'cafe', 'bakery', 'bar'].includes(normalized)) return CATEGORY_BUCKETS.FOOD;
  if (['accommodation', 'hotel', 'hostel', 'resort'].includes(normalized)) {
    return CATEGORY_BUCKETS.ACCOMMODATION;
  }
  if (['attraction', 'beach', 'bridge', 'museum', 'place_of_worship', 'park', 'arts']
    .includes(normalized)) return CATEGORY_BUCKETS.CULTURAL;
  if (['market', 'shopping', 'grocery', 'bookstore', 'convenience'].includes(normalized)) {
    return CATEGORY_BUCKETS.SHOPPING;
  }
  return CATEGORY_BUCKETS.OTHER;
}

function nameIsTravelerRelevant(name) {
  const normalized = normalizeVietnameseText(name);
  return normalized.length >= 3 && !NON_TRAVELER_NAME_PATTERNS.some((pattern) => pattern.test(normalized));
}

function categoryNameConsistent(record) {
  const category = normalizeCategory(record.category);
  const name = normalizeVietnameseText(record.name);
  if (category === 'accommodation'
    && /\b(park|cong vien|suoi|waterfall|nui|mountain|beach|bai bien)\b/.test(name)
    && !/\b(hotel|homestay|house|villa|resort|apartment|hostel|inn|lodge)\b/.test(name)) return false;
  if (category === 'attraction'
    && /\b(hotel|homestay|house|villa|resort|apartment|can ho|hostel|inn|lodge)\b/.test(name)) return false;
  const otherCategories = new Set(['travel', 'tours', 'sports', 'information', 'holiday', 'event', 'community']);
  if (otherCategories.has(category)) {
    return /\b(travel|tour|sport|visitor|information|holiday|event|convention|wedding|du lich)\b/.test(name);
  }
  return true;
}

function addressConflictsDaNang(record) {
  const address = normalizeVietnameseText(record.address);
  return /\b(ha noi|hanoi|ho chi minh|sai gon|saigon|hai phong|can tho)\b/.test(address)
    && !/\b(da nang|danang)\b/.test(address);
}

function metadataQuality(record) {
  return [record.address, normalizePhone(record.phone), isOfficialDomain(record.website), record.openingHours]
    .filter(Boolean).length;
}

function supportClass(stage4qResult) {
  const independent = stage4qResult.evidenceDetails?.independentEvidence || {};
  if (independent.strongCrossSource) return SUPPORT_CLASSES.STRONG;
  if (stage4qResult.supportIndependence === 'INDEPENDENT_SUPPORT'
    && stage4qResult.crossSourceSupport?.length) return SUPPORT_CLASSES.MODERATE;
  if (stage4qResult.supportIndependence === 'SHARED_UPSTREAM_SUPPORT') return SUPPORT_CLASSES.SHARED;
  return SUPPORT_CLASSES.SINGLE;
}

function currentCanonicalResult(record, existenceIndex) {
  const result = secondPassExistenceSearch(record, existenceIndex.combined, existenceIndex.thresholds);
  const nearestCanonical = result.candidates
    .filter((candidate) => candidate.kind === 'canonical')
    .sort((left, right) => left.distanceMeters - right.distanceMeters
      || String(left.id).localeCompare(String(right.id)))[0] || null;
  return { ...result, nearestCanonical };
}

function stableProposedCanonicalId(record, cityId) {
  return candidateId(cityId, sourceIdentity(record));
}

function duplicatePair(left, right, config) {
  const leftPhone = normalizePhone(left.record.phone);
  const rightPhone = normalizePhone(right.record.phone);
  const leftDomain = normalizeDomain(left.record.website);
  const rightDomain = normalizeDomain(right.record.website);
  const distanceMeters = haversineMeters(
    left.record.latitude, left.record.longitude, right.record.latitude, right.record.longitude,
  );
  const nameScore = nameSimilarity(left.record.name, right.record.name);
  const categoryScore = categoryCompatibility(left.record.category, right.record.category);
  const branchConflict = hasBranchConflict(left.record.name, right.record.name);
  const exactPhone = Boolean(leftPhone && leftPhone === rightPhone);
  const exactDomain = Boolean(leftDomain && leftDomain === rightDomain
    && isOfficialDomain(left.record.website) && isOfficialDomain(right.record.website));
  const nearSameIdentity = distanceMeters <= config.duplicateDistanceMeters
    && nameScore >= config.duplicateNameSimilarity && categoryScore > 0 && !branchConflict;
  const culturalCategories = new Set(['attraction', 'beach', 'bridge', 'museum', 'place_of_worship', 'park']);
  const nearLandmarkIdentity = distanceMeters <= config.duplicateDistanceMeters
    && culturalCategories.has(normalizeCategory(left.record.category))
    && normalizeCategory(left.record.category) === normalizeCategory(right.record.category)
    && (nameScore >= 0.5 || tokenOverlap(left.record.name, right.record.name) >= 0.3)
    && !branchConflict;
  if (!exactPhone && !exactDomain && !nearSameIdentity && !nearLandmarkIdentity) return null;
  return {
    leftCaseId: left.caseId,
    rightCaseId: right.caseId,
    distanceMeters: Number(distanceMeters.toFixed(2)),
    nameSimilarity: Number(nameScore.toFixed(4)),
    exactPhone,
    exactOfficialDomain: exactDomain,
    branchConflict,
    reason: exactPhone ? 'same_phone' : exactDomain ? 'same_official_domain'
      : nearLandmarkIdentity ? 'near_same_landmark_identity' : 'near_same_name_category',
  };
}

function evidencePriority(candidate) {
  const supportRank = {
    [SUPPORT_CLASSES.STRONG]: 4,
    [SUPPORT_CLASSES.MODERATE]: 3,
    [SUPPORT_CLASSES.SINGLE]: 2,
    [SUPPORT_CLASSES.SHARED]: 1,
  }[candidate.supportClass] || 0;
  return [supportRank, metadataQuality(candidate.record), Number(Boolean(candidate.record.address)),
    Number(Boolean(normalizePhone(candidate.record.phone))), Number(isOfficialDomain(candidate.record.website))];
}

function compareCandidates(left, right) {
  const leftRank = evidencePriority(left);
  const rightRank = evidencePriority(right);
  for (let index = 0; index < leftRank.length; index += 1) {
    if (leftRank[index] !== rightRank[index]) return rightRank[index] - leftRank[index];
  }
  return stableHash(left.caseId).localeCompare(stableHash(right.caseId))
    || left.caseId.localeCompare(right.caseId);
}

function deduplicateEligiblePool(candidates, config) {
  const ordered = [...candidates].sort(compareCandidates);
  const retained = [];
  const removed = [];
  for (const candidate of ordered) {
    const duplicate = retained.map((existing) => ({ existing, pair: duplicatePair(candidate, existing, config) }))
      .find((item) => item.pair);
    if (duplicate) {
      removed.push({ candidate, retainedCaseId: duplicate.existing.caseId, duplicate: duplicate.pair });
    } else {
      retained.push(candidate);
    }
  }
  return { retained, removed };
}

function candidateFingerprint(candidate, config) {
  return stableHash({
    source: candidate.record.source,
    sourceId: candidate.record.sourceId,
    name: candidate.record.name,
    coordinates: [candidate.record.latitude, candidate.record.longitude],
    category: candidate.record.category,
    address: candidate.record.address || null,
    phone: normalizePhone(candidate.record.phone),
    domain: normalizeDomain(candidate.record.website),
    supportClass: candidate.supportClass,
    crossSourceSupport: candidate.stage4qResult.crossSourceSupport || [],
    canonicalAbsence: candidate.currentExistence.existenceClass,
    canonicalCandidates: candidate.currentExistence.candidates,
    duplicateStatus: candidate.duplicateStatus,
    provenance: candidate.record.provenance,
    policyVersion: config.policyVersion,
  });
}

function evaluateCandidateForReview({ record, stage4qResult, existenceIndex, boundary, config, now }) {
  const currentExistence = currentCanonicalResult(record, existenceIndex);
  const boundaryClass = classifyAdministrativeBoundary(record, boundary);
  const provenance = provenanceAssessment(record);
  const freshness = classifyFreshness(record, now, existenceIndex.thresholds);
  const travelerRelevance = classifyTravelerCategory(record.category);
  const branch = branchAssessment(record);
  const support = supportClass(stage4qResult);
  const validCore = Boolean(record.name && hasValidCoordinates(record)
    && !['unknown', ''].includes(normalizeCategory(record.category)));
  const stableIdentity = Boolean(record.source && record.sourceId);
  const independentPath = support === SUPPORT_CLASSES.STRONG
    || support === SUPPORT_CLASSES.MODERATE
    || (support === SUPPORT_CLASSES.SINGLE && normalizePhone(record.phone)
      && isOfficialDomain(record.website) && (record.address || record.openingHours));
  const duplicateStatus = stage4qResult.duplicateStatus || 'UNRESOLVED_DUPLICATE_RISK';
  const reasonCodes = [];
  let recommendation = RECOMMENDATIONS.APPROVE;

  const requiredBoundary = [BOUNDARY_CLASSIFICATIONS.INSIDE_DANANG,
    BOUNDARY_CLASSIFICATIONS.BOUNDARY_EDGE].includes(boundaryClass);
  if (!requiredBoundary) reasonCodes.push('outside_or_invalid_danang_polygon');
  if (currentExistence.existenceClass !== EXISTENCE_CLASSES.CLEAR) {
    reasonCodes.push('current_canonical_or_historical_absence_not_clear');
  }
  if (travelerRelevance !== CATEGORY_ELIGIBILITY.TRAVELER_RELEVANT
    || !nameIsTravelerRelevant(record.name)) reasonCodes.push('traveler_relevance_hard_gate_failed');
  if (!categoryNameConsistent(record)) reasonCodes.push('category_name_semantics_conflict');
  if (addressConflictsDaNang(record)) reasonCodes.push('address_geography_conflict');
  if (!validCore) reasonCodes.push('invalid_core_identity_fields');
  if (!stableIdentity) reasonCodes.push('unstable_source_identity');
  if (provenance.rejected || !provenance.canaryStorageEligible) reasonCodes.push('provenance_or_license_gate_failed');
  if (duplicateStatus !== 'RESOLVED_OR_NONE') reasonCodes.push('stage4q_duplicate_risk_unresolved');
  if (['STALE_RISK', 'CLOSED_EVIDENCE'].includes(freshness.freshnessClass)) {
    reasonCodes.push('freshness_known_bad');
  }
  if (!branch.branchSpecific) reasonCodes.push('chain_branch_identity_insufficient');
  if (!independentPath) reasonCodes.push('independent_real_place_evidence_missing');
  if (stage4qResult.evidenceClass !== EVIDENCE_CLASSES.CANARY) {
    reasonCodes.push('stage4q_canary_gate_not_met');
  }
  if (reasonCodes.length) recommendation = reasonCodes.some((reason) => [
    'outside_or_invalid_danang_polygon', 'traveler_relevance_hard_gate_failed',
    'invalid_core_identity_fields', 'unstable_source_identity', 'provenance_or_license_gate_failed',
    'freshness_known_bad', 'category_name_semantics_conflict', 'address_geography_conflict',
  ].includes(reason)) ? RECOMMENDATIONS.REJECT : RECOMMENDATIONS.REVIEW;
  else reasonCodes.push('all_stage4r_selection_gates_passed');

  const result = {
    caseId: stage4qResult.caseId,
    record,
    stage4qResult,
    currentExistence,
    boundaryClass,
    supportClass: support,
    freshnessClass: freshness.freshnessClass,
    travelerRelevance,
    duplicateStatus,
    provenanceStatus: provenance.status,
    licenseStatus: provenance.canaryStorageEligible ? 'PASS' : 'REVIEW_OR_REJECT',
    evidenceConfidence: support === SUPPORT_CLASSES.STRONG ? 'VERY_HIGH'
      : support === SUPPORT_CLASSES.MODERATE ? 'HIGH' : 'HIGH_SINGLE_SOURCE',
    recommendation,
    reasonCodes: [...new Set(reasonCodes)].sort(),
    proposedCanonicalId: stableProposedCanonicalId(record, config.cityId),
    proposedOperation: 'CREATE_NEW',
    reviewerDecision: 'DEFER',
    reviewerNote: '',
  };
  result.decisionFingerprint = candidateFingerprint(result, config);
  return result;
}

function geoCell(candidate, config) {
  return `${Math.floor(candidate.record.latitude / config.geoCellDegrees)}:${Math.floor(
    candidate.record.longitude / config.geoCellDegrees,
  )}`;
}

function identityBase(candidate) {
  return buildIdentityProfile(candidate.record.name).baseName || normalizeVietnameseText(candidate.record.name);
}

function selectHumanCanary(candidates, config) {
  const eligible = candidates.filter((item) => item.recommendation === RECOMMENDATIONS.APPROVE);
  const deduped = deduplicateEligiblePool(eligible, config);
  const selected = [];
  const selectedIds = new Set();
  const baseCounts = new Map();
  const geoCounts = new Map();
  const canTake = (candidate) => {
    const base = identityBase(candidate);
    const cell = geoCell(candidate, config);
    return (baseCounts.get(base) || 0) < config.maximumPerIdentityBase
      && (geoCounts.get(cell) || 0) < config.maximumPerGeoCell;
  };
  const take = (candidate) => {
    if (selectedIds.has(candidate.caseId) || !canTake(candidate)) return false;
    selected.push(candidate);
    selectedIds.add(candidate.caseId);
    const base = identityBase(candidate);
    const cell = geoCell(candidate, config);
    baseCounts.set(base, (baseCounts.get(base) || 0) + 1);
    geoCounts.set(cell, (geoCounts.get(cell) || 0) + 1);
    return true;
  };
  const ordered = [...deduped.retained].sort(compareCandidates);
  for (const candidate of ordered.filter((item) => item.supportClass === SUPPORT_CLASSES.STRONG)) {
    if (selected.length >= config.maximumSelected) break;
    take(candidate);
  }
  for (const [bucket, target] of Object.entries(config.categoryTargets)) {
    for (const candidate of ordered.filter((item) => categoryBucket(item.record.category) === bucket)) {
      const count = selected.filter((item) => categoryBucket(item.record.category) === bucket).length;
      if (count >= target || selected.length >= config.maximumSelected) break;
      take(candidate);
    }
  }
  for (const candidate of ordered) {
    if (selected.length >= config.maximumSelected) break;
    take(candidate);
  }
  selected.sort(compareCandidates);
  const selectedDuplicatePairs = [];
  for (let left = 0; left < selected.length; left += 1) {
    for (let right = left + 1; right < selected.length; right += 1) {
      const duplicate = duplicatePair(selected[left], selected[right], config);
      if (duplicate) selectedDuplicatePairs.push(duplicate);
    }
  }
  if (selectedDuplicatePairs.length) throw new Error('Selected canary contains unresolved duplicate pairs.');
  return { selected, duplicateRemoved: deduped.removed, selectedDuplicatePairs };
}

function validateReviewerDecisions(rows, selectedCaseIds) {
  const expected = new Set(selectedCaseIds);
  const seen = new Set();
  for (const row of rows) {
    if (!expected.has(row.caseId)) throw new Error(`Unknown Stage 4R case: ${row.caseId}`);
    if (seen.has(row.caseId)) throw new Error(`Duplicate Stage 4R decision: ${row.caseId}`);
    if (!REVIEW_DECISIONS.has(row.reviewerDecision)) {
      throw new Error(`Invalid Stage 4R reviewer decision: ${row.reviewerDecision}`);
    }
    seen.add(row.caseId);
  }
  if (seen.size !== expected.size) throw new Error('Stage 4R decision file does not cover selected cases.');
  return true;
}

function buildDryRunCreatePlan(selected, decisions, config) {
  assertPreparationOnly(config);
  validateReviewerDecisions(decisions, selected.map((item) => item.caseId));
  const decisionById = new Map(decisions.map((item) => [item.caseId, item]));
  const proposals = selected.map((item) => {
    const decision = decisionById.get(item.caseId);
    return {
      caseId: item.caseId,
      source: item.record.source,
      sourceId: item.record.sourceId,
      proposedCanonicalId: item.proposedCanonicalId,
      coreFields: {
        name: item.record.name,
        category: normalizeCategory(item.record.category),
        latitude: item.record.latitude,
        longitude: item.record.longitude,
        address: item.record.address || null,
      },
      optionalEnrichments: {
        phone: item.record.phone || null,
        website: item.record.website || null,
        openingHours: item.record.openingHours || null,
      },
      fieldProvenance: item.record.provenance,
      license: item.record.license || item.record.provenance,
      reviewDecision: decision.reviewerDecision,
      decisionFingerprint: item.decisionFingerprint,
      rollback: { action: 'REMOVE_NEW_CANONICAL_ROW_AND_SIDECAR_RECORDS', applied: false },
    };
  });
  const approvedOperations = proposals.filter((item) => (
    item.reviewDecision === config.approvedOperationsRequiredDecision
  ));
  return {
    schemaVersion: 'stage4r-create-new-dry-run-plan-v1',
    status: 'DRY_RUN_NOT_APPLIED',
    policyVersion: config.policyVersion,
    proposals,
    approvedOperations,
    approvedCreateNewCount: approvedOperations.length,
    canonicalWritesExecuted: 0,
    runtimeInstalled: false,
    deterministicHash: stableHash(proposals),
  };
}

function summarizeSelection({ inputCount, independentInspected, evaluated, selection, plan }) {
  const selected = selection.selected;
  const supportCount = (support) => selected.filter((item) => item.supportClass === support).length;
  const recommendationCount = (value) => selected.filter((item) => item.recommendation === value).length;
  const bucketCount = (bucket) => selected.filter((item) => categoryBucket(item.record.category) === bucket).length;
  return {
    schemaVersion: 'stage4r-create-new-canary-summary-v1',
    inputCount,
    independentInspected,
    removedByCurrentCanonicalRecheck: evaluated.filter((item) => [
      EXISTENCE_CLASSES.POSSIBLE, EXISTENCE_CLASSES.LIKELY,
    ].includes(item.currentExistence.existenceClass)).length,
    removedByDuplicateRecheck: selection.duplicateRemoved.length,
    selectedCount: selected.length,
    categories: {
      foodCafe: bucketCount(CATEGORY_BUCKETS.FOOD),
      accommodation: bucketCount(CATEGORY_BUCKETS.ACCOMMODATION),
      attractionCultural: bucketCount(CATEGORY_BUCKETS.CULTURAL),
      shoppingMarket: bucketCount(CATEGORY_BUCKETS.SHOPPING),
      otherTraveler: bucketCount(CATEGORY_BUCKETS.OTHER),
    },
    support: {
      strongIndependent: supportCount(SUPPORT_CLASSES.STRONG),
      moderateIndependent: supportCount(SUPPORT_CLASSES.MODERATE),
      singleSourceStrong: supportCount(SUPPORT_CLASSES.SINGLE),
      sameUpstreamOnly: supportCount(SUPPORT_CLASSES.SHARED),
    },
    recommendations: {
      approve: recommendationCount(RECOMMENDATIONS.APPROVE),
      review: recommendationCount(RECOMMENDATIONS.REVIEW),
      reject: recommendationCount(RECOMMENDATIONS.REJECT),
    },
    approvedCreateNewOperations: plan.approvedCreateNewCount,
    autoCreateNew: false,
    canonicalWrites: 0,
    deletes: 0,
    deterministicHash: stableHash(selected.map((item) => ({
      caseId: item.caseId,
      proposedCanonicalId: item.proposedCanonicalId,
      fingerprint: item.decisionFingerprint,
    }))),
  };
}

module.exports = {
  CATEGORY_BUCKETS,
  RECOMMENDATIONS,
  REVIEW_DECISIONS,
  SUPPORT_CLASSES,
  assertPreparationOnly,
  buildDryRunCreatePlan,
  candidateFingerprint,
  categoryBucket,
  categoryNameConsistent,
  compareCandidates,
  deduplicateEligiblePool,
  duplicatePair,
  evaluateCandidateForReview,
  nameIsTravelerRelevant,
  selectHumanCanary,
  stableProposedCanonicalId,
  summarizeSelection,
  supportClass,
  validateReviewerDecisions,
};
