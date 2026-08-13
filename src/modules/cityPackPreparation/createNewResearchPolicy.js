const { classifyAdministrativeBoundary, BOUNDARY_CLASSIFICATIONS } = require('./administrativeBoundary');
const {
  DECISIONS,
  buildSpatialIndex,
  categoryCompatibility,
  detectSpatialDuplicates,
  haversineMeters,
  nameSimilarity,
  nearbySpatialRecords,
} = require('./entityResolution');
const { stableHash } = require('./incrementalSync');
const {
  CATEGORY_ELIGIBILITY,
  NEW_TIERS,
  classifyTravelerCategory,
  triageNewCandidate,
} = require('./reviewApplyPreparation');
const { normalizeName } = require('./sourceRecord');

const CREATE_NEW_STATES = Object.freeze({
  CANARY: 'CREATE_CANARY_ELIGIBLE',
  STRONG: 'NEW_STRONG_CANDIDATE',
  REVIEW: 'NEW_REVIEW',
  LOW_CONFIDENCE: 'NEW_LOW_CONFIDENCE',
  REJECT: 'NEW_REJECT',
  POSSIBLE_EXISTING: 'NEW_POSSIBLE_EXISTING_MATCH',
});

const RESEARCH_LABELS = Object.freeze({
  CONFIRMED: 'CONFIRMED_REAL_AND_ABSENT',
  LIKELY_REAL: 'LIKELY_REAL_NEEDS_REVIEW',
  LIKELY_EXISTING: 'LIKELY_EXISTING_CANONICAL',
  DUPLICATE: 'DUPLICATE_SOURCE',
  LOW_VALUE: 'LOW_VALUE',
  INVALID: 'INVALID',
  UNCERTAIN: 'UNCERTAIN',
});

const CHAIN_TOKENS = [
  'circle k', 'cong', 'gong cha', 'highlands', 'jollibee', 'kfc', 'lotteria',
  'mixue', 'pizza hut', 'starbucks', 'the house',
];

function assertResearchOnly(config) {
  if (config.autoCreateNew !== false || config.canonicalWrites !== 0
    || config.deleteOperations !== 0 || config.runtimeExposure !== false) {
    throw new Error('AUTO_CREATE_NEW must remain disabled for Stage 4P research.');
  }
}

function normalizedDomain(value) {
  if (!value) return null;
  try { return new URL(value).hostname.toLowerCase().replace(/^www\./, ''); }
  catch (_) { return null; }
}

function latestObservedAt(record) {
  const values = [
    record.sourceUpdatedAt,
    record.provenance?.retrievedAt,
    ...(record.provenance?.upstreamSources || []).map((item) => item.updateTime),
  ].filter(Boolean).map((value) => new Date(value)).filter((value) => Number.isFinite(value.getTime()));
  return values.length ? new Date(Math.max(...values.map((value) => value.getTime()))).toISOString() : null;
}

function sourceConfidence(record) {
  const values = [
    record.confidence,
    record.raw?.properties?.confidence,
    ...(record.provenance?.upstreamSources || []).map((item) => item.confidence),
  ].map(Number).filter(Number.isFinite);
  return values.length ? Math.max(...values) : null;
}

function prepareCandidateRecord(candidate) {
  const provenance = candidate.provenance || {};
  return {
    ...candidate,
    normalizedName: normalizeName(candidate.name),
    latitude: Number(candidate.latitude),
    longitude: Number(candidate.longitude),
    externalIds: candidate.externalIds || {},
    provenance,
    license: {
      license: provenance.license || null,
      policyClass: provenance.policyClass || null,
      attribution: provenance.attribution || null,
      licenseUrl: provenance.licenseUrl || null,
    },
    confidence: sourceConfidence(candidate),
    sourceUpdatedAt: latestObservedAt(candidate),
    raw: { properties: { confidence: sourceConfidence(candidate) } },
  };
}

function externalIdentityKeys(record) {
  const keys = [];
  for (const [field, value] of Object.entries(record.externalIds || {})) {
    if (value !== null && value !== undefined && String(value).trim()) {
      const normalizedField = field.toLowerCase().replace(/_id$/, '');
      keys.push(`${normalizedField}:${String(value).trim().toLowerCase()}`);
    }
  }
  for (const value of [record.sourceId, ...(record.aliases || [])]) {
    if (!value) continue;
    const text = String(value).trim().toLowerCase();
    keys.push(`any:${text}`);
    const separator = text.indexOf(':');
    if (separator > 0) keys.push(`${text.slice(0, separator)}:${text.slice(separator + 1)}`);
  }
  return [...new Set(keys)];
}

function buildCanonicalExistenceIndex(canonicalPois) {
  const external = new Map();
  for (const poi of canonicalPois) {
    for (const key of externalIdentityKeys(poi)) {
      if (!external.has(key)) external.set(key, poi);
    }
  }
  return {
    external,
    spatial: buildSpatialIndex(canonicalPois, 0.01),
  };
}

function secondPassExistenceCheck(record, index, thresholds) {
  for (const key of externalIdentityKeys(record)) {
    const exact = index.external.get(key);
    if (exact) {
      return {
        source: record.source,
        sourceId: record.sourceId,
        decision: DECISIONS.HIGH_CONFIDENCE_MATCH,
        bestCandidate: {
          canonicalPoiId: exact.id,
          canonicalName: exact.name,
          distanceMeters: haversineMeters(record.latitude, record.longitude,
            exact.latitude, exact.longitude),
          nameSimilarity: nameSimilarity(record.name, exact.name),
          categoryCompatibility: categoryCompatibility(record.category, exact.category),
        },
        reasonCodes: ['exact_existing_external_or_alias_identifier'],
      };
    }
  }
  const candidates = nearbySpatialRecords(index.spatial, record.latitude, record.longitude, 1)
    .map((poi) => ({
      canonicalPoiId: poi.id,
      canonicalName: poi.name,
      distanceMeters: haversineMeters(record.latitude, record.longitude, poi.latitude, poi.longitude),
      nameSimilarity: nameSimilarity(record.name, poi.name),
      categoryCompatibility: categoryCompatibility(record.category, poi.category),
    }))
    .filter((item) => item.distanceMeters <= 800)
    .sort((left, right) => (
      right.nameSimilarity - left.nameSimilarity
      || left.distanceMeters - right.distanceMeters
      || left.canonicalPoiId.localeCompare(right.canonicalPoiId)
    ));
  const best = candidates[0] || null;
  const likelyExisting = Boolean(best
    && best.distanceMeters <= thresholds.possibleExistingDistanceMeters
    && best.nameSimilarity >= thresholds.possibleExistingNameSimilarity
    && best.categoryCompatibility > 0);
  return {
    source: record.source,
    sourceId: record.sourceId,
    decision: likelyExisting ? DECISIONS.PROBABLE_MATCH : DECISIONS.NEW_CANDIDATE,
    bestCandidate: best,
    reasonCodes: likelyExisting
      ? ['second_pass_name_category_spatial_existing_risk']
      : ['second_pass_no_supported_existing_match'],
  };
}

function metadataCount(record) {
  return [record.address, record.phone, record.website, record.openingHours].filter(Boolean).length;
}

function provenanceAssessment(record) {
  const complete = Boolean(
    record.provenance?.source
    && record.provenance?.sourceId
    && record.license?.license
    && record.license?.policyClass
    && record.license?.attribution,
  );
  const prohibited = record.license?.policyClass === 'PROHIBITED';
  const shareAlike = record.license?.policyClass === 'OPEN_SHAREALIKE_ISOLATED';
  return {
    complete,
    rejected: !complete || prohibited,
    canaryStorageEligible: complete && !prohibited && !shareAlike,
    status: !complete ? 'INCOMPLETE' : prohibited ? 'PROHIBITED'
      : shareAlike ? 'SHAREALIKE_REVIEW_REQUIRED' : 'PASS',
  };
}

function branchAssessment(record) {
  const normalized = normalizeName(record.name);
  const chain = CHAIN_TOKENS.find((token) => normalized.includes(token)) || null;
  if (!chain) return { chain: false, branchSpecific: true };
  const addressHasNumber = /(^|\s)\d{1,5}([a-z]?)(\s|,|$)/i.test(record.address || '');
  const branchSpecific = Boolean(
    Number.isFinite(record.latitude)
    && Number.isFinite(record.longitude)
    && record.sourceId
    && (addressHasNumber || record.phone || normalizedDomain(record.website)),
  );
  return { chain: true, chainName: chain, branchSpecific };
}

function freshnessScore(record, now, freshWithinDays) {
  const observed = latestObservedAt(record);
  if (!observed) return { score: 0.5, observedAt: null, reason: 'freshness_unknown' };
  const ageDays = Math.max(0, (new Date(now).getTime() - new Date(observed).getTime()) / 86400000);
  if (ageDays <= freshWithinDays) return { score: 1, observedAt: observed, reason: 'fresh_source_observation' };
  if (ageDays <= freshWithinDays * 2) return { score: 0.6, observedAt: observed, reason: 'aging_source_observation' };
  return { score: 0.25, observedAt: observed, reason: 'stale_source_observation' };
}

function qualityRank(record) {
  const provenance = provenanceAssessment(record);
  const sourceRank = { overture: 3, wikidata: 2, wikidata_wikimedia: 2, osm: 1 }[record.source] || 0;
  return [Number(provenance.complete), metadataCount(record), sourceRank, sourceConfidence(record) || 0];
}

function compareRepresentative(left, right) {
  const leftRank = qualityRank(left);
  const rightRank = qualityRank(right);
  for (let index = 0; index < leftRank.length; index += 1) {
    if (leftRank[index] !== rightRank[index]) return rightRank[index] - leftRank[index];
  }
  return String(left.sourceId).localeCompare(String(right.sourceId));
}

function buildDuplicateContext(records, duplicates) {
  const byId = new Map(records.map((record) => [record.sourceId, record]));
  const adjacency = new Map();
  for (const duplicate of duplicates) {
    for (const [left, right] of [[duplicate.sourceIdA, duplicate.sourceIdB], [duplicate.sourceIdB, duplicate.sourceIdA]]) {
      if (!adjacency.has(left)) adjacency.set(left, new Set());
      adjacency.get(left).add(right);
    }
  }
  const context = new Map();
  const visited = new Set();
  for (const sourceId of [...adjacency.keys()].sort()) {
    if (visited.has(sourceId)) continue;
    const stack = [sourceId];
    const component = [];
    while (stack.length) {
      const current = stack.pop();
      if (visited.has(current)) continue;
      visited.add(current);
      component.push(current);
      for (const neighbor of adjacency.get(current) || []) stack.push(neighbor);
    }
    const componentRecords = component.map((id) => byId.get(id)).filter(Boolean).sort(compareRepresentative);
    if (!componentRecords.length) continue;
    const representative = componentRecords[0].sourceId;
    const sources = [...new Set(componentRecords.map((item) => item.source))].sort();
    const independentSupport = sources.length >= 2;
    for (const record of componentRecords) {
      context.set(record.sourceId, {
        representative,
        duplicateRisk: record.sourceId === representative ? 0.2 : 1,
        supportSources: sources,
        independentSupport,
        componentSize: componentRecords.length,
      });
    }
  }
  return context;
}

function absenceScore(match, thresholds) {
  const nearest = match?.bestCandidate;
  if (!nearest) return 1;
  if (match.decision !== DECISIONS.NEW_CANDIDATE) return 0;
  if (nearest.distanceMeters >= 500) return 1;
  if (nearest.distanceMeters >= thresholds.strongAbsenceDistanceMeters) return 0.95;
  if (nearest.distanceMeters >= thresholds.possibleExistingDistanceMeters) return 0.82;
  return nearest.nameSimilarity >= thresholds.possibleExistingNameSimilarity ? 0.15 : 0.65;
}

function isPossibleExisting(match, thresholds) {
  if (!match) return false;
  if (match.decision !== DECISIONS.NEW_CANDIDATE) return true;
  const nearest = match.bestCandidate;
  return Boolean(nearest
    && nearest.distanceMeters <= thresholds.possibleExistingDistanceMeters
    && nearest.nameSimilarity >= thresholds.possibleExistingNameSimilarity);
}

function evaluateCandidate({ record, match, triage, duplicateContext, config, now }) {
  const reasons = [];
  const provenance = provenanceAssessment(record);
  const branch = branchAssessment(record);
  const freshness = freshnessScore(record, now, config.thresholds.freshWithinDays);
  const eligibility = classifyTravelerCategory(record.category);
  const possibleExisting = isPossibleExisting(match, config.thresholds);
  const duplicate = duplicateContext || {
    representative: record.sourceId, duplicateRisk: 0, supportSources: [record.source],
    independentSupport: false, componentSize: 1,
  };
  const nameUsable = normalizeName(record.name).length >= 3;
  const validCoordinates = Number.isFinite(record.latitude) && Number.isFinite(record.longitude);
  const realPlace = Math.min(1, 0.35 + metadataCount(record) * 0.15
    + (sourceConfidence(record) === null ? 0.1 : sourceConfidence(record) * 0.25));
  const sourceQuality = ({ overture: 0.9, osm: 0.8, wikidata: 0.85,
    wikidata_wikimedia: 0.85 }[record.source] || 0.5);
  const identityAbsence = absenceScore(match, config.thresholds);
  const crossSourceSupport = duplicate.independentSupport ? 1 : 0;
  const travelerValue = eligibility === CATEGORY_ELIGIBILITY.TRAVELER_RELEVANT ? 1
    : eligibility === CATEGORY_ELIGIBILITY.CONTEXTUAL ? 0.5 : 0;
  const provenanceScore = provenance.complete ? (provenance.canaryStorageEligible ? 1 : 0.7) : 0;
  const totalScore = Number(Math.max(0, Math.min(1,
    identityAbsence * 0.25 + realPlace * 0.2 + travelerValue * 0.15
    + sourceQuality * 0.1 + freshness.score * 0.1 + crossSourceSupport * 0.1
    + provenanceScore * 0.1 - duplicate.duplicateRisk * 0.2,
  )).toFixed(4));

  let state = CREATE_NEW_STATES.REVIEW;
  let label = RESEARCH_LABELS.UNCERTAIN;
  if (possibleExisting) {
    state = CREATE_NEW_STATES.POSSIBLE_EXISTING;
    label = RESEARCH_LABELS.LIKELY_EXISTING;
    reasons.push('possible_existing_canonical_second_pass');
  } else if (!validCoordinates || !nameUsable || eligibility === CATEGORY_ELIGIBILITY.EXCLUDED) {
    state = CREATE_NEW_STATES.REJECT;
    label = RESEARCH_LABELS.INVALID;
    reasons.push(!validCoordinates ? 'invalid_coordinates' : !nameUsable ? 'generic_or_missing_name' : 'excluded_category');
  } else if (provenance.rejected) {
    state = CREATE_NEW_STATES.REJECT;
    label = RESEARCH_LABELS.INVALID;
    reasons.push('provenance_rejected');
  } else if (duplicate.duplicateRisk === 1) {
    state = CREATE_NEW_STATES.REJECT;
    label = RESEARCH_LABELS.DUPLICATE;
    reasons.push('non_representative_source_duplicate');
  } else if (triage.tier === NEW_TIERS.C || triage.tier === NEW_TIERS.D
    || eligibility !== CATEGORY_ELIGIBILITY.TRAVELER_RELEVANT
    || !branch.branchSpecific || metadataCount(record) < 2) {
    state = CREATE_NEW_STATES.LOW_CONFIDENCE;
    label = eligibility === CATEGORY_ELIGIBILITY.LOW_VALUE ? RESEARCH_LABELS.LOW_VALUE : RESEARCH_LABELS.UNCERTAIN;
    reasons.push(!branch.branchSpecific ? 'chain_branch_identity_insufficient' : triage.reason || 'minimum_quality_not_met');
  } else if (triage.tier === NEW_TIERS.A && totalScore >= config.thresholds.minimumStrongScore) {
    state = CREATE_NEW_STATES.STRONG;
    label = RESEARCH_LABELS.LIKELY_REAL;
    reasons.push('precision_first_strong_review_candidate');
    if (totalScore >= config.thresholds.minimumCanaryScore
      && duplicate.independentSupport && provenance.canaryStorageEligible) {
      state = CREATE_NEW_STATES.CANARY;
      reasons.push('independent_cross_source_support', 'canary_research_only');
    }
  } else {
    reasons.push('human_review_required');
  }
  if (freshness.reason) reasons.push(freshness.reason);
  if (duplicate.independentSupport) reasons.push('cross_source_support_preserved');
  if (provenance.status === 'SHAREALIKE_REVIEW_REQUIRED') reasons.push('sharealike_license_review_required');
  if (branch.chain) reasons.push(branch.branchSpecific ? 'chain_branch_identity_sufficient' : 'chain_branch_identity_missing');

  return {
    caseId: `CREATE_NEW:${record.source}:${record.sourceId}`,
    source: record.source,
    sourceId: record.sourceId,
    name: record.name,
    category: record.category,
    latitude: record.latitude,
    longitude: record.longitude,
    address: record.address || null,
    phonePresent: Boolean(record.phone),
    websitePresent: Boolean(record.website),
    nearestCanonicalId: match?.bestCandidate?.canonicalPoiId || null,
    nearestCanonicalName: match?.bestCandidate?.canonicalName || null,
    distanceMeters: match?.bestCandidate?.distanceMeters ?? null,
    crossSourceSupport: duplicate.supportSources,
    travelerRelevance: eligibility,
    freshnessSignal: freshness,
    provenanceStatus: provenance.status,
    duplicateRisk: duplicate.duplicateRisk,
    evidenceLabel: label,
    evidenceConfidence: totalScore,
    reasonCodes: [...new Set(reasons)].sort(),
    recommendation: state,
    reviewerDecision: 'DEFER',
    reviewerNote: '',
    scorecard: {
      identityAbsenceConfidence: identityAbsence,
      realPlaceConfidence: Number(realPlace.toFixed(4)),
      travelerValue,
      sourceQuality,
      freshness: freshness.score,
      crossSourceSupport,
      provenanceCompleteness: provenanceScore,
      duplicateRisk: duplicate.duplicateRisk,
      total: totalScore,
    },
    identityFingerprint: buildCreateNewFingerprint(record, duplicate.supportSources),
  };
}

function buildCreateNewFingerprint(record, supportSources = []) {
  return stableHash({
    source: record.source,
    sourceId: record.sourceId,
    name: normalizeName(record.name),
    latitude: record.latitude,
    longitude: record.longitude,
    category: record.category,
    address: record.address || null,
    externalIds: record.externalIds || {},
    provenance: record.provenance || null,
    supportSources: [...supportSources].sort(),
  });
}

function createResearchDecisionRecord(result, decision, reference) {
  if (!['APPROVE', 'REJECT', 'DEFER'].includes(decision)) throw new Error(`Unsupported decision: ${decision}`);
  return {
    schemaVersion: 'stage4p-create-new-decision-memory-v1',
    caseId: result.caseId,
    decision,
    decisionScope: 'CREATE_NEW_RESEARCH_ONLY',
    decisionReference: reference,
    identityFingerprint: result.identityFingerprint,
    canonicalWriteAuthorized: false,
  };
}

function findReusableResearchDecision(memory, result) {
  const previous = [...memory].reverse().find((item) => item.caseId === result.caseId);
  if (!previous) return { reusable: false, reason: 'NO_PRIOR_DECISION' };
  if (previous.identityFingerprint !== result.identityFingerprint) {
    return { reusable: false, reason: 'CREATE_NEW_FINGERPRINT_CHANGED' };
  }
  return {
    reusable: true,
    outcome: previous.decision === 'APPROVE' ? 'REUSE_APPROVAL'
      : previous.decision === 'REJECT' ? 'REUSE_REJECTION' : 'SUPPRESS_UNCHANGED_DEFER',
    canonicalWriteAuthorized: false,
  };
}

function deterministicSelection(items, limit) {
  return [...items].sort((left, right) => (
    right.evidenceConfidence - left.evidenceConfidence
    || stableHash(left.caseId).localeCompare(stableHash(right.caseId))
    || left.caseId.localeCompare(right.caseId)
  )).slice(0, limit);
}

function buildCanaryReviewSample(results, config) {
  const target = config.canary.targetSampleSize;
  const groups = [
    [CREATE_NEW_STATES.CANARY, 35],
    [CREATE_NEW_STATES.STRONG, 20],
    [CREATE_NEW_STATES.POSSIBLE_EXISTING, 8],
    [CREATE_NEW_STATES.REJECT, 7],
    [CREATE_NEW_STATES.LOW_CONFIDENCE, 5],
  ];
  const selected = [];
  const seen = new Set();
  for (const [state, limit] of groups) {
    for (const item of deterministicSelection(results.filter((result) => result.recommendation === state), limit)) {
      if (!seen.has(item.caseId)) { seen.add(item.caseId); selected.push(item); }
    }
  }
  for (const item of deterministicSelection(results, target)) {
    if (selected.length >= target) break;
    if (!seen.has(item.caseId)) { seen.add(item.caseId); selected.push(item); }
  }
  return selected.sort((left, right) => left.caseId.localeCompare(right.caseId)).slice(0, target);
}

function summarizeResearch(results, context) {
  const count = (predicate) => results.filter(predicate).length;
  const outcomeCounts = Object.fromEntries(Object.values(CREATE_NEW_STATES).map((state) => [state, 0]));
  for (const result of results) outcomeCounts[result.recommendation] += 1;
  return {
    schemaVersion: 'stage4p-create-new-research-summary-v1',
    status: 'RESEARCH_ONLY_NON_RUNTIME_NOT_CANONICAL',
    totalInputCandidates: context.totalInputCandidates,
    polygonEligible: context.polygonEligible,
    totalNewCandidatesEvaluated: results.length,
    possibleExistingFiltered: outcomeCounts[CREATE_NEW_STATES.POSSIBLE_EXISTING],
    duplicateRiskFiltered: count((item) => item.reasonCodes.includes('non_representative_source_duplicate')),
    lowValueOrQualityRejected: outcomeCounts[CREATE_NEW_STATES.LOW_CONFIDENCE]
      + count((item) => item.recommendation === CREATE_NEW_STATES.REJECT
        && !item.reasonCodes.includes('provenance_rejected')
        && !item.reasonCodes.includes('non_representative_source_duplicate')),
    provenanceRejected: count((item) => item.reasonCodes.includes('provenance_rejected')),
    strongReviewCandidates: outcomeCounts[CREATE_NEW_STATES.STRONG]
      + outcomeCounts[CREATE_NEW_STATES.CANARY],
    strongestCanaryCandidates: outcomeCounts[CREATE_NEW_STATES.CANARY],
    outcomeCounts,
    autoCreateNew: false,
    canonicalWrites: 0,
    deletes: 0,
    deterministicHash: stableHash(results),
  };
}

function evaluateCreateNewResearch({ candidates, canonicalPois, boundary, config, now }) {
  assertResearchOnly(config);
  const records = candidates.map(prepareCandidateRecord);
  const boundaryRecords = records.map((record) => ({
    record,
    boundaryClassification: classifyAdministrativeBoundary(record, boundary),
  }));
  const eligible = boundaryRecords.filter((item) => [
    BOUNDARY_CLASSIFICATIONS.INSIDE_DANANG,
    BOUNDARY_CLASSIFICATIONS.BOUNDARY_EDGE,
  ].includes(item.boundaryClassification)).map((item) => item.record);
  const existenceIndex = buildCanonicalExistenceIndex(canonicalPois);
  const matches = eligible.map((record) => secondPassExistenceCheck(record, existenceIndex, config.thresholds));
  const matchById = new Map(matches.map((match) => [match.sourceId, match]));
  const boundaryById = new Map(boundaryRecords.map((item) => [item.record.sourceId, item.boundaryClassification]));
  const duplicates = detectSpatialDuplicates(eligible, {
    includeCrossSource: true,
    hardened: true,
  });
  const duplicateContext = buildDuplicateContext(eligible, duplicates);
  const duplicateSourceIds = new Set(duplicates.flatMap((item) => [item.sourceIdA, item.sourceIdB]));
  const results = [];
  for (const item of boundaryRecords) {
    const match = matchById.get(item.record.sourceId) || null;
    const triage = triageNewCandidate(item.record, match, {
      boundaryClassification: item.boundaryClassification,
      duplicateSourceIds,
    });
    results.push(evaluateCandidate({
      record: item.record,
      match,
      triage,
      duplicateContext: duplicateContext.get(item.record.sourceId),
      config,
      now,
    }));
  }
  const ordered = results.sort((left, right) => left.caseId.localeCompare(right.caseId));
  const summary = summarizeResearch(ordered, {
    totalInputCandidates: records.length,
    polygonEligible: eligible.length,
  });
  return {
    results: ordered,
    canary: buildCanaryReviewSample(ordered, config),
    summary,
    resolutionSummary: {
      highConfidenceMatches: matches.filter((item) => item.decision === DECISIONS.HIGH_CONFIDENCE_MATCH).length,
      probableMatches: matches.filter((item) => item.decision === DECISIONS.PROBABLE_MATCH).length,
      newCandidates: matches.filter((item) => item.decision === DECISIONS.NEW_CANDIDATE).length,
    },
    duplicatePairCount: duplicates.length,
  };
}

module.exports = {
  CREATE_NEW_STATES,
  RESEARCH_LABELS,
  assertResearchOnly,
  branchAssessment,
  buildCanonicalExistenceIndex,
  buildCanaryReviewSample,
  buildCreateNewFingerprint,
  createResearchDecisionRecord,
  evaluateCandidate,
  evaluateCreateNewResearch,
  findReusableResearchDecision,
  prepareCandidateRecord,
  provenanceAssessment,
  secondPassExistenceCheck,
  summarizeResearch,
};
