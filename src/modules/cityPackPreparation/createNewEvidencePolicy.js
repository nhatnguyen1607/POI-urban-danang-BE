const {
  buildSpatialIndex,
  categoryCompatibility,
  haversineMeters,
  nameSimilarity,
  nearbySpatialRecords,
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
const { hasValidCoordinates, normalizeCategory } = require('./sourceRecord');
const { provenanceAssessment } = require('./createNewResearchPolicy');

const EVIDENCE_CLASSES = Object.freeze({
  CANARY: 'CREATE_CANARY_ELIGIBLE',
  STRONG: 'STRONG_HUMAN_REVIEW',
  NORMAL: 'NORMAL_REVIEW',
  POSSIBLE_EXISTING: 'POSSIBLE_EXISTING',
  DUPLICATE: 'DUPLICATE_REVIEW',
  DEFER: 'DEFER',
  REJECT: 'REJECT',
});

const EXISTENCE_CLASSES = Object.freeze({
  CLEAR: 'CLEARLY_ABSENT',
  POSSIBLE: 'POSSIBLE_EXISTING',
  LIKELY: 'LIKELY_EXISTING',
  UNCERTAIN: 'UNCERTAIN_EXISTENCE',
});

const CONSENSUS_TYPES = Object.freeze({
  EXACT: 'EXACT_CROSS_SOURCE_ID',
  STRONG: 'STRONG_INDEPENDENT_CONSENSUS',
  MODERATE: 'MODERATE_CONSENSUS',
  SAME_UPSTREAM: 'SAME_UPSTREAM_ONLY',
  SINGLE: 'SINGLE_SOURCE_ONLY',
  CONFLICTING: 'CONFLICTING_SOURCES',
});

const INDEPENDENCE = Object.freeze({
  INDEPENDENT: 'INDEPENDENT_SUPPORT',
  SHARED: 'SHARED_UPSTREAM_SUPPORT',
  UNKNOWN: 'UNKNOWN_INDEPENDENCE',
});

const FRESHNESS_CLASSES = Object.freeze({
  GOOD: 'FRESHNESS_GOOD',
  UNKNOWN: 'FRESHNESS_UNKNOWN',
  STALE: 'STALE_RISK',
  CLOSED: 'CLOSED_EVIDENCE',
});

const GENERIC_DOMAINS = new Set([
  'facebook.com', 'foursquare.com', 'instagram.com', 'linktr.ee', 'maps.app.goo.gl',
  'tripadvisor.com', 'twitter.com', 'x.com', 'yelp.com', 'youtube.com',
]);

function assertResearchOnly(config) {
  if (config.autoCreateNew !== false || config.canonicalWrites !== 0
    || config.deleteOperations !== 0 || config.runtimeExposure !== false) {
    throw new Error('Stage 4Q must remain research-only with AUTO_CREATE_NEW disabled.');
  }
}

function normalizePhone(value) {
  if (!value) return null;
  let digits = String(value).replace(/(?:ext|extension|may le|nhanh).*$/i, '').replace(/\D/g, '');
  if (digits.startsWith('00')) digits = digits.slice(2);
  if (digits.startsWith('0') && digits.length >= 9 && digits.length <= 11) digits = `84${digits.slice(1)}`;
  if (digits.length < 9 || digits.length > 12) return null;
  return digits;
}

function normalizeDomain(value) {
  if (!value) return null;
  try {
    const url = new URL(/^https?:\/\//i.test(String(value)) ? String(value) : `https://${value}`);
    return url.hostname.toLowerCase().replace(/^www\./, '').replace(/\.$/, '') || null;
  } catch (_) {
    return null;
  }
}

function isOfficialDomain(value) {
  const domain = normalizeDomain(value);
  if (!domain) return false;
  return ![...GENERIC_DOMAINS].some((generic) => domain === generic || domain.endsWith(`.${generic}`));
}

function normalizeAddress(value) {
  return normalizeVietnameseText(value)
    .replace(/\b(tp|thanh pho) da nang\b/g, ' da nang ')
    .replace(/\b(q|quan)\s+(hai chau|son tra|ngu hanh son|thanh khe|lien chieu|cam le)\b/g, ' $2 ')
    .replace(/\b(p|phuong)\s+/g, ' ')
    .replace(/\b(duong|d)\s+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function valueTokenSimilarity(left, right) {
  const leftTokens = new Set(normalizeAddress(left).split(' ').filter(Boolean));
  const rightTokens = new Set(normalizeAddress(right).split(' ').filter(Boolean));
  if (!leftTokens.size || !rightTokens.size) return 0;
  const shared = [...leftTokens].filter((token) => rightTokens.has(token)).length;
  return shared / new Set([...leftTokens, ...rightTokens]).size;
}

function normalizeUpstreamName(value) {
  const normalized = normalizeVietnameseText(value).replace(/\s+/g, '');
  if (/openstreetmap|osm/.test(normalized)) return 'openstreetmap';
  if (/wikidata/.test(normalized)) return 'wikidata';
  if (/wikimedia|commons/.test(normalized)) return 'wikimedia';
  if (/facebook|meta/.test(normalized)) return 'meta';
  return normalized || null;
}

function sourceLineage(record) {
  const upstream = (record.provenance?.upstreamSources || [])
    .map((item) => normalizeUpstreamName(item.dataset || item.source || item.name))
    .filter(Boolean);
  if (record.source === 'overture') return { known: upstream.length > 0, keys: new Set(upstream) };
  const direct = normalizeUpstreamName(record.source);
  return { known: Boolean(direct), keys: new Set(direct ? [direct] : []) };
}

function classifySupportIndependence(left, right) {
  const leftLineage = sourceLineage(left);
  const rightLineage = sourceLineage(right);
  if (!leftLineage.known || !rightLineage.known) return INDEPENDENCE.UNKNOWN;
  const overlap = [...leftLineage.keys].some((key) => rightLineage.keys.has(key));
  return overlap ? INDEPENDENCE.SHARED : INDEPENDENCE.INDEPENDENT;
}

function recordKey(record) {
  return `${record.source}:${record.sourceId}`;
}

function identityKeys(record) {
  const keys = [];
  for (const [field, rawValue] of Object.entries(record.externalIds || {})) {
    const value = String(rawValue || '').trim().toLowerCase();
    if (!value) continue;
    const normalizedField = field.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/_id$/, '');
    keys.push(`${normalizedField}:${value}`);
  }
  for (const alias of record.aliases || []) {
    if (String(alias).trim()) keys.push(`global:${String(alias).trim().toLowerCase()}`);
  }
  if (record.globalId) keys.push(`global:${String(record.globalId).trim().toLowerCase()}`);
  if (record.evidenceKind === 'canonical' && record.id) {
    keys.push(`global:${String(record.id).trim().toLowerCase()}`);
  }
  return [...new Set(keys)].sort();
}

function sharedIdentityKeys(left, right) {
  const rightKeys = new Set(identityKeys(right));
  return identityKeys(left).filter((key) => rightKeys.has(key));
}

function buildExistenceIndex(canonicalPois, historicalPois = []) {
  const records = [
    ...canonicalPois.map((record) => ({ ...record, evidenceKind: 'canonical' })),
    ...historicalPois.map((record) => ({ ...record, evidenceKind: 'historical' })),
  ];
  const external = new Map();
  for (const record of records) {
    for (const key of identityKeys(record)) {
      if (!external.has(key)) external.set(key, []);
      external.get(key).push(record);
    }
  }
  return { external, spatial: buildSpatialIndex(records, 0.01) };
}

function scoreExistenceCandidate(candidate, evidence) {
  const distanceMeters = haversineMeters(
    candidate.latitude, candidate.longitude, evidence.latitude, evidence.longitude,
  );
  return {
    kind: evidence.evidenceKind,
    id: evidence.id || evidence.globalId || evidence.sourceId,
    name: evidence.name || null,
    distanceMeters: Number(distanceMeters.toFixed(2)),
    nameSimilarity: Number(nameSimilarity(candidate.name, evidence.name).toFixed(4)),
    categoryCompatibility: categoryCompatibility(candidate.category, evidence.category),
    addressSimilarity: Number(valueTokenSimilarity(candidate.address, evidence.address || evidence.district).toFixed(4)),
    sharedIdentityKeys: sharedIdentityKeys(candidate, evidence),
    branchConflict: hasBranchConflict(candidate.name, evidence.name),
  };
}

function secondPassExistenceSearch(candidate, index, thresholds) {
  const exactRecords = identityKeys(candidate).flatMap((key) => index.external.get(key) || []);
  const nearby = nearbySpatialRecords(index.spatial, candidate.latitude, candidate.longitude, 1)
    .map((record) => scoreExistenceCandidate(candidate, record))
    .filter((item) => item.distanceMeters <= thresholds.uncertainExistingDistanceMeters)
    .sort((left, right) => (
      right.sharedIdentityKeys.length - left.sharedIdentityKeys.length
      || right.nameSimilarity - left.nameSimilarity
      || left.distanceMeters - right.distanceMeters
      || String(left.id).localeCompare(String(right.id))
    ));
  const exact = exactRecords.map((record) => scoreExistenceCandidate(candidate, record));
  const candidates = [...new Map([...exact, ...nearby].map((item) => [`${item.kind}:${item.id}`, item])).values()]
    .sort((left, right) => (
      right.sharedIdentityKeys.length - left.sharedIdentityKeys.length
      || right.nameSimilarity - left.nameSimilarity
      || left.distanceMeters - right.distanceMeters
    ));
  const best = candidates[0] || null;
  let existenceClass = EXISTENCE_CLASSES.CLEAR;
  const reasonCodes = [];
  if (candidates.some((item) => item.sharedIdentityKeys.length > 0)) {
    existenceClass = EXISTENCE_CLASSES.LIKELY;
    reasonCodes.push('existing_exact_historical_or_canonical_identifier');
  } else if (best && best.distanceMeters <= thresholds.likelyExistingDistanceMeters
    && best.nameSimilarity >= thresholds.likelyExistingNameSimilarity
    && best.categoryCompatibility > 0 && !best.branchConflict) {
    existenceClass = EXISTENCE_CLASSES.LIKELY;
    reasonCodes.push('existing_near_name_category_agreement');
  } else if (best && best.distanceMeters <= thresholds.possibleExistingDistanceMeters
    && best.nameSimilarity >= thresholds.possibleExistingNameSimilarity
    && best.categoryCompatibility > 0) {
    existenceClass = EXISTENCE_CLASSES.POSSIBLE;
    reasonCodes.push('possible_existing_spatial_name_block');
  } else if (best && best.distanceMeters <= thresholds.uncertainExistingDistanceMeters
    && (best.nameSimilarity >= 0.45 || best.categoryCompatibility > 0)) {
    existenceClass = EXISTENCE_CLASSES.UNCERTAIN;
    reasonCodes.push('nearby_existing_requires_branch_review');
  } else {
    reasonCodes.push('no_supported_canonical_or_historical_match');
  }
  return { existenceClass, bestCandidate: best, candidates: candidates.slice(0, 8), reasonCodes };
}

function sourceEdge(left, right, thresholds) {
  const distanceMeters = haversineMeters(left.latitude, left.longitude, right.latitude, right.longitude);
  if (distanceMeters > thresholds.sourceSupportDistanceMeters) return null;
  const nameScore = nameSimilarity(left.name, right.name);
  const categoryScore = categoryCompatibility(left.category, right.category);
  const ids = sharedIdentityKeys(left, right);
  const leftPhone = normalizePhone(left.phone);
  const rightPhone = normalizePhone(right.phone);
  const leftDomain = normalizeDomain(left.website);
  const rightDomain = normalizeDomain(right.website);
  const addressScore = valueTokenSimilarity(left.address, right.address);
  const branchConflict = hasBranchConflict(left.name, right.name);
  let type = null;
  if (ids.length) type = ids.some((key) => key.startsWith('wikidata:') || key.startsWith('qid:'))
    ? 'WIKIDATA_LINK' : 'EXACT_ID';
  else if (leftPhone && leftPhone === rightPhone) type = 'SAME_PHONE';
  else if (leftDomain && leftDomain === rightDomain && isOfficialDomain(left.website)
    && isOfficialDomain(right.website)) type = 'SAME_WEBSITE';
  else if (addressScore >= thresholds.addressSimilarity && nameScore >= thresholds.sourceSupportNameSimilarity) {
    type = 'SAME_ADDRESS';
  } else if (nameScore >= thresholds.strongSourceNameSimilarity && categoryScore > 0) {
    type = 'SPATIAL_NAME_SUPPORT';
  } else if (nameScore >= thresholds.sourceSupportNameSimilarity && categoryScore > 0) {
    type = 'POSSIBLE_MATCH';
  } else if (nameScore >= 0.8 && categoryScore === 0) {
    type = 'POSSIBLE_MATCH';
  }
  if (!type) return null;
  const sameSource = left.source === right.source;
  return {
    from: recordKey(left),
    to: recordKey(right),
    type: sameSource ? 'SOURCE_DUPLICATE' : type,
    underlyingType: type,
    distanceMeters: Number(distanceMeters.toFixed(2)),
    nameSimilarity: Number(nameScore.toFixed(4)),
    categoryCompatibility: categoryScore,
    addressSimilarity: Number(addressScore.toFixed(4)),
    sharedIdentityKeys: ids,
    branchConflict,
    independence: sameSource ? INDEPENDENCE.SHARED : classifySupportIndependence(left, right),
    source: right.source,
    sourceId: right.sourceId,
  };
}

function compareRecordQuality(left, right) {
  const rank = (record) => [
    Number(provenanceAssessment(record).canaryStorageEligible),
    Number(Boolean(normalizePhone(record.phone))),
    Number(isOfficialDomain(record.website)),
    Number(Boolean(record.address)),
    ({ wikidata: 3, overture: 2, osm: 1 }[record.source] || 0),
  ];
  const leftRank = rank(left);
  const rightRank = rank(right);
  for (let index = 0; index < leftRank.length; index += 1) {
    if (leftRank[index] !== rightRank[index]) return rightRank[index] - leftRank[index];
  }
  return recordKey(left).localeCompare(recordKey(right));
}

function buildSourceEntityGraph(primaryRecords, allRecords, thresholds) {
  const index = buildSpatialIndex(allRecords, 0.002);
  const graph = new Map();
  for (const record of primaryRecords) {
    const edges = nearbySpatialRecords(index, record.latitude, record.longitude, 1)
      .filter((other) => recordKey(other) !== recordKey(record))
      .map((other) => ({ edge: sourceEdge(record, other, thresholds), other }))
      .filter((item) => item.edge)
      .sort((left, right) => (
        left.edge.type.localeCompare(right.edge.type)
        || left.edge.distanceMeters - right.edge.distanceMeters
        || left.edge.to.localeCompare(right.edge.to)
      ));
    graph.set(recordKey(record), edges);
  }
  return graph;
}

function assessSourceSupport(record, graphEntries) {
  const crossSource = graphEntries.filter((item) => item.other.source !== record.source);
  const conflicting = crossSource.some(({ edge }) => edge.branchConflict
    || (edge.nameSimilarity >= 0.8 && edge.categoryCompatibility === 0));
  const strongTypes = new Set(['EXACT_ID', 'SAME_PHONE', 'SAME_WEBSITE', 'WIKIDATA_LINK']);
  const strong = crossSource.filter(({ edge }) => strongTypes.has(edge.underlyingType));
  const independentStrong = strong.filter(({ edge, other }) => (
    edge.independence === INDEPENDENCE.INDEPENDENT && other.source !== 'wikimedia_commons'
  ));
  let consensusType = CONSENSUS_TYPES.SINGLE;
  if (conflicting) consensusType = CONSENSUS_TYPES.CONFLICTING;
  else if (crossSource.some(({ edge }) => edge.underlyingType === 'EXACT_ID')) consensusType = CONSENSUS_TYPES.EXACT;
  else if (independentStrong.length) consensusType = CONSENSUS_TYPES.STRONG;
  else if (crossSource.length && crossSource.every(({ edge }) => edge.independence === INDEPENDENCE.SHARED)) {
    consensusType = CONSENSUS_TYPES.SAME_UPSTREAM;
  } else if (crossSource.length) consensusType = CONSENSUS_TYPES.MODERATE;
  const sameSourceDuplicate = graphEntries.some(({ other }) => other.source === record.source);
  const unresolvedCrossSource = crossSource.some(({ edge }) => !strongTypes.has(edge.underlyingType)
    || edge.branchConflict);
  const losesRepresentative = strong.some(({ other }) => compareRecordQuality(record, other) > 0);
  return {
    consensusType,
    crossSourceSupported: crossSource.length > 0,
    strongIndependentSupport: independentStrong.length > 0,
    sameUpstreamOnly: consensusType === CONSENSUS_TYPES.SAME_UPSTREAM,
    supportIndependence: independentStrong.length ? INDEPENDENCE.INDEPENDENT
      : crossSource.some(({ edge }) => edge.independence === INDEPENDENCE.SHARED)
        ? INDEPENDENCE.SHARED : crossSource.length ? INDEPENDENCE.UNKNOWN : null,
    duplicateStatus: sameSourceDuplicate || unresolvedCrossSource || losesRepresentative
      ? 'UNRESOLVED_DUPLICATE_RISK' : 'RESOLVED_OR_NONE',
    edges: graphEntries.map(({ edge }) => edge),
  };
}

function classifyFreshness(record, now, thresholds) {
  const explicitStatus = normalizeVietnameseText(record.operatingStatus);
  if (/permanently closed|closed|dong cua|ngung hoat dong/.test(explicitStatus)) {
    return { freshnessClass: FRESHNESS_CLASSES.CLOSED, observedAt: null, ageDays: null };
  }
  const dates = [
    record.sourceUpdatedAt,
    record.provenance?.retrievedAt,
    ...(record.provenance?.upstreamSources || []).map((item) => item.updateTime),
  ].filter(Boolean).map((value) => new Date(value)).filter((value) => Number.isFinite(value.getTime()));
  if (!dates.length) return { freshnessClass: FRESHNESS_CLASSES.UNKNOWN, observedAt: null, ageDays: null };
  const observed = new Date(Math.max(...dates.map((date) => date.getTime())));
  const ageDays = Math.max(0, (new Date(now).getTime() - observed.getTime()) / 86400000);
  return {
    freshnessClass: ageDays > thresholds.staleAfterDays
      ? FRESHNESS_CLASSES.STALE : FRESHNESS_CLASSES.GOOD,
    observedAt: observed.toISOString(),
    ageDays: Number(ageDays.toFixed(2)),
  };
}

function buildStage4qFingerprint(record, evidence, config) {
  return stableHash({
    source: record.source,
    sourceId: record.sourceId,
    coordinates: [record.latitude, record.longitude],
    name: normalizeVietnameseText(record.name),
    category: normalizeCategory(record.category),
    address: normalizeAddress(record.address),
    phone: normalizePhone(record.phone),
    domain: normalizeDomain(record.website),
    support: evidence.sourceSupport,
    existence: evidence.existence,
    provenance: record.provenance,
    policyVersion: config.policyVersion,
    evidenceVersion: config.evidenceVersion,
  });
}

function existenceGraphEdges(record, existence) {
  return existence.candidates.map((candidate) => ({
    from: recordKey(record),
    to: `${candidate.kind}:${candidate.id}`,
    type: candidate.kind === 'historical' ? 'HISTORICAL_LINK'
      : candidate.sharedIdentityKeys.length ? 'EXACT_ID'
        : candidate.nameSimilarity >= 0.85 ? 'SPATIAL_NAME_SUPPORT' : 'POSSIBLE_MATCH',
    distanceMeters: candidate.distanceMeters,
    nameSimilarity: candidate.nameSimilarity,
    categoryCompatibility: candidate.categoryCompatibility,
    addressSimilarity: candidate.addressSimilarity,
    sharedIdentityKeys: candidate.sharedIdentityKeys,
    branchConflict: candidate.branchConflict,
    independence: null,
  }));
}

function evaluateEvidenceCandidate({ record, existence, graphEntries, config, now }) {
  const sourceSupport = assessSourceSupport(record, graphEntries || []);
  const freshness = classifyFreshness(record, now, config.thresholds);
  const provenance = provenanceAssessment(record);
  const travelerRelevance = classifyTravelerCategory(record.category);
  const valid = Boolean(record.name && hasValidCoordinates(record)
    && !['unknown', ''].includes(normalizeCategory(record.category)));
  const officialIdentity = Boolean(normalizePhone(record.phone) && isOfficialDomain(record.website)
    && (record.address || record.openingHours));
  const landmark = ['attraction', 'beach', 'bridge', 'museum', 'place_of_worship']
    .includes(normalizeCategory(record.category));
  const wikidataLink = sourceSupport.edges.some((edge) => edge.underlyingType === 'WIKIDATA_LINK');
  const independentEvidencePath = sourceSupport.strongIndependentSupport
    || officialIdentity || (landmark && wikidataLink && sourceSupport.crossSourceSupported);
  const strongQuality = Boolean(record.address && (record.phone || record.website || record.openingHours));
  const reasonCodes = [...existence.reasonCodes];
  let evidenceClass = EVIDENCE_CLASSES.STRONG;

  if (!valid) {
    evidenceClass = EVIDENCE_CLASSES.REJECT;
    reasonCodes.push('invalid_name_category_or_coordinates');
  } else if (travelerRelevance !== CATEGORY_ELIGIBILITY.TRAVELER_RELEVANT) {
    evidenceClass = EVIDENCE_CLASSES.REJECT;
    reasonCodes.push('traveler_relevance_hard_gate_failed');
  } else if (provenance.rejected) {
    evidenceClass = EVIDENCE_CLASSES.REJECT;
    reasonCodes.push('provenance_hard_gate_failed');
  } else if (existence.existenceClass === EXISTENCE_CLASSES.LIKELY
    || existence.existenceClass === EXISTENCE_CLASSES.POSSIBLE) {
    evidenceClass = EVIDENCE_CLASSES.POSSIBLE_EXISTING;
    reasonCodes.push('canonical_or_historical_existence_risk');
  } else if (sourceSupport.duplicateStatus !== 'RESOLVED_OR_NONE') {
    evidenceClass = EVIDENCE_CLASSES.DUPLICATE;
    reasonCodes.push('duplicate_hard_gate_unresolved');
  } else if (freshness.freshnessClass === FRESHNESS_CLASSES.CLOSED) {
    evidenceClass = EVIDENCE_CLASSES.REJECT;
    reasonCodes.push('closed_evidence');
  } else if (freshness.freshnessClass === FRESHNESS_CLASSES.STALE) {
    evidenceClass = EVIDENCE_CLASSES.DEFER;
    reasonCodes.push('stale_risk');
  } else if (existence.existenceClass === EXISTENCE_CLASSES.UNCERTAIN) {
    evidenceClass = EVIDENCE_CLASSES.NORMAL;
    reasonCodes.push('existence_uncertain_human_review');
  } else if (!strongQuality) {
    evidenceClass = EVIDENCE_CLASSES.NORMAL;
    reasonCodes.push('strong_record_quality_not_met');
  } else if (provenance.canaryStorageEligible && independentEvidencePath
    && sourceSupport.consensusType !== CONSENSUS_TYPES.CONFLICTING) {
    evidenceClass = EVIDENCE_CLASSES.CANARY;
    reasonCodes.push(sourceSupport.strongIndependentSupport
      ? 'strong_independent_cross_source_path' : officialIdentity
        ? 'official_identity_real_place_path' : 'landmark_entity_linkage_path');
    reasonCodes.push('research_only_no_canonical_write');
  } else {
    reasonCodes.push(!provenance.canaryStorageEligible
      ? 'license_requires_human_storage_review' : 'independent_evidence_path_missing');
  }
  if (sourceSupport.sameUpstreamOnly) reasonCodes.push('same_upstream_not_independent');
  if (sourceSupport.consensusType === CONSENSUS_TYPES.SINGLE) reasonCodes.push('single_source_only');
  if (freshness.freshnessClass === FRESHNESS_CLASSES.UNKNOWN) reasonCodes.push('freshness_unknown_not_auto_rejected');

  const evidence = { existence, sourceSupport, freshness, provenance };
  return {
    caseId: `CREATE_NEW:${record.source}:${record.sourceId}`,
    source: record.source,
    sourceId: record.sourceId,
    name: record.name,
    category: record.category,
    latitude: record.latitude,
    longitude: record.longitude,
    address: record.address || null,
    nearestCanonicalId: existence.bestCandidate?.kind === 'canonical' ? existence.bestCandidate.id : null,
    nearestCanonicalName: existence.bestCandidate?.kind === 'canonical' ? existence.bestCandidate.name : null,
    nearestDistanceMeters: existence.bestCandidate?.distanceMeters ?? null,
    existenceClass: existence.existenceClass,
    crossSourceSupport: [...new Set(sourceSupport.edges.map((edge) => edge.source))].sort(),
    supportIndependence: sourceSupport.supportIndependence,
    phoneEvidence: normalizePhone(record.phone) ? 'VALID_PHONE_PRESENT' : 'NONE',
    websiteEvidence: isOfficialDomain(record.website) ? 'OFFICIAL_DOMAIN_PRESENT'
      : normalizeDomain(record.website) ? 'NON_OFFICIAL_OR_DIRECTORY_DOMAIN' : 'NONE',
    addressEvidence: record.address ? 'ADDRESS_PRESENT' : 'NONE',
    historicalEvidence: existence.candidates.some((item) => item.kind === 'historical')
      ? 'HISTORICAL_CANDIDATE_FOUND' : 'NONE',
    wikidataEvidence: wikidataLink ? 'COMPATIBLE_WIKIDATA_LINK' : 'NONE',
    freshnessClass: freshness.freshnessClass,
    travelerRelevance,
    provenanceStatus: provenance.status,
    duplicateStatus: sourceSupport.duplicateStatus,
    consensusType: sourceSupport.consensusType,
    evidenceClass,
    reasonCodes: [...new Set(reasonCodes)].sort(),
    recommendation: evidenceClass,
    reviewerDecision: 'DEFER',
    reviewerNote: '',
    identityFingerprint: buildStage4qFingerprint(record, evidence, config),
    evidenceDetails: {
      graphEdges: [...sourceSupport.edges, ...existenceGraphEdges(record, existence)],
      canonicalSecondPassCandidates: existence.candidates,
      independentEvidence: {
        pathSatisfied: independentEvidencePath,
        strongCrossSource: sourceSupport.strongIndependentSupport,
        officialIdentity,
        landmarkEntityLink: landmark && wikidataLink,
      },
      provenanceReferences: {
        source: record.provenance?.source || null,
        sourceId: record.provenance?.sourceId || null,
        snapshotRef: record.provenance?.snapshotRef || null,
        license: record.provenance?.license || record.license?.license || null,
        policyClass: record.provenance?.policyClass || record.license?.policyClass || null,
      },
    },
  };
}

function deterministicReviewSample(results, config) {
  const rank = (left, right) => left.caseId.localeCompare(right.caseId);
  const balancedTake = (items, limit, groupKey) => {
    const groups = new Map();
    for (const item of [...items].sort(rank)) {
      const key = groupKey(item);
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(item);
    }
    const selected = [];
    const orderedGroups = [...groups.entries()].sort(([left], [right]) => left.localeCompare(right));
    while (selected.length < limit && orderedGroups.some(([, values]) => values.length)) {
      for (const [, values] of orderedGroups) {
        if (selected.length >= limit) break;
        if (values.length) selected.push(values.shift());
      }
    }
    return selected;
  };
  const take = (classification, limit) => balancedTake(
    results.filter((item) => item.evidenceClass === classification),
    limit,
    (item) => `${normalizeCategory(item.category)}:${item.consensusType}`,
  );
  const selected = [
    ...take(EVIDENCE_CLASSES.CANARY, config.review.maximumCanaryRows),
    ...take(EVIDENCE_CLASSES.STRONG, config.review.strongHumanReviewRows),
  ];
  const negatives = results.filter((item) => [
    EVIDENCE_CLASSES.POSSIBLE_EXISTING, EVIDENCE_CLASSES.DUPLICATE,
    EVIDENCE_CLASSES.NORMAL, EVIDENCE_CLASSES.DEFER, EVIDENCE_CLASSES.REJECT,
  ].includes(item.evidenceClass));
  selected.push(...balancedTake(negatives, config.review.difficultNegativeRows,
    (item) => item.evidenceClass));
  return [...new Map(selected.map((item) => [item.caseId, item])).values()]
    .sort(rank).slice(0, config.review.maximumRows);
}

function summarizeEvidence(results) {
  const count = (predicate) => results.filter(predicate).length;
  const classifications = Object.fromEntries(Object.values(EVIDENCE_CLASSES).map((value) => [value, 0]));
  for (const result of results) classifications[result.evidenceClass] += 1;
  return {
    schemaVersion: 'stage4q-create-new-evidence-summary-v1',
    status: 'RESEARCH_ONLY_NON_RUNTIME_NOT_CANONICAL',
    evaluated: results.length,
    existence: {
      clearlyAbsent: count((item) => item.existenceClass === EXISTENCE_CLASSES.CLEAR),
      possibleOrLikelyExisting: count((item) => [EXISTENCE_CLASSES.POSSIBLE,
        EXISTENCE_CLASSES.LIKELY].includes(item.existenceClass)),
      uncertain: count((item) => item.existenceClass === EXISTENCE_CLASSES.UNCERTAIN),
    },
    support: {
      crossSource: count((item) => item.crossSourceSupport.length > 0),
      strongIndependent: count((item) => item.evidenceDetails.independentEvidence.strongCrossSource),
      sameUpstreamOnly: count((item) => item.supportIndependence === INDEPENDENCE.SHARED
        && !item.evidenceDetails.independentEvidence.strongCrossSource),
      singleSourceOnly: count((item) => item.consensusType === CONSENSUS_TYPES.SINGLE),
    },
    freshness: {
      good: count((item) => item.freshnessClass === FRESHNESS_CLASSES.GOOD),
      unknown: count((item) => item.freshnessClass === FRESHNESS_CLASSES.UNKNOWN),
      stale: count((item) => item.freshnessClass === FRESHNESS_CLASSES.STALE),
      closed: count((item) => item.freshnessClass === FRESHNESS_CLASSES.CLOSED),
    },
    duplicateRiskUnresolved: count((item) => item.duplicateStatus !== 'RESOLVED_OR_NONE'),
    classifications,
    autoCreateNew: false,
    canonicalWrites: 0,
    deletes: 0,
    deterministicHash: stableHash(results),
  };
}

function evaluateStrongPool({ strongRecords, allRecords, canonicalPois, historicalPois,
  config, now }) {
  assertResearchOnly(config);
  const existenceIndex = buildExistenceIndex(canonicalPois, historicalPois);
  const graph = buildSourceEntityGraph(strongRecords, allRecords, config.thresholds);
  const results = strongRecords.map((record) => evaluateEvidenceCandidate({
    record,
    existence: secondPassExistenceSearch(record, existenceIndex, config.thresholds),
    graphEntries: graph.get(recordKey(record)) || [],
    config,
    now,
  })).sort((left, right) => left.caseId.localeCompare(right.caseId));
  return { results, review: deterministicReviewSample(results, config), summary: summarizeEvidence(results) };
}

function checkIncrementalReuse(state, inputFingerprint, config) {
  if (!state) return { reusable: false, reason: 'NO_STATE' };
  if (state.policyVersion !== config.policyVersion || state.evidenceVersion !== config.evidenceVersion) {
    return { reusable: false, reason: 'POLICY_OR_EVIDENCE_VERSION_CHANGED' };
  }
  if (state.inputFingerprint !== inputFingerprint) {
    return { reusable: false, reason: 'INPUT_FINGERPRINT_CHANGED' };
  }
  return { reusable: true, reason: 'UNCHANGED_POLICY_AND_INPUT', results: state.results };
}

module.exports = {
  CONSENSUS_TYPES,
  EVIDENCE_CLASSES,
  EXISTENCE_CLASSES,
  FRESHNESS_CLASSES,
  INDEPENDENCE,
  assertResearchOnly,
  assessSourceSupport,
  buildExistenceIndex,
  buildSourceEntityGraph,
  buildStage4qFingerprint,
  checkIncrementalReuse,
  classifyFreshness,
  classifySupportIndependence,
  deterministicReviewSample,
  evaluateEvidenceCandidate,
  evaluateStrongPool,
  isOfficialDomain,
  normalizeAddress,
  normalizeDomain,
  normalizePhone,
  secondPassExistenceSearch,
  sourceEdge,
  summarizeEvidence,
};
