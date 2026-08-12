const { BOUNDARY_CLASSIFICATIONS, classifyAdministrativeBoundary } = require('./administrativeBoundary');
const { DECISIONS } = require('./entityResolution');
const { stableHash, recordKey, STATUSES } = require('./incrementalSync');
const { normalizeCategory, normalizeName } = require('./sourceRecord');

const VALIDATION_LABELS = Object.freeze({
  MATCH: 'MATCH',
  NOT_MATCH: 'NOT_MATCH',
  DUPLICATE_SAME_PLACE: 'DUPLICATE_SAME_PLACE',
  SEPARATE_PLACE: 'SEPARATE_PLACE',
  UNCERTAIN: 'UNCERTAIN',
});

const LABEL_STATUS = Object.freeze({
  MACHINE_SUGGESTED: 'MACHINE_SUGGESTED',
  EVIDENCE_SUPPORTED: 'EVIDENCE_SUPPORTED',
});

const CATEGORY_ELIGIBILITY = Object.freeze({
  TRAVELER_RELEVANT: 'TRAVELER_RELEVANT',
  CONTEXTUAL: 'CONTEXTUAL',
  LOW_VALUE: 'LOW_VALUE',
  EXCLUDED: 'EXCLUDED',
});

const NEW_TIERS = Object.freeze({
  A: 'NEW_TIER_A_STRONG',
  B: 'NEW_TIER_B_REVIEW',
  C: 'NEW_TIER_C_LOW_VALUE',
  D: 'NEW_TIER_D_INVALID_OR_EXCLUDE',
});

const REVIEW_PRIORITIES = Object.freeze({
  P0: 'P0',
  P1: 'P1',
  P2: 'P2',
  P3: 'P3',
  P4: 'P4',
  DEFERRED: 'DEFERRED',
});

const APPLY_OPERATIONS = Object.freeze({
  ENRICH_EXISTING: 'ENRICH_EXISTING',
  CREATE_NEW: 'CREATE_NEW',
  KEEP_EXISTING: 'KEEP_EXISTING',
  KEEP_SEPARATE: 'KEEP_SEPARATE',
  MERGE_SOURCE_DUPLICATE: 'MERGE_SOURCE_DUPLICATE',
  DEFER: 'DEFER',
  REJECT: 'REJECT',
});

const FIELD_POLICIES = Object.freeze({
  SAFE_ENRICHMENT: 'SAFE_ENRICHMENT',
  CONFLICT_REVIEW: 'CONFLICT_REVIEW',
  NO_CHANGE: 'NO_CHANGE',
  REJECTED_SOURCE_VALUE: 'REJECTED_SOURCE_VALUE',
});

const TRAVELER_RELEVANT_CATEGORIES = new Set([
  'accommodation', 'arts', 'attraction', 'bakery', 'bar', 'beach', 'bookstore',
  'bridge', 'cafe', 'campus', 'community', 'convenience', 'event', 'grocery',
  'holiday', 'information', 'market', 'museum', 'park', 'place_of_worship',
  'restaurant', 'resort', 'shopping', 'sports', 'tours', 'travel',
]);

const CONTEXTUAL_CATEGORIES = new Set([
  'bank', 'building', 'college', 'education', 'educational', 'financial', 'gym',
  'health', 'hospital', 'medical', 'pharmacy', 'pool', 'public', 'religious',
  'retail', 'station', 'transport', 'transportation', 'university', 'yoga',
]);

const LOW_VALUE_CATEGORIES = new Set([
  'advertising', 'architectural', 'auto', 'automotive', 'b2b', 'beauty',
  'business', 'car', 'commercial', 'computer', 'construction', 'cosmetic',
  'dentist', 'flowers', 'freight', 'furniture', 'hair', 'home', 'industrial',
  'insurance', 'interior', 'internet', 'it', 'laundromat', 'medical', 'mobile',
  'motorcycle', 'nail', 'pet', 'printing', 'professional', 'service', 'software',
  'spas', 'tattoo', 'telecommunications', 'tutoring', 'utility', 'wholesale',
]);

function compactRecord(record) {
  if (!record) return null;
  return {
    source: record.source || null,
    sourceId: record.sourceId || record.id || null,
    name: record.name || null,
    category: record.category || null,
    latitude: record.latitude ?? null,
    longitude: record.longitude ?? null,
    address: record.address || record.district || null,
    website: record.website || null,
    phone: record.phone || null,
    openingHours: record.openingHours || null,
    externalIds: record.externalIds || {},
  };
}

function sharedExternalIdentifiers(left, right) {
  const ignored = new Set(['osm_type', 'source_type', 'dataset']);
  const leftEntries = Object.entries(left?.externalIds || {})
    .filter(([field, value]) => !ignored.has(field) && value !== null && value !== '');
  const rightEntries = Object.entries(right?.externalIds || {})
    .filter(([field, value]) => !ignored.has(field) && value !== null && value !== '');
  const shared = [];
  for (const [leftField, leftValue] of leftEntries) {
    for (const [rightField, rightValue] of rightEntries) {
      if (String(leftValue) === String(rightValue)) {
        shared.push(`${leftField}:${rightField}:${leftValue}`);
      }
    }
  }
  return [...new Set(shared)].sort();
}

function evidenceLabelForMatch(match, sourceRecord, canonicalRecord) {
  const candidate = match.bestCandidate;
  const sharedIds = sharedExternalIdentifiers(sourceRecord, canonicalRecord);
  if (sharedIds.length > 0) {
    return {
      label: VALIDATION_LABELS.MATCH,
      status: LABEL_STATUS.EVIDENCE_SUPPORTED,
      confidence: 1,
      reviewer: 'STAGE4I_STRUCTURED_EVIDENCE_V1',
      evidenceType: 'EXACT_STABLE_EXTERNAL_IDENTIFIER',
      evidence: sharedIds,
      reasonCodes: ['exact_stable_external_identifier_linkage'],
    };
  }
  if (candidate?.branchConflict) {
    return {
      label: VALIDATION_LABELS.SEPARATE_PLACE,
      status: LABEL_STATUS.EVIDENCE_SUPPORTED,
      confidence: 0.98,
      reviewer: 'STAGE4I_STRUCTURED_EVIDENCE_V1',
      evidenceType: 'BRANCH_LOCATION_CONFLICT',
      evidence: [`distance_m=${candidate.distanceMeters}`, 'independent_branch_tokens_conflict'],
      reasonCodes: ['same_chain_different_branch_evidence'],
    };
  }
  if (
    candidate?.fullNameExact
    && candidate.distanceMeters <= 5
    && candidate.categoryCompatibility === 1
    && candidate.addressSimilarity >= 0.5
  ) {
    return {
      label: VALIDATION_LABELS.MATCH,
      status: LABEL_STATUS.EVIDENCE_SUPPORTED,
      confidence: 0.99,
      reviewer: 'STAGE4I_STRUCTURED_EVIDENCE_V1',
      evidenceType: 'MULTI_FIELD_SOURCE_CANONICAL_OBSERVATION',
      evidence: [
        'exact_full_normalized_name',
        `distance_m=${candidate.distanceMeters}`,
        'exact_category',
        `address_token_overlap=${candidate.addressSimilarity}`,
      ],
      reasonCodes: ['independent_record_fields_agree'],
    };
  }
  if (candidate?.categoryConflict && candidate.distanceMeters <= 30) {
    return {
      label: VALIDATION_LABELS.NOT_MATCH,
      status: LABEL_STATUS.EVIDENCE_SUPPORTED,
      confidence: 0.95,
      reviewer: 'STAGE4I_STRUCTURED_EVIDENCE_V1',
      evidenceType: 'SAME_BUILDING_DIFFERENT_BUSINESS',
      evidence: [`distance_m=${candidate.distanceMeters}`, 'incompatible_known_categories'],
      reasonCodes: ['same_building_category_conflict'],
    };
  }
  return {
    label: VALIDATION_LABELS.UNCERTAIN,
    status: LABEL_STATUS.MACHINE_SUGGESTED,
    confidence: 0,
    reviewer: 'PENDING_HUMAN_REVIEW',
    evidenceType: 'INSUFFICIENT_INDEPENDENT_EVIDENCE',
    evidence: [],
    reasonCodes: ['do_not_infer_ground_truth_from_resolver'],
  };
}

function evidenceLabelForDuplicate(duplicate, left, right) {
  const sharedIds = sharedExternalIdentifiers(left, right);
  const samePhone = Boolean(left?.phone && right?.phone && left.phone === right.phone);
  const sameWebsite = Boolean(left?.website && right?.website && left.website === right.website);
  if (sharedIds.length > 0 || samePhone) {
    return {
      label: VALIDATION_LABELS.DUPLICATE_SAME_PLACE,
      status: LABEL_STATUS.EVIDENCE_SUPPORTED,
      confidence: sharedIds.length > 0 ? 1 : 0.99,
      reviewer: 'STAGE4I_STRUCTURED_EVIDENCE_V1',
      evidenceType: sharedIds.length > 0 ? 'EXACT_STABLE_EXTERNAL_IDENTIFIER' : 'EXACT_PHONE_LINKAGE',
      evidence: sharedIds.length > 0 ? sharedIds : [`phone=${left.phone}`],
      reasonCodes: [sharedIds.length > 0 ? 'shared_stable_identifier' : 'shared_exact_phone'],
    };
  }
  if (
    sameWebsite
    && normalizeName(left?.name) === normalizeName(right?.name)
    && duplicate.distanceMeters <= 10
  ) {
    return {
      label: VALIDATION_LABELS.DUPLICATE_SAME_PLACE,
      status: LABEL_STATUS.EVIDENCE_SUPPORTED,
      confidence: 0.99,
      reviewer: 'STAGE4I_STRUCTURED_EVIDENCE_V1',
      evidenceType: 'EXACT_WEBSITE_NAME_LOCATION',
      evidence: [`website=${left.website}`, `distance_m=${duplicate.distanceMeters}`],
      reasonCodes: ['shared_exact_website_name_and_location'],
    };
  }
  return {
    label: VALIDATION_LABELS.UNCERTAIN,
    status: LABEL_STATUS.MACHINE_SUGGESTED,
    confidence: 0,
    reviewer: 'PENDING_HUMAN_REVIEW',
    evidenceType: 'INSUFFICIENT_INDEPENDENT_EVIDENCE',
    evidence: [],
    reasonCodes: ['near_name_and_location_are_not_ground_truth'],
  };
}

function deterministicSample(items, limit, keyFunction) {
  return [...items]
    .sort((left, right) => {
      const leftKey = stableHash(keyFunction(left));
      const rightKey = stableHash(keyFunction(right));
      if (leftKey !== rightKey) return leftKey.localeCompare(rightKey);
      return String(keyFunction(left)).localeCompare(String(keyFunction(right)));
    })
    .slice(0, limit);
}

function buildValidationDataset({ matches, duplicates, recordsById, canonicalById }) {
  const strata = [
    [DECISIONS.HIGH_CONFIDENCE_MATCH, 50],
    [DECISIONS.PROBABLE_MATCH, 50],
    [DECISIONS.AMBIGUOUS, 30],
    [DECISIONS.NEW_CANDIDATE, 100],
  ];
  const cases = [];
  for (const [decision, limit] of strata) {
    const selected = deterministicSample(
      matches.filter((match) => match.decision === decision),
      limit,
      (match) => `${match.source}:${match.sourceId}`,
    );
    for (const match of selected) {
      const sourceRecord = recordsById.get(match.sourceId);
      const canonicalRecord = canonicalById.get(match.bestCandidate?.canonicalPoiId);
      cases.push({
        caseId: `MATCH:${match.source}:${match.sourceId}`,
        stratum: decision,
        machineSuggested: {
          status: LABEL_STATUS.MACHINE_SUGGESTED,
          decision,
          confidence: match.confidence,
          reasonCodes: match.reasonCodes,
        },
        recordA: compactRecord(sourceRecord),
        recordB: compactRecord(canonicalRecord),
        candidateEvidence: match.bestCandidate || null,
        evidenceSupported: evidenceLabelForMatch(match, sourceRecord, canonicalRecord),
      });
    }
  }
  const selectedDuplicates = deterministicSample(
    duplicates,
    70,
    (duplicate) => `${duplicate.sourceIdA}:${duplicate.sourceIdB}`,
  );
  for (const duplicate of selectedDuplicates) {
    const left = recordsById.get(duplicate.sourceIdA);
    const right = recordsById.get(duplicate.sourceIdB);
    cases.push({
      caseId: `DUPLICATE:${duplicate.sourceIdA}:${duplicate.sourceIdB}`,
      stratum: DECISIONS.SOURCE_DUPLICATE,
      machineSuggested: {
        status: LABEL_STATUS.MACHINE_SUGGESTED,
        decision: DECISIONS.SOURCE_DUPLICATE,
        confidence: duplicate.confidence,
        reasonCodes: duplicate.reasonCodes,
      },
      recordA: compactRecord(left),
      recordB: compactRecord(right),
      candidateEvidence: {
        distanceMeters: duplicate.distanceMeters,
        nameSimilarity: duplicate.nameSimilarity,
        categoryCompatibility: duplicate.categoryCompatibility,
      },
      evidenceSupported: evidenceLabelForDuplicate(duplicate, left, right),
    });
  }
  return cases.sort((left, right) => left.caseId.localeCompare(right.caseId));
}

function calculatePrecisionGate(cases, machineDecision, options = {}) {
  const minEvidence = options.minEvidence || 20;
  const requiredPrecision = options.requiredPrecision ?? 0.995;
  const supported = cases.filter((item) => (
    item.stratum === machineDecision
    && item.evidenceSupported.status === LABEL_STATUS.EVIDENCE_SUPPORTED
  ));
  const positiveLabel = machineDecision === DECISIONS.SOURCE_DUPLICATE
    ? VALIDATION_LABELS.DUPLICATE_SAME_PLACE
    : VALIDATION_LABELS.MATCH;
  const correct = supported.filter((item) => item.evidenceSupported.label === positiveLabel).length;
  const precision = supported.length > 0 ? Number((correct / supported.length).toFixed(4)) : null;
  return {
    machineDecision,
    supportedLabels: supported.length,
    uncertainLabels: cases.filter((item) => (
      item.stratum === machineDecision
      && item.evidenceSupported.label === VALIDATION_LABELS.UNCERTAIN
    )).length,
    correct,
    precision,
    minEvidence,
    requiredPrecision,
    status: supported.length < minEvidence
      ? 'INSUFFICIENT_EVIDENCE'
      : precision >= requiredPrecision
        ? 'PASS'
        : 'FAIL',
  };
}

function classifyTravelerCategory(category) {
  const normalized = normalizeCategory(category);
  if (TRAVELER_RELEVANT_CATEGORIES.has(normalized)) return CATEGORY_ELIGIBILITY.TRAVELER_RELEVANT;
  if (CONTEXTUAL_CATEGORIES.has(normalized)) return CATEGORY_ELIGIBILITY.CONTEXTUAL;
  if (LOW_VALUE_CATEGORIES.has(normalized)) return CATEGORY_ELIGIBILITY.LOW_VALUE;
  return CATEGORY_ELIGIBILITY.EXCLUDED;
}

function provenanceComplete(record) {
  return Boolean(
    record?.provenance?.source
    && record?.provenance?.sourceId
    && record?.license?.license
    && record?.license?.policyClass
    && record?.license?.attribution,
  );
}

function nameQuality(record) {
  const normalized = normalizeName(record?.name);
  if (normalized.length < 3) return false;
  return !/^(unknown|unnamed|place|point|location|real|topic|building|service)$/.test(normalized);
}

function sourceConfidence(record) {
  const candidates = [
    record?.raw?.properties?.confidence,
    record?.raw?.confidence,
    record?.confidence,
  ].map(Number).filter(Number.isFinite);
  return candidates.length > 0 ? Math.max(...candidates) : null;
}

function triageNewCandidate(record, match, context = {}) {
  const boundaryClassification = context.boundaryClassification
    || record.boundaryClassification
    || BOUNDARY_CLASSIFICATIONS.INVALID_COORDINATE;
  const categoryEligibility = classifyTravelerCategory(record.category);
  const duplicate = context.duplicateSourceIds?.has(record.sourceId) || false;
  if (
    ![BOUNDARY_CLASSIFICATIONS.INSIDE_DANANG, BOUNDARY_CLASSIFICATIONS.BOUNDARY_EDGE]
      .includes(boundaryClassification)
    || categoryEligibility === CATEGORY_ELIGIBILITY.EXCLUDED
    || !nameQuality(record)
  ) {
    return { tier: NEW_TIERS.D, categoryEligibility, reason: 'invalid_outside_or_excluded' };
  }
  if (
    duplicate
    || [CATEGORY_ELIGIBILITY.CONTEXTUAL, CATEGORY_ELIGIBILITY.LOW_VALUE].includes(categoryEligibility)
  ) {
    return { tier: NEW_TIERS.C, categoryEligibility, reason: duplicate ? 'source_duplicate_review_first' : 'non_traveler_priority' };
  }
  const metadataCount = [record.address, record.website, record.phone, record.openingHours]
    .filter(Boolean).length;
  const confidence = sourceConfidence(record);
  const farFromCanonical = !match?.bestCandidate || match.bestCandidate.distanceMeters > 250;
  const strong = provenanceComplete(record)
    && metadataCount >= 2
    && (confidence === null || confidence >= 0.85)
    && farFromCanonical;
  return {
    tier: strong ? NEW_TIERS.A : NEW_TIERS.B,
    categoryEligibility,
    reason: strong ? 'strong_review_candidate_not_auto_create' : 'traveler_relevant_human_review',
  };
}

function priorityForMatch(match, triage, highGate) {
  if (match.decision === DECISIONS.AMBIGUOUS) return [REVIEW_PRIORITIES.P0, 'potential_false_merge_or_chain_collision'];
  if (match.decision === DECISIONS.HIGH_CONFIDENCE_MATCH && highGate.status !== 'PASS') {
    return [REVIEW_PRIORITIES.P0, 'high_confidence_gate_not_met'];
  }
  if (match.decision === DECISIONS.PROBABLE_MATCH) {
    return triage.categoryEligibility === CATEGORY_ELIGIBILITY.TRAVELER_RELEVANT
      ? [REVIEW_PRIORITIES.P1, 'high_value_probable_match']
      : [REVIEW_PRIORITIES.P4, 'probable_match_lower_product_value'];
  }
  if (match.decision === DECISIONS.NEW_CANDIDATE) {
    if (triage.tier === NEW_TIERS.A) return [REVIEW_PRIORITIES.P2, 'strong_new_traveler_candidate'];
    if (triage.tier === NEW_TIERS.B) return [REVIEW_PRIORITIES.P4, 'new_traveler_candidate_needs_evidence'];
    return [REVIEW_PRIORITIES.DEFERRED, triage.reason];
  }
  return [REVIEW_PRIORITIES.DEFERRED, 'not_current_review_scope'];
}

function buildPrioritizedReviewQueue({ matches, duplicates, recordsById, highGate }) {
  const duplicateSourceIds = new Set(duplicates.flatMap((item) => [item.sourceIdA, item.sourceIdB]));
  const queue = [];
  for (const match of matches) {
    const record = recordsById.get(match.sourceId);
    const triage = triageNewCandidate(record, match, { duplicateSourceIds });
    const [priority, priorityReason] = priorityForMatch(match, triage, highGate);
    if (match.decision === DECISIONS.HIGH_CONFIDENCE_MATCH && highGate.status === 'PASS') continue;
    queue.push({
      queueId: `MATCH:${match.source}:${match.sourceId}`,
      priority,
      priorityReason,
      decision: match.decision,
      source: match.source,
      sourceId: match.sourceId,
      name: record?.name || match.sourceName,
      category: record?.category || null,
      categoryEligibility: triage.categoryEligibility,
      newTier: match.decision === DECISIONS.NEW_CANDIDATE ? triage.tier : null,
      latitude: record?.latitude ?? null,
      longitude: record?.longitude ?? null,
      candidateCanonicalMatch: match.bestCandidate || null,
      evidence: { confidence: match.confidence, reasonCodes: match.reasonCodes },
      provenance: record?.provenance || match.provenance,
      license: record?.license || match.license,
      recommendedAction: match.decision === DECISIONS.NEW_CANDIDATE
        ? APPLY_OPERATIONS.CREATE_NEW
        : match.decision === DECISIONS.AMBIGUOUS
          ? APPLY_OPERATIONS.DEFER
          : APPLY_OPERATIONS.ENRICH_EXISTING,
    });
  }
  for (const duplicate of duplicates) {
    const left = recordsById.get(duplicate.sourceIdA);
    queue.push({
      queueId: `DUPLICATE:${duplicate.sourceIdA}:${duplicate.sourceIdB}`,
      priority: REVIEW_PRIORITIES.P3,
      priorityReason: 'source_duplicate_resolution',
      decision: DECISIONS.SOURCE_DUPLICATE,
      source: duplicate.source,
      sourceId: duplicate.sourceIdA,
      relatedSourceId: duplicate.sourceIdB,
      name: duplicate.nameA,
      category: left?.category || null,
      categoryEligibility: classifyTravelerCategory(left?.category),
      newTier: null,
      latitude: left?.latitude ?? null,
      longitude: left?.longitude ?? null,
      candidateCanonicalMatch: null,
      evidence: {
        confidence: duplicate.confidence,
        distanceMeters: duplicate.distanceMeters,
        reasonCodes: duplicate.reasonCodes,
      },
      provenance: duplicate.records?.map((item) => item.provenance) || [],
      license: duplicate.records?.map((item) => item.license) || [],
      recommendedAction: APPLY_OPERATIONS.MERGE_SOURCE_DUPLICATE,
    });
  }
  return queue.sort((left, right) => {
    if (left.priority !== right.priority) return left.priority.localeCompare(right.priority);
    return left.queueId.localeCompare(right.queueId);
  });
}

function selectCanaryReview(queue, size = 75) {
  const targets = { P0: 15, P1: 15, P2: 20, P3: 15, P4: 10 };
  const selected = [];
  const selectedIds = new Set();
  const add = (items, limit) => {
    const available = items.filter((item) => !selectedIds.has(item.queueId));
    const sampled = deterministicSample(available, limit, (item) => item.queueId);
    for (const item of sampled) selectedIds.add(item.queueId);
    selected.push(...sampled);
  };

  add(queue.filter((item) => (
    item.priority === REVIEW_PRIORITIES.P0
    && (
      item.decision === DECISIONS.AMBIGUOUS
      || item.evidence?.reasonCodes?.some((reason) => reason.includes('chain') || reason.includes('branch'))
    )
  )), 5);
  add(queue.filter((item) => item.priority === REVIEW_PRIORITIES.P0), targets.P0 - selected.length);
  add(queue.filter((item) => item.priority === REVIEW_PRIORITIES.P1), targets.P1);
  add(queue.filter((item) => item.priority === REVIEW_PRIORITIES.P2), targets.P2);
  add(queue.filter((item) => item.priority === REVIEW_PRIORITIES.P3), targets.P3);
  const beforeP4 = selected.length;
  add(queue.filter((item) => (
    item.priority === REVIEW_PRIORITIES.P4
    && item.categoryEligibility !== CATEGORY_ELIGIBILITY.TRAVELER_RELEVANT
  )), 5);
  add(queue.filter((item) => item.priority === REVIEW_PRIORITIES.P4), targets.P4 - (selected.length - beforeP4));

  if (selected.length < size) {
    add(
      queue.filter((item) => item.priority !== REVIEW_PRIORITIES.DEFERRED && !selectedIds.has(item.queueId)),
      size - selected.length,
    );
  }
  return selected.slice(0, size).sort((left, right) => left.queueId.localeCompare(right.queueId));
}

function normalizeComparable(value) {
  if (value === null || value === undefined || value === '') return null;
  if (typeof value === 'object') return JSON.stringify(value, Object.keys(value).sort());
  return normalizeName(value) || String(value).trim().toLowerCase();
}

function classifyFieldConflict(field, oldValue, proposedValue, fieldProvenance = null) {
  if (proposedValue === null || proposedValue === undefined || proposedValue === '') {
    return FIELD_POLICIES.REJECTED_SOURCE_VALUE;
  }
  if (['media', 'externalIds'].includes(field) && !fieldProvenance?.license) {
    return FIELD_POLICIES.REJECTED_SOURCE_VALUE;
  }
  if (oldValue === null || oldValue === undefined || oldValue === '') return FIELD_POLICIES.SAFE_ENRICHMENT;
  if (normalizeComparable(oldValue) === normalizeComparable(proposedValue)) return FIELD_POLICIES.NO_CHANGE;
  return FIELD_POLICIES.CONFLICT_REVIEW;
}

function buildFieldChanges(canonicalRecord, sourceRecord) {
  const fields = ['name', 'category', 'coordinates', 'address', 'website', 'phone', 'openingHours', 'media', 'externalIds'];
  return fields.map((field) => {
    const oldValue = field === 'coordinates'
      ? canonicalRecord ? { latitude: canonicalRecord.latitude, longitude: canonicalRecord.longitude } : null
      : canonicalRecord?.[field] ?? null;
    const proposedValue = field === 'coordinates'
      ? { latitude: sourceRecord?.latitude ?? null, longitude: sourceRecord?.longitude ?? null }
      : field === 'media'
        ? sourceRecord?.media || null
        : sourceRecord?.[field] ?? null;
    const provenance = sourceRecord?.provenance?.fields?.[field]
      || sourceRecord?.provenance?.fields?.latitude
      || sourceRecord?.provenance
      || null;
    return {
      field,
      oldValue,
      proposedValue,
      policy: classifyFieldConflict(field, oldValue, proposedValue, provenance),
      fieldProvenance: provenance,
    };
  });
}

function buildApplyPlan({ canary, recordsById, canonicalById, validationByCaseId }) {
  const operations = canary.map((item) => {
    const sourceRecord = recordsById.get(item.sourceId);
    const canonicalId = item.candidateCanonicalMatch?.canonicalPoiId || null;
    const canonicalRecord = canonicalById.get(canonicalId) || null;
    const validation = validationByCaseId.get(item.queueId) || null;
    return {
      operationId: item.queueId,
      operation: APPLY_OPERATIONS.DEFER,
      recommendedOperation: item.recommendedAction,
      canonicalId,
      candidateId: item.sourceId,
      sourceIds: [item.sourceId, item.relatedSourceId].filter(Boolean),
      exactFieldChanges: buildFieldChanges(canonicalRecord, sourceRecord),
      licenseAttribution: {
        license: sourceRecord?.license || item.license,
        provenance: sourceRecord?.provenance || item.provenance,
      },
      decisionEvidence: validation?.evidenceSupported || item.evidence,
      reviewerStatus: 'PENDING_HUMAN_REVIEW',
      applied: false,
    };
  });
  return {
    status: 'DRY_RUN_ONLY_NOT_APPROVED',
    canonicalWriteAuthorized: false,
    supportedOperations: Object.values(APPLY_OPERATIONS),
    operations,
    deterministicHash: stableHash(operations),
  };
}

function dryRunApplyPlan({ canonicalPois, applyPlan, candidateRecordsById }) {
  const proposed = canonicalPois.map((poi) => ({ ...poi }));
  const byId = new Map(proposed.map((poi) => [poi.id, poi]));
  const metrics = { enrichments: 0, additions: 0, conflicts: 0, rejectedActions: 0, deferred: 0 };
  for (const item of applyPlan.operations || []) {
    if (item.reviewerStatus !== 'APPROVED') {
      metrics.deferred += 1;
      continue;
    }
    if (item.operation === APPLY_OPERATIONS.ENRICH_EXISTING) {
      const target = byId.get(item.canonicalId);
      if (!target) { metrics.rejectedActions += 1; continue; }
      const safe = item.exactFieldChanges.filter((change) => change.policy === FIELD_POLICIES.SAFE_ENRICHMENT);
      const conflicts = item.exactFieldChanges.filter((change) => change.policy === FIELD_POLICIES.CONFLICT_REVIEW);
      if (conflicts.length > 0) { metrics.conflicts += conflicts.length; continue; }
      for (const change of safe) {
        if (change.field === 'coordinates') {
          target.latitude = change.proposedValue.latitude;
          target.longitude = change.proposedValue.longitude;
        } else target[change.field] = change.proposedValue;
      }
      metrics.enrichments += 1;
    } else if (item.operation === APPLY_OPERATIONS.CREATE_NEW) {
      const source = candidateRecordsById.get(item.candidateId);
      const id = `candidate:${item.candidateId}`;
      if (!source || byId.has(id)) { metrics.rejectedActions += 1; continue; }
      const created = { id, ...compactRecord(source), provenance: source.provenance, license: source.license };
      proposed.push(created);
      byId.set(id, created);
      metrics.additions += 1;
    } else if (item.operation === APPLY_OPERATIONS.DEFER) metrics.deferred += 1;
    else if (![APPLY_OPERATIONS.KEEP_EXISTING, APPLY_OPERATIONS.KEEP_SEPARATE].includes(item.operation)) {
      metrics.rejectedActions += 1;
    }
  }
  const ordered = proposed.sort((left, right) => left.id.localeCompare(right.id));
  return {
    status: 'PROPOSED_NEXT_CITY_PACK_DRY_RUN_ONLY',
    beforeCount: canonicalPois.length,
    proposedAfterCount: ordered.length,
    ...metrics,
    provenanceLicenseSummary: {
      approvedActions: (applyPlan.operations || []).filter((item) => item.reviewerStatus === 'APPROVED').length,
      allActionsRetainFieldProvenance: (applyPlan.operations || []).every(
        (item) => item.exactFieldChanges.every((change) => Boolean(change.fieldProvenance) || change.policy === FIELD_POLICIES.REJECTED_SOURCE_VALUE),
      ),
    },
    deterministicProposedPackHash: stableHash(ordered),
    proposedPack: ordered,
  };
}

function prepareIncrementalReviewDelta({ syncState, recordsByKey, boundary, resolveRecord }) {
  const metrics = {
    discovered: syncState.length,
    unchangedSkipped: 0,
    identityResolved: 0,
    changedNonIdentity: 0,
    outsideExcluded: 0,
    invalidExcluded: 0,
    reviewWork: 0,
  };
  const work = [];
  for (const state of syncState) {
    if (state.status === STATUSES.UNCHANGED) { metrics.unchangedSkipped += 1; continue; }
    const record = recordsByKey.get(recordKey(state));
    if (!record || state.status === STATUSES.INVALID) { metrics.invalidExcluded += 1; continue; }
    const boundaryClassification = classifyAdministrativeBoundary(record, boundary);
    if (![BOUNDARY_CLASSIFICATIONS.INSIDE_DANANG, BOUNDARY_CLASSIFICATIONS.BOUNDARY_EDGE].includes(boundaryClassification)) {
      metrics.outsideExcluded += 1;
      continue;
    }
    let resolution = null;
    if (state.status === STATUSES.NEW || state.processing?.identity) {
      resolution = resolveRecord(record);
      metrics.identityResolved += 1;
    } else metrics.changedNonIdentity += 1;
    work.push({
      source: record.source,
      sourceId: record.sourceId,
      status: state.status,
      boundaryClassification,
      categoryEligibility: classifyTravelerCategory(record.category),
      resolution,
    });
    metrics.reviewWork += 1;
  }
  return { work, metrics, deterministicHash: stableHash({ work, metrics }) };
}

module.exports = {
  APPLY_OPERATIONS,
  CATEGORY_ELIGIBILITY,
  FIELD_POLICIES,
  LABEL_STATUS,
  NEW_TIERS,
  REVIEW_PRIORITIES,
  VALIDATION_LABELS,
  buildApplyPlan,
  buildFieldChanges,
  buildPrioritizedReviewQueue,
  buildValidationDataset,
  calculatePrecisionGate,
  classifyFieldConflict,
  classifyTravelerCategory,
  dryRunApplyPlan,
  evidenceLabelForDuplicate,
  evidenceLabelForMatch,
  prepareIncrementalReviewDelta,
  provenanceComplete,
  selectCanaryReview,
  triageNewCandidate,
};
