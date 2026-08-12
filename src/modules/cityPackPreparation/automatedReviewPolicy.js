const { findReusableDecision, stableHash } = require('./decisionMemory');

const POLICY_OUTCOMES = Object.freeze([
  'SKIP_UNCHANGED',
  'REUSE_APPROVAL',
  'REUSE_REJECTION',
  'AUTO_ACCEPT_SAFE',
  'AUTO_REJECT',
  'HUMAN_REVIEW',
  'DEFER',
]);

const DEFAULT_VERSIONS = Object.freeze({
  policy: 'stage4l-review-policy-v1',
  resolver: 'stage4h-v1',
  evidence: 'stage4j-independent-evidence-v1',
  boundary: 'osm-relation-1891418-v72-product-scope-v1',
});

function classifyNewTier(reviewCase) {
  return reviewCase.newCandidateTier || reviewCase.tier || null;
}

function safeChanges(reviewCase, config) {
  const allowed = new Set(config.allowedAutoAcceptFields);
  return (reviewCase.fieldChanges || []).filter(
    (change) => allowed.has(change.field)
      && change.state === 'SAFE_ADDITION'
      && change.newValue !== null
      && change.newValue !== '',
  );
}

function hasHardSafetyFailure(reviewCase) {
  const geographic = reviewCase.geographicEligibility || reviewCase.geography;
  return reviewCase.invalid === true
    || ['OUTSIDE_DANANG', 'INVALID_COORDINATE'].includes(geographic)
    || reviewCase.categoryEligibility === 'EXCLUDED'
    || reviewCase.provenanceComplete !== true
    || (reviewCase.fieldChanges || []).some((item) => item.state === 'LICENSE_RESTRICTED');
}

function decision(outcome, reasonCodes, reviewCase, additions = {}) {
  if (!POLICY_OUTCOMES.includes(outcome)) throw new Error(`Unknown policy outcome: ${outcome}`);
  return {
    caseId: reviewCase.caseId,
    outcome,
    reasonCodes,
    canonicalId: reviewCase.canonical?.id || reviewCase.canonicalId || null,
    source: reviewCase.source?.source || reviewCase.source || null,
    sourceId: reviewCase.source?.sourceId || reviewCase.sourceId || null,
    ...additions,
  };
}

function evaluateReviewCase(reviewCase, {
  memory = [], config, versions = DEFAULT_VERSIONS, materializeReusableApprovals = false,
} = {}) {
  if (reviewCase.syncStatus === 'UNCHANGED') {
    return decision('SKIP_UNCHANGED', ['unchanged_identity_and_fields'], reviewCase);
  }

  const reuse = findReusableDecision(memory, reviewCase, versions);
  if (reuse.reusable) {
    const additions = safeChanges(reviewCase, config).filter((change) => (
      reuse.changedFields?.includes(change.field)
        || (materializeReusableApprovals && reuse.record.approvedFields.includes(change.field))
    ));
    if (reuse.outcome === 'REUSE_APPROVAL' && additions.length > 0) {
      return decision('AUTO_ACCEPT_SAFE', [
        'entity_approval_reused', 'new_safe_fields_validated_only',
      ], reviewCase, {
        decisionSource: 'POLICY',
        reusedDecisionReference: reuse.record.decisionReference,
        applyOperations: additions.map((change) => ({
          operation: 'ENRICH_EXISTING', field: change.field,
          oldValue: change.oldValue ?? null, newValue: change.newValue,
          provenance: change.provenance,
        })),
      });
    }
    return decision(reuse.outcome, [
      reuse.outcome === 'DEFER' ? 'unchanged_defer_suppressed' : 'prior_human_decision_reused',
    ], reviewCase, { decisionReference: reuse.record.decisionReference });
  }

  if (hasHardSafetyFailure(reviewCase)) {
    return decision('AUTO_REJECT', ['system_safety_or_provenance_failure'], reviewCase, {
      decisionSource: 'SYSTEM_SAFETY',
    });
  }

  const classification = reviewCase.resolver?.classification || reviewCase.classification;
  if (classification === 'SOURCE_DUPLICATE') {
    return decision('AUTO_REJECT', ['source_duplicate_not_a_new_entity'], reviewCase, {
      decisionSource: 'SYSTEM_SAFETY',
    });
  }
  if (['PROBABLE_MATCH', 'AMBIGUOUS'].includes(classification)) {
    return decision('HUMAN_REVIEW', ['identity_not_safe_for_automation'], reviewCase);
  }
  if (classification === 'NEW_CANDIDATE') {
    const tier = classifyNewTier(reviewCase);
    if (tier === 'NEW_TIER_D_INVALID_OR_EXCLUDE') {
      return decision('AUTO_REJECT', ['new_candidate_excluded_or_invalid'], reviewCase, {
        decisionSource: 'SYSTEM_SAFETY',
      });
    }
    if (tier === 'NEW_TIER_A_STRONG') {
      return decision('HUMAN_REVIEW', ['new_entity_requires_human_approval'], reviewCase);
    }
    return decision('DEFER', ['new_candidate_not_ready_for_review'], reviewCase);
  }

  const additions = safeChanges(reviewCase, config);
  const evidence = reviewCase.evidence || {};
  const distance = evidence.checks?.resolverDistanceMeters;
  const strictMatch = classification === 'HIGH_CONFIDENCE_MATCH'
    && evidence.confidence === 'STRONG'
    && evidence.independent === true
    && (reviewCase.resolver?.confidence ?? 0) >= config.minimumAutoAcceptConfidence
    && (distance === undefined || distance === null || distance <= config.maximumCoordinateDistanceMeters)
    && additions.length > 0;

  if (strictMatch) {
    return decision('AUTO_ACCEPT_SAFE', ['strict_existing_match_safe_fields_only'], reviewCase, {
      decisionSource: 'POLICY',
      applyOperations: additions.map((change) => ({
        operation: 'ENRICH_EXISTING',
        field: change.field,
        oldValue: change.oldValue ?? null,
        newValue: change.newValue,
        provenance: change.provenance,
      })),
    });
  }

  return decision('HUMAN_REVIEW', [
    reuse.reason || 'strict_auto_accept_gate_not_met',
  ], reviewCase);
}

function applySafetyBudgets(results, config) {
  const accepted = results.filter((item) => item.outcome === 'AUTO_ACCEPT_SAFE');
  const rejected = results.filter((item) => item.outcome === 'AUTO_REJECT');
  const acceptedFields = accepted.reduce((sum, item) => sum + item.applyOperations.length, 0);
  const rejectionRate = results.length === 0 ? 0 : rejected.length / results.length;
  const identityInvalidations = results.filter((item) => (
    item.reasonCodes || []).includes('IDENTITY_FINGERPRINT_CHANGED')
  ).length;
  const reasons = [];
  if (accepted.length > config.budgets.maximumAutoAcceptCases) reasons.push('AUTO_ACCEPT_CASE_BUDGET');
  if (acceptedFields > config.budgets.maximumAutoAcceptFields) reasons.push('AUTO_ACCEPT_FIELD_BUDGET');
  if (rejectionRate > config.budgets.maximumAutoRejectRate) reasons.push('AUTO_REJECT_RATE_BUDGET');
  if (identityInvalidations > config.budgets.maximumIdentityInvalidations) {
    reasons.push('IDENTITY_INVALIDATION_BUDGET');
  }
  return {
    tripped: reasons.length > 0,
    reasons,
    metrics: {
      autoAcceptCases: accepted.length,
      autoAcceptFields: acceptedFields,
      rejectionRate,
      identityInvalidations,
    },
  };
}

function evaluateRunAnomalies(metrics = {}, config) {
  const thresholds = config.anomalyThresholds || {};
  const reasons = [];
  if ((metrics.changedRecords || 0) > thresholds.maximumChangedRecords) {
    reasons.push('CHANGED_RECORD_SPIKE');
  }
  if ((metrics.newCandidateRate || 0) > thresholds.maximumNewCandidateRate) {
    reasons.push('NEW_CANDIDATE_RATE_SPIKE');
  }
  if (metrics.insideBoundaryRate !== undefined
    && metrics.insideBoundaryRate < thresholds.minimumInsideBoundaryRate) {
    reasons.push('BOUNDARY_INSIDE_RATE_SHIFT');
  }
  if ((metrics.provenanceFailureRate || 0) > thresholds.maximumProvenanceFailureRate) {
    reasons.push('PROVENANCE_FAILURE_SPIKE');
  }
  if (metrics.snapshotVersionRegressed === true) reasons.push('SNAPSHOT_VERSION_REGRESSION');
  if (metrics.canonicalSha256
    && metrics.canonicalSha256 !== thresholds.expectedCanonicalSha256) {
    reasons.push('UNEXPECTED_CANONICAL_BASELINE');
  }
  if ((metrics.lockedFieldMutationAttempts || 0) > 0) reasons.push('LOCKED_FIELD_MUTATION_ATTEMPT');
  return { tripped: reasons.length > 0, reasons };
}

function runAutomatedReviewPolicy(cases, options) {
  const ordered = [...cases].sort((left, right) => left.caseId.localeCompare(right.caseId));
  const results = ordered.map((reviewCase) => evaluateReviewCase(reviewCase, options));
  const circuitBreaker = applySafetyBudgets(results, options.config);
  const anomalyGate = evaluateRunAnomalies(options.runMetrics, options.config);
  const applyPlan = circuitBreaker.tripped || anomalyGate.tripped ? [] : results
    .filter((item) => item.outcome === 'AUTO_ACCEPT_SAFE')
    .flatMap((item) => item.applyOperations.map((operation) => ({
      caseId: item.caseId,
      canonicalId: item.canonicalId,
      source: item.source,
      sourceId: item.sourceId,
      policyDecision: item.outcome,
      policyVersion: (options.versions || DEFAULT_VERSIONS).policy,
      decisionSource: item.decisionSource,
      reasonCodes: item.reasonCodes,
      evidenceReference: `case:${item.caseId}`,
      decisionMemoryReference: item.reusedDecisionReference || null,
      ...operation,
      license: operation.provenance?.license || null,
      rollback: { field: operation.field, restoreValue: operation.oldValue },
    })));
  return {
    status: 'CANDIDATE_NON_RUNTIME_NOT_CANONICAL_POLICY_PLAN',
    processingVersions: options.versions || DEFAULT_VERSIONS,
    results,
    applyPlan,
    circuitBreaker,
    anomalyGate,
    deterministicHash: stableHash({ results, applyPlan, circuitBreaker, anomalyGate }),
    canonicalWriteAuthorized: false,
    createNewAuthorized: false,
  };
}

module.exports = {
  DEFAULT_VERSIONS,
  POLICY_OUTCOMES,
  applySafetyBudgets,
  evaluateRunAnomalies,
  evaluateReviewCase,
  runAutomatedReviewPolicy,
  safeChanges,
};
