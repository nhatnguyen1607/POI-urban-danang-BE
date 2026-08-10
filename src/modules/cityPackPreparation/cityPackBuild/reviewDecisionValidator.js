const { ACTIONS_BY_DECISION } = require('../reviewQueue');

const SUPPORTED_ACTIONS = Object.freeze([
  'ACCEPT_MATCH',
  'REJECT_MATCH',
  'CREATE_NEW',
  'MERGE_SOURCE_DUPLICATE',
  'KEEP_SEPARATE',
  'DEFER',
]);

function loadReviewDecisions(decisionDocument) {
  const decisions = Array.isArray(decisionDocument?.decisions) ? decisionDocument.decisions : [];

  return decisions.map((decision) => ({
    queueId: String(decision.queueId || '').trim(),
    action: String(decision.action || '').trim(),
    canonicalPoiId: decision.canonicalPoiId || null,
    reviewerNote: decision.reviewerNote || null,
  }));
}

function validateReviewDecisions({ decisions, reviewQueue, canonicalPois }) {
  const errors = [];
  const queueById = new Map(reviewQueue.map((item) => [item.queueId, item]));
  const canonicalIds = new Set(canonicalPois.map((poi) => poi.id));
  const seenQueueIds = new Set();

  for (const decision of decisions) {
    if (!decision.queueId || !queueById.has(decision.queueId)) {
      errors.push({
        code: 'UNKNOWN_QUEUE_ID',
        queueId: decision.queueId,
      });
      continue;
    }

    if (seenQueueIds.has(decision.queueId)) {
      errors.push({
        code: 'DUPLICATE_OR_CONFLICTING_DECISION',
        queueId: decision.queueId,
      });
      continue;
    }
    seenQueueIds.add(decision.queueId);

    if (!SUPPORTED_ACTIONS.includes(decision.action)) {
      errors.push({
        code: 'UNSUPPORTED_ACTION',
        queueId: decision.queueId,
        action: decision.action,
      });
      continue;
    }

    const queueItem = queueById.get(decision.queueId);
    const allowedActions = ACTIONS_BY_DECISION[queueItem.decision] || [];
    if (!allowedActions.includes(decision.action)) {
      errors.push({
        code: 'ACTION_NOT_ALLOWED_FOR_DECISION',
        queueId: decision.queueId,
        action: decision.action,
        queueDecision: queueItem.decision,
      });
      continue;
    }

    if (decision.action === 'ACCEPT_MATCH') {
      if (!decision.canonicalPoiId || !canonicalIds.has(decision.canonicalPoiId)) {
        errors.push({
          code: 'INVALID_CANONICAL_ID',
          queueId: decision.queueId,
          canonicalPoiId: decision.canonicalPoiId,
        });
      }

      const candidateIds = new Set((queueItem.candidateCanonicalPois || []).map((candidate) => candidate.canonicalPoiId));
      if (decision.canonicalPoiId && !candidateIds.has(decision.canonicalPoiId)) {
        errors.push({
          code: 'UNSAFE_AMBIGUOUS_AUTO_RESOLUTION',
          queueId: decision.queueId,
          canonicalPoiId: decision.canonicalPoiId,
        });
      }
    }
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

module.exports = {
  SUPPORTED_ACTIONS,
  loadReviewDecisions,
  validateReviewDecisions,
};
