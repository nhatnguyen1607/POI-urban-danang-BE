const { DECISIONS } = require('./entityResolution');

const ACTIONS_BY_DECISION = Object.freeze({
  [DECISIONS.AMBIGUOUS]: ['ACCEPT_MATCH', 'REJECT_MATCH', 'KEEP_SEPARATE', 'DEFER'],
  [DECISIONS.NEW_CANDIDATE]: ['CREATE_NEW', 'REJECT_MATCH', 'DEFER'],
  [DECISIONS.SOURCE_DUPLICATE]: ['MERGE_SOURCE_DUPLICATE', 'KEEP_SEPARATE', 'DEFER'],
  [DECISIONS.INVALID]: ['REJECT_MATCH', 'DEFER'],
});

function buildReviewQueue({ matches, duplicates }) {
  const queue = [];

  for (const match of matches) {
    if (![DECISIONS.AMBIGUOUS, DECISIONS.NEW_CANDIDATE, DECISIONS.INVALID].includes(match.decision)) {
      continue;
    }

    queue.push({
      queueId: `${match.decision}:${match.source}:${match.sourceId}`,
      decision: match.decision,
      sourceRecord: {
        source: match.source,
        sourceId: match.sourceId,
        name: match.sourceName,
      },
      candidateCanonicalPois: match.candidates,
      matchingEvidence: {
        confidence: match.confidence,
        reasonCodes: match.reasonCodes,
      },
      provenance: match.provenance,
      license: match.license,
      suggestedActions: ACTIONS_BY_DECISION[match.decision],
    });
  }

  for (const duplicate of duplicates) {
    queue.push({
      queueId: `${duplicate.decision}:${duplicate.source}:${duplicate.sourceIdA}:${duplicate.sourceIdB}`,
      decision: duplicate.decision,
      sourceRecord: {
        source: duplicate.sourceA || duplicate.source,
        sourceId: duplicate.sourceIdA,
        name: duplicate.nameA,
      },
      duplicateRecord: {
        source: duplicate.sourceB || duplicate.source,
        sourceId: duplicate.sourceIdB,
        name: duplicate.nameB,
      },
      candidateCanonicalPois: [],
      matchingEvidence: {
        confidence: duplicate.confidence,
        distanceMeters: duplicate.distanceMeters,
        nameSimilarity: duplicate.nameSimilarity,
        categoryCompatibility: duplicate.categoryCompatibility ?? null,
        reasonCodes: duplicate.reasonCodes,
      },
      provenance: duplicate.records.map((record) => record.provenance),
      license: duplicate.records.map((record) => record.license),
      suggestedActions: ACTIONS_BY_DECISION[duplicate.decision],
    });
  }

  return queue.sort((a, b) => a.queueId.localeCompare(b.queueId));
}

module.exports = {
  ACTIONS_BY_DECISION,
  buildReviewQueue,
};
