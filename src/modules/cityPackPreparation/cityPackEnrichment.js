const { stableHash } = require('./decisionMemory');

const ENRICHMENT_FIELDS = new Set(['address', 'phone', 'website', 'openingHours', 'media']);

function buildEnrichmentRecord(item, schemaVersion = 'stage4l-enrichment-v1') {
  if (!ENRICHMENT_FIELDS.has(item.field)) throw new Error(`Unsupported enrichment field: ${item.field}`);
  if (!item.canonical_id || !item.case_id || !item.source_id || !item.provenance) {
    throw new Error('Enrichment record requires canonical, case, source and provenance references.');
  }
  return {
    schemaVersion,
    status: 'CANDIDATE_NON_RUNTIME_NOT_CANONICAL',
    canonicalId: item.canonical_id,
    caseId: item.case_id,
    field: item.field,
    value: item.new_value,
    humanDecision: item.human_decision,
    source: item.source,
    sourceId: item.source_id,
    snapshotReference: item.provenance.snapshotRef || null,
    provenance: item.provenance,
    license: item.license || item.provenance.license || null,
    sourceExternalIds: item.source_external_ids || null,
    firstSeen: item.first_seen || null,
    lastSeen: item.last_seen || null,
    fieldUpdatedReference: item.field_updated_reference || 'stage4k-human-approval',
    processingVersion: schemaVersion,
    rollback: { action: 'REMOVE_SIDECAR_RECORD' },
  };
}

function recoverStage4kSkippedEnrichments(preApplySummary) {
  const recovered = (preApplySummary.skipped || [])
    .filter((item) => item.status === 'SKIPPED_UNSUPPORTED_CANONICAL_FIELD')
    .filter((item) => ['phone', 'website'].includes(item.field))
    .map((item) => buildEnrichmentRecord(item))
    .sort((left, right) => `${left.canonicalId}:${left.field}`.localeCompare(`${right.canonicalId}:${right.field}`));
  return {
    status: 'CANDIDATE_NON_RUNTIME_NOT_CANONICAL',
    schemaVersion: 'stage4l-enrichment-v1',
    records: recovered,
    recordCount: recovered.length,
    deterministicHash: stableHash(recovered),
    runtimeInstalled: false,
  };
}

module.exports = {
  ENRICHMENT_FIELDS,
  buildEnrichmentRecord,
  recoverStage4kSkippedEnrichments,
};
