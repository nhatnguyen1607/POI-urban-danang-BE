const { createSourceAdapter } = require('../sourceAdapterContract');
const { normalizeSourceRecord } = require('../sourceRecord');

function normalizeOvertureRecords(records) {
  return records
    .filter((record) => record.source === 'overture')
    .map((record) =>
      normalizeSourceRecord(
        {
          ...record,
          snapshotRef: record.snapshotRef || 'stage4b-overture-fixture',
        },
        'overture',
      ),
    )
    .sort((a, b) => a.sourceId.localeCompare(b.sourceId));
}

module.exports = createSourceAdapter({
  source: 'overture',
  policyClass: 'OPEN_PERMISSIVE_CANDIDATE',
  normalize: normalizeOvertureRecords,
});
