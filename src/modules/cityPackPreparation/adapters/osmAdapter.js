const { createSourceAdapter } = require('../sourceAdapterContract');
const { normalizeSourceRecord } = require('../sourceRecord');

function normalizeOsmRecords(records) {
  return records
    .filter((record) => record.source === 'osm')
    .map((record) =>
      normalizeSourceRecord(
        {
          ...record,
          snapshotRef: record.snapshotRef || 'stage4b-osm-fixture',
        },
        'osm',
      ),
    )
    .sort((a, b) => a.sourceId.localeCompare(b.sourceId));
}

module.exports = createSourceAdapter({
  source: 'osm',
  policyClass: 'OPEN_SHAREALIKE_ISOLATED',
  normalize: normalizeOsmRecords,
});
