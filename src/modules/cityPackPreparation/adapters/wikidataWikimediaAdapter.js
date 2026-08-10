const { createSourceAdapter } = require('../sourceAdapterContract');
const { normalizeSourceRecord } = require('../sourceRecord');

function normalizeWikidataWikimediaRecords(records) {
  return records
    .filter((record) => record.source === 'wikidata' || record.source === 'wikimedia_commons')
    .map((record) =>
      normalizeSourceRecord(
        {
          ...record,
          snapshotRef: record.snapshotRef || `stage4b-${record.source}-fixture`,
        },
        record.source,
      ),
    )
    .sort((a, b) => a.sourceId.localeCompare(b.sourceId));
}

module.exports = createSourceAdapter({
  source: 'wikidata_wikimedia',
  policyClass: 'OPEN_KNOWLEDGE_AND_MEDIA_ATTRIBUTION_REQUIRED',
  normalize: normalizeWikidataWikimediaRecords,
});
