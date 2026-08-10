const fs = require('node:fs');
const path = require('node:path');

const overtureAdapter = require('./adapters/overtureAdapter');
const osmAdapter = require('./adapters/osmAdapter');
const wikidataWikimediaAdapter = require('./adapters/wikidataWikimediaAdapter');
const { inspectCanonicalDataset, readCanonicalPois } = require('./canonicalDataset');
const { resolveSourceRecords } = require('./entityResolution');
const { buildReviewQueue } = require('./reviewQueue');

const DEFAULT_ADAPTERS = [overtureAdapter, osmAdapter, wikidataWikimediaAdapter];

function readStage4bFixture(samplePath) {
  const parsed = JSON.parse(fs.readFileSync(samplePath, 'utf8'));
  return parsed.records || [];
}

function normalizeWithAdapters(rawRecords, adapters = DEFAULT_ADAPTERS) {
  return adapters
    .flatMap((adapter) => adapter.normalize(rawRecords))
    .sort((a, b) => {
      if (a.source !== b.source) return a.source.localeCompare(b.source);
      return a.sourceId.localeCompare(b.sourceId);
    });
}

function summarizeEnrichmentCandidates(normalizedRecords, matchResults) {
  const acceptable = new Map(
    matchResults.matches
      .filter((match) => ['HIGH_CONFIDENCE_MATCH', 'PROBABLE_MATCH'].includes(match.decision))
      .map((match) => [match.sourceId, match]),
  );

  return normalizedRecords
    .filter((record) => acceptable.has(record.sourceId))
    .map((record) => {
      const match = acceptable.get(record.sourceId);
      return {
        source: record.source,
        sourceId: record.sourceId,
        canonicalPoiId: match.bestCandidate?.canonicalPoiId || null,
        canonicalName: match.bestCandidate?.canonicalName || null,
        hasAddress: Boolean(record.address),
        hasWebsite: Boolean(record.website),
        hasPhone: Boolean(record.phone),
        hasOpeningHours: Boolean(record.openingHours),
        hasExternalIds: Object.keys(record.externalIds || {}).length > 0,
        hasMedia: Boolean(record.media || record.externalIds?.commons_file),
        policyClass: record.provenance.policyClass,
        attribution: record.provenance.attribution,
      };
    })
    .sort((a, b) => {
      if (a.source !== b.source) return a.source.localeCompare(b.source);
      return a.sourceId.localeCompare(b.sourceId);
    });
}

function toCsv(rows, headers) {
  const escapeCell = (value) => {
    const text = Array.isArray(value)
      ? value.join('|')
      : value === null || value === undefined
        ? ''
        : String(value);
    return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  };
  return [headers.join(',')]
    .concat(rows.map((row) => headers.map((header) => escapeCell(row[header])).join(',')))
    .join('\n')
    .concat('\n');
}

function writeJson(outputDir, fileName, value) {
  fs.mkdirSync(outputDir, { recursive: true });
  fs.writeFileSync(path.join(outputDir, fileName), `${JSON.stringify(value, null, 2)}\n`);
}

function writeCsv(outputDir, fileName, rows, headers) {
  fs.mkdirSync(outputDir, { recursive: true });
  fs.writeFileSync(path.join(outputDir, fileName), toCsv(rows, headers));
}

function runStage4cDryRun({ canonicalPath, samplePath, outputDir }) {
  const canonical = inspectCanonicalDataset(canonicalPath);
  const canonicalPois = readCanonicalPois(canonicalPath);
  const rawRecords = readStage4bFixture(samplePath);
  const normalizedRecords = normalizeWithAdapters(rawRecords);
  const matchResults = resolveSourceRecords(normalizedRecords, canonicalPois);
  const reviewQueue = buildReviewQueue(matchResults);
  const enrichmentCandidates = summarizeEnrichmentCandidates(normalizedRecords, matchResults);

  const summary = {
    status: 'NON_RUNTIME_STAGE4C_DRY_RUN',
    canonical,
    adapters: DEFAULT_ADAPTERS.map((adapter) => ({
      source: adapter.source,
      policyClass: adapter.policyClass,
      contractVersion: adapter.contractVersion,
    })),
    normalizedRecords: normalizedRecords.length,
    matchSummary: matchResults.summary,
    reviewQueueItems: reviewQueue.length,
    enrichmentCandidates: enrichmentCandidates.length,
    runtimeChanged: false,
  };

  writeJson(outputDir, 'stage4c_normalized_records.json', normalizedRecords);
  writeJson(outputDir, 'stage4c_match_results.json', matchResults);
  writeJson(outputDir, 'stage4c_review_queue.json', reviewQueue);
  writeJson(outputDir, 'stage4c_summary.json', summary);
  writeCsv(outputDir, 'stage4c_enrichment_candidates.csv', enrichmentCandidates, [
    'source',
    'sourceId',
    'canonicalPoiId',
    'canonicalName',
    'hasAddress',
    'hasWebsite',
    'hasPhone',
    'hasOpeningHours',
    'hasExternalIds',
    'hasMedia',
    'policyClass',
    'attribution',
  ]);

  return {
    canonical,
    normalizedRecords,
    matchResults,
    reviewQueue,
    enrichmentCandidates,
    summary,
  };
}

module.exports = {
  DEFAULT_ADAPTERS,
  normalizeWithAdapters,
  readStage4bFixture,
  runStage4cDryRun,
  summarizeEnrichmentCandidates,
};
