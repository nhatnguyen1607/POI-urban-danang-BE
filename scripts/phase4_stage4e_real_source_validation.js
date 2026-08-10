const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const { buildCandidateCityPack } = require('../src/modules/cityPackPreparation/cityPackBuild/cityPackBuilder');
const { runStage4cDryRun } = require('../src/modules/cityPackPreparation/dryRun');
const { loadRealSnapshotRecords } = require('../src/modules/cityPackPreparation/realSnapshot');

const ROOT = path.resolve(__dirname, '..');
const SNAPSHOT_DIR = path.join(ROOT, 'data', 'spikes', 'phase4', 'stage4e', 'snapshots');
const OUTPUT_DIR = path.join(ROOT, 'data', 'spikes', 'phase4', 'stage4e');
const BUILD_DIR = path.join(ROOT, 'data', 'citypacks', 'candidates', 'danang', 'stage4e-real-bounded');
const CANONICAL_PATH = path.join(ROOT, 'data', 'canonical', 'urbanagent_poi_master_v1.csv');
const ADAPTER_INPUT_PATH = path.join(OUTPUT_DIR, 'stage4e_adapter_input.json');
const REVIEW_DECISION_PATH = path.join(OUTPUT_DIR, 'review_decisions', 'stage4e_review_decisions.json');

function writeJson(filePath, value) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${JSON.stringify(value)}\n`);
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function boolCount(records, predicate) {
  return records.filter(predicate).length;
}

function fieldCoverage(records) {
  const fields = {
    name: (record) => Boolean(record.name),
    category: (record) => Boolean(record.category && record.category !== 'unknown'),
    coordinates: (record) => Number.isFinite(record.latitude) && Number.isFinite(record.longitude),
    address: (record) => Boolean(record.address),
    website: (record) => Boolean(record.website),
    phone: (record) => Boolean(record.phone),
    openingHours: (record) => Boolean(record.openingHours),
    externalIds: (record) => Object.values(record.externalIds || {}).some(Boolean),
    media: (record) => Boolean(record.media),
  };

  const bySource = {};
  for (const source of [...new Set(records.map((record) => record.source))].sort()) {
    const sourceRecords = records.filter((record) => record.source === source);
    bySource[source] = {
      records: sourceRecords.length,
      fields: Object.fromEntries(
        Object.entries(fields).map(([field, predicate]) => {
          const populated = boolCount(sourceRecords, predicate);
          return [field, {
            populated,
            coverage: sourceRecords.length ? Number((populated / sourceRecords.length).toFixed(4)) : 0,
          }];
        }),
      ),
    };
  }
  return bySource;
}

function provenanceReport(records) {
  const recordComplete = (record) => (
    record.provenance?.source
    && record.provenance?.sourceId
    && record.provenance?.snapshotRef
    && record.provenance?.policyClass
    && record.provenance?.license
    && record.provenance?.attribution
  );
  const mediaRecords = records.filter((record) => record.media);
  const mediaComplete = mediaRecords.filter((record) => (
    (record.media?.licenseName || record.media?.license)
    && (record.media?.attributionText || record.media?.attribution)
    && record.media?.source === 'wikimedia_commons'
    && record.media?.assetIdentifier
  ));
  const fieldEntries = records.flatMap((record) => Object.values(record.provenance?.fields || {}));
  const completeFields = fieldEntries.filter((field) => (
    field.source
    && field.sourceId
    && field.field
    && field.snapshotRef
    && field.policyClass
    && field.license
    && field.attribution
  ));

  return {
    records: records.length,
    completeRecordProvenance: records.filter(recordComplete).length,
    completeRecordProvenanceRate: records.length
      ? Number((records.filter(recordComplete).length / records.length).toFixed(4))
      : 0,
    fieldProvenanceEntries: fieldEntries.length,
    completeFieldProvenance: completeFields.length,
    completeFieldProvenanceRate: fieldEntries.length
      ? Number((completeFields.length / fieldEntries.length).toFixed(4))
      : 0,
    mediaRecords: mediaRecords.length,
    completePerFileMediaLicense: mediaComplete.length,
    completePerFileMediaLicenseRate: mediaRecords.length
      ? Number((mediaComplete.length / mediaRecords.length).toFixed(4))
      : 1,
  };
}

function representativeMatch(matches, decision) {
  const item = matches.find((match) => match.decision === decision);
  if (!item) return null;
  return {
    source: item.source,
    sourceId: item.sourceId,
    sourceName: item.sourceName,
    decision: item.decision,
    confidence: item.confidence,
    bestCandidate: item.bestCandidate,
    reasonCodes: item.reasonCodes,
  };
}

function sourceReports(stage4c) {
  const matches = stage4c.matchResults.matches;
  const exact = matches.filter(
    (match) => match.bestCandidate?.nameSimilarity === 1 && match.bestCandidate?.distanceMeters <= 50,
  );
  const fuzzy = matches.filter(
    (match) => ['HIGH_CONFIDENCE_MATCH', 'PROBABLE_MATCH'].includes(match.decision)
      && match.bestCandidate?.nameSimilarity < 1,
  );
  const conflicts = matches.filter(
    (match) => match.bestCandidate
      && match.bestCandidate.categoryCompatibility === 0
      && match.bestCandidate.nameSimilarity >= 0.6
      && match.bestCandidate.distanceMeters <= 150,
  );

  return {
    exactMatchReport: exact.map((item) => ({
      sourceId: item.sourceId,
      decision: item.decision,
      canonicalPoiId: item.bestCandidate.canonicalPoiId,
      distanceMeters: item.bestCandidate.distanceMeters,
      categoryCompatibility: item.bestCandidate.categoryCompatibility,
    })),
    fuzzyMatchReport: fuzzy.map((item) => ({
      sourceId: item.sourceId,
      decision: item.decision,
      canonicalPoiId: item.bestCandidate.canonicalPoiId,
      nameSimilarity: item.bestCandidate.nameSimilarity,
      distanceMeters: item.bestCandidate.distanceMeters,
    })),
    ambiguityReport: matches.filter((item) => item.decision === 'AMBIGUOUS'),
    probableNewEntityReport: matches.filter((item) => item.decision === 'NEW_CANDIDATE'),
    duplicateReport: stage4c.matchResults.duplicates,
    sourceConflictReport: conflicts.map((item) => ({
      sourceId: item.sourceId,
      sourceName: item.sourceName,
      canonicalPoiId: item.bestCandidate.canonicalPoiId,
      canonicalName: item.bestCandidate.canonicalName,
      categoryCompatibility: item.bestCandidate.categoryCompatibility,
    })),
    costReport: {
      paidApiCostUsd: 0,
      googlePlacesUsed: false,
      note: 'Public bounded research access only; operating cost at city scale was not measured.',
    },
  };
}

function buildOnce(outputDir, sourceSnapshots) {
  return buildCandidateCityPack({
    cityId: 'da-nang',
    buildId: 'stage4e-real-bounded',
    canonicalPath: CANONICAL_PATH,
    samplePath: ADAPTER_INPUT_PATH,
    reviewDecisionPath: REVIEW_DECISION_PATH,
    outputDir,
    sourceSnapshots,
  });
}

function runStage4eValidation() {
  const manifest = readJson(path.join(SNAPSHOT_DIR, 'snapshot_manifest.json'));
  const sourceSnapshots = manifest.snapshots.map((snapshot) => snapshot.snapshotRef);
  const rawRecords = loadRealSnapshotRecords({
    overture: path.join(SNAPSHOT_DIR, 'overture_places_2026-07-22.0_bounded.json'),
    osm: path.join(SNAPSHOT_DIR, 'osm_2026-08-10_bounded.json'),
    wikidata: path.join(SNAPSHOT_DIR, 'wikidata_2026-08-10_bounded.json'),
    commons: path.join(SNAPSHOT_DIR, 'wikimedia_commons_2026-08-10_metadata.json'),
  });
  writeJson(ADAPTER_INPUT_PATH, { records: rawRecords });

  const stage4cTemp = fs.mkdtempSync(path.join(os.tmpdir(), 'urbanagent-stage4e-stage4c-'));
  const stage4c = runStage4cDryRun({
    canonicalPath: CANONICAL_PATH,
    samplePath: ADAPTER_INPUT_PATH,
    outputDir: stage4cTemp,
  });
  const coverage = fieldCoverage(stage4c.normalizedRecords);
  const provenance = provenanceReport(stage4c.normalizedRecords);
  const reports = sourceReports(stage4c);
  const spotChecks = {
    highConfidence: representativeMatch(stage4c.matchResults.matches, 'HIGH_CONFIDENCE_MATCH'),
    probable: representativeMatch(stage4c.matchResults.matches, 'PROBABLE_MATCH'),
    ambiguous: representativeMatch(stage4c.matchResults.matches, 'AMBIGUOUS'),
    newCandidate: representativeMatch(stage4c.matchResults.matches, 'NEW_CANDIDATE'),
    sourceDuplicate: stage4c.matchResults.duplicates[0] || null,
    limitation: 'Representative observations only; this bounded sample does not measure city-wide accuracy.',
  };

  const first = buildOnce(fs.mkdtempSync(path.join(os.tmpdir(), 'urbanagent-stage4e-build-a-')), sourceSnapshots);
  const second = buildOnce(fs.mkdtempSync(path.join(os.tmpdir(), 'urbanagent-stage4e-build-b-')), sourceSnapshots);
  const deterministic = (
    first.summary.artifactHashes.candidatePackHash === second.summary.artifactHashes.candidatePackHash
    && first.summary.artifactHashes.licenseManifestHash === second.summary.artifactHashes.licenseManifestHash
    && first.summary.artifactHashes.reviewDecisionHash === second.summary.artifactHashes.reviewDecisionHash
  );
  const finalBuild = buildOnce(BUILD_DIR, sourceSnapshots);

  const sourceCounts = Object.fromEntries(
    [...new Set(stage4c.normalizedRecords.map((record) => record.source))]
      .sort()
      .map((source) => [source, stage4c.normalizedRecords.filter((record) => record.source === source).length]),
  );
  const validNormalizedRecords = stage4c.normalizedRecords.filter(
    (record) => Number.isFinite(record.latitude) && Number.isFinite(record.longitude) && record.name,
  ).length;
  const summary = {
    status: 'CANDIDATE_NON_RUNTIME_NOT_CANONICAL',
    geographicBoundingBox: manifest.bbox,
    sourceCounts,
    normalizedRecords: stage4c.normalizedRecords.length,
    validNormalizedRecords,
    invalidNormalizedRecords: stage4c.normalizedRecords.length - validNormalizedRecords,
    matchSummary: stage4c.matchResults.summary,
    reviewQueueCount: stage4c.reviewQueue.length,
    candidateCount: finalBuild.summary.candidateTotal,
    fieldCoverage: coverage,
    provenance,
    candidateBuild: finalBuild.summary,
    deterministic,
    canonical: stage4c.canonical,
    runtimeChanged: false,
    firestoreChanged: false,
    productionDatabaseTouched: false,
    googlePlacesIncluded: false,
    scientificLimit: 'Small bounded convenience sample; no city-wide accuracy or coverage claim.',
  };

  writeJson(path.join(OUTPUT_DIR, 'stage4e_normalized_records.json'), stage4c.normalizedRecords);
  writeJson(path.join(OUTPUT_DIR, 'stage4e_match_results.json'), stage4c.matchResults);
  writeJson(path.join(OUTPUT_DIR, 'stage4e_review_queue.json'), stage4c.reviewQueue);
  writeJson(path.join(OUTPUT_DIR, 'stage4e_field_coverage.json'), coverage);
  writeJson(path.join(OUTPUT_DIR, 'stage4e_manual_spot_checks.json'), spotChecks);
  writeJson(path.join(OUTPUT_DIR, 'stage4e_source_reports.json'), reports);
  writeJson(path.join(OUTPUT_DIR, 'stage4e_summary.json'), summary);
  return { summary, spotChecks, reports };
}

if (require.main === module) {
  console.log(JSON.stringify(runStage4eValidation().summary, null, 2));
}

module.exports = {
  fieldCoverage,
  provenanceReport,
  runStage4eValidation,
};
