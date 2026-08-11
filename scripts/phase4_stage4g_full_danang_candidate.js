const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const { performance } = require('node:perf_hooks');

const { buildCandidateCityPack, stableHash } = require('../src/modules/cityPackPreparation/cityPackBuild/cityPackBuilder');
const { inspectCanonicalDataset } = require('../src/modules/cityPackPreparation/canonicalDataset');
const { runStage4cDryRun } = require('../src/modules/cityPackPreparation/dryRun');
const { hasValidCoordinates } = require('../src/modules/cityPackPreparation/sourceRecord');
const { loadRealSnapshotRecords } = require('../src/modules/cityPackPreparation/realSnapshot');
const { fieldCoverage, provenanceReport } = require('./phase4_stage4e_real_source_validation');

const ROOT = path.resolve(__dirname, '..');
const CONFIG_PATH = path.join(ROOT, 'config', 'phase4_stage4g_sources.json');
const CANONICAL_PATH = path.join(ROOT, 'data', 'canonical', 'urbanagent_poi_master_v1.csv');
const STAGE4G_DIR = path.join(ROOT, 'data', 'spikes', 'phase4', 'stage4g');
const CACHE_DIR = path.join(STAGE4G_DIR, 'cache');
const SNAPSHOT_MANIFEST_PATH = path.join(STAGE4G_DIR, 'snapshot_manifest.json');
const BUILD_DIR = process.env.URBANAGENT_STAGE4G_OUTPUT_DIR
  || path.join(ROOT, 'data', 'citypacks', 'candidates', 'danang', 'stage4g-full-city');
const STATUS = 'CANDIDATE_NON_RUNTIME_NOT_CANONICAL';

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function writeJson(filePath, value) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${JSON.stringify(value)}\n`);
}

function fileHash(filePath) {
  return crypto.createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');
}

function snapshotPaths() {
  return {
    overture: path.join(CACHE_DIR, 'overture_full_danang.json'),
    osm: path.join(CACHE_DIR, 'osm_full_danang.json'),
    wikidata: path.join(CACHE_DIR, 'wikidata_full_danang.json'),
    commons: path.join(CACHE_DIR, 'wikimedia_commons_full_danang_metadata.json'),
  };
}

function verifySnapshotHashes(manifest, paths) {
  const bySource = new Map(manifest.snapshots.map((snapshot) => [snapshot.source, snapshot]));
  for (const [source, filePath] of Object.entries(paths)) {
    const manifestSource = source === 'commons' ? 'wikimedia_commons' : source;
    const expected = bySource.get(manifestSource);
    if (!expected || !fs.existsSync(filePath) || fileHash(filePath) !== expected.sha256) {
      throw new Error(`Stage 4G snapshot verification failed for ${manifestSource}.`);
    }
  }
}

function insideBoundary(record, boundary) {
  return (
    hasValidCoordinates(record)
    && record.latitude >= boundary.south
    && record.latitude <= boundary.north
    && record.longitude >= boundary.west
    && record.longitude <= boundary.east
  );
}

function duplicateGroupCount(duplicates) {
  const parent = new Map();
  const find = (value) => {
    if (!parent.has(value)) parent.set(value, value);
    if (parent.get(value) !== value) parent.set(value, find(parent.get(value)));
    return parent.get(value);
  };
  const union = (left, right) => {
    const leftRoot = find(left);
    const rightRoot = find(right);
    if (leftRoot !== rightRoot) parent.set(rightRoot, leftRoot);
  };
  for (const duplicate of duplicates) union(duplicate.sourceIdA, duplicate.sourceIdB);
  return new Set([...parent.keys()].map(find)).size;
}

function buildConflictQueue(stage4c, acquisitionManifest) {
  const recordsById = new Map(stage4c.normalizedRecords.map((record) => [record.sourceId, record]));
  const conflicts = stage4c.matchResults.matches
    .filter((match) => (
      ['HIGH_CONFIDENCE_MATCH', 'PROBABLE_MATCH'].includes(match.decision)
      && match.bestCandidate
      && match.bestCandidate.categoryCompatibility < 1
    ))
    .map((match) => {
      const record = recordsById.get(match.sourceId);
      return {
        queueId: `CONFLICTING_ENRICHMENT:${match.source}:${match.sourceId}`,
        decision: 'CONFLICTING_ENRICHMENT',
        sourceRecord: { source: match.source, sourceId: match.sourceId, name: match.sourceName },
        candidateCanonicalPois: match.candidates,
        matchingEvidence: {
          confidence: match.confidence,
          categoryCompatibility: match.bestCandidate.categoryCompatibility,
          reasonCodes: ['compatible_but_nonidentical_category'],
        },
        provenance: record.provenance,
        license: record.license,
        suggestedActions: ['ACCEPT_MATCH', 'REJECT_MATCH', 'KEEP_SEPARATE', 'DEFER'],
      };
    });
  const licenseConflicts = acquisitionManifest.snapshots
    .filter((snapshot) => snapshot.licenseConflictCount > 0)
    .map((snapshot) => ({
      queueId: `PROVENANCE_LICENSE_CONFLICT:${snapshot.source}:${snapshot.snapshotRef}`,
      decision: 'PROVENANCE_LICENSE_CONFLICT',
      sourceRecord: { source: snapshot.source, sourceId: snapshot.snapshotRef, name: null },
      candidateCanonicalPois: [],
      matchingEvidence: {
        confidence: 0,
        reasonCodes: ['source_media_missing_required_license_metadata'],
        affectedRecords: snapshot.licenseConflictCount,
      },
      provenance: { source: snapshot.source, snapshotRef: snapshot.snapshotRef },
      license: snapshot.license,
      suggestedActions: ['REJECT_MATCH', 'DEFER'],
    }));
  return [...conflicts, ...licenseConflicts].sort((left, right) => left.queueId.localeCompare(right.queueId));
}

function representative(matches, decision, limit = 3) {
  return matches
    .filter((match) => match.decision === decision)
    .sort((left, right) => {
      if (right.confidence !== left.confidence) return right.confidence - left.confidence;
      return left.sourceId.localeCompare(right.sourceId);
    })
    .slice(0, limit)
    .map((match) => ({
      source: match.source,
      sourceId: match.sourceId,
      sourceName: match.sourceName,
      decision: match.decision,
      confidence: match.confidence,
      reasonCodes: match.reasonCodes,
      bestCandidate: match.bestCandidate,
    }));
}

function spotChecks(stage4c, acquisitionManifest) {
  return {
    highConfidence: representative(stage4c.matchResults.matches, 'HIGH_CONFIDENCE_MATCH'),
    probable: representative(stage4c.matchResults.matches, 'PROBABLE_MATCH'),
    ambiguous: representative(stage4c.matchResults.matches, 'AMBIGUOUS'),
    newCandidate: representative(stage4c.matchResults.matches, 'NEW_CANDIDATE'),
    invalid: representative(stage4c.matchResults.matches, 'INVALID'),
    duplicates: stage4c.matchResults.duplicates.slice(0, 5).map((duplicate) => ({
      sourcePair: duplicate.sourcePair,
      sourceIdA: duplicate.sourceIdA,
      sourceIdB: duplicate.sourceIdB,
      nameA: duplicate.nameA,
      nameB: duplicate.nameB,
      distanceMeters: duplicate.distanceMeters,
      categoryCompatibility: duplicate.categoryCompatibility,
      reasonCodes: duplicate.reasonCodes,
    })),
    outOfBoundary: acquisitionManifest.snapshots.map((snapshot) => ({
      source: snapshot.source,
      clearlyOutOfCityCount: snapshot.clearlyOutOfCityCount,
      invalidRecordCount: snapshot.invalidRecordCount,
    })),
    reviewNote: 'Representative deterministic samples only; no scientific precision or recall claim.',
  };
}

function buildOnce({ outputDir, stage4c, sourceSnapshots, writeArtifacts = true }) {
  const decisionsPath = path.join(CACHE_DIR, 'stage4g_review_decisions.json');
  if (!fs.existsSync(decisionsPath)) writeJson(decisionsPath, { decisions: [] });
  return buildCandidateCityPack({
    cityId: 'da-nang',
    buildId: 'stage4g-full-danang-candidate',
    canonicalPath: CANONICAL_PATH,
    samplePath: path.join(CACHE_DIR, 'stage4g_adapter_input.json'),
    reviewDecisionPath: decisionsPath,
    outputDir,
    sourceSnapshots,
    includeProposedNewCandidates: true,
    resolutionOptions: {
      useSpatialIndex: true,
      useSpatialDuplicateIndex: true,
      includeCrossSourceDuplicates: true,
      maxCandidateDistanceMeters: 800,
    },
    preparedStage4c: stage4c,
    writeArtifacts,
  });
}

function runStage4g() {
  const config = readJson(CONFIG_PATH);
  const acquisitionManifest = readJson(SNAPSHOT_MANIFEST_PATH);
  const paths = snapshotPaths();
  verifySnapshotHashes(acquisitionManifest, paths);
  const canonicalHashBefore = fileHash(CANONICAL_PATH);
  const canonical = {
    ...inspectCanonicalDataset(CANONICAL_PATH),
    path: 'data/canonical/urbanagent_poi_master_v1.csv',
  };
  const timings = {};

  const normalizeStarted = performance.now();
  const rawRecords = loadRealSnapshotRecords(paths);
  const boundaryRejected = rawRecords.filter((record) => !insideBoundary(record, config.boundary));
  const inBoundaryRecords = rawRecords.filter((record) => insideBoundary(record, config.boundary));
  writeJson(path.join(CACHE_DIR, 'stage4g_adapter_input.json'), { records: inBoundaryRecords });
  timings.snapshotLoadAndBoundarySeconds = Number(((performance.now() - normalizeStarted) / 1000).toFixed(3));

  const resolutionStarted = performance.now();
  const stage4c = runStage4cDryRun({
    canonicalPath: CANONICAL_PATH,
    samplePath: path.join(CACHE_DIR, 'stage4g_adapter_input.json'),
    outputDir: path.join(CACHE_DIR, 'stage4c-output'),
    resolutionOptions: {
      useSpatialIndex: true,
      useSpatialDuplicateIndex: true,
      includeCrossSourceDuplicates: true,
      maxCandidateDistanceMeters: 800,
    },
    writeOutputs: false,
  });
  const conflictQueue = buildConflictQueue(stage4c, acquisitionManifest);
  stage4c.reviewQueue = [...stage4c.reviewQueue, ...conflictQueue]
    .sort((left, right) => left.queueId.localeCompare(right.queueId));
  stage4c.summary.reviewQueueItems = stage4c.reviewQueue.length;
  timings.entityResolutionSeconds = Number(((performance.now() - resolutionStarted) / 1000).toFixed(3));

  const secondResolutionStarted = performance.now();
  const stage4cSecond = runStage4cDryRun({
    canonicalPath: CANONICAL_PATH,
    samplePath: path.join(CACHE_DIR, 'stage4g_adapter_input.json'),
    outputDir: path.join(CACHE_DIR, 'stage4c-output-repeat'),
    resolutionOptions: {
      useSpatialIndex: true,
      useSpatialDuplicateIndex: true,
      includeCrossSourceDuplicates: true,
      maxCandidateDistanceMeters: 800,
    },
    writeOutputs: false,
  });
  const secondConflictQueue = buildConflictQueue(stage4cSecond, acquisitionManifest);
  stage4cSecond.reviewQueue = [...stage4cSecond.reviewQueue, ...secondConflictQueue]
    .sort((left, right) => left.queueId.localeCompare(right.queueId));
  stage4cSecond.summary.reviewQueueItems = stage4cSecond.reviewQueue.length;
  timings.secondEntityResolutionSeconds = Number(((performance.now() - secondResolutionStarted) / 1000).toFixed(3));

  const sourceSnapshots = acquisitionManifest.snapshots.map((snapshot) => snapshot.snapshotRef).sort();
  const firstStarted = performance.now();
  const first = buildOnce({
    outputDir: BUILD_DIR,
    stage4c,
    sourceSnapshots,
  });
  timings.firstCandidateBuildSeconds = Number(((performance.now() - firstStarted) / 1000).toFixed(3));
  const secondStarted = performance.now();
  const second = buildOnce({
    outputDir: BUILD_DIR,
    stage4c: stage4cSecond,
    sourceSnapshots,
    writeArtifacts: false,
  });
  timings.secondCandidateBuildSeconds = Number(((performance.now() - secondStarted) / 1000).toFixed(3));
  const finalBuild = first;

  const deterministic = (
    stableHash(first.pack) === stableHash(second.pack)
    && stableHash(first.licenseManifest) === stableHash(second.licenseManifest)
    && stableHash(first.summary) === stableHash(second.summary)
    && stableHash(stage4c.normalizedRecords) === stableHash(stage4cSecond.normalizedRecords)
    && stableHash(stage4c.matchResults) === stableHash(stage4cSecond.matchResults)
    && stableHash(stage4c.reviewQueue) === stableHash(stage4cSecond.reviewQueue)
  );
  const coverage = fieldCoverage(stage4c.normalizedRecords);
  const provenance = provenanceReport(stage4c.normalizedRecords);
  const sourceCounts = Object.fromEntries(
    ['overture', 'osm', 'wikidata'].map((source) => [source, {
      raw: acquisitionManifest.snapshots.find((snapshot) => snapshot.source === source)?.rawRecordCount || 0,
      normalized: stage4c.normalizedRecords.filter((record) => record.source === source).length,
    }]),
  );
  const acquisitionInvalid = acquisitionManifest.snapshots.reduce(
    (total, snapshot) => total + snapshot.invalidRecordCount,
    0,
  );
  const acquisitionOutOfCity = acquisitionManifest.snapshots.reduce(
    (total, snapshot) => total + snapshot.clearlyOutOfCityCount,
    0,
  );
  const duplicateGroups = duplicateGroupCount(stage4c.matchResults.duplicates);
  const summary = {
    status: STATUS,
    cityId: config.cityId,
    boundary: config.boundary,
    canonical,
    sourceCounts,
    wikimediaCommons: {
      requested: acquisitionManifest.snapshots.find((snapshot) => snapshot.source === 'wikimedia_commons')?.rawRecordCount || 0,
      complete: finalBuild.licenseManifest.mediaLicenseEntries.length,
    },
    normalizedRecords: stage4c.normalizedRecords.length,
    invalidRecords: stage4c.matchResults.summary.invalid + acquisitionInvalid,
    clearlyOutOfCityRecords: acquisitionOutOfCity + boundaryRejected.length,
    resolution: stage4c.matchResults.summary,
    duplicateGroups,
    proposedEnrichmentCount: finalBuild.pack.matchedEnrichment.length,
    proposedNewCandidateCount: finalBuild.pack.proposedNewCandidates.length,
    candidatePackTotal: finalBuild.summary.candidateTotal,
    reviewQueueCount: stage4c.reviewQueue.length,
    conflictQueueCount: conflictQueue.length,
    rejectedCount: finalBuild.pack.rejectedRecords.length,
    fieldCoverage: coverage,
    provenance,
    licenseManifestCounts: finalBuild.licenseManifest.counts,
    deterministic,
    timings: {
      acquisitionSeconds: acquisitionManifest.acquisitionSeconds,
      ...timings,
    },
    approximatePeakRssMb: Number((process.memoryUsage().rss / 1024 / 1024).toFixed(2)),
    runtimeChanged: false,
    firestoreChanged: false,
    productionDatabaseTouched: false,
    googlePlacesIncluded: false,
    scientificLimit: 'Full supported bbox preparation run; no labeled ground truth or precision/recall claim.',
  };
  const quality = spotChecks(stage4c, acquisitionManifest);
  const proposedChanges = {
    status: STATUS,
    canonicalEnrichments: summary.proposedEnrichmentCount,
    proposedNewPois: summary.proposedNewCandidateCount,
    conflictingEnrichments: conflictQueue.filter((item) => item.decision === 'CONFLICTING_ENRICHMENT').length,
    provenanceLicenseConflicts: conflictQueue.filter((item) => item.decision === 'PROVENANCE_LICENSE_CONFLICT').length,
    manualReviewItems: summary.reviewQueueCount,
    sourceDuplicateGroups: duplicateGroups,
    invalidRecords: summary.invalidRecords,
    outOfCityRecords: summary.clearlyOutOfCityRecords,
    canonicalWriteAuthorized: false,
  };
  const artifactManifest = {
    status: STATUS,
    buildId: finalBuild.summary.buildId,
    sourceSnapshots,
    candidateArtifactsCommitted: false,
    candidateOutputPath: path.resolve(BUILD_DIR).startsWith(ROOT)
      ? path.relative(ROOT, BUILD_DIR).replace(/\\/g, '/')
      : 'EXTERNAL_NONCANONICAL_STAGE4G_OUTPUT',
    artifactHashes: finalBuild.summary.artifactHashes,
    normalizedOrderingHash: stableHash(stage4c.normalizedRecords.map((record) => record.sourceId)),
    classificationHash: stableHash(stage4c.matchResults),
    reviewQueueHash: stableHash(stage4c.reviewQueue),
    licenseManifestHash: stableHash(finalBuild.licenseManifest),
  };

  writeJson(path.join(STAGE4G_DIR, 'stage4g_summary.json'), summary);
  writeJson(path.join(STAGE4G_DIR, 'stage4g_quality_spot_checks.json'), quality);
  writeJson(path.join(STAGE4G_DIR, 'stage4g_proposed_change_summary.json'), proposedChanges);
  writeJson(path.join(STAGE4G_DIR, 'stage4g_candidate_artifact_manifest.json'), artifactManifest);

  if (fileHash(CANONICAL_PATH) !== canonicalHashBefore) {
    throw new Error('Canonical CSV changed during Stage 4G.');
  }
  if (!deterministic) throw new Error('Stage 4G deterministic build validation failed.');
  return { summary, quality, proposedChanges, artifactManifest };
}

if (require.main === module) {
  console.log(JSON.stringify(runStage4g().summary, null, 2));
}

module.exports = {
  duplicateGroupCount,
  runStage4g,
  verifySnapshotHashes,
};
