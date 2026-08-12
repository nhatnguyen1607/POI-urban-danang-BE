const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const { performance } = require('node:perf_hooks');

const {
  BOUNDARY_CLASSIFICATIONS,
  loadBoundaryArtifact,
  summarizeBoundaryClassifications,
} = require('../src/modules/cityPackPreparation/administrativeBoundary');
const {
  inspectCanonicalDataset,
  parseCsv,
  readCanonicalPois,
} = require('../src/modules/cityPackPreparation/canonicalDataset');
const { normalizeWithAdapters } = require('../src/modules/cityPackPreparation/dryRun');
const { DECISIONS, resolveSourceRecords } = require('../src/modules/cityPackPreparation/entityResolution');
const { stableHash } = require('../src/modules/cityPackPreparation/incrementalSync');
const { loadRealSnapshotRecords } = require('../src/modules/cityPackPreparation/realSnapshot');
const {
  NEW_TIERS,
  REVIEW_PRIORITIES,
  VALIDATION_LABELS,
  buildApplyPlan,
  buildPrioritizedReviewQueue,
  buildValidationDataset,
  calculatePrecisionGate,
  classifyTravelerCategory,
  dryRunApplyPlan,
  selectCanaryReview,
  triageNewCandidate,
} = require('../src/modules/cityPackPreparation/reviewApplyPreparation');
const { verifySnapshots } = require('./phase4_stage4h_full_city_comparison');

const ROOT = path.resolve(__dirname, '..');
const CANONICAL_PATH = path.join(ROOT, 'data', 'canonical', 'urbanagent_poi_master_v1.csv');
const BOUNDARY_PATH = path.join(
  ROOT,
  'data',
  'spikes',
  'phase4',
  'stage4i',
  'boundary',
  'osm-relation-1891418-v72.geojson',
);
const OUTPUT_DIR = path.join(ROOT, 'data', 'spikes', 'phase4', 'stage4i');
const MANIFEST_PATH = path.join(ROOT, 'data', 'spikes', 'phase4', 'stage4g', 'snapshot_manifest.json');
const STAGE4G_SUMMARY_PATH = path.join(ROOT, 'data', 'spikes', 'phase4', 'stage4g', 'stage4g_summary.json');
const STAGE4H_REVIEW_QUEUE_BEFORE = 79078;

function sha256(filePath) {
  return crypto.createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');
}

function snapshotPaths(cacheDir) {
  return {
    overture: path.join(cacheDir, 'overture_full_danang.json'),
    osm: path.join(cacheDir, 'osm_full_danang.json'),
    wikidata: path.join(cacheDir, 'wikidata_full_danang.json'),
    commons: path.join(cacheDir, 'wikimedia_commons_full_danang_metadata.json'),
  };
}

function insideBbox(record, boundary) {
  return Number.isFinite(record.latitude)
    && Number.isFinite(record.longitude)
    && record.longitude >= boundary.west
    && record.longitude <= boundary.east
    && record.latitude >= boundary.south
    && record.latitude <= boundary.north;
}

function parseSourceIds(value) {
  const ids = {};
  for (const item of String(value || '').split('|').map((part) => part.trim()).filter(Boolean)) {
    const separator = item.indexOf(':');
    if (separator < 0) continue;
    const source = item.slice(0, separator);
    const id = item.slice(separator + 1);
    ids[source] = id;
  }
  return ids;
}

function readCanonicalEvidencePois(canonicalPath) {
  const canonical = readCanonicalPois(canonicalPath);
  const rows = parseCsv(fs.readFileSync(canonicalPath, 'utf8'));
  const headers = rows[0].map((header, index) => (
    index === 0 ? header.replace(/^\uFEFF/, '').trim() : header.trim()
  ));
  const rawById = new Map(rows.slice(1).map((cells) => {
    const raw = Object.fromEntries(headers.map((header, index) => [header, cells[index] ?? '']));
    return [raw.Global_ID, raw];
  }));
  return canonical.map((poi) => {
    const raw = rawById.get(poi.id) || {};
    return {
      ...poi,
      sourceId: poi.id,
      externalIds: parseSourceIds(raw.Source_IDs),
      aliases: String(raw.Alias_Global_IDs || '').split('|').filter(Boolean),
      evidenceMetadata: {
        source: raw.Source || null,
        restaurantIdIsSourceIdentifier: raw.RestaurantID || null,
      },
    };
  });
}

function countValues(items, valueFunction, allowedValues = []) {
  const counts = Object.fromEntries(allowedValues.map((value) => [value, 0]));
  for (const item of items) {
    const value = valueFunction(item);
    counts[value] = (counts[value] || 0) + 1;
  }
  return counts;
}

function csvEscape(value) {
  if (value === null || value === undefined) return '';
  const text = typeof value === 'object' ? JSON.stringify(value) : String(value);
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function writeCsv(filePath, rows, fields) {
  const body = [
    fields.join(','),
    ...rows.map((row) => fields.map((field) => csvEscape(row[field])).join(',')),
  ].join('\n');
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${body}\n`, 'utf8');
}

function writeJson(filePath, value) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${JSON.stringify(value)}\n`, 'utf8');
}

function validationWorksheetRows(validation) {
  return validation.map((item) => ({
    caseId: item.caseId,
    stratum: item.stratum,
    machineDecision: item.machineSuggested.decision,
    machineConfidence: item.machineSuggested.confidence,
    evidenceStatus: item.evidenceSupported.status,
    evidenceLabel: item.evidenceSupported.label,
    evidenceType: item.evidenceSupported.evidenceType,
    evidence: item.evidenceSupported.evidence,
    reviewer: item.evidenceSupported.reviewer,
    source: item.recordA?.source,
    sourceId: item.recordA?.sourceId,
    sourceName: item.recordA?.name,
    sourceCategory: item.recordA?.category,
    candidateId: item.recordB?.sourceId,
    candidateName: item.recordB?.name,
    candidateCategory: item.recordB?.category,
    distanceMeters: item.candidateEvidence?.distanceMeters,
    humanLabel: '',
    humanReviewer: '',
    humanEvidenceReference: '',
    humanNotes: '',
  }));
}

function canaryRows(canary) {
  return canary.map((item) => ({
    queueId: item.queueId,
    priority: item.priority,
    priorityReason: item.priorityReason,
    decision: item.decision,
    recommendedAction: item.recommendedAction,
    source: item.source,
    sourceId: item.sourceId,
    relatedSourceId: item.relatedSourceId || '',
    name: item.name,
    category: item.category,
    categoryEligibility: item.categoryEligibility,
    newTier: item.newTier || '',
    latitude: item.latitude,
    longitude: item.longitude,
    canonicalId: item.candidateCanonicalMatch?.canonicalPoiId || '',
    canonicalName: item.candidateCanonicalMatch?.canonicalName || '',
    evidence: item.evidence,
    reviewerDecision: '',
    reviewerNotes: '',
  }));
}

function runStage4i({ cacheDir }) {
  const started = performance.now();
  const manifest = JSON.parse(fs.readFileSync(MANIFEST_PATH, 'utf8'));
  const stage4gSummary = JSON.parse(fs.readFileSync(STAGE4G_SUMMARY_PATH, 'utf8'));
  const paths = snapshotPaths(cacheDir);
  verifySnapshots(manifest, paths);
  const canonicalBefore = inspectCanonicalDataset(CANONICAL_PATH);
  const canonicalPois = readCanonicalEvidencePois(CANONICAL_PATH);
  const canonicalById = new Map(canonicalPois.map((record) => [record.id, record]));
  const boundary = loadBoundaryArtifact(BOUNDARY_PATH);
  const rawRecords = loadRealSnapshotRecords(paths);
  const bboxRecords = normalizeWithAdapters(rawRecords.filter((record) => insideBbox(record, stage4gSummary.boundary)));
  if (bboxRecords.length !== stage4gSummary.normalizedRecords) {
    throw new Error(`Fixed Stage 4G bbox record mismatch: ${bboxRecords.length}.`);
  }

  const boundaryResult = summarizeBoundaryClassifications(bboxRecords, boundary);
  const eligibleRecords = boundaryResult.classified.filter((record) => (
    [BOUNDARY_CLASSIFICATIONS.INSIDE_DANANG, BOUNDARY_CLASSIFICATIONS.BOUNDARY_EDGE]
      .includes(record.boundaryClassification)
  ));
  const recordsById = new Map(eligibleRecords.map((record) => [record.sourceId, record]));
  const resolution = resolveSourceRecords(eligibleRecords, canonicalPois, undefined, {
    useSpatialIndex: true,
    useSpatialDuplicateIndex: true,
    includeCrossSourceDuplicates: true,
    maxCandidateDistanceMeters: 800,
    hardened: true,
    includeMetrics: true,
    resolutionVersion: 'stage4i-v1',
  });

  const validation = buildValidationDataset({
    matches: resolution.matches,
    duplicates: resolution.duplicates,
    recordsById,
    canonicalById,
  });
  const gates = {
    highConfidence: calculatePrecisionGate(validation, DECISIONS.HIGH_CONFIDENCE_MATCH, {
      minEvidence: 30,
      requiredPrecision: 0.995,
    }),
    probable: calculatePrecisionGate(validation, DECISIONS.PROBABLE_MATCH, {
      minEvidence: 20,
      requiredPrecision: 0.99,
    }),
    duplicate: calculatePrecisionGate(validation, DECISIONS.SOURCE_DUPLICATE, {
      minEvidence: 20,
      requiredPrecision: 0.995,
    }),
  };

  const queue = buildPrioritizedReviewQueue({
    matches: resolution.matches,
    duplicates: resolution.duplicates,
    recordsById,
    highGate: gates.highConfidence,
  });
  const canary = selectCanaryReview(queue, 75);
  const validationByCaseId = new Map(validation.map((item) => [item.caseId, item]));
  const applyPlan = buildApplyPlan({ canary, recordsById, canonicalById, validationByCaseId });
  const dryRun = dryRunApplyPlan({
    canonicalPois,
    applyPlan,
    candidateRecordsById: recordsById,
  });
  const dryRunRepeat = dryRunApplyPlan({
    canonicalPois,
    applyPlan,
    candidateRecordsById: recordsById,
  });
  const deterministic = dryRun.deterministicProposedPackHash === dryRunRepeat.deterministicProposedPackHash
    && stableHash({
      enrichments: dryRun.enrichments,
      additions: dryRun.additions,
      conflicts: dryRun.conflicts,
      rejectedActions: dryRun.rejectedActions,
      deferred: dryRun.deferred,
    }) === stableHash({
      enrichments: dryRunRepeat.enrichments,
      additions: dryRunRepeat.additions,
      conflicts: dryRunRepeat.conflicts,
      rejectedActions: dryRunRepeat.rejectedActions,
      deferred: dryRunRepeat.deferred,
    })
    && applyPlan.deterministicHash === buildApplyPlan({ canary, recordsById, canonicalById, validationByCaseId }).deterministicHash;

  const duplicateSourceIds = new Set(resolution.duplicates.flatMap((item) => [item.sourceIdA, item.sourceIdB]));
  const newMatches = resolution.matches.filter((match) => match.decision === DECISIONS.NEW_CANDIDATE);
  const newTriage = newMatches.map((match) => {
    const record = recordsById.get(match.sourceId);
    return {
      sourceId: match.sourceId,
      ...triageNewCandidate(record, match, { duplicateSourceIds }),
    };
  });
  const tierCounts = countValues(newTriage, (item) => item.tier, Object.values(NEW_TIERS));
  const categoryCounts = countValues(
    newMatches,
    (match) => classifyTravelerCategory(recordsById.get(match.sourceId)?.category),
    ['TRAVELER_RELEVANT', 'CONTEXTUAL', 'LOW_VALUE', 'EXCLUDED'],
  );
  const priorityCounts = countValues(queue, (item) => item.priority, Object.values(REVIEW_PRIORITIES));
  priorityCounts.DEFERRED += boundaryResult.counts.OUTSIDE_DANANG
    + boundaryResult.counts.INVALID_COORDINATE;
  const activeQueueCount = ['P0', 'P1', 'P2', 'P3', 'P4']
    .reduce((total, priority) => total + (priorityCounts[priority] || 0), 0);
  const evidenceCounts = countValues(
    validation,
    (item) => item.evidenceSupported.label,
    Object.values(VALIDATION_LABELS),
  );
  const canonicalBoundary = summarizeBoundaryClassifications(canonicalPois, boundary).counts;
  const canonicalAfter = inspectCanonicalDataset(CANONICAL_PATH);
  if (canonicalBefore.sha256 !== canonicalAfter.sha256 || canonicalAfter.rows !== 4166) {
    throw new Error('Canonical dataset changed during Stage 4I.');
  }
  if (!deterministic) throw new Error('Stage 4I dry-run determinism failed.');

  const summary = {
    status: 'CANDIDATE_NON_RUNTIME_NOT_CANONICAL_DRY_RUN_ONLY',
    cityId: 'da-nang',
    boundary: {
      productScope: 'PRE_2025_MERGER_DA_NANG_CITY_PACK',
      currentLegalBoundary: false,
      source: boundary.properties.source,
      sourceIdentifier: boundary.properties.sourceIdentifier,
      sourceVersion: boundary.properties.sourceVersion,
      sourceTimestamp: boundary.properties.sourceTimestamp,
      sourceChangeset: boundary.properties.sourceChangeset,
      overpassSnapshotAt: boundary.properties.overpassSnapshotAt,
      administrativeLevel: boundary.properties.administrativeLevel,
      crs: boundary.properties.crs,
      license: boundary.properties.license,
      artifactPath: path.relative(ROOT, BOUNDARY_PATH).replace(/\\/g, '/'),
      artifactSha256: sha256(BOUNDARY_PATH),
      policyReason: 'Hoi An remains a separate future City Pack in canonical project context.',
    },
    geography: {
      bboxRecords: bboxRecords.length,
      ...boundaryResult.counts,
      canonicalBoundaryClassification: canonicalBoundary,
      landMaskApplied: false,
      landSignalPolicy: 'Polygon membership is primary; no simplistic land mask excludes coasts, bridges, waterfronts or peninsulas.',
    },
    resolution: resolution.summary,
    validation: {
      sampleSize: validation.length,
      evidenceSupported: validation.filter((item) => item.evidenceSupported.status === 'EVIDENCE_SUPPORTED').length,
      evidenceCounts,
      uncertain: evidenceCounts.UNCERTAIN,
      methodology: 'Stratified resolver sampling; labels use separately evaluated structured record evidence and never copy resolver decisions.',
      limitations: 'Operational sample, not population-wide ground truth; unresolved cases remain UNCERTAIN.',
    },
    precisionGates: gates,
    thresholds: {
      changed: false,
      reason: 'No evidence-supported safety failure justifies threshold tuning; false-merge avoidance remains preferred.',
    },
    newCandidateTiers: tierCounts,
    categoryEligibility: categoryCounts,
    reviewQueue: {
      beforeStage4i: STAGE4H_REVIEW_QUEUE_BEFORE,
      prioritizedActiveAfter: activeQueueCount,
      priorityCounts,
      canarySize: canary.length,
    },
    applyPlan: {
      operations: applyPlan.operations.length,
      reviewerApproved: applyPlan.operations.filter((item) => item.reviewerStatus === 'APPROVED').length,
      allOperationsDefaultToDefer: applyPlan.operations.every((item) => item.operation === 'DEFER'),
      deterministicHash: applyPlan.deterministicHash,
    },
    dryRun: {
      beforeCount: dryRun.beforeCount,
      proposedAfterCount: dryRun.proposedAfterCount,
      enrichments: dryRun.enrichments,
      additions: dryRun.additions,
      conflicts: dryRun.conflicts,
      rejectedActions: dryRun.rejectedActions,
      deferred: dryRun.deferred,
      deterministicProposedPackHash: dryRun.deterministicProposedPackHash,
    },
    incrementalIntegration: {
      flow: 'sync -> polygon -> normalization -> resolution -> category/value -> review priority -> apply-plan dry run',
      unchangedSkipsExpensiveStages: true,
      deltaOnly: true,
      testedSeparately: true,
    },
    deterministic,
    canonical: {
      ...canonicalAfter,
      path: 'data/canonical/urbanagent_poi_master_v1.csv',
    },
    safety: {
      canonicalModified: false,
      runtimeModified: false,
      frontendModified: false,
      productionDatabaseOrFirebaseTouched: false,
      googleBulkIncluded: false,
      secondCityCreated: false,
      canonicalApplyExecuted: false,
    },
    readyForControlledHumanApprovedApply: false,
    blocker: 'Independent evidence is insufficient for at least one operational precision gate, and the product boundary differs from the current legal post-merger boundary.',
    proposedStage4j: 'Human evidence hardening for canary labels and explicit product-boundary approval before any controlled apply.',
  };

  writeJson(path.join(OUTPUT_DIR, 'stage4i_summary.json'), summary);
  fs.writeFileSync(
    path.join(OUTPUT_DIR, 'validation_labels.jsonl'),
    `${validation.map((item) => JSON.stringify(item)).join('\n')}\n`,
    'utf8',
  );
  writeCsv(
    path.join(OUTPUT_DIR, 'review_worksheet.csv'),
    validationWorksheetRows(validation),
    [
      'caseId', 'stratum', 'machineDecision', 'machineConfidence', 'evidenceStatus',
      'evidenceLabel', 'evidenceType', 'evidence', 'reviewer', 'source', 'sourceId',
      'sourceName', 'sourceCategory', 'candidateId', 'candidateName',
      'candidateCategory', 'distanceMeters', 'humanLabel', 'humanReviewer',
      'humanEvidenceReference', 'humanNotes',
    ],
  );
  writeCsv(
    path.join(OUTPUT_DIR, 'canary_review.csv'),
    canaryRows(canary),
    [
      'queueId', 'priority', 'priorityReason', 'decision', 'recommendedAction',
      'source', 'sourceId', 'relatedSourceId', 'name', 'category',
      'categoryEligibility', 'newTier', 'latitude', 'longitude', 'canonicalId',
      'canonicalName', 'evidence', 'reviewerDecision', 'reviewerNotes',
    ],
  );
  writeJson(path.join(OUTPUT_DIR, 'apply_plan.json'), applyPlan);
  return {
    summary,
    processingSeconds: Number(((performance.now() - started) / 1000).toFixed(3)),
  };
}

if (require.main === module) {
  const cacheDir = process.argv[2];
  if (!cacheDir) {
    console.error('Usage: node scripts/phase4_stage4i_review_apply_preparation.js <stage4g-cache-dir>');
    process.exitCode = 1;
  } else {
    const result = runStage4i({ cacheDir: path.resolve(cacheDir) });
    console.log(JSON.stringify(result, null, 2));
  }
}

module.exports = {
  BOUNDARY_PATH,
  CANONICAL_PATH,
  STAGE4H_REVIEW_QUEUE_BEFORE,
  parseSourceIds,
  readCanonicalEvidencePois,
  runStage4i,
  validationWorksheetRows,
};
