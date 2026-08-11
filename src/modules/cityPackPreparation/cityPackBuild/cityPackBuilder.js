const crypto = require('node:crypto');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const { inspectCanonicalDataset, readCanonicalPois } = require('../canonicalDataset');
const { runStage4cDryRun } = require('../dryRun');
const {
  generateProvenanceLicenseManifest,
  validateProvenanceLicenseManifest,
} = require('./provenanceManifest');
const { loadReviewDecisions, validateReviewDecisions } = require('./reviewDecisionValidator');

const BUILD_VERSION = 'phase4-stage4d-v1';
const CANDIDATE_STATUS = 'CANDIDATE_NON_RUNTIME_NOT_CANONICAL';

function* jsonTokens(value, arrayValue = false) {
  if (value && typeof value.toJSON === 'function') {
    yield* jsonTokens(value.toJSON(), arrayValue);
    return;
  }
  if (value === null || typeof value !== 'object') {
    const encoded = JSON.stringify(value);
    yield encoded === undefined && arrayValue ? 'null' : encoded;
    return;
  }
  if (Array.isArray(value)) {
    yield '[';
    for (let index = 0; index < value.length; index += 1) {
      if (index > 0) yield ',';
      yield* jsonTokens(value[index], true);
    }
    yield ']';
    return;
  }
  yield '{';
  let first = true;
  for (const key of Object.keys(value)) {
    const child = value[key];
    if (['undefined', 'function', 'symbol'].includes(typeof child)) continue;
    if (!first) yield ',';
    first = false;
    yield JSON.stringify(key);
    yield ':';
    yield* jsonTokens(child);
  }
  yield '}';
}

function stableHash(value) {
  const hash = crypto.createHash('sha256');
  for (const token of jsonTokens(value)) hash.update(token);
  return hash.digest('hex');
}

function candidateId(cityId, sourceId) {
  return `candidate:${cityId}:${stableHash(sourceId).slice(0, 16)}`;
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function writeJson(outputDir, fileName, value) {
  fs.mkdirSync(outputDir, { recursive: true });
  const fileDescriptor = fs.openSync(path.join(outputDir, fileName), 'w');
  let buffer = '';
  try {
    for (const token of jsonTokens(value)) {
      buffer += token;
      if (buffer.length >= 1024 * 1024) {
        fs.writeSync(fileDescriptor, buffer);
        buffer = '';
      }
    }
    fs.writeSync(fileDescriptor, `${buffer}\n`);
  } finally {
    fs.closeSync(fileDescriptor);
  }
}

function writeCsv(outputDir, fileName, rows, headers) {
  fs.mkdirSync(outputDir, { recursive: true });
  const escapeCell = (value) => {
    const text = Array.isArray(value)
      ? value.join('|')
      : value === null || value === undefined
        ? ''
        : String(value);
    return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  };
  const csv = [headers.join(',')]
    .concat(rows.map((row) => headers.map((header) => escapeCell(row[header])).join(',')))
    .join('\n')
    .concat('\n');
  fs.writeFileSync(path.join(outputDir, fileName), csv);
}

function indexBy(items, keyFn) {
  return new Map(items.map((item) => [keyFn(item), item]));
}

function compactProvenance(record) {
  return {
    source: record.provenance.source,
    sourceId: record.provenance.sourceId,
    snapshotRef: record.provenance.snapshotRef,
    policyClass: record.provenance.policyClass,
    license: record.provenance.license,
    attribution: record.provenance.attribution,
    licenseUrl: record.provenance.licenseUrl,
    attributionUrl: record.provenance.attributionUrl,
    policyReference: record.provenance.policyReference,
    upstreamSources: record.provenance.upstreamSources,
    fields: record.provenance.fields,
  };
}

function buildEnrichmentRecord({ sourceRecord, canonicalPoiId, decisionSource }) {
  return {
    status: CANDIDATE_STATUS,
    type: 'matched_enrichment',
    canonicalPoiId,
    source: sourceRecord.source,
    sourceId: sourceRecord.sourceId,
    name: sourceRecord.name,
    category: sourceRecord.category,
    fields: {
      address: sourceRecord.address,
      website: sourceRecord.website,
      phone: sourceRecord.phone,
      openingHours: sourceRecord.openingHours,
      externalIds: sourceRecord.externalIds,
      media: sourceRecord.media,
    },
    provenance: compactProvenance(sourceRecord),
    decisionSource,
  };
}

function buildNewCandidateRecord({ cityId, sourceRecord, decisionSource }) {
  return {
    status: CANDIDATE_STATUS,
    type: 'approved_new_candidate',
    candidateId: candidateId(cityId, sourceRecord.sourceId),
    cityId,
    source: sourceRecord.source,
    sourceId: sourceRecord.sourceId,
    name: sourceRecord.name,
    category: sourceRecord.category,
    latitude: sourceRecord.latitude,
    longitude: sourceRecord.longitude,
    address: sourceRecord.address,
    website: sourceRecord.website,
    phone: sourceRecord.phone,
    openingHours: sourceRecord.openingHours,
    externalIds: sourceRecord.externalIds,
    media: sourceRecord.media,
    provenance: compactProvenance(sourceRecord),
    decisionSource,
  };
}

function isInsideDaNangBounds(record) {
  return (
    Number.isFinite(record.latitude) &&
    Number.isFinite(record.longitude) &&
    record.latitude >= 15.8 &&
    record.latitude <= 16.3 &&
    record.longitude >= 107.8 &&
    record.longitude <= 108.5
  );
}

function validateCandidatePack({ matchedEnrichment, approvedNewCandidates, proposedNewCandidates = [], licenseManifest }) {
  const errors = [];
  const candidateIds = new Set();
  const canonicalIds = new Set();

  for (const candidate of [...approvedNewCandidates, ...proposedNewCandidates]) {
    if (candidateIds.has(candidate.candidateId)) {
      errors.push({ code: 'DUPLICATE_CANDIDATE_ID', candidateId: candidate.candidateId });
    }
    candidateIds.add(candidate.candidateId);

    if (!candidate.name) {
      errors.push({ code: 'MISSING_NAME', sourceId: candidate.sourceId });
    }

    if (!isInsideDaNangBounds(candidate)) {
      errors.push({ code: 'OUTSIDE_DA_NANG_BOUNDS', sourceId: candidate.sourceId });
    }

    if (!candidate.provenance?.policyClass) {
      errors.push({ code: 'MISSING_PROVENANCE', sourceId: candidate.sourceId });
    }
  }

  for (const enrichment of matchedEnrichment) {
    if (!enrichment.canonicalPoiId) {
      errors.push({ code: 'MISSING_CANONICAL_ID', sourceId: enrichment.sourceId });
    }

    const key = `${enrichment.canonicalPoiId}:${enrichment.sourceId}`;
    if (canonicalIds.has(key)) {
      errors.push({ code: 'DUPLICATE_MATCHED_ENRICHMENT', key });
    }
    canonicalIds.add(key);

    if (!enrichment.provenance?.policyClass) {
      errors.push({ code: 'MISSING_PROVENANCE', sourceId: enrichment.sourceId });
    }
  }

  const provenanceValidation = validateProvenanceLicenseManifest(licenseManifest);
  errors.push(...provenanceValidation.errors);

  return {
    valid: errors.length === 0,
    errors,
  };
}

function generateLicenseManifest({ normalizedRecords, buildId, cityId }) {
  return generateProvenanceLicenseManifest({
    normalizedRecords,
    buildId,
    cityId,
    buildVersion: BUILD_VERSION,
    status: CANDIDATE_STATUS,
  });
}

function buildCandidateCityPack({
  cityId = 'da-nang',
  buildId = 'stage4d-fixture-build',
  canonicalPath,
  samplePath,
  reviewDecisionPath,
  outputDir,
  sourceSnapshots = ['stage4b-fixture'],
  includeProposedNewCandidates = false,
  resolutionOptions = {},
  preparedStage4c = null,
  writeArtifacts = true,
}) {
  const canonical = inspectCanonicalDataset(canonicalPath);
  const canonicalPois = readCanonicalPois(canonicalPath);
  const stage4c = preparedStage4c || runStage4cDryRun({
    canonicalPath,
    samplePath,
    outputDir: fs.mkdtempSync(path.join(os.tmpdir(), 'urbanagent-stage4d-stage4c-')),
    resolutionOptions,
  });
  const reviewDocument = readJson(reviewDecisionPath);
  const decisions = loadReviewDecisions(reviewDocument);
  const validation = validateReviewDecisions({
    decisions,
    reviewQueue: stage4c.reviewQueue,
    canonicalPois,
  });

  if (!validation.valid) {
    const error = new Error('Invalid Stage 4D review decisions.');
    error.validation = validation;
    throw error;
  }

  const decisionByQueueId = indexBy(decisions, (decision) => decision.queueId);
  const recordsBySourceId = indexBy(stage4c.normalizedRecords, (record) => record.sourceId);
  const matchedEnrichment = [];
  const approvedNewCandidates = [];
  const rejectedRecords = [];
  const deferredReviewItems = [];
  const sourceDuplicateGroups = [];
  const proposedNewCandidates = includeProposedNewCandidates
    ? stage4c.matchResults.matches
      .filter((match) => match.decision === 'NEW_CANDIDATE')
      .map((match) => ({
        ...buildNewCandidateRecord({
          cityId,
          sourceRecord: recordsBySourceId.get(match.sourceId),
          decisionSource: 'stage4c_review_required_not_approved',
        }),
        type: 'proposed_new_candidate',
      }))
    : [];
  const autoMatched = stage4c.matchResults.matches.filter((match) =>
    ['HIGH_CONFIDENCE_MATCH', 'PROBABLE_MATCH'].includes(match.decision),
  );

  for (const match of autoMatched) {
    const sourceRecord = recordsBySourceId.get(match.sourceId);
    matchedEnrichment.push(
      buildEnrichmentRecord({
        sourceRecord,
        canonicalPoiId: match.bestCandidate.canonicalPoiId,
        decisionSource: 'stage4c_auto_candidate_not_canonical_apply',
      }),
    );
  }

  for (const queueItem of stage4c.reviewQueue) {
    const decision = decisionByQueueId.get(queueItem.queueId);

    if (!decision || decision.action === 'DEFER' || decision.action === 'KEEP_SEPARATE') {
      deferredReviewItems.push({
        queueId: queueItem.queueId,
        decision: queueItem.decision,
        sourceRecord: queueItem.sourceRecord,
        reason: decision?.action || 'NO_DECISION',
      });
      continue;
    }

    if (decision.action === 'REJECT_MATCH') {
      rejectedRecords.push({
        queueId: queueItem.queueId,
        sourceRecord: queueItem.sourceRecord,
        reviewerNote: decision.reviewerNote,
      });
      continue;
    }

    if (decision.action === 'ACCEPT_MATCH') {
      const sourceRecord = recordsBySourceId.get(queueItem.sourceRecord.sourceId);
      matchedEnrichment.push(
        buildEnrichmentRecord({
          sourceRecord,
          canonicalPoiId: decision.canonicalPoiId,
          decisionSource: 'review_decision_accept_match',
        }),
      );
      continue;
    }

    if (decision.action === 'CREATE_NEW') {
      const sourceRecord = recordsBySourceId.get(queueItem.sourceRecord.sourceId);
      approvedNewCandidates.push(
        buildNewCandidateRecord({
          cityId,
          sourceRecord,
          decisionSource: 'review_decision_create_new',
        }),
      );
      continue;
    }

    if (decision.action === 'MERGE_SOURCE_DUPLICATE') {
      sourceDuplicateGroups.push({
        queueId: queueItem.queueId,
        records: [queueItem.sourceRecord, queueItem.duplicateRecord],
        decisionSource: 'review_decision_merge_source_duplicate',
        reviewerNote: decision.reviewerNote,
      });
    }
  }

  const licenseManifest = generateLicenseManifest({
    normalizedRecords: stage4c.normalizedRecords,
    buildId,
    cityId,
  });
  const pack = {
    status: CANDIDATE_STATUS,
    buildVersion: BUILD_VERSION,
    buildId,
    cityId,
    canonicalBaseline: {
      rows: canonical.rows,
      sha256: canonical.sha256,
      shaMatchesExpected: canonical.shaMatchesExpected,
    },
    sourceSnapshots: [...sourceSnapshots].sort(),
    matchedEnrichment: matchedEnrichment.sort((a, b) => {
      if (a.canonicalPoiId !== b.canonicalPoiId) return a.canonicalPoiId.localeCompare(b.canonicalPoiId);
      return a.sourceId.localeCompare(b.sourceId);
    }),
    approvedNewCandidates: approvedNewCandidates.sort((a, b) => a.candidateId.localeCompare(b.candidateId)),
    ...(includeProposedNewCandidates
      ? { proposedNewCandidates: proposedNewCandidates.sort((a, b) => a.candidateId.localeCompare(b.candidateId)) }
      : {}),
    sourceDuplicateGroups: sourceDuplicateGroups.sort((a, b) => a.queueId.localeCompare(b.queueId)),
    rejectedRecords: rejectedRecords.sort((a, b) => a.queueId.localeCompare(b.queueId)),
    deferredReviewItems: deferredReviewItems.sort((a, b) => a.queueId.localeCompare(b.queueId)),
  };

  const packValidation = validateCandidatePack({
    matchedEnrichment: pack.matchedEnrichment,
    approvedNewCandidates: pack.approvedNewCandidates,
    proposedNewCandidates: pack.proposedNewCandidates || [],
    licenseManifest,
  });

  if (!packValidation.valid) {
    const error = new Error('Invalid candidate City Pack.');
    error.validation = packValidation;
    throw error;
  }

  const artifactHashes = {
    candidatePackHash: stableHash(pack),
    licenseManifestHash: stableHash(licenseManifest),
    reviewDecisionHash: stableHash(decisions),
  };
  const summary = {
    status: CANDIDATE_STATUS,
    buildVersion: BUILD_VERSION,
    buildId,
    cityId,
    canonicalBaselineCount: canonical.rows,
    canonicalSha256: canonical.sha256,
    canonicalShaMatchesExpected: canonical.shaMatchesExpected,
    matchedEnriched: pack.matchedEnrichment.length,
    approvedNewCandidates: pack.approvedNewCandidates.length,
    ...(includeProposedNewCandidates
      ? { proposedNewCandidates: pack.proposedNewCandidates.length }
      : {}),
    rejected: pack.rejectedRecords.length,
    deferred: pack.deferredReviewItems.filter((item) => item.reason === 'DEFER').length,
    unresolved: pack.deferredReviewItems.filter((item) => item.reason === 'NO_DECISION').length,
    sourceDuplicateGroups: pack.sourceDuplicateGroups.length,
    candidateTotal: pack.matchedEnrichment.length
      + pack.approvedNewCandidates.length
      + (pack.proposedNewCandidates?.length || 0),
    sourceContributionCounts: stage4c.summary.adapters.reduce((acc, adapter) => {
      acc[adapter.source] = stage4c.normalizedRecords.filter((record) =>
        adapter.source === 'wikidata_wikimedia'
          ? record.source === 'wikidata' || record.source === 'wikimedia_commons'
          : record.source === adapter.source,
      ).length;
      return acc;
    }, {}),
    runtimeChanged: false,
    artifactHashes,
  };

  if (writeArtifacts) {
    writeJson(outputDir, 'candidate_city_pack.json', pack);
    writeJson(outputDir, 'license_attribution_manifest.json', licenseManifest);
    writeJson(outputDir, 'build_summary.json', summary);
    writeCsv(outputDir, 'candidate_city_pack_index.csv', [
      ...pack.matchedEnrichment.map((record) => ({
        type: record.type,
        id: record.canonicalPoiId,
        source: record.source,
        sourceId: record.sourceId,
        name: record.name,
      })),
      ...pack.approvedNewCandidates.map((record) => ({
        type: record.type,
        id: record.candidateId,
        source: record.source,
        sourceId: record.sourceId,
        name: record.name,
      })),
      ...(pack.proposedNewCandidates || []).map((record) => ({
        type: record.type,
        id: record.candidateId,
        source: record.source,
        sourceId: record.sourceId,
        name: record.name,
      })),
    ], ['type', 'id', 'source', 'sourceId', 'name']);
  }

  return {
    canonical,
    decisions,
    pack,
    licenseManifest,
    summary,
    validation: {
      decisions: validation,
      candidatePack: packValidation,
    },
  };
}

module.exports = {
  BUILD_VERSION,
  CANDIDATE_STATUS,
  buildCandidateCityPack,
  candidateId,
  generateLicenseManifest,
  stableHash,
  validateCandidatePack,
};
