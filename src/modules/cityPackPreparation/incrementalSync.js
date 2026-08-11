const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');

const { hasValidCoordinates } = require('./sourceRecord');
const { normalizeVietnameseText } = require('./historicalNormalization');

const STATUSES = Object.freeze({
  NEW: 'NEW',
  CHANGED: 'CHANGED',
  UNCHANGED: 'UNCHANGED',
  MISSING: 'MISSING',
  INVALID: 'INVALID',
});

const SOURCE_CAPABILITIES = Object.freeze({
  CHANGE_FEED: 'CHANGE_FEED',
  DELTA: 'DELTA',
  SNAPSHOT: 'SNAPSHOT',
  POLLING: 'POLLING',
});

const DEFAULT_SOURCE_CAPABILITIES = Object.freeze({
  overture: SOURCE_CAPABILITIES.SNAPSHOT,
  osm: SOURCE_CAPABILITIES.POLLING,
  wikidata: SOURCE_CAPABILITIES.POLLING,
  wikimedia_commons: SOURCE_CAPABILITIES.POLLING,
});

const DEFAULT_VERSIONS = Object.freeze({
  resolutionVersion: 'stage4h-v1',
  provenanceVersion: 'stage4f-v1',
  multimodalVersion: 'stage4h-cache-v1',
  visionProcessingVersion: 'disabled',
  embeddingVersion: 'not-computed',
});

function sortValue(value) {
  if (Array.isArray(value)) return value.map(sortValue);
  if (!value || typeof value !== 'object') return value;
  return Object.fromEntries(
    Object.keys(value).sort().map((key) => [key, sortValue(value[key])]),
  );
}

function stableHash(value) {
  return crypto.createHash('sha256').update(JSON.stringify(sortValue(value))).digest('hex');
}

function recordKey(record) {
  return `${String(record.source || '').trim()}:${String(record.sourceId || '').trim()}`;
}

function compactExternalIds(value) {
  return Object.fromEntries(
    Object.entries(value || {})
      .filter(([, item]) => item !== null && item !== undefined && item !== '')
      .sort(([left], [right]) => left.localeCompare(right)),
  );
}

function mediaItems(record) {
  const raw = Array.isArray(record.media) ? record.media : record.media ? [record.media] : [];
  return raw
    .map((media, index) => ({
      source: media.source || record.source,
      sourceRecordId: record.sourceId,
      mediaId: String(media.mediaId || media.sourceId || media.assetIdentifier || index),
      reference: media.url || media.mediaReference || media.reference || null,
      contentHash: media.contentHash || null,
      etag: media.etag || media.ETag || null,
      lastModified: media.lastModified || media['Last-Modified'] || null,
      license: media.licenseName || media.license || null,
      provenance: {
        attributionText: media.attributionText || null,
        attributionUrl: media.attributionUrl || null,
        licenseUrl: media.licenseUrl || null,
        sourcePage: media.sourcePage || null,
      },
    }))
    .sort((left, right) => left.mediaId.localeCompare(right.mediaId));
}

function stageFingerprints(record) {
  const identity = {
    name: normalizeVietnameseText(record.name),
    latitude: record.latitude,
    longitude: record.longitude,
    address: normalizeVietnameseText(record.address),
    category: record.category || null,
    externalIds: compactExternalIds(record.externalIds || record.externalIdentifiers),
  };
  const media = mediaItems(record).map((asset) => ({
    mediaId: asset.mediaId,
    reference: asset.reference,
    contentHash: asset.contentHash,
    etag: asset.etag,
    lastModified: asset.lastModified,
  }));
  const provenance = {
    license: record.license || null,
    provenance: record.provenance || null,
    upstreamSources: record.upstreamSources || record.provenance?.upstreamSources || [],
  };
  return {
    rawContentHash: stableHash({ identity, media, provenance }),
    normalizedContentHash: stableHash(identity),
    identityHash: stableHash(identity),
    mediaHash: stableHash(media),
    provenanceHash: stableHash(provenance),
  };
}

function indexByKey(records = []) {
  return new Map(records.map((record) => [recordKey(record), record]));
}

function updateMediaRegistry({
  record,
  snapshotId,
  previousRegistry,
  versions,
  activeMediaKeys,
  metrics,
}) {
  const registry = [];
  for (const asset of mediaItems(record)) {
    const key = `${asset.source}:${asset.sourceRecordId}:${asset.mediaId}`;
    activeMediaKeys.add(key);
    const previous = previousRegistry.get(key);
    const referenceFingerprint = stableHash({
      reference: asset.reference,
      contentHash: asset.contentHash,
      etag: asset.etag,
      lastModified: asset.lastModified,
    });
    const provenanceFingerprint = stableHash({ license: asset.license, provenance: asset.provenance });
    const changed = !previous || previous.referenceFingerprint !== referenceFingerprint;
    const processingVersionChanged = Boolean(previous) && (
      previous.visionProcessingVersion !== versions.visionProcessingVersion
      || previous.embeddingVersion !== versions.embeddingVersion
    );
    if (!previous) metrics.mediaNew += 1;
    else if (changed || processingVersionChanged) metrics.mediaInvalidated += 1;
    else metrics.mediaReused += 1;
    registry.push({
      key,
      ...asset,
      firstSeen: previous?.firstSeen || snapshotId,
      lastSeen: snapshotId,
      status: previous ? (changed ? STATUSES.CHANGED : STATUSES.UNCHANGED) : STATUSES.NEW,
      referenceFingerprint,
      provenanceFingerprint,
      visionProcessingVersion: versions.visionProcessingVersion,
      embeddingVersion: versions.embeddingVersion,
      invalidateVision: changed || processingVersionChanged,
      invalidateEmbedding: changed || processingVersionChanged,
    });
  }
  return registry;
}

function buildStateRecord({ record, previous, snapshotId, versions, resolution, fingerprints, status }) {
  return {
    source: record.source,
    sourceId: record.sourceId,
    firstSeenSnapshot: previous?.firstSeenSnapshot || snapshotId,
    lastSeenSnapshot: snapshotId,
    sourceUpdatedAt: record.sourceUpdatedAt || previous?.sourceUpdatedAt || null,
    rawContentHash: fingerprints.rawContentHash,
    normalizedContentHash: fingerprints.normalizedContentHash,
    identityHash: fingerprints.identityHash,
    mediaHash: fingerprints.mediaHash,
    provenanceHash: fingerprints.provenanceHash,
    status,
    canonicalMatchId: resolution?.bestCandidate?.canonicalPoiId
      || resolution?.canonicalMatchId
      || previous?.canonicalMatchId
      || null,
    resolutionClassification: resolution?.decision
      || previous?.resolutionClassification
      || null,
    resolutionVersion: versions.resolutionVersion,
    provenanceVersion: versions.provenanceVersion,
    multimodalVersion: versions.multimodalVersion,
    lastProcessedSnapshot: status === STATUSES.UNCHANGED
      ? previous?.lastProcessedSnapshot || snapshotId
      : snapshotId,
    processing: {
      identity: !previous || previous.identityHash !== fingerprints.identityHash
        || previous.resolutionVersion !== versions.resolutionVersion,
      media: !previous || previous.mediaHash !== fingerprints.mediaHash
        || previous.multimodalVersion !== versions.multimodalVersion,
      provenance: !previous || previous.provenanceHash !== fingerprints.provenanceHash
        || previous.provenanceVersion !== versions.provenanceVersion,
    },
    multimodalCacheKey: stableHash({
      normalizedContentHash: fingerprints.normalizedContentHash,
      mediaHash: fingerprints.mediaHash,
      multimodalVersion: versions.multimodalVersion,
    }),
  };
}

function emptyMetrics(total = 0) {
  return {
    discovered: total,
    processed: 0,
    skipped: 0,
    resolutionProcessed: 0,
    mediaProcessed: 0,
    provenanceProcessed: 0,
    mediaNew: 0,
    mediaInvalidated: 0,
    mediaReused: 0,
    missing: 0,
    invalid: 0,
  };
}

function runIncrementalSync({
  snapshotId,
  records,
  previousState = [],
  previousMediaRegistry = [],
  versions = {},
  resolveRecord = () => null,
  checkpoint = null,
  checkpointEvery = 1000,
  stopAfter = null,
  onCheckpoint = null,
}) {
  if (!snapshotId) throw new Error('snapshotId is required.');
  const resolvedVersions = { ...DEFAULT_VERSIONS, ...versions };
  const ordered = [...records].sort((left, right) => recordKey(left).localeCompare(recordKey(right)));
  const previousByKey = indexByKey(previousState);
  const previousMediaByKey = new Map(previousMediaRegistry.map((item) => [item.key, item]));
  const completed = new Set(checkpoint?.completedKeys || []);
  const state = [...(checkpoint?.partialState || [])];
  const mediaRegistry = [...(checkpoint?.partialMediaRegistry || [])];
  const metrics = { ...emptyMetrics(ordered.length), ...(checkpoint?.metrics || {}) };
  metrics.discovered = ordered.length;
  const activeKeys = new Set(completed);
  const activeMediaKeys = new Set(
    (checkpoint?.partialMediaRegistry || []).map((item) => item.key),
  );
  let completedThisRun = 0;

  for (const record of ordered) {
    const key = recordKey(record);
    activeKeys.add(key);
    if (completed.has(key)) continue;
    const previous = previousByKey.get(key);
    const fingerprints = stageFingerprints(record);
    const invalid = !record.source || !record.sourceId || !hasValidCoordinates(record);
    const changed = Boolean(previous) && (
      previous.identityHash !== fingerprints.identityHash
      || previous.mediaHash !== fingerprints.mediaHash
      || previous.provenanceHash !== fingerprints.provenanceHash
      || previous.resolutionVersion !== resolvedVersions.resolutionVersion
      || previous.provenanceVersion !== resolvedVersions.provenanceVersion
      || previous.multimodalVersion !== resolvedVersions.multimodalVersion
    );
    const status = invalid
      ? STATUSES.INVALID
      : !previous
        ? STATUSES.NEW
        : changed
          ? STATUSES.CHANGED
          : STATUSES.UNCHANGED;
    const identityChanged = !previous
      || previous.identityHash !== fingerprints.identityHash
      || previous.resolutionVersion !== resolvedVersions.resolutionVersion;
    const mediaChanged = !previous
      || previous.mediaHash !== fingerprints.mediaHash
      || previous.multimodalVersion !== resolvedVersions.multimodalVersion;
    const provenanceChanged = !previous
      || previous.provenanceHash !== fingerprints.provenanceHash
      || previous.provenanceVersion !== resolvedVersions.provenanceVersion;

    let resolution = null;
    if (invalid) metrics.invalid += 1;
    else if (identityChanged) {
      resolution = resolveRecord(record);
      metrics.resolutionProcessed += 1;
    }
    if (mediaChanged) metrics.mediaProcessed += 1;
    if (provenanceChanged) metrics.provenanceProcessed += 1;
    if (status === STATUSES.UNCHANGED) metrics.skipped += 1;
    else metrics.processed += 1;

    state.push(buildStateRecord({
      record,
      previous,
      snapshotId,
      versions: resolvedVersions,
      resolution,
      fingerprints,
      status,
    }));
    mediaRegistry.push(...updateMediaRegistry({
      record,
      snapshotId,
      previousRegistry: previousMediaByKey,
      versions: resolvedVersions,
      activeMediaKeys,
      metrics,
    }));
    completed.add(key);
    completedThisRun += 1;

    const shouldCheckpoint = completedThisRun % checkpointEvery === 0 || completedThisRun === stopAfter;
    if (shouldCheckpoint && onCheckpoint) {
      onCheckpoint({
        snapshotId,
        lastCompletedStage: 'record_processing',
        completedKeys: [...completed].sort(),
        partialState: [...state].sort((a, b) => recordKey(a).localeCompare(recordKey(b))),
        partialMediaRegistry: [...mediaRegistry].sort((a, b) => a.key.localeCompare(b.key)),
        metrics,
        failedSourceIds: [],
        processingVersions: resolvedVersions,
      });
    }
    if (stopAfter && completedThisRun >= stopAfter) {
      return {
        complete: false,
        checkpoint: {
          snapshotId,
          lastCompletedStage: 'record_processing',
          completedKeys: [...completed].sort(),
          partialState: state,
          partialMediaRegistry: mediaRegistry,
          metrics,
          failedSourceIds: [],
          processingVersions: resolvedVersions,
        },
        metrics,
      };
    }
  }

  for (const previous of previousState) {
    const key = recordKey(previous);
    if (activeKeys.has(key)) continue;
    state.push({
      ...previous,
      status: STATUSES.MISSING,
      lastProcessedSnapshot: snapshotId,
      retirementReviewRequired: true,
    });
    metrics.missing += 1;
  }
  for (const previous of previousMediaRegistry) {
    if (activeMediaKeys.has(previous.key)) continue;
    mediaRegistry.push({ ...previous, status: STATUSES.MISSING, lastSeen: previous.lastSeen });
  }

  const stateByKey = new Map();
  for (const item of state) stateByKey.set(recordKey(item), item);
  const mediaByKey = new Map();
  for (const item of mediaRegistry) mediaByKey.set(item.key, item);
  const finalState = [...stateByKey.values()].sort((a, b) => recordKey(a).localeCompare(recordKey(b)));
  const finalMedia = [...mediaByKey.values()].sort((a, b) => a.key.localeCompare(b.key));

  return {
    complete: true,
    snapshotId,
    state: finalState,
    mediaRegistry: finalMedia,
    metrics,
    deterministicHash: stableHash({ state: finalState, mediaRegistry: finalMedia, metrics }),
    reviewCandidates: finalState
      .filter((item) => item.status === STATUSES.MISSING)
      .map((item) => ({
        source: item.source,
        sourceId: item.sourceId,
        decision: 'SOURCE_RECORD_MISSING_REVIEW',
        autoDeleteAuthorized: false,
      })),
  };
}

function readJsonl(filePath) {
  if (!filePath || !fs.existsSync(filePath)) return [];
  return fs.readFileSync(filePath, 'utf8')
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

function writeJsonl(filePath, records) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  const body = records.map((record) => JSON.stringify(sortValue(record))).join('\n');
  fs.writeFileSync(filePath, body ? `${body}\n` : '', 'utf8');
}

module.exports = {
  DEFAULT_SOURCE_CAPABILITIES,
  DEFAULT_VERSIONS,
  SOURCE_CAPABILITIES,
  STATUSES,
  mediaItems,
  readJsonl,
  recordKey,
  runIncrementalSync,
  stableHash,
  stageFingerprints,
  writeJsonl,
};
