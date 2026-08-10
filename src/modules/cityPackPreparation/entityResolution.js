const {
  hasValidCoordinates,
  normalizeCategory,
  normalizeName,
} = require('./sourceRecord');

const DECISIONS = Object.freeze({
  HIGH_CONFIDENCE_MATCH: 'HIGH_CONFIDENCE_MATCH',
  PROBABLE_MATCH: 'PROBABLE_MATCH',
  AMBIGUOUS: 'AMBIGUOUS',
  NEW_CANDIDATE: 'NEW_CANDIDATE',
  SOURCE_DUPLICATE: 'SOURCE_DUPLICATE',
  INVALID: 'INVALID',
});

const DEFAULT_THRESHOLDS = Object.freeze({
  highName: 0.9,
  highDistanceMeters: 50,
  probableName: 0.6,
  probableDistanceMeters: 150,
  chainCollisionName: 0.92,
  chainCollisionDistanceMeters: 750,
});

function categoryCompatibility(sourceCategory, canonicalCategory) {
  const source = normalizeCategory(sourceCategory);
  const target = normalizeCategory(canonicalCategory);

  if (source === target) return 1;

  const compatibleGroups = [
    new Set(['restaurant', 'cafe', 'bakery']),
    new Set(['attraction', 'beach', 'bridge', 'museum', 'park', 'market']),
    new Set(['media', 'attraction', 'beach', 'bridge', 'museum']),
  ];

  return compatibleGroups.some((group) => group.has(source) && group.has(target)) ? 0.55 : 0;
}

function levenshtein(a, b) {
  if (a === b) return 0;
  if (!a) return b.length;
  if (!b) return a.length;

  const previous = Array.from({ length: b.length + 1 }, (_, index) => index);
  const current = Array.from({ length: b.length + 1 }, () => 0);

  for (let i = 1; i <= a.length; i += 1) {
    current[0] = i;
    for (let j = 1; j <= b.length; j += 1) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      current[j] = Math.min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost);
    }
    for (let j = 0; j < current.length; j += 1) previous[j] = current[j];
  }

  return previous[b.length];
}

function tokenSimilarity(a, b) {
  const aTokens = new Set(a.split(' ').filter(Boolean));
  const bTokens = new Set(b.split(' ').filter(Boolean));

  if (aTokens.size === 0 || bTokens.size === 0) return 0;

  const intersection = [...aTokens].filter((token) => bTokens.has(token)).length;
  const union = new Set([...aTokens, ...bTokens]).size;
  return intersection / union;
}

function nameSimilarity(a, b) {
  const left = normalizeName(a);
  const right = normalizeName(b);

  return normalizedNameSimilarity(left, right);
}

function normalizedNameSimilarity(left, right) {

  if (!left || !right) return 0;
  if (left === right) return 1;

  const maxLength = Math.max(left.length, right.length);
  const editRatio = maxLength === 0 ? 0 : 1 - levenshtein(left, right) / maxLength;
  return Math.max(editRatio, tokenSimilarity(left, right));
}

function haversineMeters(aLat, aLon, bLat, bLon) {
  const radius = 6371000;
  const toRad = (degrees) => (degrees * Math.PI) / 180;
  const dLat = toRad(bLat - aLat);
  const dLon = toRad(bLon - aLon);
  const lat1 = toRad(aLat);
  const lat2 = toRad(bLat);
  const sinLat = Math.sin(dLat / 2);
  const sinLon = Math.sin(dLon / 2);
  const h = sinLat * sinLat + Math.cos(lat1) * Math.cos(lat2) * sinLon * sinLon;
  return 2 * radius * Math.asin(Math.sqrt(h));
}

function spatialCell(latitude, longitude, cellDegrees = 0.01) {
  return `${Math.floor(latitude / cellDegrees)}:${Math.floor(longitude / cellDegrees)}`;
}

function buildSpatialIndex(records, cellDegrees = 0.01) {
  const index = new Map();
  for (const record of records) {
    if (!hasValidCoordinates(record)) continue;
    const preparedRecord = {
      ...record,
      resolutionNormalizedName: normalizeName(record.name),
    };
    const key = spatialCell(preparedRecord.latitude, preparedRecord.longitude, cellDegrees);
    if (!index.has(key)) index.set(key, []);
    index.get(key).push(preparedRecord);
  }
  for (const values of index.values()) {
    values.sort((left, right) => String(left.id || left.sourceId).localeCompare(String(right.id || right.sourceId)));
  }
  return { cellDegrees, index };
}

function nearbySpatialRecords(spatialIndex, latitude, longitude, radiusCells = 1) {
  const { cellDegrees, index } = spatialIndex;
  const latitudeCell = Math.floor(latitude / cellDegrees);
  const longitudeCell = Math.floor(longitude / cellDegrees);
  const records = [];
  for (let latitudeOffset = -radiusCells; latitudeOffset <= radiusCells; latitudeOffset += 1) {
    for (let longitudeOffset = -radiusCells; longitudeOffset <= radiusCells; longitudeOffset += 1) {
      const key = `${latitudeCell + latitudeOffset}:${longitudeCell + longitudeOffset}`;
      records.push(...(index.get(key) || []));
    }
  }
  return records;
}

function scoreCandidate(sourceRecord, canonicalPoi, precomputedDistanceMeters = null) {
  const distanceMeters = precomputedDistanceMeters ?? haversineMeters(
    sourceRecord.latitude,
    sourceRecord.longitude,
    canonicalPoi.latitude,
    canonicalPoi.longitude,
  );
  const nSim = normalizedNameSimilarity(
    sourceRecord.normalizedName || normalizeName(sourceRecord.name),
    canonicalPoi.resolutionNormalizedName || normalizeName(canonicalPoi.name),
  );
  const cat = categoryCompatibility(sourceRecord.category, canonicalPoi.category);
  const addressEvidence =
    sourceRecord.address && canonicalPoi.district
      ? normalizeName(sourceRecord.address).includes(normalizeName(canonicalPoi.district))
      : false;
  const confidence = Number(
    Math.max(0, Math.min(1, nSim * 0.58 + cat * 0.22 + Math.max(0, 1 - distanceMeters / 500) * 0.2)).toFixed(4),
  );

  return {
    canonicalPoiId: canonicalPoi.id,
    canonicalName: canonicalPoi.name,
    canonicalCategory: canonicalPoi.category,
    distanceMeters: Number(distanceMeters.toFixed(2)),
    nameSimilarity: Number(nSim.toFixed(4)),
    categoryCompatibility: Number(cat.toFixed(2)),
    addressEvidence,
    confidence,
  };
}

function classifySourceRecord(sourceRecord, canonicalPois, thresholds = DEFAULT_THRESHOLDS, options = {}) {
  if (!hasValidCoordinates(sourceRecord)) {
    return {
      source: sourceRecord.source,
      sourceId: sourceRecord.sourceId,
      sourceName: sourceRecord.name,
      decision: DECISIONS.INVALID,
      confidence: 0,
      reasonCodes: ['invalid_or_missing_coordinates'],
      candidates: [],
      provenance: sourceRecord.provenance,
      license: sourceRecord.license,
    };
  }

  const candidates = canonicalPois
    .filter((poi) => hasValidCoordinates(poi))
    .map((poi) => ({
      poi,
      distanceMeters: haversineMeters(
        sourceRecord.latitude,
        sourceRecord.longitude,
        poi.latitude,
        poi.longitude,
      ),
    }))
    .filter(({ distanceMeters }) => (
      options.maxCandidateDistanceMeters == null
      || distanceMeters <= options.maxCandidateDistanceMeters
    ))
    .map(({ poi, distanceMeters }) => scoreCandidate(sourceRecord, poi, distanceMeters))
    .filter((candidate) => candidate.nameSimilarity >= 0.45 || candidate.distanceMeters <= 250)
    .sort((a, b) => {
      if (b.confidence !== a.confidence) return b.confidence - a.confidence;
      if (a.distanceMeters !== b.distanceMeters) return a.distanceMeters - b.distanceMeters;
      return a.canonicalPoiId.localeCompare(b.canonicalPoiId);
    })
    .slice(0, 5);

  const credible = candidates.filter(
    (candidate) =>
      candidate.nameSimilarity >= thresholds.probableName &&
      candidate.distanceMeters <= thresholds.probableDistanceMeters &&
      candidate.categoryCompatibility >= 0.5,
  );
  const high = credible.filter(
    (candidate) =>
      candidate.nameSimilarity >= thresholds.highName &&
      candidate.distanceMeters <= thresholds.highDistanceMeters &&
      candidate.categoryCompatibility >= 0.7,
  );
  const chainLikeCollision = candidates.filter(
    (candidate) =>
      candidate.nameSimilarity >= thresholds.chainCollisionName &&
      candidate.categoryCompatibility >= 0.7 &&
      candidate.distanceMeters <= thresholds.chainCollisionDistanceMeters,
  );

  let decision = DECISIONS.NEW_CANDIDATE;
  const reasonCodes = [];

  if (chainLikeCollision.length > 1) {
    decision = DECISIONS.AMBIGUOUS;
    reasonCodes.push('chain_branch_collision', 'multiple_same_name_nearby_candidates');
  } else if (high.length === 1) {
    decision = DECISIONS.HIGH_CONFIDENCE_MATCH;
    reasonCodes.push('tight_distance', 'strong_name_similarity', 'category_compatible');
  } else if (high.length > 1) {
    decision = DECISIONS.AMBIGUOUS;
    reasonCodes.push('multiple_high_confidence_candidates');
  } else if (credible.length === 1) {
    decision = DECISIONS.PROBABLE_MATCH;
    reasonCodes.push('nearby_coordinates', 'usable_name_similarity', 'category_compatible');
  } else if (credible.length > 1) {
    decision = DECISIONS.AMBIGUOUS;
    reasonCodes.push('multiple_probable_candidates');
  } else {
    reasonCodes.push('no_credible_canonical_match');
  }

  return {
    source: sourceRecord.source,
    sourceId: sourceRecord.sourceId,
    sourceName: sourceRecord.name,
    decision,
    confidence: candidates[0]?.confidence || 0,
    reasonCodes,
    candidates,
    bestCandidate: candidates[0] || null,
    provenance: sourceRecord.provenance,
    license: sourceRecord.license,
  };
}

function isSupportedDuplicatePair(left, right, includeCrossSource) {
  if (left.source === right.source) return true;
  if (!includeCrossSource) return false;
  const pair = new Set([left.source, right.source]);
  if (pair.size !== 2) return false;
  if (pair.has('overture') && pair.has('osm')) return true;
  return pair.has('wikidata') && (pair.has('overture') || pair.has('osm'));
}

function isEntityExternalIdField(field) {
  return !['osm_type', 'source_type', 'dataset'].includes(String(field).toLowerCase());
}

function hasSharedExternalId(left, right) {
  const leftIds = left.externalIds || {};
  const rightIds = right.externalIds || {};
  return Object.entries(leftIds).some(([field, value]) => (
    isEntityExternalIdField(field)
    && value !== null
    && value !== undefined
    && value !== ''
    && rightIds[field] !== null
    && rightIds[field] !== undefined
    && String(rightIds[field]) === String(value)
  ));
}

function duplicateEvidence(left, right) {
  const distanceMeters = haversineMeters(left.latitude, left.longitude, right.latitude, right.longitude);
  const nSim = nameSimilarity(left.name, right.name);
  const samePhone = Boolean(left.phone && right.phone && left.phone === right.phone);
  const sharedExternalId = hasSharedExternalId(left, right);
  const categoryScore = categoryCompatibility(left.category, right.category);
  const duplicate = (
    sharedExternalId
    || (samePhone && distanceMeters <= 150)
    || (nSim >= 0.92 && categoryScore >= 0.55 && distanceMeters <= 60)
  );
  return { categoryScore, distanceMeters, duplicate, nSim, samePhone, sharedExternalId };
}

function hasConservativeDuplicatePrefilter(left, right) {
  if (left.normalizedName && left.normalizedName === right.normalizedName) return true;
  if (left.phone && right.phone && left.phone === right.phone) return true;
  return hasSharedExternalId(left, right);
}

function duplicateResult(left, right, evidence) {
  const sourcePair = [left.source, right.source].sort();
  return {
    decision: DECISIONS.SOURCE_DUPLICATE,
    source: left.source === right.source ? left.source : 'cross_source',
    sourcePair,
    sourceIdA: left.sourceId,
    sourceIdB: right.sourceId,
    sourceA: left.source,
    sourceB: right.source,
    nameA: left.name,
    nameB: right.name,
    distanceMeters: Number(evidence.distanceMeters.toFixed(2)),
    nameSimilarity: Number(evidence.nSim.toFixed(4)),
    categoryCompatibility: Number(evidence.categoryScore.toFixed(2)),
    confidence: evidence.sharedExternalId || evidence.samePhone
      ? 0.98
      : Number(Math.max(evidence.nSim, 1 - evidence.distanceMeters / 100).toFixed(4)),
    reasonCodes: [
      evidence.sharedExternalId
        ? 'shared_external_id'
        : evidence.samePhone
          ? 'same_phone'
          : 'near_duplicate_name_category_and_location',
    ],
    records: [left, right],
  };
}

function detectSourceDuplicates(sourceRecords) {
  const duplicates = [];

  for (let i = 0; i < sourceRecords.length; i += 1) {
    for (let j = i + 1; j < sourceRecords.length; j += 1) {
      const left = sourceRecords[i];
      const right = sourceRecords[j];
      if (left.source !== right.source) continue;
      if (!hasValidCoordinates(left) || !hasValidCoordinates(right)) continue;

      const distanceMeters = haversineMeters(left.latitude, left.longitude, right.latitude, right.longitude);
      const nSim = nameSimilarity(left.name, right.name);
      const samePhone = left.phone && right.phone && left.phone === right.phone;

      if ((nSim >= 0.92 && distanceMeters <= 60) || samePhone) {
        duplicates.push({
          decision: DECISIONS.SOURCE_DUPLICATE,
          source: left.source,
          sourceIdA: left.sourceId,
          sourceIdB: right.sourceId,
          nameA: left.name,
          nameB: right.name,
          distanceMeters: Number(distanceMeters.toFixed(2)),
          nameSimilarity: Number(nSim.toFixed(4)),
          confidence: samePhone ? 0.98 : Number(Math.max(nSim, 1 - distanceMeters / 100).toFixed(4)),
          reasonCodes: [samePhone ? 'same_phone' : 'near_duplicate_name_and_location'],
          records: [left, right],
        });
      }
    }
  }

  return duplicates.sort((a, b) => {
    if (a.source !== b.source) return a.source.localeCompare(b.source);
    if (a.sourceIdA !== b.sourceIdA) return a.sourceIdA.localeCompare(b.sourceIdA);
    return a.sourceIdB.localeCompare(b.sourceIdB);
  });
}

function detectSpatialDuplicates(sourceRecords, { includeCrossSource = true } = {}) {
  const evidenceBuckets = new Map();
  const addToBucket = (key, record) => {
    if (!key) return;
    if (!evidenceBuckets.has(key)) evidenceBuckets.set(key, []);
    evidenceBuckets.get(key).push(record);
  };
  for (const record of sourceRecords) {
    if (!hasValidCoordinates(record)) continue;
    if (record.normalizedName?.length >= 3) addToBucket(`name:${record.normalizedName}`, record);
    if (record.phone) addToBucket(`phone:${record.phone}`, record);
    for (const [field, value] of Object.entries(record.externalIds || {})) {
      if (value && isEntityExternalIdField(field)) addToBucket(`external:${field}:${String(value)}`, record);
    }
  }
  const duplicateParents = new Map();
  const duplicates = [];

  const findDuplicateRoot = (key) => {
    if (!duplicateParents.has(key)) duplicateParents.set(key, key);
    let root = key;
    while (duplicateParents.get(root) !== root) root = duplicateParents.get(root);
    let current = key;
    while (duplicateParents.get(current) !== root) {
      const parent = duplicateParents.get(current);
      duplicateParents.set(current, root);
      current = parent;
    }
    return root;
  };

  const unionDuplicateRoots = (leftKey, rightKey) => {
    const leftRoot = findDuplicateRoot(leftKey);
    const rightRoot = findDuplicateRoot(rightKey);
    if (leftRoot === rightRoot) return false;
    const [parent, child] = [leftRoot, rightRoot].sort();
    duplicateParents.set(child, parent);
    return true;
  };

  const evaluatePair = (left, right) => {
    const leftKey = `${left.source}:${left.sourceId}`;
    const rightKey = `${right.source}:${right.sourceId}`;
    if (leftKey === rightKey || !isSupportedDuplicatePair(left, right, includeCrossSource)) return;
    if (findDuplicateRoot(leftKey) === findDuplicateRoot(rightKey)) return;
    if (!hasConservativeDuplicatePrefilter(left, right)) return;
    const evidence = duplicateEvidence(left, right);
    if (evidence.duplicate && unionDuplicateRoots(leftKey, rightKey)) {
      duplicates.push(duplicateResult(left, right, evidence));
    }
  };

  for (const [bucketKey, bucket] of evidenceBuckets.entries()) {
    if (bucketKey.startsWith('external:') && bucket.length > 100) {
      const ordered = [...bucket].sort((left, right) => (
        `${left.source}:${left.sourceId}`.localeCompare(`${right.source}:${right.sourceId}`)
      ));
      for (let index = 1; index < ordered.length; index += 1) {
        evaluatePair(ordered[0], ordered[index]);
      }
      continue;
    }
    if (bucket.length <= 100) {
      for (let leftIndex = 0; leftIndex < bucket.length; leftIndex += 1) {
        for (let rightIndex = leftIndex + 1; rightIndex < bucket.length; rightIndex += 1) {
          evaluatePair(bucket[leftIndex], bucket[rightIndex]);
        }
      }
      continue;
    }

    const cellDegrees = 0.0003;
    const representativeIndex = { cellDegrees, index: new Map() };
    const radiusCells = bucketKey.startsWith('phone:') ? 5 : 2;
    const ordered = [...bucket].sort((left, right) => (
      `${left.source}:${left.sourceId}`.localeCompare(`${right.source}:${right.sourceId}`)
    ));
    for (const left of ordered) {
      for (const right of nearbySpatialRecords(
        representativeIndex,
        left.latitude,
        left.longitude,
        radiusCells,
      )) {
        evaluatePair(left, right);
      }
      const key = spatialCell(left.latitude, left.longitude, cellDegrees);
      if (!representativeIndex.index.has(key)) representativeIndex.index.set(key, []);
      const representatives = representativeIndex.index.get(key);
      const signature = bucketKey.startsWith('phone:')
        ? left.source
        : `${left.source}:${normalizeCategory(left.category)}`;
      const alreadyRepresented = representatives.some((record) => {
        const recordSignature = bucketKey.startsWith('phone:')
          ? record.source
          : `${record.source}:${normalizeCategory(record.category)}`;
        return recordSignature === signature;
      });
      if (!alreadyRepresented) representatives.push(left);
    }
  }

  return duplicates.sort((left, right) => {
    if (left.sourceIdA !== right.sourceIdA) return left.sourceIdA.localeCompare(right.sourceIdA);
    return left.sourceIdB.localeCompare(right.sourceIdB);
  });
}

function resolveSourceRecords(sourceRecords, canonicalPois, thresholds = DEFAULT_THRESHOLDS, options = {}) {
  const canonicalSpatialIndex = options.useSpatialIndex ? buildSpatialIndex(canonicalPois, 0.01) : null;
  const matches = sourceRecords
    .map((record) => classifySourceRecord(
      record,
      canonicalSpatialIndex
        ? nearbySpatialRecords(canonicalSpatialIndex, record.latitude, record.longitude, 1)
        : canonicalPois,
      thresholds,
      options,
    ))
    .sort((a, b) => {
      if (a.source !== b.source) return a.source.localeCompare(b.source);
      return a.sourceId.localeCompare(b.sourceId);
    });
  const duplicates = options.useSpatialDuplicateIndex
    ? detectSpatialDuplicates(sourceRecords, { includeCrossSource: options.includeCrossSourceDuplicates !== false })
    : detectSourceDuplicates(sourceRecords);

  return {
    matches,
    duplicates,
    summary: {
      highConfidenceMatches: matches.filter((match) => match.decision === DECISIONS.HIGH_CONFIDENCE_MATCH).length,
      probableMatches: matches.filter((match) => match.decision === DECISIONS.PROBABLE_MATCH).length,
      ambiguous: matches.filter((match) => match.decision === DECISIONS.AMBIGUOUS).length,
      newCandidates: matches.filter((match) => match.decision === DECISIONS.NEW_CANDIDATE).length,
      invalid: matches.filter((match) => match.decision === DECISIONS.INVALID).length,
      sourceDuplicates: duplicates.length,
    },
  };
}

module.exports = {
  DECISIONS,
  DEFAULT_THRESHOLDS,
  categoryCompatibility,
  buildSpatialIndex,
  classifySourceRecord,
  detectSpatialDuplicates,
  detectSourceDuplicates,
  haversineMeters,
  hasSharedExternalId,
  nameSimilarity,
  nearbySpatialRecords,
  resolveSourceRecords,
};
