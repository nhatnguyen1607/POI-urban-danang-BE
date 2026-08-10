const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');

const ROOT = path.resolve(__dirname, '..');
const CANONICAL_PATH = path.join(ROOT, 'data', 'canonical', 'urbanagent_poi_master_v1.csv');
const SAMPLE_PATH = path.join(
  ROOT,
  'data',
  'spikes',
  'phase4',
  'stage4b',
  'source_samples',
  'phase4_stage4b_source_samples.json',
);
const OUTPUT_DIR = path.join(ROOT, 'data', 'spikes', 'phase4', 'stage4b');
const EXPECTED_SHA = '5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae';

const THRESHOLDS = {
  conservative: { highName: 0.9, highDistanceMeters: 50, probableName: 0.6, probableDistanceMeters: 150 },
  balanced: { highName: 0.88, highDistanceMeters: 75, probableName: 0.72, probableDistanceMeters: 250 },
  loose: { highName: 0.84, highDistanceMeters: 100, probableName: 0.65, probableDistanceMeters: 400 },
};

function sha256(filePath) {
  return crypto.createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let value = '';
  let inQuotes = false;

  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const next = text[i + 1];

    if (char === '"') {
      if (inQuotes && next === '"') {
        value += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }

    if (char === ',' && !inQuotes) {
      row.push(value);
      value = '';
      continue;
    }

    if ((char === '\n' || char === '\r') && !inQuotes) {
      if (char === '\r' && next === '\n') i += 1;
      row.push(value);
      if (row.some((cell) => cell !== '')) rows.push(row);
      row = [];
      value = '';
      continue;
    }

    value += char;
  }

  if (value || row.length > 0) {
    row.push(value);
    rows.push(row);
  }

  return rows;
}

function numberOrNull(value) {
  if (value === null || value === undefined || value === '') return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function normalizeName(value) {
  return String(value || '')
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/đ/g, 'd')
    .replace(/Đ/g, 'd')
    .toLowerCase()
    .replace(/&/g, ' and ')
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\b(da nang|danang|dn|quan|tiem|cua hang|nha hang|coffee|cafe|bakery)\b/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function normalizeCategory(value) {
  const raw = normalizeName(value);
  if (!raw) return 'unknown';
  if (/(bakery|banh|bread)/.test(raw)) return 'bakery';
  if (/(cafe|coffee|tea|tra sua|bubble|milk tea)/.test(raw)) return 'cafe';
  if (/(restaurant|food|mon|mi quang|bo ne|nhau|seafood)/.test(raw)) return 'restaurant';
  if (/(beach|bai bien)/.test(raw)) return 'beach';
  if (/(bridge|cau)/.test(raw)) return 'bridge';
  if (/(museum|bao tang)/.test(raw)) return 'museum';
  if (/(market|cho)/.test(raw)) return 'market';
  if (/(park|attraction|tourism|landmark)/.test(raw)) return 'attraction';
  if (/(media|photo|image)/.test(raw)) return 'media';
  return raw.split(' ')[0] || 'unknown';
}

function readCanonicalPois() {
  const rows = parseCsv(fs.readFileSync(CANONICAL_PATH, 'utf8'));
  const headers = rows[0].map((header, index) =>
    index === 0 ? header.replace(/^\uFEFF/, '').trim() : header.trim(),
  );

  return rows.slice(1).map((cells) => {
    const record = Object.fromEntries(headers.map((header, index) => [header, cells[index] ?? '']));
    return {
      id: record.Global_ID,
      name: record['Restaurant Name'],
      category: normalizeCategory(record.Category_Normalized || record.Category),
      sourceCategory: record.Category,
      latitude: numberOrNull(record.Lat),
      longitude: numberOrNull(record.Lon),
      address: record.Address_Current || record.Address_Raw || '',
      district: record.District || record.District_Raw || '',
      source: record.Source,
    };
  });
}

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

function hasValidCoordinates(record) {
  return (
    Number.isFinite(record.latitude) &&
    Number.isFinite(record.longitude) &&
    record.latitude >= -90 &&
    record.latitude <= 90 &&
    record.longitude >= -180 &&
    record.longitude <= 180
  );
}

function normalizeSourceRecord(record) {
  return {
    ...record,
    source: String(record.source || '').trim(),
    sourceId: String(record.sourceId || '').trim(),
    name: String(record.name || '').trim(),
    normalizedName: normalizeName(record.name),
    normalizedCategory: normalizeCategory(record.category),
    latitude: numberOrNull(record.latitude),
    longitude: numberOrNull(record.longitude),
    address: String(record.address || '').trim(),
    website: record.website || null,
    phone: record.phone || null,
    openingHours: record.openingHours || null,
    externalIdentifiers: record.externalIdentifiers || {},
    license: record.license || {},
  };
}

function scoreCandidate(sourceRecord, canonicalPoi) {
  const distanceMeters = haversineMeters(
    sourceRecord.latitude,
    sourceRecord.longitude,
    canonicalPoi.latitude,
    canonicalPoi.longitude,
  );
  const nSim = nameSimilarity(sourceRecord.name, canonicalPoi.name);
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

function emptyMatch(sourceRecord) {
  return {
    source: sourceRecord.source,
    sourceId: sourceRecord.sourceId,
    sourceName: sourceRecord.name,
    sourceCategory: sourceRecord.normalizedCategory,
    canonicalPoiId: null,
    canonicalName: null,
    distanceMeters: null,
    nameSimilarity: null,
    categoryCompatibility: null,
    addressEvidence: false,
  };
}

function classifyRecord(sourceRecord, canonicalPois, thresholds) {
  if (!hasValidCoordinates(sourceRecord)) {
    return {
      ...emptyMatch(sourceRecord),
      decision: 'invalid_coordinates',
      confidence: 0,
      reasonCodes: ['invalid_or_missing_coordinates'],
      candidates: [],
      licensePolicyClass: sourceRecord.license?.policyClass || null,
      attribution: sourceRecord.license?.attribution || null,
    };
  }

  const candidates = canonicalPois
    .filter((poi) => hasValidCoordinates(poi))
    .map((poi) => scoreCandidate(sourceRecord, poi))
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

  let decision = 'new_entity_candidate';
  const reasonCodes = [];

  const chainLikeCollision = candidates.filter(
    (candidate) =>
      candidate.nameSimilarity >= 0.92 &&
      candidate.categoryCompatibility >= 0.7 &&
      candidate.distanceMeters <= 750,
  );

  if (chainLikeCollision.length > 1) {
    decision = 'ambiguous';
    reasonCodes.push('chain_branch_collision', 'multiple_same_name_nearby_candidates');
  } else if (high.length === 1) {
    decision = 'high_confidence_match';
    reasonCodes.push('tight_distance', 'strong_name_similarity', 'category_compatible');
  } else if (high.length > 1) {
    decision = 'ambiguous';
    reasonCodes.push('multiple_high_confidence_candidates');
  } else if (credible.length === 1) {
    decision = 'probable_match';
    reasonCodes.push('nearby_coordinates', 'usable_name_similarity', 'category_compatible');
  } else if (credible.length > 1) {
    decision = 'ambiguous';
    reasonCodes.push('multiple_probable_candidates');
  } else {
    reasonCodes.push('no_credible_canonical_match');
  }

  const best = candidates[0] || null;
  return {
    source: sourceRecord.source,
    sourceId: sourceRecord.sourceId,
    sourceName: sourceRecord.name,
    sourceCategory: sourceRecord.normalizedCategory,
    canonicalPoiId: best?.canonicalPoiId || null,
    canonicalName: best?.canonicalName || null,
    distanceMeters: best?.distanceMeters ?? null,
    nameSimilarity: best?.nameSimilarity ?? null,
    categoryCompatibility: best?.categoryCompatibility ?? null,
    addressEvidence: best?.addressEvidence ?? false,
    decision,
    confidence: best?.confidence ?? 0,
    reasonCodes,
    candidates,
    licensePolicyClass: sourceRecord.license?.policyClass || null,
    attribution: sourceRecord.license?.attribution || null,
  };
}

function detectSourceDuplicates(records) {
  const duplicates = [];
  for (let i = 0; i < records.length; i += 1) {
    for (let j = i + 1; j < records.length; j += 1) {
      const left = records[i];
      const right = records[j];
      if (left.source !== right.source) continue;
      if (!hasValidCoordinates(left) || !hasValidCoordinates(right)) continue;
      const distanceMeters = haversineMeters(left.latitude, left.longitude, right.latitude, right.longitude);
      const nSim = nameSimilarity(left.name, right.name);
      const samePhone = left.phone && right.phone && left.phone === right.phone;
      if ((nSim >= 0.92 && distanceMeters <= 60) || samePhone) {
        duplicates.push({
          source: left.source,
          sourceIdA: left.sourceId,
          sourceIdB: right.sourceId,
          nameA: left.name,
          nameB: right.name,
          distanceMeters: Number(distanceMeters.toFixed(2)),
          nameSimilarity: Number(nSim.toFixed(4)),
          reasonCodes: [samePhone ? 'same_phone' : 'near_duplicate_name_and_location'],
        });
      }
    }
  }
  return duplicates;
}

function summarizeByDecision(matches) {
  return {
    highConfidenceMatches: matches.filter((match) => match.decision === 'high_confidence_match').length,
    probableMatches: matches.filter((match) => match.decision === 'probable_match').length,
    ambiguous: matches.filter((match) => match.decision === 'ambiguous').length,
    newCandidates: matches.filter((match) => match.decision === 'new_entity_candidate').length,
    invalidOrMissingCoordinates: matches.filter((match) => match.decision === 'invalid_coordinates').length,
  };
}

function summarizeEnrichment(records, matches) {
  const accepted = new Set(
    matches
      .filter((match) => ['high_confidence_match', 'probable_match'].includes(match.decision))
      .map((match) => match.sourceId),
  );
  const matchedRecords = records.filter((record) => accepted.has(record.sourceId));
  const withField = (predicate) => matchedRecords.filter(predicate).length;

  return {
    matchedRecords: matchedRecords.length,
    addressAvailable: withField((record) => Boolean(record.address)),
    websiteAvailable: withField((record) => Boolean(record.website)),
    phoneAvailable: withField((record) => Boolean(record.phone)),
    openingHoursAvailable: withField((record) => Boolean(record.openingHours)),
    wikidataIdAvailable: withField((record) => Boolean(record.externalIdentifiers.wikidata_qid)),
    commonsMediaAvailable: withField((record) => Boolean(record.externalIdentifiers.commons_file || record.media)),
    sourceLicenseTracked: withField((record) => Boolean(record.license?.policyClass)),
  };
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

function writeJson(fileName, value) {
  fs.writeFileSync(path.join(OUTPUT_DIR, fileName), `${JSON.stringify(value, null, 2)}\n`);
}

function writeCsv(fileName, rows, headers) {
  fs.writeFileSync(path.join(OUTPUT_DIR, fileName), toCsv(rows, headers));
}

function main() {
  const canonicalSha = sha256(CANONICAL_PATH);
  const canonicalPois = readCanonicalPois();
  const fixture = JSON.parse(fs.readFileSync(SAMPLE_PATH, 'utf8'));
  const records = fixture.records.map(normalizeSourceRecord);
  const duplicates = detectSourceDuplicates(records);
  const matches = records.map((record) => classifyRecord(record, canonicalPois, THRESHOLDS.conservative));
  const thresholdSensitivity = Object.fromEntries(
    Object.entries(THRESHOLDS).map(([name, thresholds]) => [
      name,
      summarizeByDecision(records.map((record) => classifyRecord(record, canonicalPois, thresholds))),
    ]),
  );
  const enrichmentCoverage = summarizeEnrichment(records, matches);
  const sourceCounts = records.reduce((acc, record) => {
    acc[record.source] = (acc[record.source] || 0) + 1;
    return acc;
  }, {});

  const summary = {
    status: 'NON_CANONICAL_SPIKE_ONLY',
    canonical: {
      path: 'data/canonical/urbanagent_poi_master_v1.csv',
      rows: canonicalPois.length,
      sha256: canonicalSha,
      shaMatchesExpected: canonicalSha === EXPECTED_SHA,
    },
    samples: {
      sourceSampleCount: records.length,
      sourceCounts,
      successfullyNormalized: records.filter((record) => record.sourceId && record.name).length,
      invalidOrMissingCoordinates: matches.filter((match) => match.decision === 'invalid_coordinates').length,
      addressAvailable: records.filter((record) => Boolean(record.address)).length,
      openingHoursAvailable: records.filter((record) => Boolean(record.openingHours)).length,
      externalIdAvailable: records.filter((record) => Object.keys(record.externalIdentifiers).length > 0).length,
    },
    matches: {
      ...summarizeByDecision(matches),
      duplicateSourceCandidates: duplicates.length,
    },
    enrichmentCoverage,
    thresholdSensitivity,
    licenseObservations: [
      'Overture sample rows must preserve per-record source/license metadata before any future canonical import.',
      'OSM sample rows are license-isolated because ODbL share-alike obligations may affect derived databases.',
      'Wikidata entity data can be handled as CC0 knowledge enrichment, but source identifiers and statement provenance should remain namespaced.',
      'Wikimedia Commons media requires file-level license, author, source URL, attribution, and derivative-use metadata.',
      'Google Places was not queried; it remains request-time/live enrichment only and not canonical offline storage.',
    ],
  };

  const candidateHeaders = [
    'source',
    'sourceId',
    'sourceName',
    'canonicalPoiId',
    'canonicalName',
    'distanceMeters',
    'nameSimilarity',
    'categoryCompatibility',
    'addressEvidence',
    'decision',
    'confidence',
    'reasonCodes',
    'licensePolicyClass',
    'attribution',
  ];
  const candidateRows = matches.map((match) => ({
    source: match.source,
    sourceId: match.sourceId,
    sourceName: match.sourceName,
    canonicalPoiId: match.canonicalPoiId,
    canonicalName: match.canonicalName,
    distanceMeters: match.distanceMeters,
    nameSimilarity: match.nameSimilarity,
    categoryCompatibility: match.categoryCompatibility,
    addressEvidence: match.addressEvidence,
    decision: match.decision,
    confidence: match.confidence,
    reasonCodes: match.reasonCodes,
    licensePolicyClass: match.licensePolicyClass,
    attribution: match.attribution,
  }));
  const enrichmentRows = records.map((record) => ({
    source: record.source,
    sourceId: record.sourceId,
    name: record.name,
    hasAddress: Boolean(record.address),
    hasWebsite: Boolean(record.website),
    hasPhone: Boolean(record.phone),
    hasOpeningHours: Boolean(record.openingHours),
    hasExternalIds: Object.keys(record.externalIdentifiers).length > 0,
    hasMedia: Boolean(record.media || record.externalIdentifiers.commons_file),
    licensePolicyClass: record.license?.policyClass || '',
    attribution: record.license?.attribution || '',
  }));

  writeJson('phase4_stage4b_match_summary.json', summary);
  writeJson('phase4_stage4b_threshold_sensitivity.json', thresholdSensitivity);
  writeJson('phase4_stage4b_duplicate_source_candidates.json', duplicates);
  writeCsv('phase4_stage4b_match_candidates.csv', candidateRows, candidateHeaders);
  writeCsv(
    'phase4_stage4b_ambiguous.csv',
    candidateRows.filter((row) => row.decision === 'ambiguous'),
    candidateHeaders,
  );
  writeCsv(
    'phase4_stage4b_new_candidates.csv',
    candidateRows.filter((row) => row.decision === 'new_entity_candidate'),
    candidateHeaders,
  );
  writeCsv('phase4_stage4b_enrichment_coverage.csv', enrichmentRows, [
    'source',
    'sourceId',
    'name',
    'hasAddress',
    'hasWebsite',
    'hasPhone',
    'hasOpeningHours',
    'hasExternalIds',
    'hasMedia',
    'licensePolicyClass',
    'attribution',
  ]);

  console.log(JSON.stringify(summary, null, 2));
}

main();
