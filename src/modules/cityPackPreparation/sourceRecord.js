function numberOrNull(value) {
  if (value === null || value === undefined || value === '') {
    return null;
  }

  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function normalizeName(value) {
  return String(value || '')
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/\u0111/g, 'd')
    .replace(/\u0110/g, 'd')
    .toLowerCase()
    .replace(/&/g, ' and ')
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\b(da nang|danang|dn|quan|tiem|cua hang|nha hang|coffee|cafe|bakery)\b/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function normalizeCategoryText(value) {
  return String(value || '')
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/\u0111/g, 'd')
    .replace(/\u0110/g, 'd')
    .toLowerCase()
    .replace(/&/g, ' and ')
    .replace(/[^a-z0-9\s_]/g, ' ')
    .replace(/[_\s]+/g, ' ')
    .trim();
}

function normalizeCategory(value) {
  const raw = normalizeCategoryText(value);

  if (!raw) return 'unknown';
  if (/(bakery|banh|bread)/.test(raw)) return 'bakery';
  if (/(cafe|coffee|ca phe|tea room|tra sua|bubble|milk tea)/.test(raw)) return 'cafe';
  if (/(bar|pub|beer|bia|cocktail|karaoke)/.test(raw)) return 'bar';
  if (/(restaurant|food|diner|fast food|mon|mi quang|bo ne|quan an|nha hang|nhau|seafood)/.test(raw)) {
    return 'restaurant';
  }
  if (/(hotel|hostel|lodging|accommodation|guest house|khach san|nha nghi)/.test(raw)) {
    return 'accommodation';
  }
  if (/(beach|bai bien)/.test(raw)) return 'beach';
  if (/(bridge|cau)/.test(raw)) return 'bridge';
  if (/(museum|bao tang)/.test(raw)) return 'museum';
  if (/(market|cho)/.test(raw)) return 'market';
  if (/(church|cathedral|temple|pagoda|place of worship|nha tho|chua)/.test(raw)) return 'place_of_worship';
  if (/(park|garden|attraction|tourism|landmark|historic|artwork)/.test(raw)) return 'attraction';
  if (/(media|photo|image)/.test(raw)) return 'media';

  return raw.split(' ')[0] || 'unknown';
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

function createFieldProvenance(sourceRecord, fieldName) {
  return {
    source: sourceRecord.source,
    sourceId: sourceRecord.sourceId,
    field: fieldName,
    license: sourceRecord.license?.license || null,
    policyClass: sourceRecord.license?.policyClass || null,
    attribution: sourceRecord.license?.attribution || null,
    licenseUrl: sourceRecord.license?.licenseUrl || null,
    attributionUrl: sourceRecord.license?.attributionUrl || null,
    policyReference: sourceRecord.license?.policyReference || null,
    snapshotRef: sourceRecord.snapshotRef || sourceRecord.provenance?.snapshotRef || null,
  };
}

function normalizeSourceRecord(rawRecord, adapterName) {
  const source = String(rawRecord.source || adapterName || '').trim();
  const sourceId = String(rawRecord.sourceId || '').trim();
  const normalized = {
    source,
    sourceId,
    name: String(rawRecord.name || '').trim(),
    normalizedName: normalizeName(rawRecord.name),
    category: normalizeCategory(rawRecord.category),
    sourceCategory: rawRecord.category || null,
    latitude: numberOrNull(rawRecord.latitude),
    longitude: numberOrNull(rawRecord.longitude),
    address: String(rawRecord.address || '').trim() || null,
    website: rawRecord.website || null,
    phone: rawRecord.phone || null,
    openingHours: rawRecord.openingHours || null,
    externalIds: rawRecord.externalIdentifiers || rawRecord.externalIds || {},
    media: rawRecord.media || null,
    raw: rawRecord.raw || null,
    provenance: {
      source,
      sourceId,
      adapter: adapterName,
      snapshotRef: rawRecord.snapshotRef || 'stage4b-fixture',
      license: rawRecord.license?.license || null,
      policyClass: rawRecord.license?.policyClass || null,
      attribution: rawRecord.license?.attribution || null,
      licenseUrl: rawRecord.license?.licenseUrl || null,
      attributionUrl: rawRecord.license?.attributionUrl || null,
      policyReference: rawRecord.license?.policyReference || null,
      upstreamSources: Array.isArray(rawRecord.upstreamSources)
        ? [...rawRecord.upstreamSources]
        : [],
      fields: {},
    },
    license: rawRecord.license || {},
  };

  for (const fieldName of [
    'name',
    'category',
    'latitude',
    'longitude',
    'address',
    'website',
    'phone',
    'openingHours',
    'externalIds',
    'media',
  ]) {
    normalized.provenance.fields[fieldName] = createFieldProvenance(normalized, fieldName);
  }

  return normalized;
}

module.exports = {
  createFieldProvenance,
  hasValidCoordinates,
  normalizeCategory,
  normalizeCategoryText,
  normalizeName,
  normalizeSourceRecord,
  numberOrNull,
};
