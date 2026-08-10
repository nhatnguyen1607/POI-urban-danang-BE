const crypto = require('node:crypto');

const POLICY_REFERENCE = 'docs/rebuild/PHASE4_SOURCE_LICENSE_REGISTRY.md';
const LICENSE_URLS = Object.freeze({
  'CC0-1.0': 'https://creativecommons.org/publicdomain/zero/1.0/',
  'CDLA-Permissive-2.0': 'https://cdla.dev/permissive-2-0/',
  'ODbL-1.0': 'https://opendatacommons.org/licenses/odbl/1-0/',
});

function hashId(prefix, value) {
  const digest = crypto.createHash('sha256').update(JSON.stringify(value)).digest('hex');
  return `${prefix}:${digest.slice(0, 20)}`;
}

function arrayOf(value) {
  if (value === null || value === undefined) return [];
  return Array.isArray(value) ? value : [value];
}

function compactText(value) {
  return value === null || value === undefined || value === '' ? null : String(value);
}

function licenseUrl(licenseName, explicitUrl) {
  return compactText(explicitUrl) || LICENSE_URLS[licenseName] || null;
}

function sortedUnique(items, keyFn) {
  const byKey = new Map();
  for (const item of items) {
    const key = keyFn(item);
    if (!byKey.has(key)) byKey.set(key, item);
  }
  return [...byKey.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([, item]) => item);
}

function normalizeUpstreamSources(sources) {
  return arrayOf(sources)
    .map((source) => ({
      dataset: compactText(source.dataset),
      recordId: compactText(source.recordId ?? source.record_id),
      property: compactText(source.property),
      licenseName: compactText(source.licenseName ?? source.license),
      updateTime: compactText(source.updateTime ?? source.update_time),
      confidence: Number.isFinite(source.confidence) ? source.confidence : null,
    }))
    .sort((left, right) => JSON.stringify(left).localeCompare(JSON.stringify(right)));
}

function sourceLicenseEntry(record) {
  const provenance = record.provenance || {};
  const licenseName = compactText(provenance.license);
  const entry = {
    source: compactText(record.source),
    sourceId: null,
    assetIdentifier: null,
    licenseName,
    licenseUrl: licenseUrl(licenseName, provenance.licenseUrl),
    policyReference: compactText(provenance.policyReference) || POLICY_REFERENCE,
    policyClass: compactText(provenance.policyClass),
    attributionText: compactText(provenance.attribution),
    attributionUrl: compactText(provenance.attributionUrl),
    contributor: null,
    snapshotRef: compactText(provenance.snapshotRef),
    scope: 'source_records',
    provenanceRelationship: 'source_snapshot_governs_records',
  };
  return {
    ...entry,
    entryId: hashId('license', entry),
    license: entry.licenseName,
    attribution: entry.attributionText,
    sourceIds: [],
  };
}

function recordProvenanceEntry(record) {
  const provenance = record.provenance || {};
  const licenseName = compactText(provenance.license);
  const entry = {
    source: compactText(record.source),
    sourceId: compactText(record.sourceId),
    assetIdentifier: null,
    licenseName,
    licenseUrl: licenseUrl(licenseName, provenance.licenseUrl),
    policyReference: compactText(provenance.policyReference) || POLICY_REFERENCE,
    policyClass: compactText(provenance.policyClass),
    attributionText: compactText(provenance.attribution),
    attributionUrl: compactText(provenance.attributionUrl),
    contributor: null,
    snapshotRef: compactText(provenance.snapshotRef),
    scope: 'record',
    provenanceRelationship: 'normalized_from_external_record',
    upstreamSources: normalizeUpstreamSources(provenance.upstreamSources),
  };
  return { ...entry, entryId: hashId('record-provenance', entry) };
}

function fieldProvenanceEntries(record) {
  return Object.entries(record.provenance?.fields || {})
    .map(([fieldName, field]) => {
      const licenseName = compactText(field.license);
      const entry = {
        source: compactText(field.source),
        sourceId: compactText(field.sourceId),
        assetIdentifier: null,
        fieldName,
        licenseName,
        licenseUrl: licenseUrl(licenseName, field.licenseUrl),
        policyReference: compactText(field.policyReference) || POLICY_REFERENCE,
        policyClass: compactText(field.policyClass),
        attributionText: compactText(field.attribution),
        attributionUrl: compactText(field.attributionUrl),
        contributor: null,
        snapshotRef: compactText(field.snapshotRef),
        scope: 'field',
        provenanceRelationship: 'field_derived_from_external_record',
      };
      return { ...entry, entryId: hashId('field-provenance', entry) };
    })
    .sort((left, right) => left.entryId.localeCompare(right.entryId));
}

function mediaLicenseEntries(record) {
  return arrayOf(record.media)
    .filter(Boolean)
    .map((media) => {
      const licenseName = compactText(media.licenseName ?? media.license);
      const assetIdentifier = compactText(
        media.assetIdentifier ?? media.assetId ?? media.pageId ?? media.title ?? record.sourceId,
      );
      const entry = {
        source: compactText(media.source) || compactText(record.source),
        sourceId: compactText(media.sourceId) || compactText(record.sourceId),
        assetIdentifier,
        mediaReference: compactText(media.mediaReference ?? media.url ?? media.sourcePage),
        licenseName,
        licenseUrl: licenseUrl(licenseName, media.licenseUrl),
        policyReference: compactText(media.policyReference) || POLICY_REFERENCE,
        policyClass: compactText(media.policyClass) || compactText(record.provenance?.policyClass),
        attributionText: compactText(media.attributionText ?? media.attribution),
        attributionUrl: compactText(media.attributionUrl ?? media.sourcePage),
        contributor: compactText(media.creator ?? media.author),
        sourcePage: compactText(media.sourcePage),
        snapshotRef: compactText(media.snapshotRef) || compactText(record.provenance?.snapshotRef),
        retrievalReference: compactText(media.retrievedAt),
        scope: 'media',
        provenanceRelationship: compactText(media.provenanceRelationship)
          || 'media_linked_from_external_record',
        parentSource: compactText(record.source),
        parentSourceId: compactText(record.sourceId),
      };
      return { ...entry, entryId: hashId('media-license', entry) };
    })
    .sort((left, right) => left.entryId.localeCompare(right.entryId));
}

function attributionEntries(sourceManifest, mediaEntries) {
  const sourceAttributions = sourceManifest.map((source) => {
    const entry = {
      source: source.source,
      sourceId: null,
      assetIdentifier: null,
      attributionText: source.attributionText,
      attributionUrl: source.attributionUrl,
      contributor: null,
      licenseName: source.licenseName,
      licenseUrl: source.licenseUrl,
      snapshotRef: source.snapshotRef,
      scope: 'source_records',
      provenanceRelationship: 'attribution_for_source_snapshot',
    };
    return { ...entry, entryId: hashId('attribution', entry) };
  });
  const mediaAttributions = mediaEntries.map((media) => {
    const entry = {
      source: media.source,
      sourceId: media.sourceId,
      assetIdentifier: media.assetIdentifier,
      attributionText: media.attributionText,
      attributionUrl: media.attributionUrl,
      contributor: media.contributor,
      licenseName: media.licenseName,
      licenseUrl: media.licenseUrl,
      snapshotRef: media.snapshotRef,
      scope: 'media',
      provenanceRelationship: 'attribution_for_media_asset',
    };
    return { ...entry, entryId: hashId('attribution', entry) };
  });
  return [...sourceAttributions, ...mediaAttributions]
    .sort((left, right) => left.entryId.localeCompare(right.entryId));
}

function generateProvenanceLicenseManifest({ normalizedRecords, buildId, cityId, buildVersion, status }) {
  const sourceGroups = new Map();
  const recordProvenance = normalizedRecords
    .map(recordProvenanceEntry)
    .sort((left, right) => left.entryId.localeCompare(right.entryId));
  const fieldProvenance = normalizedRecords
    .flatMap(fieldProvenanceEntries)
    .sort((left, right) => left.entryId.localeCompare(right.entryId));
  const mediaEntries = normalizedRecords
    .flatMap(mediaLicenseEntries)
    .sort((left, right) => left.entryId.localeCompare(right.entryId));

  for (const record of normalizedRecords) {
    const sourceEntry = sourceLicenseEntry(record);
    const key = JSON.stringify({
      source: sourceEntry.source,
      snapshotRef: sourceEntry.snapshotRef,
      licenseName: sourceEntry.licenseName,
      policyClass: sourceEntry.policyClass,
      attributionText: sourceEntry.attributionText,
    });
    if (!sourceGroups.has(key)) sourceGroups.set(key, sourceEntry);
    sourceGroups.get(key).sourceIds.push(record.sourceId);
  }

  const sourceManifest = [...sourceGroups.values()]
    .map((entry) => ({ ...entry, sourceIds: [...new Set(entry.sourceIds)].sort() }))
    .sort((left, right) => left.entryId.localeCompare(right.entryId));
  const attributions = attributionEntries(sourceManifest, mediaEntries);

  return {
    status,
    schemaVersion: 'phase4-stage4f-provenance-license-v1',
    buildVersion,
    buildId,
    cityId,
    googlePlacesIncluded: false,
    sourceManifest,
    sources: sourceManifest,
    recordProvenance,
    fieldProvenance,
    mediaLicenseEntries: mediaEntries,
    attributionEntries: attributions,
    licenseEntries: [...sourceManifest, ...mediaEntries]
      .sort((left, right) => left.entryId.localeCompare(right.entryId)),
    counts: {
      sources: sourceManifest.length,
      records: recordProvenance.length,
      fields: fieldProvenance.length,
      mediaLicenses: mediaEntries.length,
      attributions: attributions.length,
    },
  };
}

function validateProvenanceLicenseManifest(manifest) {
  const errors = [];
  const required = (entry, fields, code) => {
    const missing = fields.filter((field) => entry[field] === null || entry[field] === undefined || entry[field] === '');
    if (missing.length) errors.push({ code, entryId: entry.entryId || null, missing });
  };
  const hasGoogle = [
    ...(manifest.recordProvenance || []),
    ...(manifest.mediaLicenseEntries || []),
  ].some((entry) => String(entry.source || '').toLowerCase().includes('google'));

  for (const entry of manifest.recordProvenance || []) {
    required(
      entry,
      ['source', 'sourceId', 'licenseName', 'policyClass', 'attributionText', 'snapshotRef'],
      'MISSING_RECORD_PROVENANCE',
    );
    if (entry.source === 'overture' && entry.snapshotRef?.startsWith('overture-') && !entry.upstreamSources.length) {
      errors.push({ code: 'MISSING_OVERTURE_UPSTREAM_PROVENANCE', sourceId: entry.sourceId });
    }
    if (entry.source === 'wikidata' && (!entry.sourceId.startsWith('wikidata:') || !entry.licenseName.includes('CC0'))) {
      errors.push({ code: 'INVALID_WIKIDATA_PROVENANCE', sourceId: entry.sourceId });
    }
  }

  for (const entry of manifest.fieldProvenance || []) {
    if (entry.source !== 'osm') continue;
    required(
      entry,
      ['source', 'sourceId', 'fieldName', 'licenseName', 'policyClass', 'attributionText', 'snapshotRef'],
      'MISSING_OSM_FIELD_PROVENANCE',
    );
    if (entry.policyClass !== 'OPEN_SHAREALIKE_ISOLATED') {
      errors.push({ code: 'INVALID_OSM_POLICY_BOUNDARY', entryId: entry.entryId });
    }
  }

  for (const entry of manifest.mediaLicenseEntries || []) {
    if (entry.source !== 'wikimedia_commons') continue;
    required(
      entry,
      ['source', 'sourceId', 'assetIdentifier', 'licenseName', 'attributionText', 'snapshotRef'],
      'MISSING_COMMONS_MEDIA_LICENSE',
    );
    if (entry.snapshotRef?.startsWith('wikimedia-commons-api-')) {
      required(
        entry,
        ['licenseUrl', 'attributionUrl', 'contributor', 'sourcePage'],
        'MISSING_REAL_COMMONS_MEDIA_ATTRIBUTION',
      );
    }
  }

  if (hasGoogle || manifest.googlePlacesIncluded !== false) {
    errors.push({ code: 'GOOGLE_SOURCE_NOT_ALLOWED' });
  }

  return { valid: errors.length === 0, errors };
}

module.exports = {
  POLICY_REFERENCE,
  generateProvenanceLicenseManifest,
  mediaLicenseEntries,
  normalizeUpstreamSources,
  validateProvenanceLicenseManifest,
};
