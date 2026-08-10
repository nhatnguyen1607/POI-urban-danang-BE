const fs = require('node:fs');

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function first(value) {
  return Array.isArray(value) ? value.find((item) => item !== null && item !== undefined) : value;
}

function compactAddress(address) {
  if (!address) return null;
  return [address.freeform, address.locality, address.region, address.country]
    .filter(Boolean)
    .filter((value, index, values) => values.indexOf(value) === index)
    .join(', ') || null;
}

function overtureToAdapterRecord(record, snapshot) {
  const properties = record.properties || {};
  const coordinates = record.geometry?.coordinates || [];
  const primarySource = (properties.sources || []).find((source) => source.dataset !== 'Overture')
    || first(properties.sources)
    || {};

  return {
    source: 'overture',
    sourceId: `overture:${record.id}`,
    name: properties.names?.primary || '',
    category: properties.categories?.primary || properties.basic_category || '',
    latitude: coordinates[1],
    longitude: coordinates[0],
    address: compactAddress(first(properties.addresses)),
    website: first(properties.websites) || null,
    phone: first(properties.phones) || null,
    openingHours: null,
    externalIdentifiers: {
      overture_id: record.id,
      source_record_id: primarySource.record_id || null,
    },
    raw: record,
    snapshotRef: snapshot.snapshotRef,
    license: snapshot.license,
  };
}

function osmCategory(tags = {}) {
  return tags.tourism || tags.amenity || tags.historic || tags.leisure || tags.shop || '';
}

function osmAddress(tags = {}) {
  const parts = [
    [tags['addr:housenumber'], tags['addr:street']].filter(Boolean).join(' '),
    tags['addr:city'],
  ].filter(Boolean);
  return parts.join(', ') || null;
}

function osmToAdapterRecord(record, snapshot) {
  const tags = record.tags || {};
  return {
    source: 'osm',
    sourceId: `osm:${record.type}:${record.id}`,
    name: tags.name || tags['name:en'] || tags['name:vi'] || '',
    category: osmCategory(tags),
    latitude: record.latitude,
    longitude: record.longitude,
    address: osmAddress(tags),
    website: tags.website || tags['contact:website'] || null,
    phone: tags.phone || tags['contact:phone'] || null,
    openingHours: tags.opening_hours || null,
    externalIdentifiers: {
      osm_type: record.type,
      osm_id: String(record.id),
      wikidata: tags.wikidata || null,
    },
    raw: record,
    snapshotRef: snapshot.snapshotRef,
    license: snapshot.license,
  };
}

function parseWktPoint(value) {
  const match = String(value || '').match(/^Point\((-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\)$/);
  return match ? { longitude: Number(match[1]), latitude: Number(match[2]) } : {};
}

function wikidataToAdapterRecord(record, snapshot) {
  const point = parseWktPoint(record.coordinate);
  return {
    source: 'wikidata',
    sourceId: `wikidata:${record.qid}`,
    name: record.label || '',
    category: first(record.instanceLabels) || first(record.instanceIds) || '',
    latitude: point.latitude,
    longitude: point.longitude,
    website: record.website || null,
    externalIdentifiers: { wikidata: record.qid },
    media: record.imageTitle
      ? {
          source: 'wikimedia_commons',
          title: record.imageTitle,
          license: record.mediaLicense || null,
          attribution: record.mediaAttribution || null,
        }
      : null,
    raw: record,
    snapshotRef: snapshot.snapshotRef,
    license: snapshot.license,
  };
}

function loadRealSnapshotRecords(paths) {
  const overture = readJson(paths.overture);
  const osm = readJson(paths.osm);
  const wikidata = readJson(paths.wikidata);

  return [
    ...overture.records.map((record) => overtureToAdapterRecord(record, overture)),
    ...osm.records.map((record) => osmToAdapterRecord(record, osm)),
    ...wikidata.records.map((record) => wikidataToAdapterRecord(record, wikidata)),
  ].sort((a, b) => {
    if (a.source !== b.source) return a.source.localeCompare(b.source);
    return a.sourceId.localeCompare(b.sourceId);
  });
}

module.exports = {
  loadRealSnapshotRecords,
  osmToAdapterRecord,
  overtureToAdapterRecord,
  parseWktPoint,
  wikidataToAdapterRecord,
};
