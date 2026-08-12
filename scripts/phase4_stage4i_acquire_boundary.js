const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');

const RELATION_ID = 1891418;
const RELATION_VERSION = 72;
const RELATION_TIMESTAMP = '2025-03-11T13:49:11Z';
const SNAPSHOT_AT = '2025-05-31T23:59:59Z';
const CHANGESET_ID = 163485854;
const DEFAULT_OUTPUT = path.resolve(
  __dirname,
  '..',
  'data',
  'spikes',
  'phase4',
  'stage4i',
  'boundary',
  'osm-relation-1891418-v72.geojson',
);

function endpointKey(coordinate) {
  return `${coordinate[0].toFixed(7)},${coordinate[1].toFixed(7)}`;
}

function signedArea(ring) {
  let area = 0;
  for (let index = 0; index < ring.length - 1; index += 1) {
    area += ring[index][0] * ring[index + 1][1] - ring[index + 1][0] * ring[index][1];
  }
  return area / 2;
}

function canonicalizeRing(openRing) {
  let ring = openRing;
  if (signedArea([...ring, ring[0]]) < 0) ring = [...ring].reverse();
  const keys = ring.map(endpointKey);
  let start = 0;
  for (let index = 1; index < keys.length; index += 1) {
    if (keys[index].localeCompare(keys[start]) < 0) start = index;
  }
  const rotated = [...ring.slice(start), ...ring.slice(0, start)];
  return [...rotated, rotated[0]];
}

function stitchOuterWays(members) {
  const ways = members
    .filter((member) => member.type === 'way' && member.role === 'outer' && member.geometry?.length >= 2)
    .map((member) => ({
      ref: String(member.ref),
      coordinates: member.geometry.map((point) => [Number(point.lon), Number(point.lat)]),
    }))
    .sort((left, right) => left.ref.localeCompare(right.ref));
  if (ways.length === 0) throw new Error('Historical relation contains no outer ways.');

  const unused = new Map(ways.map((way) => [way.ref, way]));
  const rings = [];
  while (unused.size > 0) {
    const seed = [...unused.values()].sort((left, right) => left.ref.localeCompare(right.ref))[0];
    unused.delete(seed.ref);
    const ring = [...seed.coordinates];
    while (endpointKey(ring[0]) !== endpointKey(ring[ring.length - 1])) {
      const end = endpointKey(ring[ring.length - 1]);
      const candidates = [...unused.values()]
        .filter((way) => (
          endpointKey(way.coordinates[0]) === end
          || endpointKey(way.coordinates[way.coordinates.length - 1]) === end
        ))
        .sort((left, right) => left.ref.localeCompare(right.ref));
      if (candidates.length === 0) throw new Error(`Boundary ring is open at ${end}.`);
      const next = candidates[0];
      unused.delete(next.ref);
      const oriented = endpointKey(next.coordinates[0]) === end
        ? next.coordinates
        : [...next.coordinates].reverse();
      ring.push(...oriented.slice(1));
    }
    rings.push(canonicalizeRing(ring.slice(0, -1)));
  }
  return rings.sort((left, right) => endpointKey(left[0]).localeCompare(endpointKey(right[0])));
}

function featureFromOverpass(payload) {
  const relation = payload.elements?.find(
    (element) => element.type === 'relation' && Number(element.id) === RELATION_ID,
  );
  if (!relation) throw new Error(`OSM relation ${RELATION_ID} is missing.`);
  const rings = stitchOuterWays(relation.members || []);
  if (rings.length !== 1) {
    throw new Error(`Expected one outer product-scope polygon, received ${rings.length}.`);
  }
  return {
    type: 'Feature',
    properties: {
      artifactStatus: 'VERSIONED_PRODUCT_SCOPE_BOUNDARY',
      cityId: 'da-nang',
      name: 'Da Nang pre-2025-merger product boundary',
      source: 'OpenStreetMap',
      sourceIdentifier: `relation/${RELATION_ID}`,
      sourceVersion: RELATION_VERSION,
      sourceTimestamp: RELATION_TIMESTAMP,
      sourceChangeset: CHANGESET_ID,
      overpassSnapshotAt: SNAPSHOT_AT,
      administrativeLevel: 4,
      crs: 'EPSG:4326',
      license: 'ODbL-1.0',
      attribution: 'OpenStreetMap contributors',
      licenseUrl: 'https://opendatacommons.org/licenses/odbl/1-0/',
      currentLegalBoundary: false,
      scopeNote: 'Versioned UrbanAgent Da Nang City Pack boundary; Hoi An remains a separate future City Pack.',
    },
    geometry: {
      type: 'Polygon',
      coordinates: rings,
    },
  };
}

function overpassQuery() {
  return `[out:json][timeout:120][date:"${SNAPSHOT_AT}"];relation(${RELATION_ID});out body geom;`;
}

async function fetchHistoricalRelation() {
  const body = new URLSearchParams({ data: overpassQuery() });
  const response = await fetch('https://overpass-api.de/api/interpreter', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
      'User-Agent': 'UrbanAgent-Phase4-Research/1.0',
    },
    body,
  });
  if (!response.ok) throw new Error(`Overpass boundary request failed: HTTP ${response.status}.`);
  return response.json();
}

function sha256(bytes) {
  return crypto.createHash('sha256').update(bytes).digest('hex');
}

async function acquireBoundary({ inputPath = null, outputPath = DEFAULT_OUTPUT } = {}) {
  const payload = inputPath
    ? JSON.parse(fs.readFileSync(inputPath, 'utf8'))
    : await fetchHistoricalRelation();
  const feature = featureFromOverpass(payload);
  const bytes = Buffer.from(`${JSON.stringify(feature)}\n`, 'utf8');
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, bytes);
  return {
    outputPath,
    sha256: sha256(bytes),
    relationId: RELATION_ID,
    relationVersion: RELATION_VERSION,
    snapshotAt: SNAPSHOT_AT,
    outerPoints: feature.geometry.coordinates[0].length,
  };
}

if (require.main === module) {
  const inputIndex = process.argv.indexOf('--input');
  const outputIndex = process.argv.indexOf('--output');
  acquireBoundary({
    inputPath: inputIndex >= 0 ? path.resolve(process.argv[inputIndex + 1]) : null,
    outputPath: outputIndex >= 0 ? path.resolve(process.argv[outputIndex + 1]) : DEFAULT_OUTPUT,
  }).then((result) => console.log(JSON.stringify(result, null, 2))).catch((error) => {
    console.error(error.message);
    process.exitCode = 1;
  });
}

module.exports = {
  CHANGESET_ID,
  DEFAULT_OUTPUT,
  RELATION_ID,
  RELATION_TIMESTAMP,
  RELATION_VERSION,
  SNAPSHOT_AT,
  acquireBoundary,
  canonicalizeRing,
  featureFromOverpass,
  overpassQuery,
  stitchOuterWays,
};
