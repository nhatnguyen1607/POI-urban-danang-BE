const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');

const { inspectCanonicalDataset, EXPECTED_CANONICAL_SHA } = require('../../src/modules/cityPackPreparation/canonicalDataset');
const { stableHash } = require('../../src/modules/cityPackPreparation/cityPackBuild/cityPackBuilder');
const {
  hasSharedExternalId,
  resolveSourceRecords,
} = require('../../src/modules/cityPackPreparation/entityResolution');

const ROOT = path.resolve(__dirname, '..', '..');
const STAGE4G_DIR = path.join(ROOT, 'data', 'spikes', 'phase4', 'stage4g');
const CANONICAL_PATH = path.join(ROOT, 'data', 'canonical', 'urbanagent_poi_master_v1.csv');

function readJson(fileName) {
  return JSON.parse(fs.readFileSync(path.join(STAGE4G_DIR, fileName), 'utf8'));
}

const snapshotManifest = readJson('snapshot_manifest.json');
const summary = readJson('stage4g_summary.json');
const quality = readJson('stage4g_quality_spot_checks.json');
const artifactManifest = readJson('stage4g_candidate_artifact_manifest.json');
const proposedChanges = readJson('stage4g_proposed_change_summary.json');

test('Stage 4G records bounded real-source snapshot metadata without committing payloads', () => {
  assert.equal(snapshotManifest.status, 'CANDIDATE_NON_RUNTIME_NOT_CANONICAL');
  assert.deepEqual(snapshotManifest.boundary, {
    source: 'src/services/canonicalCsvPoiRepository.js:DA_NANG_BBOX',
    version: 'pending_spatial_join',
    crs: 'EPSG:4326',
    west: 107.8,
    south: 15.8,
    east: 108.5,
    north: 16.3,
  });
  assert.deepEqual(
    Object.fromEntries(snapshotManifest.snapshots.map((item) => [item.source, item.rawRecordCount])),
    { osm: 6021, overture: 78625, wikidata: 994, wikimedia_commons: 81 },
  );
  assert.ok(snapshotManifest.snapshots.every((item) => /^[a-f0-9]{64}$/.test(item.sha256)));
  assert.equal(snapshotManifest.googlePlacesIncluded, false);
  assert.equal(snapshotManifest.canonicalRuntimeAffected, false);
});

test('Stage 4G preserves the measured full-city classification and candidate metrics', () => {
  assert.equal(summary.normalizedRecords, 76666);
  assert.deepEqual(summary.resolution, {
    highConfidenceMatches: 393,
    probableMatches: 253,
    ambiguous: 52,
    newCandidates: 75968,
    invalid: 0,
    sourceDuplicates: 3112,
  });
  assert.equal(summary.duplicateGroups, 2640);
  assert.equal(summary.proposedEnrichmentCount, 646);
  assert.equal(summary.candidatePackTotal, 76614);
  assert.equal(summary.reviewQueueCount, 79174);
  assert.equal(summary.deterministic, true);
  assert.equal(proposedChanges.canonicalWriteAuthorized, false);
});

test('Stage 4G treats structural source metadata as non-identity evidence', () => {
  assert.equal(
    hasSharedExternalId(
      { externalIds: { osm_type: 'node' } },
      { externalIds: { osm_type: 'node' } },
    ),
    false,
  );
  assert.equal(
    hasSharedExternalId(
      { externalIds: { wikidata: 'Q25282' } },
      { externalIds: { wikidata: 'Q25282' } },
    ),
    true,
  );
});

test('Stage 4G detects conservative cross-source duplicates without false structural-ID matches', () => {
  const sourceRecords = [
    {
      source: 'osm', sourceId: 'osm:node:1', name: 'Coffee Nano', normalizedName: 'nano',
      category: 'cafe', latitude: 16.05, longitude: 108.2,
      phone: null, externalIds: { osm_type: 'node', osm_id: '1' },
    },
    {
      source: 'overture', sourceId: 'overture:1', name: 'NANO coffee', normalizedName: 'nano',
      category: 'cafe', latitude: 16.05002, longitude: 108.20002,
      phone: null, externalIds: { overture_id: '1' },
    },
    {
      source: 'osm', sourceId: 'osm:node:2', name: 'Different Hotel', normalizedName: 'different hotel',
      category: 'accommodation', latitude: 16.2, longitude: 108.35,
      phone: null, externalIds: { osm_type: 'node', osm_id: '2' },
    },
  ];
  const result = resolveSourceRecords(sourceRecords, [], undefined, {
    useSpatialIndex: true,
    useSpatialDuplicateIndex: true,
    includeCrossSourceDuplicates: true,
    maxCandidateDistanceMeters: 800,
  });

  assert.equal(result.duplicates.length, 1);
  assert.deepEqual(result.duplicates[0].sourcePair, ['osm', 'overture']);
  assert.ok(result.duplicates[0].distanceMeters < 5);
});

test('Stage 4G duplicate grouping remains bounded for dense common names', () => {
  const records = Array.from({ length: 2000 }, (_, index) => ({
    source: index % 2 === 0 ? 'osm' : 'overture',
    sourceId: `scale:${index}`,
    name: 'ATM',
    normalizedName: 'atm',
    category: 'bank',
    latitude: 16.05 + (index % 100) * 1e-7,
    longitude: 108.2 + (index % 100) * 1e-7,
    phone: null,
    externalIds: {},
  }));
  const result = resolveSourceRecords(records, [], undefined, {
    useSpatialIndex: true,
    useSpatialDuplicateIndex: true,
    includeCrossSourceDuplicates: true,
    maxCandidateDistanceMeters: 800,
  });

  assert.equal(result.duplicates.length, records.length - 1);
});

test('Stage 4G retained provenance and license metadata are complete', () => {
  assert.equal(summary.provenance.completeRecordProvenanceRate, 1);
  assert.equal(summary.provenance.completeFieldProvenanceRate, 1);
  assert.equal(summary.provenance.completePerFileMediaLicenseRate, 1);
  assert.equal(summary.googlePlacesIncluded, false);
  assert.equal(summary.runtimeChanged, false);
  assert.equal(summary.firestoreChanged, false);
  assert.equal(summary.productionDatabaseTouched, false);
  assert.equal(quality.duplicates.every((item) => item.distanceMeters <= 60), true);
});

test('Stage 4G metadata records deterministic non-runtime artifacts only', () => {
  assert.equal(artifactManifest.status, 'CANDIDATE_NON_RUNTIME_NOT_CANONICAL');
  assert.equal(artifactManifest.candidateArtifactsCommitted, false);
  assert.equal(artifactManifest.candidateOutputPath, 'EXTERNAL_NONCANONICAL_STAGE4G_OUTPUT');
  assert.ok(Object.values(artifactManifest.artifactHashes).every((hash) => /^[a-f0-9]{64}$/.test(hash)));

  const sample = { a: ['Da Nang', 4166, null], b: { enabled: false } };
  const expected = crypto.createHash('sha256').update(JSON.stringify(sample)).digest('hex');
  assert.equal(stableHash(sample), expected);
});

test('Stage 4G leaves the canonical baseline immutable', () => {
  const canonical = inspectCanonicalDataset(CANONICAL_PATH);

  assert.equal(canonical.rows, 4166);
  assert.equal(canonical.sha256, EXPECTED_CANONICAL_SHA);
  assert.equal(summary.canonical.rows, 4166);
  assert.equal(summary.canonical.sha256, EXPECTED_CANONICAL_SHA);
  assert.equal(summary.canonical.path, 'data/canonical/urbanagent_poi_master_v1.csv');
});
