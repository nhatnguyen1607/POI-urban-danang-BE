const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const { inspectCanonicalDataset, EXPECTED_CANONICAL_SHA } = require('../../src/modules/cityPackPreparation/canonicalDataset');
const {
  buildCandidateCityPack,
  validateCandidatePack,
} = require('../../src/modules/cityPackPreparation/cityPackBuild/cityPackBuilder');
const {
  validateProvenanceLicenseManifest,
} = require('../../src/modules/cityPackPreparation/cityPackBuild/provenanceManifest');
const { loadRealSnapshotRecords } = require('../../src/modules/cityPackPreparation/realSnapshot');

const ROOT = path.resolve(__dirname, '..', '..');
const SNAPSHOT_DIR = path.join(ROOT, 'data', 'spikes', 'phase4', 'stage4e', 'snapshots');
const CANONICAL_PATH = path.join(ROOT, 'data', 'canonical', 'urbanagent_poi_master_v1.csv');
const REVIEW_DECISION_PATH = path.join(
  ROOT,
  'data',
  'spikes',
  'phase4',
  'stage4e',
  'review_decisions',
  'stage4e_review_decisions.json',
);

function tempDir(prefix) {
  return fs.mkdtempSync(path.join(os.tmpdir(), prefix));
}

function fixedSnapshotBuild() {
  const rawRecords = loadRealSnapshotRecords({
    overture: path.join(SNAPSHOT_DIR, 'overture_places_2026-07-22.0_bounded.json'),
    osm: path.join(SNAPSHOT_DIR, 'osm_2026-08-10_bounded.json'),
    wikidata: path.join(SNAPSHOT_DIR, 'wikidata_2026-08-10_bounded.json'),
    commons: path.join(SNAPSHOT_DIR, 'wikimedia_commons_2026-08-10_metadata.json'),
  });
  const inputDir = tempDir('urbanagent-stage4f-input-');
  const samplePath = path.join(inputDir, 'records.json');
  fs.writeFileSync(samplePath, `${JSON.stringify({ records: rawRecords })}\n`);

  return buildCandidateCityPack({
    cityId: 'da-nang',
    buildId: 'stage4f-fixed-stage4e-snapshots',
    canonicalPath: CANONICAL_PATH,
    samplePath,
    reviewDecisionPath: REVIEW_DECISION_PATH,
    outputDir: tempDir('urbanagent-stage4f-build-'),
    sourceSnapshots: [
      'osm-map-api-2026-08-10-da-nang-core-bounded',
      'overture-2026-07-22.0-da-nang-core-bounded',
      'wikidata-wdqs-2026-08-10-da-nang-core-bounded',
      'wikimedia-commons-api-2026-08-10-linked-media-metadata',
    ],
  });
}

const first = fixedSnapshotBuild();

test('Stage 4F emits one complete license entry per Commons file', () => {
  const entries = first.licenseManifest.mediaLicenseEntries;

  assert.equal(entries.length, 4);
  assert.equal(new Set(entries.map((entry) => entry.assetIdentifier)).size, 4);
  assert.ok(entries.every((entry) => entry.source === 'wikimedia_commons'));
  assert.ok(entries.every((entry) => entry.licenseName && entry.licenseUrl));
  assert.ok(entries.every((entry) => entry.contributor && entry.attributionText));
  assert.ok(entries.every((entry) => entry.sourcePage && entry.snapshotRef));
});

test('Stage 4F preserves different Commons file licenses independently', () => {
  const entries = first.licenseManifest.mediaLicenseEntries;
  const byLicense = new Map(entries.map((entry) => [entry.licenseName, entry.licenseUrl]));

  assert.equal(byLicense.get('CC BY 2.0'), 'https://creativecommons.org/licenses/by/2.0');
  assert.equal(byLicense.get('CC BY-SA 3.0'), 'https://creativecommons.org/licenses/by-sa/3.0');
  assert.equal(first.licenseManifest.attributionEntries.filter((entry) => entry.scope === 'media').length, 4);
});

test('Stage 4F keeps every OSM field traceable inside the ODbL boundary', () => {
  const osmFields = first.licenseManifest.fieldProvenance.filter((entry) => entry.source === 'osm');

  assert.equal(osmFields.length, 330);
  assert.ok(osmFields.every((entry) => entry.licenseName === 'ODbL-1.0'));
  assert.ok(osmFields.every((entry) => entry.policyClass === 'OPEN_SHAREALIKE_ISOLATED'));
  assert.ok(osmFields.every((entry) => entry.sourceId && entry.fieldName && entry.snapshotRef));
});

test('Stage 4F keeps Wikidata entity provenance separate from Commons media', () => {
  const wikidata = first.licenseManifest.recordProvenance.filter((entry) => entry.source === 'wikidata');
  const commons = first.licenseManifest.mediaLicenseEntries;

  assert.equal(wikidata.length, 12);
  assert.ok(wikidata.every((entry) => entry.sourceId.startsWith('wikidata:')));
  assert.ok(wikidata.every((entry) => entry.licenseName === 'CC0-1.0'));
  assert.ok(commons.every((entry) => entry.parentSource === 'wikidata'));
});

test('Stage 4F retains Overture record and upstream source license distinctions', () => {
  const overture = first.licenseManifest.recordProvenance.filter((entry) => entry.source === 'overture');

  assert.equal(overture.length, 31);
  assert.ok(overture.every((entry) => entry.licenseName === 'CDLA-Permissive-2.0'));
  assert.ok(overture.every((entry) => entry.upstreamSources.length >= 1));
  assert.ok(overture.every((entry) => entry.upstreamSources.every((source) => source.dataset)));
});

test('Stage 4F rejects missing required per-file license metadata', () => {
  const manifest = structuredClone(first.licenseManifest);
  manifest.mediaLicenseEntries[0].licenseName = null;

  const manifestValidation = validateProvenanceLicenseManifest(manifest);
  const packValidation = validateCandidatePack({
    matchedEnrichment: first.pack.matchedEnrichment,
    approvedNewCandidates: first.pack.approvedNewCandidates,
    licenseManifest: manifest,
  });

  assert.equal(manifestValidation.valid, false);
  assert.equal(packValidation.valid, false);
  assert.ok(packValidation.errors.some((error) => error.code === 'MISSING_COMMONS_MEDIA_LICENSE'));
});

test('Stage 4F manifest and artifact hashes are deterministic', () => {
  const second = fixedSnapshotBuild();

  assert.deepEqual(first.licenseManifest, second.licenseManifest);
  assert.deepEqual(first.pack, second.pack);
  assert.deepEqual(first.summary.artifactHashes, second.summary.artifactHashes);
});

test('Stage 4F excludes Google from the candidate manifest', () => {
  assert.equal(first.licenseManifest.googlePlacesIncluded, false);
  assert.equal(
    first.licenseManifest.recordProvenance.some((entry) => entry.source.includes('google')),
    false,
  );
});

test('Stage 4F keeps the canonical baseline intact', () => {
  const canonical = inspectCanonicalDataset(CANONICAL_PATH);

  assert.equal(canonical.rows, 4166);
  assert.equal(canonical.sha256, EXPECTED_CANONICAL_SHA);
  assert.equal(first.summary.runtimeChanged, false);
});
