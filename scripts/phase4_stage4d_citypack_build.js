const path = require('node:path');

const { buildCandidateCityPack } = require('../src/modules/cityPackPreparation/cityPackBuild/cityPackBuilder');

const ROOT = path.resolve(__dirname, '..');
const buildId = process.env.URBANAGENT_STAGE4D_BUILD_ID || 'stage4d-fixture-build';

const result = buildCandidateCityPack({
  cityId: 'da-nang',
  buildId,
  canonicalPath: path.join(ROOT, 'data', 'canonical', 'urbanagent_poi_master_v1.csv'),
  samplePath: path.join(
    ROOT,
    'data',
    'spikes',
    'phase4',
    'stage4b',
    'source_samples',
    'phase4_stage4b_source_samples.json',
  ),
  reviewDecisionPath: path.join(
    ROOT,
    'data',
    'spikes',
    'phase4',
    'stage4d',
    'review_decisions',
    'stage4d_review_decisions.fixture.json',
  ),
  outputDir: path.join(ROOT, 'data', 'citypacks', 'candidates', 'danang', buildId),
});

console.log(JSON.stringify(result.summary, null, 2));
