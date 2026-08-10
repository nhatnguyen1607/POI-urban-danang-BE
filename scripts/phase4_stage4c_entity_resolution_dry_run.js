const path = require('node:path');

const { runStage4cDryRun } = require('../src/modules/cityPackPreparation/dryRun');

const ROOT = path.resolve(__dirname, '..');

const result = runStage4cDryRun({
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
  outputDir: path.join(ROOT, 'data', 'spikes', 'phase4', 'stage4c'),
});

console.log(JSON.stringify(result.summary, null, 2));
