const fs = require('fs');
const path = require('path');

const ROOT_DIR = path.resolve(__dirname, '..', '..');
const POI_URBAN_DIR = path.resolve(ROOT_DIR, '..', 'poi_urban');

function readJson(filePath, fallback = null) {
  try {
    if (!fs.existsSync(filePath)) return fallback;
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
  } catch (error) {
    return { error: error.message, file: filePath };
  }
}

function fileInfo(filePath) {
  if (!fs.existsSync(filePath)) return null;
  const stat = fs.statSync(filePath);
  return {
    path: filePath,
    size: stat.size,
    updatedAt: stat.mtime.toISOString(),
  };
}

function getAgentTrainingStatus() {
  const syntheticSummaryPath = path.join(ROOT_DIR, 'data', 'synthetic', 'urbanagent_synthetic_v1.summary.json');
  const trainingSummaryPath = path.join(ROOT_DIR, 'data', 'training', 'agent_representation_pairs_v1.summary.json');
  const learningReportPath = path.join(ROOT_DIR, 'reports', 'evaluation', 'learning_loop_eval_latest.json');
  const twoTowerMetricsPath = path.join(
    POI_URBAN_DIR,
    'results',
    'agent_representation_two_tower',
    'two_tower_metrics.json',
  );
  const rerankerMetricsPath = path.join(
    POI_URBAN_DIR,
    'results',
    'agent_representation',
    'agent_representation_metrics.json',
  );

  return {
    generatedAt: new Date().toISOString(),
    backend: {
      synthetic: readJson(syntheticSummaryPath, null),
      representationData: readJson(trainingSummaryPath, null),
      learningLoop: readJson(learningReportPath, null),
      figures: {
        learningBeforeAfter: fileInfo(path.join(ROOT_DIR, 'reports', 'evaluation', 'figures', 'learning_before_after.svg')),
        learningSummaryCard: fileInfo(path.join(ROOT_DIR, 'reports', 'evaluation', 'figures', 'learning_summary_card.svg')),
      },
    },
    research: {
      repo: POI_URBAN_DIR,
      rerankerMetrics: readJson(rerankerMetricsPath, null),
      twoTowerMetrics: readJson(twoTowerMetricsPath, null),
      figures: {
        rerankerPng: fileInfo(path.join(POI_URBAN_DIR, 'results', 'agent_representation', 'agent_representation_metrics.png')),
        rerankerSvg: fileInfo(path.join(POI_URBAN_DIR, 'results', 'agent_representation', 'agent_representation_metrics.svg')),
        twoTowerPng: fileInfo(
          path.join(POI_URBAN_DIR, 'results', 'agent_representation_two_tower', 'two_tower_training_report.png'),
        ),
        twoTowerSvg: fileInfo(
          path.join(POI_URBAN_DIR, 'results', 'agent_representation_two_tower', 'two_tower_training_report.svg'),
        ),
        twoTowerCheckpoint: fileInfo(
          path.join(POI_URBAN_DIR, 'results', 'agent_representation_two_tower', 'agent_two_tower_representation.pt'),
        ),
      },
    },
    commands: [
      'cd D:\\POI-urban-danang-BE && npm run generate:synthetic -- --count=120',
      'cd D:\\POI-urban-danang-BE && npm run export:representation-data',
      'cd D:\\poi_urban && python research_pipeline/train_agent_representation_reranker.py',
      'cd D:\\poi_urban && python research_pipeline/train_agent_two_tower_representation.py --epochs 80',
      'cd D:\\POI-urban-danang-BE && npm run evaluate:learning && npm run visualize:evaluation',
    ],
  };
}

module.exports = {
  getAgentTrainingStatus,
};
