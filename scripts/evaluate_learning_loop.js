const fs = require('fs');
const path = require('path');
const { runEvaluation } = require('./evaluate_agent');
const { buildAgentMemory } = require('./build_agent_memory');
const { trainReranker } = require('./train_reranker');
const { RERANKER_PATH, resetRerankerCache } = require('../src/services/rerankerService');
const { resetMemoryCache } = require('../src/services/agentMemoryService');

const ROOT_DIR = path.resolve(__dirname, '..');
const OUT_DIR = path.join(ROOT_DIR, 'reports', 'evaluation');
const OUT_FILE = path.join(OUT_DIR, 'learning_loop_eval_latest.json');
function diff(before, after) {
  const keys = ['precisionAtExpectedK', 'recallAtReturnedK', 'f1', 'intentCoverage', 'exactPoiHitRate'];
  return Object.fromEntries(
    keys.map((key) => [key, Number(((after[key] || 0) - (before[key] || 0)).toFixed(4))]),
  );
}

async function evaluateLearningLoop() {
  const previousDisable = process.env.DISABLE_RERANKER;
  process.env.DISABLE_RERANKER = '1';
  resetRerankerCache();
  resetMemoryCache();
  const before = await runEvaluation();

  process.env.DISABLE_RERANKER = previousDisable || '';
  const memory = await buildAgentMemory();
  const reranker = await trainReranker();
  resetRerankerCache();
  resetMemoryCache();
  const after = await runEvaluation();

  const report = {
    createdAt: new Date().toISOString(),
    before: before.aggregate,
    after: after.aggregate,
    delta: diff(before.aggregate, after.aggregate),
    memory: {
      profileCount: Object.keys(memory.profiles || {}).length,
      sources: memory.sources,
    },
    reranker: {
      artifact: RERANKER_PATH,
      trainingSources: reranker.trainingSources,
    },
  };

  fs.mkdirSync(OUT_DIR, { recursive: true });
  fs.writeFileSync(OUT_FILE, JSON.stringify(report, null, 2), 'utf8');
  return report;
}

if (require.main === module) {
  evaluateLearningLoop()
    .then((report) => {
      console.log(JSON.stringify({ before: report.before, after: report.after, delta: report.delta }, null, 2));
      console.log(`Wrote learning loop report to ${OUT_FILE}`);
    })
    .catch((error) => {
      resetRerankerCache();
      resetMemoryCache();
      console.error(error);
      process.exit(1);
    });
}

module.exports = {
  evaluateLearningLoop,
};
