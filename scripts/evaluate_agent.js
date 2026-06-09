const fs = require('fs');
const path = require('path');
const { createItinerary } = require('../src/services/itineraryPlannerService');

const ROOT_DIR = path.resolve(__dirname, '..');
const DEFAULT_DATASET = path.join(ROOT_DIR, 'data', 'synthetic', 'urbanagent_synthetic_v1.jsonl');
const OUT_DIR = path.join(ROOT_DIR, 'reports', 'evaluation');
const OUT_FILE = path.join(OUT_DIR, 'agent_eval_latest.json');

function readJsonl(filePath) {
  return fs
    .readFileSync(filePath, 'utf8')
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

function precisionAtK(predictedIds, expectedIds, k) {
  const topK = predictedIds.slice(0, k);
  if (!topK.length) return 0;
  const hits = topK.filter((id) => expectedIds.has(id)).length;
  return hits / topK.length;
}

function recallAtK(predictedIds, expectedIds, k) {
  if (!expectedIds.size) return 0;
  const topK = predictedIds.slice(0, k);
  const hits = topK.filter((id) => expectedIds.has(id)).length;
  return hits / expectedIds.size;
}

function f1(precision, recall) {
  if (!precision && !recall) return 0;
  return (2 * precision * recall) / (precision + recall);
}

function intentCoverage(predictedItems, expectedItinerary) {
  const expectedRoles = new Set(expectedItinerary.map((item) => item.expected_role));
  if (!expectedRoles.size) return 0;
  const predictedText = predictedItems
    .map((item) => `${item.poi.category} ${item.poi.reason} ${item.poi.title}`)
    .join(' ')
    .toLowerCase();
  const roleTerms = {
    cafe: ['cafe', 'coffee', 'dessert', 'trà', 'tra'],
    food: ['quán ăn', 'quan an', 'food', 'ăn', 'an'],
    seafood: ['hải sản', 'hai san', 'seafood'],
    pub: ['nhậu', 'nhau', 'beer', 'bar'],
    travel: ['du lịch', 'du lich', 'check', 'biển', 'bien'],
  };
  const covered = [...expectedRoles].filter((role) => {
    const terms = roleTerms[role] || [role];
    return terms.some((term) => predictedText.includes(term));
  });
  return covered.length / expectedRoles.size;
}

async function evaluateSample(sample) {
  const expected = sample.expected_output.expected_itinerary || [];
  const expectedIds = new Set(expected.map((item) => item.poi_id));
  const itinerary = await createItinerary({
    query: sample.user_query,
    context: {
      location: sample.query_context.origin,
      personaId: sample.persona?.persona_id,
      userId: sample.query_context?.user_id,
    },
    transport: sample.query_context.transport,
    limit: Math.max(4, expected.length),
  });
  const predictedItems = itinerary.itinerary || [];
  const predictedIds = predictedItems.map((item) => item.poi.id);
  const pAtK = precisionAtK(predictedIds, expectedIds, Math.max(1, expected.length));
  const rAtK = recallAtK(predictedIds, expectedIds, Math.max(1, predictedIds.length));
  const coverage = intentCoverage(predictedItems, expected);

  return {
    sampleId: sample.sample_id,
    personaId: sample.persona?.persona_id,
    query: sample.user_query,
    expectedIds: [...expectedIds],
    predictedIds,
    metrics: {
      precisionAtExpectedK: Number(pAtK.toFixed(4)),
      recallAtReturnedK: Number(rAtK.toFixed(4)),
      f1: Number(f1(pAtK, rAtK).toFixed(4)),
      intentCoverage: Number(coverage.toFixed(4)),
      exactPoiHit: predictedIds.some((id) => expectedIds.has(id)),
    },
  };
}

function aggregate(results) {
  const count = Math.max(results.length, 1);
  const avg = (key) => results.reduce((sum, item) => sum + item.metrics[key], 0) / count;
  return {
    sampleCount: results.length,
    precisionAtExpectedK: Number(avg('precisionAtExpectedK').toFixed(4)),
    recallAtReturnedK: Number(avg('recallAtReturnedK').toFixed(4)),
    f1: Number(avg('f1').toFixed(4)),
    intentCoverage: Number(avg('intentCoverage').toFixed(4)),
    exactPoiHitRate: Number((results.filter((item) => item.metrics.exactPoiHit).length / count).toFixed(4)),
  };
}

async function runEvaluation(datasetPath = DEFAULT_DATASET) {
  if (!fs.existsSync(datasetPath)) {
    throw new Error(`Dataset not found: ${datasetPath}. Run npm run generate:synthetic first.`);
  }
  const samples = readJsonl(datasetPath);
  const results = [];
  for (const sample of samples) {
    results.push(await evaluateSample(sample));
  }
  const report = {
    datasetPath,
    createdAt: new Date().toISOString(),
    aggregate: aggregate(results),
    results,
  };
  fs.mkdirSync(OUT_DIR, { recursive: true });
  fs.writeFileSync(OUT_FILE, JSON.stringify(report, null, 2), 'utf8');
  return report;
}

if (require.main === module) {
  runEvaluation(process.argv[2] || DEFAULT_DATASET)
    .then((report) => {
      console.log(JSON.stringify(report.aggregate, null, 2));
      console.log(`Wrote evaluation report to ${OUT_FILE}`);
    })
    .catch((error) => {
      console.error(error);
      process.exit(1);
    });
}

module.exports = {
  runEvaluation,
  evaluateSample,
};
