const fs = require('fs');
const path = require('path');
const { loadPOIs, normalizeText } = require('../src/services/poiDataService');
const { detectIntents } = require('../src/services/intentService');
const { RERANKER_PATH, resetRerankerCache } = require('../src/services/rerankerService');
const { MEMORY_PATH } = require('../src/services/agentMemoryService');

const ROOT_DIR = path.resolve(__dirname, '..');
const SYNTHETIC_FILE = path.join(ROOT_DIR, 'data', 'synthetic', 'urbanagent_synthetic_v1.jsonl');
const FEEDBACK_FILE = path.join(ROOT_DIR, 'storage', 'feedback', 'agent-feedback.jsonl');

function readJsonlIfExists(filePath) {
  if (!fs.existsSync(filePath)) return [];
  return fs
    .readFileSync(filePath, 'utf8')
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

function addWeight(map, key, delta) {
  if (!key) return;
  map[key] = Number(((map[key] || 0) + delta).toFixed(4));
}

function normalizeWeights(map, maxAbs = 4) {
  Object.keys(map).forEach((key) => {
    map[key] = Number(Math.max(-maxAbs, Math.min(maxAbs, map[key])).toFixed(4));
  });
  return map;
}

function ensureIntent(intentCategoryWeights, intentId) {
  if (!intentCategoryWeights[intentId]) intentCategoryWeights[intentId] = {};
  return intentCategoryWeights[intentId];
}

function trainFromSynthetic(samples, poiById, model) {
  samples.forEach((sample) => {
    const intents = sample.expected_output.intent_labels || detectIntents(sample.user_query).map((intent) => intent.id);
    const positives = sample.expected_output.expected_itinerary || [];
    positives.forEach((item) => {
      const poi = poiById.get(item.poi_id);
      if (!poi) return;
      addWeight(model.poiWeights, poi.id, 1.0);
      addWeight(model.categoryWeights, poi.category, 0.45);
      intents.forEach((intentId) => {
        addWeight(ensureIntent(model.intentCategoryWeights, intentId), poi.category, 0.6);
      });
    });

    (sample.expected_output.negative_pois || []).forEach((item) => {
      addWeight(model.poiPenalties, item.poi_id, 1.0);
    });
  });
}

function trainFromFeedback(events, model) {
  events.forEach((event) => {
    const payload = event.payload || {};
    const poiId = payload.poiId;
    const category = payload.category;
    if (['add_to_itinerary', 'poi_useful', 'route_requested'].includes(event.eventType)) {
      addWeight(model.poiWeights, poiId, event.eventType === 'route_requested' ? 0.25 : 0.8);
      addWeight(model.categoryWeights, category, 0.25);
    }
    if (['remove_from_itinerary', 'poi_not_fit'].includes(event.eventType)) {
      addWeight(model.poiPenalties, poiId, event.eventType === 'poi_not_fit' ? 0.8 : 0.5);
      if (category) addWeight(model.categoryWeights, category, -0.15);
    }
  });
}

function compactCategoryWeights(categoryWeights) {
  const compact = {};
  Object.entries(categoryWeights).forEach(([category, weight]) => {
    const normalized = normalizeText(category)
      .split(',')
      .map((item) => item.trim())
      .filter(Boolean)[0] || normalizeText(category);
    addWeight(compact, normalized, Number(weight));
  });
  return normalizeWeights(compact);
}

function trainFromMemory(memory, model) {
  if (!memory?.profiles) return;
  Object.values(memory.profiles).forEach((profile) => {
    Object.entries(profile.preferences?.poiAffinity || {}).forEach(([poiId, weight]) => {
      addWeight(model.poiWeights, poiId, Number(weight) * 0.55);
    });
    Object.entries(profile.preferences?.poiPenalty || {}).forEach(([poiId, weight]) => {
      addWeight(model.poiPenalties, poiId, Number(weight) * 0.55);
    });
    Object.entries(profile.preferences?.categoryAffinity || {}).forEach(([category, weight]) => {
      addWeight(model.categoryWeights, category, Number(weight) * 0.35);
    });
    Object.entries(profile.preferences?.categoryPenalty || {}).forEach(([category, weight]) => {
      addWeight(model.categoryWeights, category, -Number(weight) * 0.25);
    });
  });
}

async function trainReranker() {
  const pois = await loadPOIs();
  const poiById = new Map(pois.map((poi) => [poi.id, poi]));
  const synthetic = readJsonlIfExists(SYNTHETIC_FILE);
  const feedback = readJsonlIfExists(FEEDBACK_FILE);
  const memory = fs.existsSync(MEMORY_PATH) ? JSON.parse(fs.readFileSync(MEMORY_PATH, 'utf8')) : null;

  const model = {
    version: 'agent_reranker_v1',
    createdAt: new Date().toISOString(),
    learningRate: 0.12,
    trainingSources: {
      syntheticFile: SYNTHETIC_FILE,
      feedbackFile: FEEDBACK_FILE,
      memoryFile: MEMORY_PATH,
      syntheticSamples: synthetic.length,
      feedbackEvents: feedback.length,
      memoryProfiles: memory?.profiles ? Object.keys(memory.profiles).length : 0,
    },
    poiWeights: {},
    poiPenalties: {},
    categoryWeights: {},
    intentCategoryWeights: {},
  };

  trainFromSynthetic(synthetic, poiById, model);
  trainFromFeedback(feedback, model);
  trainFromMemory(memory, model);
  model.poiWeights = normalizeWeights(model.poiWeights);
  model.poiPenalties = normalizeWeights(model.poiPenalties);
  model.categoryWeights = compactCategoryWeights(model.categoryWeights);
  Object.keys(model.intentCategoryWeights).forEach((intentId) => {
    model.intentCategoryWeights[intentId] = compactCategoryWeights(model.intentCategoryWeights[intentId]);
  });

  fs.mkdirSync(path.dirname(RERANKER_PATH), { recursive: true });
  fs.writeFileSync(RERANKER_PATH, JSON.stringify(model, null, 2), 'utf8');
  resetRerankerCache();
  return model;
}

if (require.main === module) {
  trainReranker()
    .then((model) => {
      console.log(
        JSON.stringify(
          {
            artifact: RERANKER_PATH,
            syntheticSamples: model.trainingSources.syntheticSamples,
            feedbackEvents: model.trainingSources.feedbackEvents,
            poiWeights: Object.keys(model.poiWeights).length,
            categoryWeights: Object.keys(model.categoryWeights).length,
          },
          null,
          2,
        ),
      );
    })
    .catch((error) => {
      console.error(error);
      process.exit(1);
    });
}

module.exports = {
  trainReranker,
};
