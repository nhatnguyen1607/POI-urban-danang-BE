const fs = require('fs');
const path = require('path');
const { normalizeText } = require('./poiDataService');
const { detectIntents } = require('./intentService');
const { clamp01 } = require('./scoringUtils');
const { getProfile, scoreCategoryPreference, scorePoiPreference } = require('./agentMemoryService');

const ROOT_DIR = path.resolve(__dirname, '..', '..');
const RERANKER_PATH = path.join(ROOT_DIR, 'artifacts', 'reranker', 'agent_reranker_v1.json');

let cache = null;

function loadReranker() {
  if (process.env.DISABLE_RERANKER === '1') return null;
  if (cache !== null) return cache;
  if (!fs.existsSync(RERANKER_PATH)) {
    cache = null;
    return null;
  }
  try {
    cache = JSON.parse(fs.readFileSync(RERANKER_PATH, 'utf8'));
    return cache;
  } catch (error) {
    console.warn(`[Reranker] Failed to load ${RERANKER_PATH}: ${error.message}`);
    cache = null;
    return null;
  }
}

function resetRerankerCache() {
  cache = null;
}

function valueFromMap(map, key, fallback = 0) {
  if (!map || !key) return fallback;
  return Number(map[key]) || fallback;
}

function getCategoryBoost(reranker, poi) {
  const category = normalizeText(poi.category);
  const categoryWeights = reranker?.categoryWeights || {};
  return Object.entries(categoryWeights).reduce((best, [categoryKey, weight]) => {
    return category.includes(normalizeText(categoryKey)) ? Math.max(best, Number(weight) || 0) : best;
  }, 0);
}

function getIntentBoost(reranker, query, poi) {
  const intents = detectIntents(query);
  const intentWeights = reranker?.intentCategoryWeights || {};
  const category = normalizeText(poi.category);
  return intents.reduce((sum, intent) => {
    const weights = intentWeights[intent.id] || {};
    const hit = Object.entries(weights).reduce((best, [categoryKey, weight]) => {
      return category.includes(normalizeText(categoryKey)) ? Math.max(best, Number(weight) || 0) : best;
    }, 0);
    return sum + hit;
  }, 0);
}

function getIntentPoiPenalty(reranker, query, poi) {
  const intents = detectIntents(query);
  const penalties = reranker?.intentPoiPenalties || {};
  return intents.reduce((sum, intent) => {
    return sum + valueFromMap(penalties[intent.id], poi.id);
  }, 0);
}

function applyReranker(scoredItem, query, context = {}) {
  const reranker = loadReranker();
  const profile = getProfile(context);
  if (!reranker && !profile) return scoredItem;

  const poi = scoredItem.poi;
  const poiBoost = valueFromMap(reranker?.poiWeights, poi.id);
  const categoryBoost = getCategoryBoost(reranker, poi);
  const intentBoost = getIntentBoost(reranker, query, poi);
  const penalty = valueFromMap(reranker?.poiPenalties, poi.id) + getIntentPoiPenalty(reranker, query, poi);
  const memoryPoiBoost = scorePoiPreference(profile, poi.id);
  const memoryCategoryBoost = scoreCategoryPreference(profile, poi.category);
  const learningRate = reranker?.learningRate || 0.08;
  const memoryRate = profile?.memoryRate || 0.08;
  const delta = clamp01((poiBoost + categoryBoost + intentBoost) * learningRate);
  const memoryDelta = Math.max(-0.32, Math.min(0.32, (memoryPoiBoost + memoryCategoryBoost) * memoryRate));
  const penaltyDelta = clamp01(penalty * learningRate);
  const score = clamp01(scoredItem.score + delta + memoryDelta - penaltyDelta);

  return {
    ...scoredItem,
    score,
    reranker: {
      artifactVersion: reranker?.version || null,
      integrationMode: reranker?.researchModel?.integrationMode || 'runtime_json_reranker',
      memoryVersion: profile?.version || null,
      memoryKey: profile?.profileId || null,
      poiBoost,
      categoryBoost,
      intentBoost,
      memoryPoiBoost,
      memoryCategoryBoost,
      penalty,
      delta: Number((delta + memoryDelta - penaltyDelta).toFixed(4)),
    },
  };
}

module.exports = {
  RERANKER_PATH,
  loadReranker,
  resetRerankerCache,
  applyReranker,
};
