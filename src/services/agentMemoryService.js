const fs = require('fs');
const path = require('path');
const { normalizeText } = require('./poiDataService');

const ROOT_DIR = path.resolve(__dirname, '..', '..');
const MEMORY_PATH = path.join(ROOT_DIR, 'artifacts', 'memory', 'agent_memory_v1.json');

let cache = null;

function loadMemory() {
  if (cache !== null) return cache;
  if (!fs.existsSync(MEMORY_PATH)) {
    cache = null;
    return null;
  }
  try {
    cache = JSON.parse(fs.readFileSync(MEMORY_PATH, 'utf8'));
    return cache;
  } catch (error) {
    console.warn(`[Memory] Failed to load ${MEMORY_PATH}: ${error.message}`);
    cache = null;
    return null;
  }
}

function resetMemoryCache() {
  cache = null;
}

function resolveMemoryKey(context = {}) {
  return context.userId || context.personaId || context.sessionId || 'global';
}

function getProfile(context = {}) {
  if (context.agentMemory) {
    return {
      profileId: context.agentMemory.userId || context.userId || 'firestore',
      version: context.agentMemory.version || 'firestore',
      memoryRate: Number(context.agentMemory.memoryRate) || 0.12,
      preferences: {
        categoryAffinity: context.agentMemory.categoryAffinity || {},
        categoryPenalty: context.agentMemory.categoryPenalty || {},
        poiAffinity: context.agentMemory.poiAffinity || {},
        poiPenalty: context.agentMemory.poiPenalty || {},
      },
    };
  }
  const memory = loadMemory();
  if (!memory) return null;
  const key = resolveMemoryKey(context);
  return memory.profiles[key] || memory.profiles.global || null;
}

function scoreCategoryPreference(profile, category) {
  if (!profile || !category) return 0;
  const normalizedCategory = normalizeText(category);
  const positives = profile.preferences?.categoryAffinity || {};
  const negatives = profile.preferences?.categoryPenalty || {};
  const positive = Object.entries(positives).reduce((best, [key, value]) => {
    return normalizedCategory.includes(normalizeText(key)) ? Math.max(best, Number(value) || 0) : best;
  }, 0);
  const negative = Object.entries(negatives).reduce((best, [key, value]) => {
    return normalizedCategory.includes(normalizeText(key)) ? Math.max(best, Number(value) || 0) : best;
  }, 0);
  return positive - negative;
}

function scorePoiPreference(profile, poiId) {
  if (!profile || !poiId) return 0;
  const positive = Number(profile.preferences?.poiAffinity?.[poiId]) || 0;
  const negative = Number(profile.preferences?.poiPenalty?.[poiId]) || 0;
  return positive - negative;
}

function buildReflection(profile) {
  if (!profile) return '';
  const topCategories = Object.entries(profile.preferences?.categoryAffinity || {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, 4)
    .map(([category]) => category);
  const weakCategories = Object.entries(profile.preferences?.categoryPenalty || {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(([category]) => category);
  const liked = topCategories.length ? `Uu tien ${topCategories.join(', ')}` : 'Chua co so thich danh muc ro rang';
  const disliked = weakCategories.length ? `han che ${weakCategories.join(', ')}` : 'chua co tin hieu khong thich ro rang';
  return `${liked}; ${disliked}.`;
}

module.exports = {
  MEMORY_PATH,
  loadMemory,
  resetMemoryCache,
  resolveMemoryKey,
  getProfile,
  scoreCategoryPreference,
  scorePoiPreference,
  buildReflection,
};
