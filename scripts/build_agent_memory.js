const fs = require('fs');
const path = require('path');
const { loadPOIs, normalizeText } = require('../src/services/poiDataService');
const { MEMORY_PATH, resetMemoryCache, buildReflection } = require('../src/services/agentMemoryService');

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
  const normalized = normalizeText(key)
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean)[0] || normalizeText(key);
  map[normalized] = Number(((map[normalized] || 0) + delta).toFixed(4));
}

function addRawWeight(map, key, delta) {
  if (!key) return;
  map[key] = Number(((map[key] || 0) + delta).toFixed(4));
}

function getOrCreateProfile(profiles, profileId, seed = {}) {
  if (!profiles[profileId]) {
    profiles[profileId] = {
      profileId,
      version: 'agent_memory_v1',
      memoryRate: 0.14,
      segment: seed.segment || 'unknown',
      observations: [],
      preferences: {
        categoryAffinity: {},
        categoryPenalty: {},
        poiAffinity: {},
        poiPenalty: {},
        tags: {},
      },
      reflection: '',
    };
  }
  return profiles[profileId];
}

function trainFromSynthetic(samples, poiById, profiles) {
  samples.forEach((sample) => {
    const persona = sample.persona || {};
    const profileId = persona.persona_id || sample.sample_id;
    const profile = getOrCreateProfile(profiles, profileId, { segment: persona.segment });
    (persona.preferences?.liked_tags || []).forEach((tag) => addWeight(profile.preferences.tags, tag, 0.5));
    (persona.preferences?.preferred_categories || []).forEach((category) => {
      addWeight(profile.preferences.categoryAffinity, category, 0.7);
    });

    (sample.expected_output?.expected_itinerary || []).forEach((item) => {
      const poi = poiById.get(item.poi_id);
      if (!poi) return;
      addRawWeight(profile.preferences.poiAffinity, poi.id, 1.0);
      addWeight(profile.preferences.categoryAffinity, poi.category, 0.8);
      profile.observations.push({
        type: 'synthetic_positive',
        poiId: poi.id,
        category: poi.category,
        query: sample.user_query,
        evidence: item.reason_evidence || [],
      });
    });

    (sample.expected_output?.negative_pois || []).forEach((item) => {
      const poi = poiById.get(item.poi_id);
      addRawWeight(profile.preferences.poiPenalty, item.poi_id, 1.0);
      if (poi) addWeight(profile.preferences.categoryPenalty, poi.category, 0.4);
    });
  });
}

function trainFromFeedback(events, profiles) {
  const global = getOrCreateProfile(profiles, 'global', { segment: 'global_feedback' });
  events.forEach((event) => {
    const payload = event.payload || {};
    const profileId = event.userId || payload.userId || event.personaId || payload.personaId || 'global';
    const profile = getOrCreateProfile(profiles, profileId);
    const targetProfiles = profileId === 'global' ? [global] : [profile, global];
    targetProfiles.forEach((target) => {
      target.observations.push({
        type: event.eventType,
        poiId: payload.poiId,
        category: payload.category,
        query: event.query,
        timestamp: event.timestamp,
      });
      if (['add_to_itinerary', 'poi_useful', 'route_requested'].includes(event.eventType)) {
        addRawWeight(target.preferences.poiAffinity, payload.poiId, event.eventType === 'route_requested' ? 0.25 : 0.8);
        addWeight(target.preferences.categoryAffinity, payload.category, 0.35);
      }
      if (['remove_from_itinerary', 'poi_not_fit'].includes(event.eventType)) {
        addRawWeight(target.preferences.poiPenalty, payload.poiId, event.eventType === 'poi_not_fit' ? 0.8 : 0.5);
        addWeight(target.preferences.categoryPenalty, payload.category, 0.35);
      }
    });
  });
}

function finalizeProfiles(profiles) {
  Object.values(profiles).forEach((profile) => {
    profile.observationCount = profile.observations.length;
    profile.reflection = buildReflection(profile);
    profile.updatedAt = new Date().toISOString();
  });
}

async function buildAgentMemory() {
  const pois = await loadPOIs();
  const poiById = new Map(pois.map((poi) => [poi.id, poi]));
  const synthetic = readJsonlIfExists(SYNTHETIC_FILE);
  const feedback = readJsonlIfExists(FEEDBACK_FILE);
  const profiles = {};

  getOrCreateProfile(profiles, 'global', { segment: 'global' });
  trainFromSynthetic(synthetic, poiById, profiles);
  trainFromFeedback(feedback, profiles);
  finalizeProfiles(profiles);

  const artifact = {
    version: 'agent_memory_v1',
    createdAt: new Date().toISOString(),
    sources: {
      syntheticFile: SYNTHETIC_FILE,
      feedbackFile: FEEDBACK_FILE,
      syntheticSamples: synthetic.length,
      feedbackEvents: feedback.length,
    },
    profiles,
  };

  fs.mkdirSync(path.dirname(MEMORY_PATH), { recursive: true });
  fs.writeFileSync(MEMORY_PATH, JSON.stringify(artifact, null, 2), 'utf8');
  resetMemoryCache();
  return artifact;
}

if (require.main === module) {
  buildAgentMemory()
    .then((artifact) => {
      console.log(
        JSON.stringify(
          {
            artifact: MEMORY_PATH,
            profiles: Object.keys(artifact.profiles).length,
            syntheticSamples: artifact.sources.syntheticSamples,
            feedbackEvents: artifact.sources.feedbackEvents,
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
  buildAgentMemory,
};
