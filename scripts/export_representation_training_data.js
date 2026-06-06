const fs = require('fs');
const path = require('path');
const { loadPOIs } = require('../src/services/poiDataService');

const ROOT_DIR = path.resolve(__dirname, '..');
const SYNTHETIC_FILE = path.join(ROOT_DIR, 'data', 'synthetic', 'urbanagent_synthetic_v1.jsonl');
const FEEDBACK_FILE = path.join(ROOT_DIR, 'storage', 'feedback', 'agent-feedback.jsonl');
const OUT_DIR = path.join(ROOT_DIR, 'data', 'training');
const OUT_FILE = path.join(OUT_DIR, 'agent_representation_pairs_v1.jsonl');
const SUMMARY_FILE = path.join(OUT_DIR, 'agent_representation_pairs_v1.summary.json');

function readJsonl(filePath) {
  if (!fs.existsSync(filePath)) return [];
  return fs
    .readFileSync(filePath, 'utf8')
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      try {
        return JSON.parse(line);
      } catch (error) {
        return null;
      }
    })
    .filter(Boolean);
}

function compactPoi(poi) {
  if (!poi) return null;
  return {
    poi_id: poi.id,
    name: poi.name,
    category: poi.category,
    district: poi.district,
    lat: poi.lat,
    lon: poi.lon,
    rating: poi.rating,
    review_count: poi.reviewCount,
    semantic_text: [poi.name, poi.category, poi.district, poi.text].filter(Boolean).join(' | ').slice(0, 1200),
    numeric_features: {
      rating: poi.rating,
      review_count: poi.reviewCount,
    },
  };
}

function pickNegativePois(sample, poiById, positiveIds) {
  const explicitNegatives = (sample.expected_output?.negative_pois || [])
    .map((item) => (typeof item === 'string' ? item : item.poi_id))
    .filter(Boolean)
    .filter((poiId) => !positiveIds.has(poiId))
    .map((poiId) => poiById.get(poiId))
    .filter(Boolean);
  if (explicitNegatives.length) {
    return explicitNegatives.slice(0, 10);
  }

  const expectedRoles = new Set(
    (sample.expected_output?.expected_itinerary || []).map((item) => String(item.expected_role || '').toLowerCase())
  );
  return (sample.candidate_pool || [])
    .filter((poiId) => !positiveIds.has(poiId))
    .map((poiId) => poiById.get(poiId))
    .filter(Boolean)
    .filter((poi) => {
      const category = String(poi.category || '').toLowerCase();
      if (!expectedRoles.size) return true;
      return !Array.from(expectedRoles).some((role) => category.includes(role));
    })
    .slice(0, 8);
}

function buildSyntheticPairRecords(samples, poiById) {
  const records = [];
  for (const sample of samples) {
    const positives = sample.expected_output?.expected_itinerary || [];
    const positiveIds = new Set(positives.map((item) => item.poi_id));
    const negatives = pickNegativePois(sample, poiById, positiveIds);

    for (const positive of positives) {
      const poi = poiById.get(positive.poi_id);
      if (!poi) continue;
      records.push({
        record_type: 'query_positive_poi',
        source: 'grounded_synthetic',
        sample_id: sample.sample_id,
        persona_id: sample.persona?.persona_id,
        query: sample.user_query,
        query_context: sample.query_context,
        label: 1,
        target_role: positive.expected_role,
        poi: compactPoi(poi),
        evidence: positive.reason_evidence || [],
      });
    }

    for (const negativePoi of negatives) {
      records.push({
        record_type: 'query_negative_poi',
        source: 'grounded_synthetic',
        sample_id: sample.sample_id,
        persona_id: sample.persona?.persona_id,
        query: sample.user_query,
        query_context: sample.query_context,
        label: 0,
        target_role: 'hard_negative_candidate',
        poi: compactPoi(negativePoi),
        evidence: ['hard_negative_grounded', 'category_or_role_mismatch'],
      });
    }

    for (const memoryItem of sample.persona?.memory || []) {
      const poi = poiById.get(memoryItem.poi_id);
      if (!poi) continue;
      records.push({
        record_type: 'persona_memory_positive',
        source: 'grounded_synthetic_memory',
        sample_id: sample.sample_id,
        persona_id: sample.persona?.persona_id,
        query: sample.persona?.reflection || sample.user_query,
        query_context: {
          segment: sample.persona?.segment,
          preferences: sample.persona?.preferences,
        },
        label: memoryItem.sentiment === 'negative' ? 0 : 1,
        target_role: 'long_term_preference',
        poi: compactPoi(poi),
        evidence: [memoryItem.evidence || 'persona_memory'],
      });
    }
  }
  return records;
}

function buildFeedbackRecords(events) {
  return events.map((event, index) => ({
    record_type: 'feedback_event',
    source: 'agent_feedback',
    feedback_index: index,
    timestamp: event.timestamp,
    role: event.role,
    event_type: event.eventType,
    query: event.query || '',
    label: ['like', 'add_to_itinerary', 'route_opened', 'booking_intent'].includes(event.eventType) ? 1 : null,
    payload: event.payload || {},
  }));
}

async function exportTrainingData() {
  console.log('[1/5] Loading POIs, synthetic samples, and feedback events');
  const pois = await loadPOIs();
  const poiById = new Map(pois.map((poi) => [poi.id, poi]));
  const syntheticSamples = readJsonl(SYNTHETIC_FILE);
  const feedbackEvents = readJsonl(FEEDBACK_FILE);
  console.log(`[2/5] Building representation pairs from ${syntheticSamples.length} synthetic samples`);
  const records = [
    ...buildSyntheticPairRecords(syntheticSamples, poiById),
    ...buildFeedbackRecords(feedbackEvents),
  ];

  console.log(`[3/5] Built ${records.length} records including ${feedbackEvents.length} feedback events`);
  fs.mkdirSync(OUT_DIR, { recursive: true });
  console.log(`[4/5] Writing JSONL training data to ${OUT_FILE}`);
  fs.writeFileSync(OUT_FILE, records.map((record) => JSON.stringify(record)).join('\n') + '\n', 'utf8');

  const summary = {
    created_at: new Date().toISOString(),
    output_file: OUT_FILE,
    poi_count: pois.length,
    synthetic_samples: syntheticSamples.length,
    feedback_events: feedbackEvents.length,
    record_count: records.length,
    by_record_type: records.reduce((acc, record) => {
      acc[record.record_type] = (acc[record.record_type] || 0) + 1;
      return acc;
    }, {}),
    next_training_target: 'Use this JSONL in poi_urban to fine-tune embedding/reranker/representation models.',
  };
  console.log(`[5/5] Writing summary to ${SUMMARY_FILE}`);
  fs.writeFileSync(SUMMARY_FILE, JSON.stringify(summary, null, 2), 'utf8');
  return summary;
}

if (require.main === module) {
  exportTrainingData()
    .then((summary) => console.log(JSON.stringify(summary, null, 2)))
    .catch((error) => {
      console.error(error);
      process.exit(1);
    });
}

module.exports = {
  exportTrainingData,
};
