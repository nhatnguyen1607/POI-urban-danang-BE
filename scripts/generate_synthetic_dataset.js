const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { loadPOIs, normalizeText } = require('../src/services/poiDataService');
const { detectIntents, categoryMatchScore } = require('../src/services/intentService');
const { haversineKm } = require('../src/services/scoringUtils');

const ROOT_DIR = path.resolve(__dirname, '..');
const OUT_DIR = path.join(ROOT_DIR, 'data', 'synthetic');
const OUT_FILE = path.join(OUT_DIR, 'urbanagent_synthetic_v1.jsonl');

const PERSONA_TEMPLATES = [
  {
    segment: 'student',
    budgetLevel: 'medium',
    mobility: 'motorbike',
    likedTags: ['quiet', 'study', 'wifi', 'moderate_price'],
    dislikedTags: ['too_crowded', 'hard_to_park'],
    query: 'Toi muon di an nhe roi ghe quan cafe yen tinh de hoc bai, di xe may.',
    constraints: ['co quan an', 'co cafe', 'yen tinh'],
  },
  {
    segment: 'tourist',
    budgetLevel: 'medium',
    mobility: 'car',
    likedTags: ['near_beach', 'seafood', 'check_in', 'high_rating'],
    dislikedTags: ['too_far', 'low_rating'],
    query: 'Tao lich trinh toi nay co hai san va cafe gan bien cho khach du lich.',
    constraints: ['hai san', 'cafe', 'gan bien'],
  },
  {
    segment: 'young_professional',
    budgetLevel: 'high',
    mobility: 'car',
    likedTags: ['dessert', 'premium', 'meeting', 'central'],
    dislikedTags: ['noisy', 'unclear_price'],
    query: 'Goi y noi gap ban co banh ngot, cafe dep va de di bang Grab.',
    constraints: ['banh ngot', 'cafe', 'de di chuyen'],
  },
];

function stableId(prefix, input) {
  return `${prefix}_${crypto.createHash('sha1').update(input).digest('hex').slice(0, 10)}`;
}

function pickByIntent(pois, intent, usedIds) {
  return pois
    .filter((poi) => !usedIds.has(poi.id))
    .map((poi) => ({ poi, match: categoryMatchScore(poi, intent) }))
    .filter((item) => item.match >= 0.7)
    .sort((a, b) => b.match - a.match || (b.poi.rating || 0) - (a.poi.rating || 0))[0]?.poi;
}

function buildSyntheticSample(template, pois, index) {
  const intents = detectIntents(template.query);
  const usedIds = new Set();
  const expected = [];

  intents.forEach((intent) => {
    const poi = pickByIntent(pois, intent, usedIds);
    if (!poi) return;
    usedIds.add(poi.id);
    expected.push({
      order: expected.length + 1,
      poi_id: poi.id,
      expected_role: intent.id,
      reason_evidence: ['category_match', 'rating_signal', 'local_csv_grounding'],
    });
  });

  const origin = { lat: 16.0544, lon: 108.2022 };
  const candidatePool = pois
    .filter((poi) => haversineKm(origin, poi) <= 12)
    .sort((a, b) => (b.rating || 0) - (a.rating || 0))
    .slice(index * 15, index * 15 + 20)
    .map((poi) => poi.id);

  return {
    sample_id: stableId('synth_dn', `${template.segment}_${index}_${template.query}`),
    version: '2026-06-urbanagent-v1',
    source: {
      generator: 'grounded_synthetic_pipeline_mvp',
      created_at: new Date().toISOString(),
      grounding: 'local_poi_csv_ids_only',
    },
    persona: {
      persona_id: stableId('persona', `${template.segment}_${template.likedTags.join('_')}`),
      segment: template.segment,
      budget_level: template.budgetLevel,
      mobility: template.mobility,
      preferences: {
        liked_tags: template.likedTags,
        disliked_tags: template.dislikedTags,
        preferred_categories: intents.map((intent) => intent.label),
      },
      memory: expected.map((item) => ({
        poi_id: item.poi_id,
        sentiment: 'positive',
        evidence: 'selected by grounded synthetic generator',
      })),
      reflection: `Persona ${template.segment} uu tien ${template.likedTags.join(', ')}.`,
    },
    query_context: {
      language: 'vi',
      city: 'Da Nang',
      origin,
      time_slot: 'evening',
      transport: template.mobility,
      weather: 'unknown',
      group_size: 2,
      constraints: template.constraints,
    },
    user_query: template.query,
    candidate_pool: Array.from(new Set([...candidatePool, ...expected.map((item) => item.poi_id)])),
    expected_output: {
      intent_labels: intents.map((intent) => intent.id),
      expected_itinerary: expected,
      acceptable_alternatives: {},
      negative_pois: [],
    },
    evaluation: {
      metrics: ['intent_coverage', 'poi_validity', 'category_precision', 'route_feasibility'],
      must_pass: {
        all_poi_ids_exist: true,
        covers_required_intents: expected.length === intents.length,
        no_hallucinated_place: true,
      },
    },
  };
}

function validateSample(sample, poiIds) {
  const ids = [
    ...sample.candidate_pool,
    ...sample.expected_output.expected_itinerary.map((item) => item.poi_id),
  ];
  const missing = ids.filter((id) => !poiIds.has(id));
  return {
    valid: missing.length === 0 && sample.expected_output.expected_itinerary.length > 0,
    missing,
  };
}

async function main() {
  const pois = await loadPOIs();
  const poiIds = new Set(pois.map((poi) => poi.id));
  const groundedPOIs = pois.filter((poi) => normalizeText(`${poi.name} ${poi.category} ${poi.text}`).length > 10);
  const samples = PERSONA_TEMPLATES.map((template, index) => buildSyntheticSample(template, groundedPOIs, index));
  const validation = samples.map((sample) => ({ sampleId: sample.sample_id, ...validateSample(sample, poiIds) }));
  const invalid = validation.filter((item) => !item.valid);
  if (invalid.length) {
    throw new Error(`Synthetic validation failed: ${JSON.stringify(invalid, null, 2)}`);
  }

  fs.mkdirSync(OUT_DIR, { recursive: true });
  fs.writeFileSync(OUT_FILE, `${samples.map((sample) => JSON.stringify(sample)).join('\n')}\n`, 'utf8');
  console.log(`Wrote ${samples.length} grounded synthetic samples to ${OUT_FILE}`);
}

if (require.main === module) {
  main().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}

module.exports = {
  buildSyntheticSample,
  validateSample,
};
