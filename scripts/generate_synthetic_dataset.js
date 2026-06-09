const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { loadPOIs, normalizeText } = require('../src/services/poiDataService');
const { detectIntents, categoryMatchScore, INTENTS } = require('../src/services/intentService');
const { haversineKm } = require('../src/services/scoringUtils');

const ROOT_DIR = path.resolve(__dirname, '..');
const OUT_DIR = path.join(ROOT_DIR, 'data', 'synthetic');
const OUT_FILE = path.join(OUT_DIR, 'urbanagent_synthetic_v1.jsonl');
const SUMMARY_FILE = path.join(OUT_DIR, 'urbanagent_synthetic_v1.summary.json');

const ORIGIN_PRESETS = [
  { label: 'Hai Chau center', lat: 16.0544, lon: 108.2022 },
  { label: 'My Khe beach', lat: 16.0617, lon: 108.2474 },
  { label: 'Da Nang University area', lat: 16.0731, lon: 108.1539 },
  { label: 'Son Tra evening area', lat: 16.0901, lon: 108.2452 },
];

const PERSONAS = [
  {
    segment: 'student',
    budgetLevel: 'medium',
    mobility: 'motorbike',
    likedTags: ['quiet', 'study', 'wifi', 'moderate_price'],
    dislikedTags: ['too_crowded', 'hard_to_park'],
    constraints: ['yen tinh', 'gia vua phai', 'co cho ngoi lau'],
  },
  {
    segment: 'tourist',
    budgetLevel: 'medium',
    mobility: 'car',
    likedTags: ['near_beach', 'seafood', 'check_in', 'high_rating'],
    dislikedTags: ['too_far', 'low_rating'],
    constraints: ['gan bien', 'de check-in', 'khong qua xa'],
  },
  {
    segment: 'young_professional',
    budgetLevel: 'high',
    mobility: 'car',
    likedTags: ['dessert', 'premium', 'meeting', 'central'],
    dislikedTags: ['noisy', 'unclear_price'],
    constraints: ['khong gian dep', 'hop gap ban', 'de di Grab'],
  },
  {
    segment: 'family',
    budgetLevel: 'medium',
    mobility: 'car',
    likedTags: ['family_friendly', 'clean', 'easy_route', 'safe'],
    dislikedTags: ['too_late', 'crowded_bar'],
    constraints: ['phu hop gia dinh', 'duong de di', 'co bua an'],
  },
  {
    segment: 'business_owner',
    budgetLevel: 'startup',
    mobility: 'motorbike',
    likedTags: ['demand_signal', 'complementary_poi', 'low_competition'],
    dislikedTags: ['direct_competition', 'bad_access'],
    constraints: ['co nhu cau an uong', 'doi thu vua phai', 'gan cum khach'],
  },
];

const SCENARIOS = [
  {
    id: 'study_cafe_food',
    intents: ['food', 'cafe'],
    timeSlot: 'afternoon',
    queryTemplates: [
      'Toi muon an nhe roi tim cafe yen tinh de hoc bai trong {duration} gio.',
      'Tao lich trinh cho sinh vien: co quan an vua tien va cafe ngoi lau, di bang {transport}.',
    ],
  },
  {
    id: 'tourist_seafood_cafe_beach',
    intents: ['seafood', 'travel', 'cafe'],
    timeSlot: 'evening',
    queryTemplates: [
      'Tao lich trinh toi nay co hai san, diem check-in va cafe gan bien trong {duration} gio.',
      'Khach du lich muon an hai san roi di choi va ghe cafe dep gan bien.',
    ],
  },
  {
    id: 'dessert_cafe_meeting',
    intents: ['cafe', 'food'],
    timeSlot: 'evening',
    queryTemplates: [
      'Goi y noi gap ban co banh ngot, cafe dep va de di bang {transport}.',
      'Toi can lich trinh nhe co dessert va cafe khong gian dep trong {duration} gio.',
    ],
  },
  {
    id: 'pub_late_food',
    intents: ['food', 'pub'],
    timeSlot: 'night',
    queryTemplates: [
      'Nhom ban muon an toi roi ghe quan nhau vui nhung duong di khong qua kho.',
      'Lap lich trinh buoi dem co quan an va cho uong bia phu hop nhom ban.',
    ],
  },
  {
    id: 'family_food_travel',
    intents: ['food', 'travel'],
    timeSlot: 'morning',
    queryTemplates: [
      'Gia dinh toi muon di an va tham quan nhe nha, uu tien duong de di.',
      'Tao lich trinh {duration} gio cho gia dinh co bua an va diem di choi an toan.',
    ],
  },
  {
    id: 'business_location_cafe',
    intents: ['cafe', 'food'],
    timeSlot: 'business_analysis',
    queryTemplates: [
      'Mo tiem banh ngot va cafe gan khu sinh vien, can khu co nhu cau nhung doi thu vua phai.',
      'Phan tich vi tri mo cafe dessert o Da Nang, uu tien POI bo tro va de tiep can.',
    ],
  },
  {
    id: 'business_location_seafood',
    intents: ['seafood', 'travel'],
    timeSlot: 'business_analysis',
    queryTemplates: [
      'Mo nha hang hai san gan cum du lich, can danh gia demand proxy va canh tranh.',
      'Tim khu vuc phu hop cho quan hai san phuc vu khach du lich Da Nang.',
    ],
  },
];

const DURATIONS = [120, 180, 240, 360];

function stableId(prefix, input) {
  return `${prefix}_${crypto.createHash('sha1').update(input).digest('hex').slice(0, 12)}`;
}

function seededHash(input) {
  return Number.parseInt(crypto.createHash('sha1').update(input).digest('hex').slice(0, 8), 16);
}

function deterministicPick(items, seed, count = 1) {
  return [...items]
    .map((item, index) => ({ item, rank: seededHash(`${seed}_${index}_${JSON.stringify(item).slice(0, 80)}`) }))
    .sort((a, b) => a.rank - b.rank)
    .slice(0, count)
    .map((entry) => entry.item);
}

function intentById(intentId) {
  return INTENTS.find((intent) => intent.id === intentId);
}

function pickByIntent(pois, intent, usedIds, origin, seed) {
  const candidates = pois
    .filter((poi) => !usedIds.has(poi.id))
    .map((poi) => ({
      poi,
      match: categoryMatchScore(poi, intent),
      distance: haversineKm(origin, poi),
      rating: poi.rating || 0,
      reviews: poi.reviewCount || 0,
    }))
    .filter((item) => item.match >= 0.7 && item.distance <= 14)
    .sort((a, b) => {
      const scoreA = a.match * 5 + a.rating * 0.7 + Math.log1p(a.reviews) * 0.1 - a.distance * 0.08;
      const scoreB = b.match * 5 + b.rating * 0.7 + Math.log1p(b.reviews) * 0.1 - b.distance * 0.08;
      return scoreB - scoreA;
    });

  const top = candidates.slice(0, 18).map((item) => item.poi);
  return deterministicPick(top, seed, 1)[0];
}

function pickHardNegatives(pois, expectedIntents, usedIds, origin, seed, count = 8) {
  const expected = expectedIntents.map(intentById).filter(Boolean);
  const candidates = pois
    .filter((poi) => !usedIds.has(poi.id))
    .map((poi) => {
      const bestExpectedMatch = Math.max(...expected.map((intent) => categoryMatchScore(poi, intent)), 0);
      const textLen = normalizeText(`${poi.name} ${poi.category} ${poi.text}`).length;
      return {
        poi,
        mismatch: 1 - bestExpectedMatch,
        distance: haversineKm(origin, poi),
        quality: (poi.rating || 0) + Math.log1p(poi.reviewCount || 0) * 0.15 + Math.min(textLen / 1200, 1),
      };
    })
    .filter((item) => item.mismatch >= 0.28 && item.distance <= 14)
    .sort((a, b) => b.quality - a.quality || a.distance - b.distance)
    .slice(0, 80)
    .map((item) => item.poi);

  return deterministicPick(candidates, seed, count);
}

function renderQuery(template, durationMinutes, transport) {
  return template
    .replace('{duration}', `${Math.round(durationMinutes / 60)}`)
    .replace('{transport}', transport === 'car' ? 'oto/Grab' : 'xe may');
}

function buildSyntheticSample({ persona, scenario, pois, index }) {
  const origin = ORIGIN_PRESETS[index % ORIGIN_PRESETS.length];
  const durationMinutes = DURATIONS[index % DURATIONS.length];
  const queryTemplate = deterministicPick(scenario.queryTemplates, `${persona.segment}_${scenario.id}_${index}`, 1)[0];
  const query = renderQuery(queryTemplate, durationMinutes, persona.mobility);
  const detected = detectIntents(query);
  const intents = scenario.intents
    .map(intentById)
    .filter(Boolean)
    .map((intent) => detected.find((detectedIntent) => detectedIntent.id === intent.id) || intent);

  const usedIds = new Set();
  const expected = [];
  intents.forEach((intent) => {
    const poi = pickByIntent(pois, intent, usedIds, origin, `${persona.segment}_${scenario.id}_${intent.id}_${index}`);
    if (!poi) return;
    usedIds.add(poi.id);
    expected.push({
      order: expected.length + 1,
      poi_id: poi.id,
      expected_role: intent.id,
      reason_evidence: ['category_match', 'rating_signal', 'local_csv_grounding'],
    });
  });

  const hardNegatives = pickHardNegatives(
    pois,
    scenario.intents,
    usedIds,
    origin,
    `${persona.segment}_${scenario.id}_neg_${index}`,
    10,
  );

  const nearbyPool = pois
    .filter((poi) => haversineKm(origin, poi) <= 12 && !usedIds.has(poi.id))
    .sort((a, b) => (b.rating || 0) - (a.rating || 0) || (b.reviewCount || 0) - (a.reviewCount || 0))
    .slice(index % 20, index % 20 + 18)
    .map((poi) => poi.id);

  const sampleId = stableId('synth_dn', `${persona.segment}_${scenario.id}_${index}_${query}`);
  const personaId = stableId('persona', `${persona.segment}_${persona.likedTags.join('_')}`);
  const negativeIds = hardNegatives.map((poi) => poi.id);

  return {
    sample_id: sampleId,
    version: '2026-06-urbanagent-v2-grounded',
    source: {
      generator: 'grounded_synthetic_pipeline_expanded',
      created_at: new Date().toISOString(),
      grounding: 'local_poi_csv_ids_only',
      scenario_id: scenario.id,
    },
    persona: {
      persona_id: personaId,
      segment: persona.segment,
      budget_level: persona.budgetLevel,
      mobility: persona.mobility,
      preferences: {
        liked_tags: persona.likedTags,
        disliked_tags: persona.dislikedTags,
        preferred_categories: intents.map((intent) => intent.label),
      },
      memory: expected.map((item) => ({
        poi_id: item.poi_id,
        sentiment: 'positive',
        evidence: 'selected by grounded synthetic generator',
      })),
      reflection: `Persona ${persona.segment} uu tien ${persona.likedTags.join(', ')} va tranh ${persona.dislikedTags.join(', ')}.`,
    },
    query_context: {
      language: 'vi',
      city: 'Da Nang',
      origin: { lat: origin.lat, lon: origin.lon, label: origin.label },
      time_slot: scenario.timeSlot,
      transport: persona.mobility,
      weather: 'unknown',
      group_size: persona.segment === 'family' ? 4 : 2,
      duration_minutes: durationMinutes,
      constraints: persona.constraints,
    },
    user_query: query,
    candidate_pool: Array.from(new Set([...nearbyPool, ...expected.map((item) => item.poi_id), ...negativeIds])),
    expected_output: {
      intent_labels: intents.map((intent) => intent.id),
      expected_itinerary: expected,
      acceptable_alternatives: {},
      negative_pois: negativeIds.map((poiId) => ({
        poi_id: poiId,
        reason: 'hard_negative_same_city_wrong_intent_or_category',
      })),
    },
    evaluation: {
      metrics: ['intent_coverage', 'poi_validity', 'category_precision', 'route_feasibility', 'hard_negative_rejection'],
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
    ...sample.expected_output.negative_pois.map((item) => item.poi_id),
  ];
  const missing = ids.filter((id) => !poiIds.has(id));
  return {
    valid: missing.length === 0 && sample.expected_output.expected_itinerary.length > 0,
    missing,
  };
}

function parseArgs() {
  const args = process.argv.slice(2);
  const countArg = args.find((arg) => arg.startsWith('--count='));
  return {
    count: countArg ? Number.parseInt(countArg.split('=')[1], 10) : 96,
  };
}

async function main() {
  const { count } = parseArgs();
  console.log(`[1/5] Loading local POI CSV data for grounded generation`);
  const pois = await loadPOIs();
  const poiIds = new Set(pois.map((poi) => poi.id));
  const groundedPOIs = pois.filter((poi) => normalizeText(`${poi.name} ${poi.category} ${poi.text}`).length > 10);
  console.log(`[2/5] Loaded ${pois.length} POIs, usable grounded POIs: ${groundedPOIs.length}`);

  const combos = [];
  PERSONAS.forEach((persona) => {
    SCENARIOS.forEach((scenario) => {
      combos.push({ persona, scenario });
    });
  });

  const samples = [];
  let index = 0;
  console.log(`[3/5] Generating ${count} synthetic samples from persona x scenario combinations`);
  while (samples.length < count) {
    const combo = combos[index % combos.length];
    const sample = buildSyntheticSample({ ...combo, pois: groundedPOIs, index });
    const validation = validateSample(sample, poiIds);
    if (validation.valid) samples.push(sample);
    index += 1;
    if (index > count * 5) break;
  }

  console.log(`[4/5] Validating all generated POI ids against local CSV`);
  const validation = samples.map((sample) => ({ sampleId: sample.sample_id, ...validateSample(sample, poiIds) }));
  const invalid = validation.filter((item) => !item.valid);
  if (invalid.length) {
    throw new Error(`Synthetic validation failed: ${JSON.stringify(invalid, null, 2)}`);
  }

  const summary = {
    created_at: new Date().toISOString(),
    output_file: OUT_FILE,
    sample_count: samples.length,
    persona_count: new Set(samples.map((sample) => sample.persona.persona_id)).size,
    scenario_count: new Set(samples.map((sample) => sample.source.scenario_id)).size,
    expected_poi_count: samples.reduce((sum, sample) => sum + sample.expected_output.expected_itinerary.length, 0),
    hard_negative_count: samples.reduce((sum, sample) => sum + sample.expected_output.negative_pois.length, 0),
    grounding: 'Every expected/negative/candidate POI id is validated against local CSV.',
  };

  console.log(`[5/5] Writing synthetic dataset and summary to ${OUT_DIR}`);
  fs.mkdirSync(OUT_DIR, { recursive: true });
  fs.writeFileSync(OUT_FILE, `${samples.map((sample) => JSON.stringify(sample)).join('\n')}\n`, 'utf8');
  fs.writeFileSync(SUMMARY_FILE, JSON.stringify(summary, null, 2), 'utf8');
  console.log(JSON.stringify(summary, null, 2));
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
