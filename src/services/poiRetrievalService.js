const { loadPOIs, normalizeText } = require('./poiDataService');
const { detectIntent, categoryMatchScore } = require('./intentService');
const {
  clamp01,
  haversineKm,
  keywordScore,
  ratingScore,
  distanceScore,
  reviewSignal,
} = require('./scoringUtils');
const { applyReranker } = require('./rerankerService');
const { runSemanticRetrieval } = require('./semanticModelService');

const DANANG_CENTER = { lat: 16.0544, lon: 108.2022 };

function buildActions(poi) {
  const mapsUrl = `https://www.google.com/maps/search/?api=1&query=${poi.lat},${poi.lon}`;
  return [
    { type: 'map', label: 'Mở bản đồ', url: mapsUrl },
    { type: 'route', label: 'Tính lộ trình', payload: { lat: poi.lat, lon: poi.lon } },
  ];
}

function memoryFromContext(context = {}) {
  const memory = context.agentMemory || {};
  return {
    persona: memory.persona || '',
    categoryAffinity: memory.categoryAffinity || {},
    poiPenalty: memory.poiPenalty || {},
    categoryPenalty: memory.categoryPenalty || {},
    explicitSignals: memory.explicitSignals || {},
  };
}

function isPoiDisliked(poi, context = {}) {
  const { poiPenalty } = memoryFromContext(context);
  return Number(poiPenalty?.[poi.id] || 0) > 0;
}

function categoryPenaltyScore(poi, context = {}) {
  const { categoryPenalty } = memoryFromContext(context);
  const normalizedCategory = normalizeText(`${poi.category || ''} ${poi.name || ''} ${poi.text || ''}`);
  return Object.entries(categoryPenalty).reduce((max, [category, value]) => {
    if (!normalizedCategory.includes(normalizeText(category))) return max;
    return Math.max(max, Number(value) || 0);
  }, 0);
}

const EXPLICIT_TAG_KEYWORDS = {
  beach_view: ['bien', 'beach', 'sea', 'ocean', 'ngam bien'],
  riverfront: ['song', 'river', 'riverfront', 'ven song', 'han'],
  hidden_gems: ['hidden', 'an minh', 'hem', 'local', 'yen tinh'],
  amusement_parks: ['khu vui choi', 'amusement', 'park', 'cong vien', 'game'],
  rooftop_city_view: ['rooftop', 'city view', 'view', 'tang thuong'],
  deep_work_study: ['hoc bai', 'study', 'work', 'cowork', 'yen tinh', 'cafe'],
  casual_socializing: ['social', 'gap go', 'ban be', 'cafe', 'restaurant'],
  weekend_chill: ['chill', 'weekend', 'thu gian', 'cafe', 'beach'],
  late_night_hangouts: ['dem', 'night', 'late', 'bar', 'nhau'],
  solo: ['solo', 'mot minh', 'quiet', 'yen tinh'],
  dating_romantic: ['hen ho', 'romantic', 'lang man', 'view', 'rooftop'],
  friends_gathering: ['ban be', 'group', 'nhom', 'nhau', 'quan an'],
  family_friendly: ['family', 'gia dinh', 'tre em', 'nha hang'],
  overcrowded_places: ['dong', 'crowded', 'hot', 'famous'],
  loud_music_noisy: ['karaoke', 'bar', 'pub', 'music', 'nhac', 'on ao'],
  no_parking_space: ['hem', 'small', 'nho', 'parking'],
  high_pricing: ['premium', 'luxury', 'cao cap', 'expensive'],
  student: ['hoc bai', 'study', 'sinh vien', 'cafe', 'budget'],
  remote_worker: ['work', 'cowork', 'wifi', 'yen tinh', 'cafe'],
  tourist: ['tourist', 'du lich', 'check in', 'beach', 'heritage'],
  local_explorer: ['local', 'hidden', 'an minh', 'quan an', 'hem'],
};

function keywordHitScore(poi, keys = []) {
  const haystack = normalizeText(`${poi.name || ''} ${poi.category || ''} ${poi.text || ''} ${poi.district || ''}`);
  const hits = keys.reduce((sum, key) => {
    const terms = EXPLICIT_TAG_KEYWORDS[key] || [key];
    return sum + (terms.some((term) => haystack.includes(normalizeText(term))) ? 1 : 0);
  }, 0);
  return clamp01(hits / Math.max(keys.length || 1, 1));
}

function explicitPreferenceScore(poi, context = {}) {
  const { persona, categoryAffinity, explicitSignals } = memoryFromContext(context);
  const likedKeys = Object.keys(categoryAffinity || {}).filter((key) => Number(categoryAffinity[key]) > 0);
  const explicit = explicitSignals || {};
  const taste = explicit.tasteProfile || {};
  const tagKeys = [
    ...(taste.sceneryVibes || []),
    ...(taste.activitiesPurposes || []),
    ...(taste.companionContexts || []),
    ...(explicit.likedTags || []),
    ...(explicit.likedCategories || []),
  ];
  const personaScore = persona ? keywordHitScore(poi, [persona]) : 0.35;
  const tagScore = keywordHitScore(poi, [...new Set([...likedKeys, ...tagKeys])]);
  return clamp01(personaScore * 0.34 + tagScore * 0.66);
}

function explicitNegativePenalty(poi, context = {}) {
  const { explicitSignals } = memoryFromContext(context);
  const filters = explicitSignals?.negativeFilters || explicitSignals?.dislikedTags || [];
  return keywordHitScore(poi, filters);
}

function scorePOI(poi, query, context = {}, semanticScore = null) {
  const normalizedQuery = normalizeText(query);
  const intent = detectIntent(query);
  const userLocation = context.location || DANANG_CENTER;
  const distanceKm = haversineKm(
    { lat: userLocation.lat, lon: userLocation.lon || userLocation.lng },
    { lat: poi.lat, lon: poi.lon },
  );
  const semantic = keywordScore(normalizedQuery, poi.normalized);
  const category = categoryMatchScore(poi, intent);
  const rating = ratingScore(poi.rating);
  const distance = distanceScore(distanceKm, context.maxDistanceKm || 14);
  const review = reviewSignal(poi.reviewCount);
  const preference = explicitPreferenceScore(poi, context);
  const memoryCategoryPenalty = categoryPenaltyScore(poi, context);
  const explicitPenalty = explicitNegativePenalty(poi, context);
  const hasModelSemantic = Number.isFinite(semanticScore);
  const baseScore = hasModelSemantic
    ? semantic * 0.20 +
      category * 0.20 +
      preference * 0.17 +
      rating * 0.12 +
      distance * 0.08 +
      review * 0.05 +
      semanticScore * 0.18
    : semantic * 0.28 +
      category * 0.22 +
      preference * 0.18 +
      rating * 0.13 +
      distance * 0.11 +
      review * 0.08;
  const finalScore = clamp01(
    baseScore - Math.min(memoryCategoryPenalty * 0.08 + explicitPenalty * 0.16, 0.34),
  );

  const warnings = [];
  if (distanceKm > 10) warnings.push('Địa điểm hơi xa so với tâm/vị trí hiện tại.');
  if (category < 0.3 && intent) warnings.push('Danh mục không trùng khớp mạnh với intent, cần kiểm tra lại.');
  if (!poi.rating) warnings.push('Thiếu rating rõ ràng.');

  return {
    poi,
    score: finalScore,
    signals: {
      semantic,
      modelSemantic: hasModelSemantic ? semanticScore : null,
      category,
      preference,
      rating,
      distance,
      review,
      distanceKm,
      explicitPenalty,
    },
    warnings,
    intent,
  };
}

function toRecommendation(scored) {
  const { poi, score, signals, warnings, intent } = scored;
  const reasonParts = [];
  if (intent) reasonParts.push(`phù hợp intent ${intent.label}`);
  if (signals.preference >= 0.45) reasonParts.push('khớp hồ sơ sở thích cá nhân');
  if (signals.semantic > 0) reasonParts.push('có từ khóa/review gần với nhu cầu');
  if (signals.rating >= 0.75) reasonParts.push('đánh giá tốt');
  if (signals.distance >= 0.7) reasonParts.push('không lệch tuyến quá xa');

  return {
    id: poi.id,
    type: 'poi',
    title: poi.name,
    name: poi.name,
    category: poi.category,
    district: poi.district,
    lat: poi.lat,
    lon: poi.lon,
    rating: poi.rating,
    price: poi.price,
    source: poi.source,
    score: Math.round(score * 100),
    scoreRaw: score,
    reason: reasonParts.length
      ? `Gợi ý vì ${reasonParts.join(', ')}.`
      : 'Gợi ý dựa trên độ tương đồng nội dung và vị trí.',
    desc: poi.text.slice(0, 180),
    signals,
    warnings,
    actions: buildActions(poi),
  };
}

async function recommendPOIs({ query, context = {}, limit = 8 }) {
  const semanticConfig = context.semanticModel || {};
  const semanticEnabled = semanticConfig.enabled === true;
  const pois = await loadPOIs();
  const candidateLimit = semanticConfig.candidateLimit || 200;
  const semanticCandidates = semanticEnabled
    ? pois
        .filter((poi) => !isPoiDisliked(poi, context))
        .map((poi) => scorePOI(poi, query, context))
        .map((item) => applyReranker(item, query, context))
        .sort((a, b) => b.score - a.score)
        .slice(0, candidateLimit)
        .map(({ poi }) => ({
          _agent_id: String(poi.id),
          RestaurantID: String(poi.id),
          'Restaurant Name': poi.name,
          Category: poi.category,
          District: poi.district,
          Lat: poi.lat,
          Lon: poi.lon,
          LLM_Input_Text: poi.text || `${poi.name}. ${poi.category}. ${poi.district}.`,
        }))
    : [];
  const semanticCandidateIds = new Set(semanticCandidates.map((candidate) => candidate._agent_id));
  const semanticTool = semanticEnabled
    ? await runSemanticRetrieval({
        query,
        version: semanticConfig.version || 'v4',
        topK: Math.min(semanticConfig.topK || candidateLimit, semanticCandidates.length),
        candidateLimit,
        timeoutMs: semanticConfig.timeoutMs || 45000,
        candidates: semanticCandidates,
      })
    : {
        enabled: false,
        available: false,
        version: semanticConfig.version || null,
        reason: 'Semantic model tool was not requested.',
        scores: new Map(),
      };
  const scored = pois
    .filter((poi) => !isPoiDisliked(poi, context))
    .filter((poi) => !semanticTool.available || semanticCandidateIds.has(String(poi.id)))
    .map((poi) => scorePOI(poi, query, context, semanticTool.scores.get(String(poi.id))))
    .map((item) => applyReranker(item, query, context))
    .filter((item) => item.score > 0.08)
    .sort((a, b) => {
      const scoreDelta = b.score - a.score;
      if (scoreDelta !== 0) return scoreDelta;
      return (b.reranker?.delta || 0) - (a.reranker?.delta || 0);
    })
    .filter((item, index, items) => items.findIndex((candidate) => candidate.poi.id === item.poi.id) === index)
    .slice(0, limit)
    .map(toRecommendation);

  return {
    role: 'traveler',
    query,
    results: scored,
    semanticTool: {
      enabled: semanticTool.enabled,
      available: semanticTool.available,
      version: semanticTool.version,
      resultCount: semanticTool.resultCount || 0,
      reason: semanticTool.reason,
      role: 'optional_perception_tool',
    },
    warnings: scored.length ? [] : ['Chưa tìm thấy POI phù hợp, hãy thử mô tả rõ hơn.'],
  };
}

module.exports = {
  recommendPOIs,
  scorePOI,
  DANANG_CENTER,
};
