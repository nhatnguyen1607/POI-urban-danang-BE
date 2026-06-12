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

function scorePOI(poi, query, context = {}) {
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
  const finalScore = clamp01(
    semantic * 0.28 +
      category * 0.22 +
      preference * 0.18 +
      rating * 0.13 +
      distance * 0.11 +
      review * 0.08 -
      Math.min(memoryCategoryPenalty * 0.08 + explicitPenalty * 0.16, 0.34),
  );

  const warnings = [];
  if (distanceKm > 10) warnings.push('Địa điểm hơi xa so với tâm/vị trí hiện tại.');
  if (category < 0.3 && intent) warnings.push('Danh mục không trùng khớp mạnh với intent, cần kiểm tra lại.');
  if (!poi.rating) warnings.push('Thiếu rating rõ ràng.');

  return {
    poi,
    score: finalScore,
    signals: { semantic, category, preference, rating, distance, review, distanceKm, explicitPenalty },
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
  const pois = await loadPOIs();
  const scored = pois
    .filter((poi) => !isPoiDisliked(poi, context))
    .map((poi) => scorePOI(poi, query, context))
    .map((item) => applyReranker(item, query, context))
    .filter((item) => item.score > 0.08)
    .sort((a, b) => {
      const scoreDelta = b.score - a.score;
      if (scoreDelta !== 0) return scoreDelta;
      return (b.reranker?.delta || 0) - (a.reranker?.delta || 0);
    })
    .slice(0, limit)
    .map(toRecommendation);

  return {
    role: 'traveler',
    query,
    results: scored,
    warnings: scored.length ? [] : ['Chưa tìm thấy POI phù hợp, hãy thử mô tả rõ hơn.'],
  };
}

module.exports = {
  recommendPOIs,
  scorePOI,
  DANANG_CENTER,
};
