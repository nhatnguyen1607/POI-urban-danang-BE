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
    poiPenalty: memory.poiPenalty || {},
    categoryPenalty: memory.categoryPenalty || {},
  };
}

function isPoiDisliked(poi, context = {}) {
  const { poiPenalty } = memoryFromContext(context);
  return Number(poiPenalty?.[poi.id] || 0) > 0;
}

function categoryPenaltyScore(poi, context = {}) {
  const { categoryPenalty } = memoryFromContext(context);
  const normalizedCategory = normalizeText(poi.category || '');
  return Object.entries(categoryPenalty).reduce((max, [category, value]) => {
    if (!normalizedCategory.includes(normalizeText(category))) return max;
    return Math.max(max, Number(value) || 0);
  }, 0);
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
  const memoryCategoryPenalty = categoryPenaltyScore(poi, context);
  const finalScore = clamp01(
    semantic * 0.34 +
      category * 0.26 +
      rating * 0.16 +
      distance * 0.14 +
      review * 0.10 -
      Math.min(memoryCategoryPenalty * 0.08, 0.28),
  );

  const warnings = [];
  if (distanceKm > 10) warnings.push('Địa điểm hơi xa so với tâm/vị trí hiện tại.');
  if (category < 0.3 && intent) warnings.push('Danh mục không trùng khớp mạnh với intent, cần kiểm tra lại.');
  if (!poi.rating) warnings.push('Thiếu rating rõ ràng.');

  return {
    poi,
    score: finalScore,
    signals: { semantic, category, rating, distance, review, distanceKm },
    warnings,
    intent,
  };
}

function toRecommendation(scored) {
  const { poi, score, signals, warnings, intent } = scored;
  const reasonParts = [];
  if (intent) reasonParts.push(`phù hợp intent ${intent.label}`);
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
