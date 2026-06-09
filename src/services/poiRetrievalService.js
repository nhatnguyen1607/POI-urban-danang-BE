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
    { type: 'map', label: 'Mo ban do', url: mapsUrl },
    { type: 'route', label: 'Tinh lo trinh', payload: { lat: poi.lat, lon: poi.lon } },
  ];
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
  const finalScore = clamp01(
    semantic * 0.34 +
      category * 0.26 +
      rating * 0.16 +
      distance * 0.14 +
      review * 0.10,
  );

  const warnings = [];
  if (distanceKm > 10) warnings.push('Dia diem hoi xa so voi tam/vi tri hien tai.');
  if (category < 0.3 && intent) warnings.push('Danh muc khong trung khop manh voi intent, can kiem tra lai.');
  if (!poi.rating) warnings.push('Thieu rating ro rang.');

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
  if (intent) reasonParts.push(`phu hop intent ${intent.label}`);
  if (signals.semantic > 0) reasonParts.push('co tu khoa/review gan voi nhu cau');
  if (signals.rating >= 0.75) reasonParts.push('danh gia tot');
  if (signals.distance >= 0.7) reasonParts.push('khong lech tuyen qua xa');

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
      ? `Goi y vi ${reasonParts.join(', ')}.`
      : 'Goi y dua tren do tuong dong noi dung va vi tri.',
    desc: poi.text.slice(0, 180),
    signals,
    warnings,
    actions: buildActions(poi),
  };
}

async function recommendPOIs({ query, context = {}, limit = 8 }) {
  const pois = await loadPOIs();
  const scored = pois
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
    warnings: scored.length ? [] : ['Chua tim thay POI phu hop, hay thu mo ta ro hon.'],
  };
}

module.exports = {
  recommendPOIs,
  scorePOI,
  DANANG_CENTER,
};
