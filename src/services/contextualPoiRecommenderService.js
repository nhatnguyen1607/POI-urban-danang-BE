const { requireFirestoreDb } = require('../config/firebaseAdmin');
const { clamp01, distanceScore, haversineKm, ratingScore, reviewSignal } = require('./scoringUtils');
const { getUserPreferences } = require('./userPreferenceService');

const DEFAULT_LOCATION = { lat: 16.0544, lon: 108.2022 };

function cleanKey(value, fallback = 'unknown') {
  return String(value || fallback)
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '') || fallback;
}

function contextFromInput(input = {}) {
  const now = input.timestamp ? new Date(input.timestamp) : new Date();
  const hour = Number(input.hour ?? now.getHours());
  const timeOfDay = input.timeOfDay || (hour < 11 ? 'morning' : hour < 17 ? 'afternoon' : hour < 22 ? 'evening' : 'night');
  const dayOfWeek = input.dayOfWeek || now.toLocaleDateString('en-US', { weekday: 'long' }).toLowerCase();
  const weekPart = ['saturday', 'sunday'].includes(cleanKey(dayOfWeek)) ? 'weekend' : 'weekday';
  return {
    timeOfDay: cleanKey(timeOfDay),
    dayOfWeek: cleanKey(dayOfWeek),
    weekPart,
    mood: cleanKey(input.mood, ''),
    visitPurpose: cleanKey(input.visitPurpose, ''),
  };
}

function normalizePoi(doc) {
  const data = doc.data();
  return {
    poiId: data.poiId || doc.id,
    name: data.name || 'POI',
    category: data.category || '',
    normalizedCategory: cleanKey(data.category),
    address: data.location?.address || data.address || '',
    district: data.location?.district || data.district || '',
    lat: Number(data.location?.lat ?? data.lat),
    lon: Number(data.location?.lng ?? data.location?.lon ?? data.lon ?? data.lng),
    rating: Number(data.rating || 0),
    reviewCount: Number(data.reviewCount || 0),
    ratingSum: Number(data.ratingSum || 0),
    timesVisited: Number(data.timesVisited || 0),
    timesAddedToItinerary: Number(data.timesAddedToItinerary || 0),
    timesRouted: Number(data.timesRouted || 0),
    openingHours: data.openingHours || null,
    status: data.status || 'active',
    semanticText: data.semanticText || '',
  };
}

function isOpenForContext(poi, context) {
  if (!poi.openingHours) return true;
  // MVP-safe filter: when openingHours is structured later, this hook can enforce day/time windows.
  return true;
}

function popularityScore(poi) {
  const visit = clamp01(Math.log10(poi.timesVisited + 1) / 2.2);
  const added = clamp01(Math.log10(poi.timesAddedToItinerary + 1) / 2.2);
  const routed = clamp01(Math.log10(poi.timesRouted + 1) / 2.4);
  const review = reviewSignal(poi.reviewCount);
  const rating = ratingScore(poi.rating || (poi.reviewCount ? poi.ratingSum / poi.reviewCount : 0));
  return clamp01(visit * 0.28 + added * 0.24 + routed * 0.16 + review * 0.14 + rating * 0.18);
}

function mapScore(map, key) {
  if (!map || !key) return 0;
  const max = Math.max(...Object.values(map).map(Number), 1);
  return clamp01(Number(map[key] || 0) / max);
}

const EXPLICIT_TAG_KEYWORDS = {
  beach_view: ['bien', 'beach', 'sea', 'ngam bien'],
  riverfront: ['song', 'river', 'ven song', 'han'],
  hidden_gems: ['hidden', 'an minh', 'hem', 'local', 'yen tinh'],
  amusement_parks: ['khu vui choi', 'amusement', 'park', 'cong vien'],
  rooftop_city_view: ['rooftop', 'city view', 'view', 'tang thuong'],
  deep_work_study: ['hoc bai', 'study', 'work', 'cowork', 'yen tinh', 'cafe'],
  casual_socializing: ['gap go', 'social', 'ban be', 'cafe', 'restaurant'],
  weekend_chill: ['chill', 'weekend', 'thu gian', 'cafe', 'beach'],
  late_night_hangouts: ['dem', 'night', 'late', 'bar', 'nhau'],
  solo: ['solo', 'mot minh', 'quiet', 'yen tinh'],
  dating_romantic: ['hen ho', 'romantic', 'lang man', 'view'],
  friends_gathering: ['ban be', 'group', 'nhom', 'nhau', 'quan an'],
  family_friendly: ['family', 'gia dinh', 'tre em', 'nha hang'],
  student: ['hoc bai', 'study', 'sinh vien', 'cafe', 'budget'],
  remote_worker: ['work', 'cowork', 'wifi', 'yen tinh', 'cafe'],
  tourist: ['tourist', 'du lich', 'check in', 'beach'],
  local_explorer: ['local', 'hidden', 'an minh', 'quan an', 'hem'],
};

function explicitHitScore(poi, keys = []) {
  const haystack = cleanKey(`${poi.name} ${poi.category} ${poi.semanticText} ${poi.district}`);
  const hits = keys.reduce((sum, key) => {
    const terms = EXPLICIT_TAG_KEYWORDS[key] || [key];
    return sum + (terms.some((term) => haystack.includes(cleanKey(term))) ? 1 : 0);
  }, 0);
  return clamp01(hits / Math.max(keys.length || 1, 1));
}

function personalizationScore(poi, profile, context) {
  if (!profile) return 0.35;
  const category = cleanKey(poi.category);
  const contextKey = `${context.timeOfDay}.${context.weekPart}`;
  const contextAffinity = profile.contextAffinities?.[contextKey] || {};
  const categoryScore = mapScore(profile.categoryWeights, category);
  const contextCategory = mapScore(contextAffinity.categories, category);
  const moodScore = context.mood ? mapScore(profile.moodWeights, context.mood) : 0.35;
  const contextMood = context.mood ? mapScore(contextAffinity.moods, context.mood) : 0.35;
  const purposeScore = context.visitPurpose
    ? Math.max(
        mapScore(profile.purposeWeights, context.visitPurpose),
        mapScore(profile.purposeByCategory?.[category], context.visitPurpose),
      )
    : mapScore(profile.purposeByCategory?.[category], Object.keys(profile.purposeByCategory?.[category] || {})[0]);
  const dwellTarget = Number(profile.dwellPreference?.targetMinutes || 45);
  const categoryDwellFit = dwellTarget >= 75 && /cafe|coffee|study|cowork|restaurant|bar/i.test(`${poi.category} ${poi.name}`)
    ? 0.85
    : dwellTarget < 25 && /takeaway|bakery|tea|juice|fast/i.test(`${poi.category} ${poi.name}`)
      ? 0.8
      : 0.5;
  const explicit = profile.explicitSignals || {};
  const taste = explicit.tasteProfile || {};
  const explicitKeys = [
    profile.persona,
    ...(taste.sceneryVibes || []),
    ...(taste.activitiesPurposes || []),
    ...(taste.companionContexts || []),
    ...(explicit.likedTags || []),
    ...(explicit.likedCategories || []),
  ].filter(Boolean);
  const explicitScore = explicitHitScore(poi, explicitKeys);

  return clamp01(
    categoryScore * 0.20 +
      contextCategory * 0.19 +
      explicitScore * 0.22 +
      purposeScore * 0.15 +
      moodScore * 0.09 +
      contextMood * 0.06 +
      categoryDwellFit * 0.09,
  );
}

function recommendationReason({ poi, signals, context }) {
  const parts = [];
  if (signals.personalization >= 0.65) parts.push(`hợp thói quen ${context.timeOfDay}`);
  if (signals.popularity >= 0.55) parts.push('được nhiều người ghé/lưu vào lịch trình');
  if (signals.distance >= 0.7) parts.push('gần vị trí hiện tại');
  if (!parts.length) parts.push('cân bằng giữa vị trí, độ phổ biến và hồ sơ sở thích');
  return `Gợi ý vì ${parts.join(', ')}.`;
}

async function loadActivePois(limit = 400) {
  const db = requireFirestoreDb();
  const snap = await db.collection('pois').where('status', '==', 'active').limit(Math.min(Number(limit) || 400, 800)).get();
  return snap.docs.map(normalizePoi).filter((poi) => Number.isFinite(poi.lat) && Number.isFinite(poi.lon));
}

async function recommendContextualPOIs({ userId, currentLocation, currentContext = {}, limit = 8 }) {
  const location = {
    lat: Number(currentLocation?.lat) || DEFAULT_LOCATION.lat,
    lon: Number(currentLocation?.lon ?? currentLocation?.lng) || DEFAULT_LOCATION.lon,
  };
  const context = contextFromInput(currentContext);
  const [profile, pois] = await Promise.all([
    getUserPreferences(userId, { rebuildIfStale: true }).catch(() => null),
    loadActivePois(500),
  ]);

  const scored = pois
    .filter((poi) => isOpenForContext(poi, context))
    .map((poi) => {
      const distanceKm = haversineKm(location, { lat: poi.lat, lon: poi.lon });
      const popularity = popularityScore(poi);
      const personalization = personalizationScore(poi, profile, context);
      const distance = distanceScore(distanceKm, currentContext.maxDistanceKm || 12);
      const finalScore = clamp01(popularity * 0.30 + personalization * 0.42 + distance * 0.28);
      return {
        poi,
        score: finalScore,
        signals: { popularity, personalization, distance, distanceKm },
      };
    })
    .filter((item) => item.score > 0.08)
    .sort((a, b) => b.score - a.score)
    .slice(0, Math.min(Number(limit) || 8, 20));

  return {
    userId: userId || null,
    context,
    profileSummary: profile
      ? {
          dwellPreference: profile.dwellPreference,
          topCategories: Object.keys(profile.categoryWeights || {}).slice(0, 5),
          sampleCounts: profile.sampleCounts,
        }
      : null,
    results: scored.map(({ poi, score, signals }) => ({
      id: poi.poiId,
      poiId: poi.poiId,
      name: poi.name,
      title: poi.name,
      category: poi.category,
      address: poi.address,
      district: poi.district,
      lat: poi.lat,
      lon: poi.lon,
      rating: poi.rating,
      score: Math.round(score * 100),
      scoreRaw: score,
      reason: recommendationReason({ poi, signals, context }),
      signals,
    })),
  };
}

module.exports = {
  recommendContextualPOIs,
  contextFromInput,
  popularityScore,
  personalizationScore,
};
