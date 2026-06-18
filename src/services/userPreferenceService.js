const { requireFirestoreDb } = require('../config/firebaseAdmin');

const MAX_PROFILE_EVENTS = 500;

function cleanKey(value, fallback = 'unknown') {
  return String(value || fallback)
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '') || fallback;
}

function increment(map, key, amount = 1) {
  if (!key) return;
  map[key] = Number(map[key] || 0) + amount;
}

function topEntries(map, limit = 12) {
  return Object.entries(map)
    .sort((a, b) => Number(b[1]) - Number(a[1]))
    .slice(0, limit)
    .reduce((acc, [key, value]) => ({ ...acc, [key]: Number(value) }), {});
}

function weekPart(dayOfWeek) {
  return ['saturday', 'sunday'].includes(cleanKey(dayOfWeek)) ? 'weekend' : 'weekday';
}

function dwellStyle(minutes) {
  if (minutes < 25) return 'quick_stop';
  if (minutes < 75) return 'standard_visit';
  return 'long_stay';
}

async function fetchUserDocs(db, collectionName, userId) {
  const snap = await db.collection(collectionName).where('userId', '==', userId).limit(MAX_PROFILE_EVENTS).get();
  return snap.docs.map((doc) => ({ id: doc.id, ...doc.data() }));
}

function buildPreferenceProfile({ userId, analytics, reviews }) {
  const categoryWeights = {};
  const moodWeights = {};
  const purposeWeights = {};
  const contextAffinities = {};
  const purposeByCategory = {};
  const moodByCategory = {};
  const transportWeights = {};
  const dwellSamples = [];
  let totalVisitEvents = 0;
  let totalRouteEvents = 0;

  reviews.forEach((review) => {
    const category = cleanKey(review.category || review.poiCategory || review.poiName);
    const ratingWeight = Math.max(Number(review.rating || 0), 1);
    const timeOfDay = cleanKey(review.context?.timeOfDay);
    const part = weekPart(review.context?.dayOfWeek);
    const purpose = cleanKey(review.visitPurpose, '');
    const mood = cleanKey(review.visitMood, '');

    increment(categoryWeights, category, ratingWeight);
    if (purpose) increment(purposeWeights, purpose, ratingWeight);
    if (mood) increment(moodWeights, mood, ratingWeight);

    const contextKey = `${timeOfDay}.${part}`;
    contextAffinities[contextKey] ||= { categories: {}, purposes: {}, moods: {} };
    increment(contextAffinities[contextKey].categories, category, ratingWeight);
    if (purpose) increment(contextAffinities[contextKey].purposes, purpose, ratingWeight);
    if (mood) increment(contextAffinities[contextKey].moods, mood, ratingWeight);

    if (purpose) {
      purposeByCategory[category] ||= {};
      increment(purposeByCategory[category], purpose, ratingWeight);
    }
    if (mood) {
      moodByCategory[category] ||= {};
      increment(moodByCategory[category], mood, ratingWeight);
    }
  });

  analytics.forEach((event) => {
    const eventType = event.eventType;
    const category = cleanKey(event.poiCategory || event.category || event.poiName);
    const timeOfDay = cleanKey(event.context?.timeOfDay);
    const part = weekPart(event.context?.dayOfWeek);
    const contextKey = `${timeOfDay}.${part}`;

    if (eventType === 'poi_visit') {
      totalVisitEvents += 1;
      const dwell = Number(event.dwellMinutes || 0);
      if (dwell > 0) dwellSamples.push(Math.min(dwell, 360));
      const weight = Math.max(Math.min(dwell / 30, 4), 0.75);
      increment(categoryWeights, category, weight);
      contextAffinities[contextKey] ||= { categories: {}, purposes: {}, moods: {} };
      increment(contextAffinities[contextKey].categories, category, weight);
    }

    if (eventType === 'route_segment') {
      totalRouteEvents += 1;
      increment(transportWeights, cleanKey(event.inferredTransport), 1);
    }
  });

  const avgDwellMinutes = dwellSamples.length
    ? dwellSamples.reduce((sum, value) => sum + value, 0) / dwellSamples.length
    : 45;

  Object.keys(contextAffinities).forEach((key) => {
    contextAffinities[key] = {
      categories: topEntries(contextAffinities[key].categories),
      purposes: topEntries(contextAffinities[key].purposes),
      moods: topEntries(contextAffinities[key].moods),
    };
  });

  Object.keys(purposeByCategory).forEach((key) => {
    purposeByCategory[key] = topEntries(purposeByCategory[key], 6);
  });
  Object.keys(moodByCategory).forEach((key) => {
    moodByCategory[key] = topEntries(moodByCategory[key], 6);
  });

  return {
    userId,
    version: 'v1',
    dwellPreference: {
      targetMinutes: Math.round(avgDwellMinutes),
      style: dwellStyle(avgDwellMinutes),
      sampleCount: dwellSamples.length,
    },
    categoryWeights: topEntries(categoryWeights, 20),
    moodWeights: topEntries(moodWeights, 12),
    purposeWeights: topEntries(purposeWeights, 12),
    contextAffinities,
    purposeByCategory,
    moodByCategory,
    transportWeights: topEntries(transportWeights, 8),
    sampleCounts: {
      reviews: reviews.length,
      analytics: analytics.length,
      visits: totalVisitEvents,
      routeSegments: totalRouteEvents,
    },
    updatedAt: new Date(),
  };
}

async function rebuildUserPreferences(userId) {
  if (!userId) {
    const error = new Error('Missing userId');
    error.status = 400;
    throw error;
  }
  const db = requireFirestoreDb();
  const [analytics, reviews] = await Promise.all([
    fetchUserDocs(db, 'user_analytics', userId),
    fetchUserDocs(db, 'reviews', userId),
  ]);
  const profile = buildPreferenceProfile({ userId, analytics, reviews });
  await db.collection('user_preferences').doc(userId).set(profile, { merge: true });
  return profile;
}

async function getUserPreferences(userId, { rebuildIfStale = true } = {}) {
  if (!userId) return null;
  const db = requireFirestoreDb();
  const ref = db.collection('user_preferences').doc(userId);
  const snap = await ref.get();
  const data = snap.exists ? snap.data() : null;
  const updatedMs = Number(data?.updatedAt?.toMillis?.() || data?.updatedAt?.getTime?.() || 0);
  const stale = !updatedMs || Date.now() - updatedMs > 6 * 60 * 60 * 1000;
  if ((!data || stale) && rebuildIfStale) return rebuildUserPreferences(userId);
  return data;
}

module.exports = {
  rebuildUserPreferences,
  getUserPreferences,
  buildPreferenceProfile,
};
