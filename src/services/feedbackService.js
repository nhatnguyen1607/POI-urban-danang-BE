const { requireFirestoreDb } = require('../config/firebaseAdmin');

function cleanString(value, max = 1200) {
  return String(value || '').trim().slice(0, max);
}

function normalizeRole(role) {
  if (role === 'traveler') return 'customer';
  return ['customer', 'seller', 'admin'].includes(role) ? role : 'customer';
}

function normalizeEventType(eventType) {
  const aliases = {
    poi_useful: 'poi_useful',
    poi_not_fit: 'poi_not_fit',
    add_to_itinerary: 'add_to_itinerary',
    remove_from_itinerary: 'remove_from_itinerary',
    agent_run_business_insight: 'business_insight_run',
    itinerary_saved: 'itinerary_saved',
  };
  return aliases[eventType] || cleanString(eventType || 'unknown', 80);
}

function sanitizeFeedback(input = {}) {
  const payload = input.payload || {};
  return {
    userId: input.userId ? cleanString(input.userId, 160) : null,
    sessionId: cleanString(input.sessionId || input.userId || 'anonymous', 160),
    role: normalizeRole(input.role),
    eventType: normalizeEventType(input.eventType),
    query: cleanString(input.query, 1200),
    poiId: cleanString(input.poiId || payload.poiId || payload.poi?.id || payload.poi?.poiId, 220) || null,
    itineraryId: cleanString(input.itineraryId || payload.itineraryId, 220) || null,
    businessAnalysisId: cleanString(input.businessAnalysisId || payload.businessAnalysisId || payload.analysisId, 220) || null,
    payload,
    createdAt: new Date(),
  };
}

function increment(target, key, delta) {
  if (!key) return;
  target[key] = Number(target[key] || 0) + delta;
}

async function updateAgentMemoryFromEvent(db, event) {
  if (!event.userId) return;
  const ref = db.collection('agentMemories').doc(event.userId);
  const snap = await ref.get();
  const current = snap.exists ? snap.data() : {};
  const categoryAffinity = { ...(current.categoryAffinity || {}) };
  const categoryPenalty = { ...(current.categoryPenalty || {}) };
  const poiAffinity = { ...(current.poiAffinity || {}) };
  const poiPenalty = { ...(current.poiPenalty || {}) };
  const category = cleanString(event.payload?.category || event.payload?.poi?.category, 160);

  if (['poi_useful', 'add_to_itinerary'].includes(event.eventType)) {
    increment(poiAffinity, event.poiId, 1);
    increment(categoryAffinity, category, 1);
  }
  if (['poi_not_fit', 'remove_from_itinerary'].includes(event.eventType)) {
    increment(poiPenalty, event.poiId, 1);
    increment(categoryPenalty, category, 1);
  }

  await ref.set(
    {
      userId: event.userId,
      categoryAffinity,
      categoryPenalty,
      poiAffinity,
      poiPenalty,
      personaSummary: Object.keys(categoryAffinity).length ? `Ưu tiên ${Object.keys(categoryAffinity).slice(0, 5).join(', ')}.` : current.personaSummary || '',
      updatedAt: new Date(),
      version: 'v1',
    },
    { merge: true },
  );
}

async function recordFeedback(input) {
  const event = sanitizeFeedback(input);
  const db = requireFirestoreDb();
  const ref = db.collection('agentEvents').doc();
  await ref.set({
    eventId: ref.id,
    ...event,
  });
  await updateAgentMemoryFromEvent(db, event);
  return {
    saved: true,
    firestoreSynced: true,
    eventId: ref.id,
    event: { ...event, eventId: ref.id },
    learningUse: 'This event can be used later for reranking, preference memory, and supervised fine-tuning.',
  };
}

module.exports = {
  recordFeedback,
};
