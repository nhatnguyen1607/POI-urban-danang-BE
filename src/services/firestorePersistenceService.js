const { requireFirestoreDb } = require('../config/firebaseAdmin');

function now() {
  return new Date();
}

function cleanString(value, max = 1200) {
  return String(value || '').trim().slice(0, max);
}

function cleanArray(value, maxItems = 40) {
  if (!Array.isArray(value)) return [];
  return value.map((item) => cleanString(item, 160)).filter(Boolean).slice(0, maxItems);
}

function numberOrZero(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : 0;
}

function numberOrNull(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function normalizeName(value) {
  return cleanString(value, 300)
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase();
}

function validRole(role) {
  return ['admin', 'customer', 'seller'].includes(role) ? role : 'customer';
}

function validLanguage(language) {
  return language === 'en' ? 'en' : 'vi';
}

async function ensureUserDocument(input) {
  const db = requireFirestoreDb();
  const uid = cleanString(input.uid, 160);
  if (!uid) {
    const error = new Error('Missing uid');
    error.status = 400;
    throw error;
  }
  const ref = db.collection('users').doc(uid);
  const snap = await ref.get();
  const existing = snap.exists ? snap.data() : null;
  const doc = {
    uid,
    email: cleanString(input.email, 320),
    displayName: cleanString(input.displayName || input.name || input.email, 320),
    ...(input.phone ? { phone: cleanString(input.phone, 80) } : {}),
    ...(input.photoURL || input.picture ? { photoURL: cleanString(input.photoURL || input.picture, 1200) } : {}),
    role: validRole(input.role || existing?.role),
    status: existing?.status || 'active',
    language: validLanguage(input.language || existing?.language),
    updatedAt: now(),
    lastLoginAt: now(),
  };
  if (!snap.exists) doc.createdAt = now();
  await ref.set(doc, { merge: true });
  return { saved: true, created: !snap.exists, user: { ...(existing || {}), ...doc } };
}

async function updateUserRole({ uid, role, language }) {
  return ensureUserDocument({ uid, role, language });
}

async function listUsers({ limit = 50, role, status } = {}) {
  const db = requireFirestoreDb();
  let query = db.collection('users');
  if (role) query = query.where('role', '==', role);
  if (status) query = query.where('status', '==', status);
  const snap = await query.limit(Math.min(Number(limit) || 50, 200)).get();
  return snap.docs.map((doc) => ({ uid: doc.id, ...doc.data() }));
}

async function updateUserStatus({ uid, status }) {
  const db = requireFirestoreDb();
  const normalized = ['active', 'pending', 'banned'].includes(status) ? status : 'active';
  await db.collection('users').doc(uid).set({ status: normalized, updatedAt: now() }, { merge: true });
  const snap = await db.collection('users').doc(uid).get();
  return { saved: true, user: { uid, ...snap.data() } };
}

function normalizeCustomerPreferences(input = {}) {
  const budgetLevel = ['low', 'medium', 'high'].includes(input.budgetLevel) ? input.budgetLevel : 'medium';
  const mobilityValues = Array.isArray(input.mobility) ? input.mobility : input.mobility ? [input.mobility] : [];
  const mobility = mobilityValues.filter((item) => ['walking', 'motorbike', 'car', 'grab'].includes(item));
  return {
    likedCategories: cleanArray(input.likedCategories),
    dislikedCategories: cleanArray(input.dislikedCategories),
    likedTags: [],
    dislikedTags: cleanArray(input.dislikedTags),
    budgetLevel,
    mobility: mobility.length ? Array.from(new Set(mobility)) : ['motorbike'],
    preferredLanguage: validLanguage(input.preferredLanguage),
  };
}

function buildAgentMemoryFromPreferences(userId, preferences = {}) {
  const categoryAffinity = {};
  const categoryPenalty = {};
  cleanArray(preferences.likedCategories).forEach((category) => {
    categoryAffinity[category] = Math.max(categoryAffinity[category] || 0, 1);
  });
  cleanArray(preferences.dislikedCategories).forEach((category) => {
    categoryPenalty[category] = Math.max(categoryPenalty[category] || 0, 1);
  });
  return {
    userId,
    categoryAffinity,
    categoryPenalty,
    poiAffinity: {},
    poiPenalty: {},
    personaSummary: Object.keys(categoryAffinity).length ? `Ưu tiên ${Object.keys(categoryAffinity).join(', ')}.` : '',
    updatedAt: now(),
    version: 'v1',
  };
}

async function getCustomerProfile(userId) {
  const db = requireFirestoreDb();
  const snap = await db.collection('customerProfiles').doc(userId).get();
  return snap.exists ? snap.data() : null;
}

async function saveCustomerProfile(input) {
  const db = requireFirestoreDb();
  const userId = cleanString(input.userId, 160);
  const preferences = normalizeCustomerPreferences(input.preferences);
  const doc = {
    userId,
    preferences,
    defaultLocation: input.defaultLocation || { lat: 16.0544, lng: 108.2022, label: 'Đà Nẵng' },
    ...(input.agentMemorySummary ? { agentMemorySummary: cleanString(input.agentMemorySummary, 2000) } : {}),
    updatedAt: now(),
  };
  await db.collection('customerProfiles').doc(userId).set(doc, { merge: true });
  await db.collection('agentMemories').doc(userId).set(buildAgentMemoryFromPreferences(userId, preferences), { merge: true });
  return { saved: true, firestoreSynced: true, profile: doc };
}

function normalizePoi(input = {}) {
  const poiId = cleanString(input.poiId || input.id || input.place_id || input.RestaurantID, 220) || `poi_${Date.now()}`;
  const name = cleanString(input.name || input.title || input['Restaurant Name'], 300) || 'Unnamed POI';
  const category = cleanString(input.category || input.Category || 'Khác', 160);
  const lat = numberOrNull(input.location?.lat ?? input.lat ?? input.Lat);
  const lng = numberOrNull(input.location?.lng ?? input.location?.lon ?? input.lng ?? input.lon ?? input.Lon);
  const address = cleanString(input.location?.address || input.address || input.Address || '', 600);
  const district = cleanString(input.location?.district || input.district || input.District || address || 'Đà Nẵng', 200);
  const semanticText = cleanString(
    input.semanticText ||
      input.text ||
      input.LLM_Input_Text ||
      input.Aggregated_Reviews ||
      `${name}. ${category}. ${district}. ${address}`,
    6000,
  );
  return {
    poiId,
    source: ['foody', 'google_maps', 'manual', 'seller'].includes(input.source) ? input.source : 'manual',
    name,
    normalizedName: normalizeName(name),
    category,
    tags: cleanArray(input.tags || [category], 30),
    location: {
      lat: lat ?? 16.0544,
      lng: lng ?? 108.2022,
      geohash: cleanString(input.location?.geohash || input.geohash || '', 120),
      district,
      address,
    },
    rating: numberOrZero(input.rating || input['Overall Rating']),
    reviewCount: numberOrZero(input.reviewCount || input.Total_Reviews_Scraped || input['User Rating Count']),
    ...(input.priceLevel !== undefined ? { priceLevel: numberOrZero(input.priceLevel) } : {}),
    ...(input.description ? { description: cleanString(input.description, 2000) } : {}),
    semanticText,
    images: cleanArray(input.images, 20),
    ...(input.openingHours ? { openingHours: input.openingHours } : {}),
    ...(input.ownerId ? { ownerId: cleanString(input.ownerId, 160) } : {}),
    verified: Boolean(input.verified),
    status: ['active', 'pending', 'hidden'].includes(input.status) ? input.status : 'active',
    createdAt: input.createdAt || now(),
    updatedAt: now(),
  };
}

async function upsertPoi(input) {
  const db = requireFirestoreDb();
  const poi = normalizePoi(input);
  const ref = db.collection('pois').doc(poi.poiId);
  const snap = await ref.get();
  await ref.set({ ...poi, createdAt: snap.exists ? snap.data().createdAt || poi.createdAt : poi.createdAt }, { merge: true });
  return { saved: true, firestoreSynced: true, poiId: poi.poiId, poi };
}

async function listPois({ limit = 50, status = 'active' } = {}) {
  const db = requireFirestoreDb();
  let query = db.collection('pois');
  if (status) query = query.where('status', '==', status);
  const snap = await query.limit(Math.min(Number(limit) || 50, 200)).get();
  return snap.docs.map((doc) => ({ poiId: doc.id, ...doc.data() }));
}

async function updatePoiStatus({ poiId, status, verified }) {
  const db = requireFirestoreDb();
  const normalized = ['active', 'pending', 'hidden'].includes(status) ? status : 'pending';
  await db.collection('pois').doc(poiId).set(
    {
      status: normalized,
      ...(verified !== undefined ? { verified: Boolean(verified) } : {}),
      updatedAt: now(),
    },
    { merge: true },
  );
  const snap = await db.collection('pois').doc(poiId).get();
  return { saved: true, poi: { poiId, ...snap.data() } };
}

function normalizeStop(item, index) {
  const poi = item.poi || item;
  return {
    poiId: String(poi.id || poi.poiId || `poi_${index + 1}`),
    order: Number(item.order || index + 1),
    stayMinutes: Number(item.suggestedStayMinutes || item.stayMinutes || 45),
    reason: String(item.reason || poi.reason || ''),
    addedBy: item.addedBy || 'agent',
    poiSnapshot: {
      id: poi.id || poi.poiId || null,
      title: poi.title || poi.name || '',
      name: poi.name || poi.title || '',
      category: poi.category || '',
      district: poi.district || '',
      lat: Number(poi.lat) || null,
      lon: Number(poi.lon || poi.lng) || null,
      rating: poi.rating || null,
      score: poi.score || null,
    },
  };
}

function buildItineraryDocument(input) {
  const stops = Array.isArray(input.itinerary)
    ? input.itinerary.map(normalizeStop)
    : Array.isArray(input.stops)
      ? input.stops.map(normalizeStop)
      : [];
  const totalDurationMinutes =
    Number(input.routeSummary?.totalDurationMinutes) ||
    stops.reduce((sum, stop) => sum + (Number(stop.travelFromPrevious?.estimatedMinutes) || 0), 0);
  return {
    userId: input.userId,
    query: cleanString(input.query, 1200),
    durationMinutes: Number(input.durationMinutes || input.tripDurationMinutes || 0),
    transport: input.transport || 'motorbike',
    origin: input.origin || input.context?.location || null,
    stops,
    routeSummary: {
      totalDistanceKm: Number(input.routeSummary?.totalDistanceKm || 0),
      totalDurationMinutes,
      warnings: Array.isArray(input.routeSummary?.warnings) ? input.routeSummary.warnings : [],
    },
    status: input.status || 'saved',
    createdAt: now(),
    updatedAt: now(),
  };
}

async function saveItinerary(input) {
  const db = requireFirestoreDb();
  const doc = buildItineraryDocument(input);
  const ref = await db.collection('itineraries').add(doc);
  await ref.update({ itineraryId: ref.id });
  return { saved: true, firestoreSynced: true, itineraryId: ref.id, itinerary: { ...doc, itineraryId: ref.id } };
}

async function listItineraries(userId) {
  const db = requireFirestoreDb();
  const snap = await db.collection('itineraries').where('userId', '==', userId).limit(20).get();
  return snap.docs
    .map((doc) => ({ itineraryId: doc.id, ...doc.data() }))
    .sort((a, b) => Number(b.updatedAt?.toMillis?.() || 0) - Number(a.updatedAt?.toMillis?.() || 0));
}

async function getAgentMemory(userId) {
  const db = requireFirestoreDb();
  const snap = await db.collection('agentMemories').doc(userId).get();
  return snap.exists ? snap.data() : null;
}

async function saveAgentMemory(input) {
  const db = requireFirestoreDb();
  const userId = cleanString(input.userId, 160);
  const doc = {
    userId,
    categoryAffinity: input.categoryAffinity || {},
    categoryPenalty: input.categoryPenalty || {},
    poiAffinity: input.poiAffinity || {},
    poiPenalty: input.poiPenalty || {},
    personaSummary: cleanString(input.personaSummary || '', 2000),
    updatedAt: now(),
    version: cleanString(input.version || 'v1', 40),
  };
  await db.collection('agentMemories').doc(userId).set(doc, { merge: true });
  return { saved: true, firestoreSynced: true, memory: doc };
}

function normalizeCandidateArea(area, index) {
  return {
    areaId: area.id || `area_${index + 1}`,
    name: area.name || `Khu vực ${index + 1}`,
    center: { lat: Number(area.lat || area.center?.lat), lng: Number(area.lon || area.lng || area.center?.lng) },
    scores: {
      opportunity: Number(area.score || area.scores?.opportunity || 0),
      demandProxy: Number(area.signals?.demandProxy || area.scores?.demandProxy || 0),
      competition: Number(area.signals?.competitionPenalty || area.scores?.competition || 0),
      complementary: Number(area.signals?.complementary || area.scores?.complementary || 0),
      accessibility: Number(area.signals?.accessibility || area.scores?.accessibility || 0),
      conceptFit: Number(area.signals?.conceptFit || area.scores?.conceptFit || 0),
    },
    evidencePoiIds: [
      ...(area.samplePOIs || []).map((poi) => poi.id).filter(Boolean),
      ...(area.llmInsight?.used_evidence_ids || []),
    ],
    warnings: Array.isArray(area.warnings) ? area.warnings : [],
    recommendation: area.llmInsight?.summary || area.reason || '',
    rawEvidence: area.evidence || null,
  };
}

async function saveBusinessAnalysis(input) {
  const db = requireFirestoreDb();
  const doc = {
    sellerId: input.sellerId,
    concept: cleanString(input.concept, 1200),
    targetCategory: input.targetCategory || '',
    candidateAreas: (input.areas || input.candidateAreas || []).map(normalizeCandidateArea),
    rawResult: input.rawResult || null,
    createdAt: now(),
  };
  const ref = await db.collection('businessAnalyses').add(doc);
  await ref.update({ analysisId: ref.id });
  return { saved: true, firestoreSynced: true, analysisId: ref.id, analysis: { ...doc, analysisId: ref.id } };
}

async function listBusinessAnalyses(sellerId) {
  const db = requireFirestoreDb();
  const snap = await db.collection('businessAnalyses').where('sellerId', '==', sellerId).limit(30).get();
  return snap.docs
    .map((doc) => ({ analysisId: doc.id, ...doc.data() }))
    .sort((a, b) => Number(b.createdAt?.toMillis?.() || 0) - Number(a.createdAt?.toMillis?.() || 0));
}

async function saveSellerConcept(input) {
  const db = requireFirestoreDb();
  const sellerId = cleanString(input.sellerId, 160);
  const concept = cleanString(input.query || input.concept, 1200);
  if (!sellerId || !concept) {
    const error = new Error('Missing seller concept fields');
    error.status = 400;
    throw error;
  }
  const doc = {
    sellerId,
    query: concept,
    concept,
    suggestions: Array.isArray(input.suggestions) ? input.suggestions.slice(0, 20) : [],
    analysis: input.analysis || input.rawResult || null,
    createdAt: now(),
    updatedAt: now(),
  };
  const ref = await db.collection('sellerProfiles').doc(sellerId).collection('concepts').add(doc);
  await ref.update({ conceptId: ref.id });
  const profileRef = db.collection('sellerProfiles').doc(sellerId);
  const profileSnap = await profileRef.get();
  const currentConcepts = Array.isArray(profileSnap.data()?.targetConcepts) ? profileSnap.data().targetConcepts : [];
  await profileRef.set(
    {
      userId: sellerId,
      targetConcepts: Array.from(new Set([concept, ...currentConcepts])).slice(0, 30),
      verified: false,
      plan: profileSnap.data()?.plan || 'free',
      createdAt: profileSnap.exists ? profileSnap.data().createdAt || now() : now(),
      updatedAt: now(),
    },
    { merge: true },
  );
  return { saved: true, firestoreSynced: true, conceptId: ref.id, concept: { ...doc, conceptId: ref.id } };
}

async function listSellerConcepts(sellerId) {
  const db = requireFirestoreDb();
  const snap = await db.collection('sellerProfiles').doc(sellerId).collection('concepts').orderBy('updatedAt', 'desc').limit(30).get();
  return snap.docs.map((doc) => ({ conceptId: doc.id, ...doc.data() }));
}

async function saveSellerBusiness(input) {
  const db = requireFirestoreDb();
  const userId = cleanString(input.userId, 160);
  const business = {
    ownerId: userId,
    name: cleanString(input.name, 300),
    category: cleanString(input.category, 160),
    address: cleanString(input.address, 600),
    imageUrl: cleanString(input.imageUrl, 1200),
    status: 'pending',
    verified: false,
    createdAt: now(),
    updatedAt: now(),
  };
  const profileRef = db.collection('sellerProfiles').doc(userId);
  const profileSnap = await profileRef.get();
  await profileRef.set(
    {
      userId,
      businessName: cleanString(input.businessName || input.name, 300),
      businessType: cleanString(input.businessType || input.category, 160),
      targetConcepts: input.targetConcept ? [cleanString(input.targetConcept, 1200)] : profileSnap.data()?.targetConcepts || [],
      verified: false,
      plan: profileSnap.data()?.plan || 'free',
      contact: {
        phone: cleanString(input.phone, 80),
        website: cleanString(input.website, 300),
        facebook: cleanString(input.facebook, 300),
      },
      createdAt: profileSnap.exists ? profileSnap.data().createdAt || now() : now(),
      updatedAt: now(),
    },
    { merge: true },
  );
  const ref = await profileRef.collection('businesses').add(business);
  await ref.update({ businessId: ref.id });
  await upsertPoi({
    poiId: `seller_${ref.id}`,
    source: 'seller',
    name: business.name,
    category: business.category,
    address: business.address,
    semanticText: `${business.name}. ${business.category}. ${business.address}.`,
    ownerId: userId,
    verified: false,
    status: 'pending',
    images: business.imageUrl ? [business.imageUrl] : [],
  });
  return { saved: true, firestoreSynced: true, businessId: ref.id, business: { ...business, businessId: ref.id } };
}

async function listSellerBusinesses(userId) {
  const db = requireFirestoreDb();
  const snap = await db.collection('sellerProfiles').doc(userId).collection('businesses').orderBy('updatedAt', 'desc').limit(30).get();
  return snap.docs.map((doc) => ({ businessId: doc.id, ...doc.data() }));
}

async function saveAdminReview(input) {
  const db = requireFirestoreDb();
  const doc = {
    reviewId: cleanString(input.reviewId, 160),
    targetType: cleanString(input.targetType || 'system', 80),
    targetId: cleanString(input.targetId || '', 220),
    title: cleanString(input.title || '', 300),
    reason: cleanString(input.reason || '', 1200),
    status: ['pending', 'approved', 'rejected', 'resolved'].includes(input.status) ? input.status : 'pending',
    reviewerId: cleanString(input.reviewerId || '', 160),
    payload: input.payload || {},
    createdAt: input.createdAt || now(),
    updatedAt: now(),
  };
  const ref = doc.reviewId ? db.collection('adminReviews').doc(doc.reviewId) : db.collection('adminReviews').doc();
  doc.reviewId = ref.id;
  await ref.set(doc, { merge: true });
  return { saved: true, firestoreSynced: true, reviewId: ref.id, review: doc };
}

async function listAdminReviews({ status, limit = 50 } = {}) {
  const db = requireFirestoreDb();
  let query = db.collection('adminReviews');
  if (status) query = query.where('status', '==', status);
  const snap = await query.limit(Math.min(Number(limit) || 50, 200)).get();
  return snap.docs.map((doc) => ({ reviewId: doc.id, ...doc.data() }));
}

module.exports = {
  ensureUserDocument,
  updateUserRole,
  listUsers,
  updateUserStatus,
  getCustomerProfile,
  saveCustomerProfile,
  upsertPoi,
  listPois,
  updatePoiStatus,
  saveItinerary,
  listItineraries,
  getAgentMemory,
  saveAgentMemory,
  saveBusinessAnalysis,
  listBusinessAnalyses,
  saveSellerConcept,
  listSellerConcepts,
  saveSellerBusiness,
  listSellerBusinesses,
  saveAdminReview,
  listAdminReviews,
};
