const MAX_ADMIN_LIST_LIMIT = 100;
const MAX_ADMIN_COUNT_PAGES = 20;

function cleanString(value, max = 500) {
  return String(value || '').trim().slice(0, max);
}

function parseLimit(value, fallback = 50) {
  const parsed = Number.parseInt(String(value || ''), 10);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.min(Math.max(parsed, 1), MAX_ADMIN_LIST_LIMIT);
}

function parseOffset(value) {
  const parsed = Number.parseInt(String(value || ''), 10);
  if (!Number.isFinite(parsed)) return 0;
  return Math.min(Math.max(parsed, 0), 10000);
}

function timestampToIso(value) {
  if (!value) return null;
  if (typeof value === 'string') return value;
  if (value instanceof Date) return value.toISOString();
  if (typeof value.toDate === 'function') return value.toDate().toISOString();
  return null;
}

function publicAdminPoi(poi = {}) {
  return {
    poiId: cleanString(poi.globalId || poi.id, 240),
    name: cleanString(poi.name, 240) || 'Unknown place',
    category: cleanString(poi.category || poi.categoryNormalized, 160) || 'unknown',
    address: cleanString(poi.addressCurrent || poi.address, 500) || null,
    district: cleanString(poi.district, 160) || null,
    source: cleanString(poi.source, 120) || null,
    rating: poi.rating !== null && poi.rating !== undefined && poi.rating !== '' && Number.isFinite(Number(poi.rating))
      ? Number(poi.rating)
      : null,
    reviewCount: poi.reviewCount !== null && poi.reviewCount !== undefined && poi.reviewCount !== '' && Number.isFinite(Number(poi.reviewCount))
      ? Number(poi.reviewCount)
      : null,
    imageUrl: /^https?:\/\//i.test(String(poi.imageUrl || '')) ? poi.imageUrl : null,
    location: Number.isFinite(Number(poi.lat)) && Number.isFinite(Number(poi.lon))
      ? { lat: Number(poi.lat), lng: Number(poi.lon) }
      : null,
    coordinateStatus: cleanString(poi.coordinateStatus, 80) || null,
  };
}

function listAdminPois({ pois = [], query, category, source, limit, offset } = {}) {
  const safeQuery = cleanString(query, 200).toLocaleLowerCase('vi-VN');
  const safeCategory = cleanString(category, 160).toLocaleLowerCase('vi-VN');
  const safeSource = cleanString(source, 120).toLocaleLowerCase('vi-VN');
  const safeLimit = parseLimit(limit);
  const safeOffset = parseOffset(offset);
  const filtered = pois.filter((poi) => {
    const candidate = publicAdminPoi(poi);
    const searchable = [candidate.poiId, candidate.name, candidate.address, candidate.district]
      .filter(Boolean)
      .join(' ')
      .toLocaleLowerCase('vi-VN');
    return (!safeQuery || searchable.includes(safeQuery))
      && (!safeCategory || candidate.category.toLocaleLowerCase('vi-VN') === safeCategory)
      && (!safeSource || String(candidate.source || '').toLocaleLowerCase('vi-VN') === safeSource);
  });
  const categories = Array.from(new Set(pois.map((poi) => publicAdminPoi(poi).category)))
    .sort((a, b) => a.localeCompare(b, 'vi'));
  const sources = Array.from(new Set(pois.map((poi) => publicAdminPoi(poi).source).filter(Boolean)))
    .sort((a, b) => a.localeCompare(b, 'en'));
  return {
    pois: filtered.slice(safeOffset, safeOffset + safeLimit).map(publicAdminPoi),
    total: filtered.length,
    limit: safeLimit,
    offset: safeOffset,
    filters: { categories, sources },
  };
}

function getAdminPoi({ pois = [], poiId } = {}) {
  const safeId = cleanString(poiId, 240);
  const poi = pois.find((candidate) => cleanString(candidate.globalId || candidate.id, 240) === safeId);
  return poi ? publicAdminPoi(poi) : null;
}

function stopSummary(stop = {}) {
  const poi = stop.poi || {};
  return {
    stopId: cleanString(stop.stopId, 240) || null,
    dayNumber: Number(stop.dayNumber) || 1,
    order: Number(stop.order) || 0,
    arrivalTime: cleanString(stop.arrivalTime, 20) || null,
    departureTime: cleanString(stop.departureTime, 20) || null,
    poi: {
      poiId: cleanString(poi.globalId || poi.id || stop.poiId, 240) || null,
      name: cleanString(poi.name, 240) || 'Unknown place',
      category: cleanString(poi.category, 160) || 'unknown',
    },
  };
}

function publicAdminTrip(id, data = {}, { detail = false } = {}) {
  const stops = Array.isArray(data.itinerary)
    ? data.itinerary
    : Array.isArray(data.preview?.stops) ? data.preview.stops : [];
  const trip = {
    tripId: cleanString(id, 240),
    ownerId: cleanString(data.ownerId, 240) || null,
    title: cleanString(data.title, 240) || 'Lịch trình Đà Nẵng',
    cityId: cleanString(data.cityId, 80) || 'da-nang',
    startDate: cleanString(data.startDate, 40) || null,
    dayCount: Math.max(1, Number(data.dayCount) || 1),
    stopCount: stops.length,
    status: cleanString(data.status, 80) || 'saved',
    needsReplan: Boolean(data.needsReplan),
    transport: cleanString(data.transport, 80) || null,
    createdAt: timestampToIso(data.createdAt),
    updatedAt: timestampToIso(data.updatedAt),
  };
  return detail ? { ...trip, stops: stops.map(stopSummary) } : trip;
}

function publicAdminFeedback(id, data = {}) {
  const payload = data.payload && typeof data.payload === 'object' ? data.payload : {};
  const ratingValue = payload.rating ?? payload.score;
  const numericRating = Number(ratingValue);
  return {
    eventId: cleanString(data.eventId || id, 240),
    userId: cleanString(data.userId, 240) || null,
    eventType: cleanString(data.eventType, 100) || 'unknown',
    rating: ratingValue !== null && ratingValue !== undefined && ratingValue !== '' && Number.isFinite(numericRating)
      ? numericRating
      : null,
    message: cleanString(payload.message || payload.comment || payload.reason, 1000) || null,
    poiId: cleanString(data.poiId || payload.poiId, 240) || null,
    itineraryId: cleanString(data.itineraryId || payload.itineraryId, 240) || null,
    createdAt: timestampToIso(data.createdAt),
  };
}

async function collectionDocuments(db, collectionName, limit = 100) {
  if (!db) throw new Error('admin_firestore_unavailable');
  const snapshot = await db.collection(collectionName).limit(limit).get();
  return (snapshot.docs || []).map((doc) => ({ id: doc.id, data: doc.data() || {} }));
}

async function collectionCount(db, collectionName) {
  if (!db) return { value: null, exact: false };
  const reference = db.collection(collectionName);
  if (typeof reference.count === 'function') {
    const snapshot = await reference.count().get();
    return { value: Number(snapshot.data()?.count) || 0, exact: true };
  }
  const snapshot = await reference.limit(501).get();
  const size = Number(snapshot.size ?? snapshot.docs?.length ?? 0);
  return { value: Math.min(size, 500), exact: size <= 500 };
}

async function countAuthUsers(auth) {
  if (!auth) return { value: null, exact: false };
  let count = 0;
  let pageToken;
  for (let page = 0; page < MAX_ADMIN_COUNT_PAGES; page += 1) {
    const result = await auth.listUsers(1000, pageToken);
    count += Array.isArray(result.users) ? result.users.length : 0;
    pageToken = result.pageToken;
    if (!pageToken) return { value: count, exact: true };
  }
  return { value: count, exact: false };
}

async function getAdminOverview({ auth, db } = {}) {
  const [users, trips, feedback, tripDocs, feedbackDocs] = await Promise.all([
    countAuthUsers(auth),
    collectionCount(db, 'travelerSavedTrips'),
    collectionCount(db, 'agentEvents'),
    collectionDocuments(db, 'travelerSavedTrips', 8),
    collectionDocuments(db, 'agentEvents', 8),
  ]);
  const recentActivity = [
    ...tripDocs.map(({ id, data }) => ({
      id: `trip:${id}`,
      type: 'trip',
      label: cleanString(data.title, 240) || 'Saved trip',
      ownerId: cleanString(data.ownerId, 240) || null,
      occurredAt: timestampToIso(data.updatedAt || data.createdAt),
    })),
    ...feedbackDocs.map(({ id, data }) => ({
      id: `feedback:${id}`,
      type: 'feedback',
      label: cleanString(data.eventType, 100) || 'Feedback',
      ownerId: cleanString(data.userId, 240) || null,
      occurredAt: timestampToIso(data.createdAt),
    })),
  ].sort((a, b) => String(b.occurredAt || '').localeCompare(String(a.occurredAt || ''))).slice(0, 8);
  return { counts: { users, trips, feedback }, recentActivity };
}

async function listAdminTrips({ db, limit } = {}) {
  const documents = await collectionDocuments(db, 'travelerSavedTrips', parseLimit(limit, 50));
  const trips = documents.map(({ id, data }) => publicAdminTrip(id, data));
  trips.sort((a, b) => String(b.updatedAt || '').localeCompare(String(a.updatedAt || '')));
  return { trips };
}

async function getAdminTrip({ db, tripId } = {}) {
  if (!db) throw new Error('admin_firestore_unavailable');
  const snapshot = await db.collection('travelerSavedTrips').doc(cleanString(tripId, 240)).get();
  return snapshot.exists ? publicAdminTrip(snapshot.id, snapshot.data() || {}, { detail: true }) : null;
}

async function listAdminFeedback({ db, limit } = {}) {
  const documents = await collectionDocuments(db, 'agentEvents', parseLimit(limit, 50));
  const feedback = documents.map(({ id, data }) => publicAdminFeedback(id, data));
  feedback.sort((a, b) => String(b.createdAt || '').localeCompare(String(a.createdAt || '')));
  return { feedback };
}

module.exports = {
  getAdminOverview,
  getAdminPoi,
  getAdminTrip,
  listAdminFeedback,
  listAdminPois,
  listAdminTrips,
  parseLimit,
  publicAdminFeedback,
  publicAdminPoi,
  publicAdminTrip,
};
