const crypto = require('crypto');
const { requireFirestoreDb } = require('../../config/firebaseAdmin');

const MAX_TITLE_LENGTH = 180;
const MAX_QUERY_LENGTH = 1200;
const MAX_ARRAY_ITEMS = 100;
const COLLECTION = 'travelerSavedTrips';
const memoryTrips = new Map();

function nowIso() {
  return new Date().toISOString();
}

function cleanString(value, max = 1200) {
  return String(value || '').trim().slice(0, max);
}

function cleanStringArray(value, maxItems = MAX_ARRAY_ITEMS) {
  if (!Array.isArray(value)) return [];
  return Array.from(new Set(value.map((item) => cleanString(item, 240)).filter(Boolean))).slice(0, maxItems);
}

function toPlainJson(value, fallback) {
  if (value === undefined) return fallback;
  try {
    return JSON.parse(JSON.stringify(value));
  } catch {
    return fallback;
  }
}

function timestampToIso(value) {
  if (!value) return null;
  if (typeof value === 'string') return value;
  if (value instanceof Date) return value.toISOString();
  if (typeof value.toDate === 'function') return value.toDate().toISOString();
  return null;
}

function useMemoryStore() {
  return process.env.URBANAGENT_SAVED_TRIPS_STORE === 'memory' && process.env.NODE_ENV !== 'production';
}

function getStore() {
  if (useMemoryStore()) return null;
  return requireFirestoreDb();
}

function normalizeDailyWindow(value) {
  if (!value || typeof value !== 'object') return null;
  const start = cleanString(value.startTime || value.start, 20);
  const end = cleanString(value.endTime || value.end, 20);
  if (!start || !end) return null;
  return { startTime: start, endTime: end };
}

function normalizeDayWindows(value) {
  if (!Array.isArray(value)) return [];
  return value
    .map((item) => ({
      dayNumber: Number(item?.dayNumber),
      startTime: cleanString(item?.startTime || item?.start, 20),
      endTime: cleanString(item?.endTime || item?.end, 20),
    }))
    .filter((item) => Number.isInteger(item.dayNumber) && item.dayNumber >= 1 && item.startTime && item.endTime)
    .slice(0, 7);
}

function normalizeSavedTripInput(input = {}) {
  const request = input.request && typeof input.request === 'object' ? input.request : {};
  const requestTrip = request.trip && typeof request.trip === 'object' ? request.trip : {};
  const trip = input.trip && typeof input.trip === 'object' ? input.trip : {};
  const constraints = request.constraints && typeof request.constraints === 'object' ? request.constraints : {};
  const preview = input.preview || input.tripPreview || input.calculatedTrip || null;
  const cityId = cleanString(input.cityId || request.cityId || preview?.cityId || 'da-nang', 80) || 'da-nang';
  const query = cleanString(input.query || request.query, MAX_QUERY_LENGTH);
  const title = cleanString(input.title || input.name || query || 'Lịch trình Đà Nẵng', MAX_TITLE_LENGTH);
  const startDate = cleanString(input.startDate || trip.startDate || trip.date || requestTrip.startDate || requestTrip.date, 40);
  const dayCount = Math.min(Math.max(Number(input.dayCount || trip.dayCount || requestTrip.dayCount || preview?.dayCount || 1) || 1, 1), 7);
  const dailyWindow = normalizeDailyWindow(input.dailyWindow || trip.dailyWindow || requestTrip.dailyWindow);
  const dayWindows = normalizeDayWindows(input.dayWindows || trip.dayWindows || requestTrip.dayWindows);
  const includedPoiIds = cleanStringArray(input.includedPoiIds || constraints.mustIncludePoiIds);
  const excludedPoiIds = cleanStringArray(input.excludedPoiIds || constraints.excludePoiIds);
  return {
    title,
    cityId,
    query,
    startDate,
    dayCount,
    dailyWindow,
    dayWindows,
    preferences: toPlainJson(input.preferences || request.preferences || {}, {}),
    pace: cleanString(input.pace || trip.pace || requestTrip.pace || 'balanced', 40),
    transport: cleanString(input.transport || trip.transport || requestTrip.transport || 'motorbike', 40),
    includedPoiIds,
    excludedPoiIds,
    request: toPlainJson(request, {}),
    preview: toPlainJson(preview, null),
    itinerary: toPlainJson(input.itinerary || preview?.stops || [], []),
    warnings: toPlainJson(input.warnings || preview?.warnings || [], []),
    status: cleanString(input.status || 'saved', 40) || 'saved',
  };
}

function serializeTrip(id, data = {}) {
  return {
    tripId: id,
    title: data.title || 'Lịch trình Đà Nẵng',
    cityId: data.cityId || 'da-nang',
    query: data.query || '',
    startDate: data.startDate || '',
    dayCount: Number(data.dayCount) || 1,
    dailyWindow: data.dailyWindow || null,
    dayWindows: Array.isArray(data.dayWindows) ? data.dayWindows : [],
    preferences: data.preferences || {},
    pace: data.pace || 'balanced',
    transport: data.transport || 'motorbike',
    includedPoiIds: Array.isArray(data.includedPoiIds) ? data.includedPoiIds : [],
    excludedPoiIds: Array.isArray(data.excludedPoiIds) ? data.excludedPoiIds : [],
    request: data.request || {},
    preview: data.preview || null,
    itinerary: Array.isArray(data.itinerary) ? data.itinerary : [],
    warnings: Array.isArray(data.warnings) ? data.warnings : [],
    status: data.status || 'saved',
    createdAt: timestampToIso(data.createdAt),
    updatedAt: timestampToIso(data.updatedAt),
  };
}

async function createSavedTrip({ ownerId, payload }) {
  const doc = {
    ...normalizeSavedTripInput(payload),
    ownerId,
    createdAt: nowIso(),
    updatedAt: nowIso(),
  };
  if (useMemoryStore()) {
    const tripId = `trip_${crypto.randomUUID()}`;
    memoryTrips.set(tripId, { ...doc });
    return serializeTrip(tripId, doc);
  }
  const db = getStore();
  const ref = await db.collection(COLLECTION).add(doc);
  return serializeTrip(ref.id, doc);
}

async function listSavedTrips(ownerId) {
  if (useMemoryStore()) {
    return Array.from(memoryTrips.entries())
      .filter(([, trip]) => trip.ownerId === ownerId)
      .map(([id, trip]) => serializeTrip(id, trip))
      .sort((a, b) => String(b.updatedAt || '').localeCompare(String(a.updatedAt || '')));
  }
  const db = getStore();
  const snap = await db.collection(COLLECTION).where('ownerId', '==', ownerId).limit(50).get();
  return snap.docs
    .map((doc) => serializeTrip(doc.id, doc.data()))
    .sort((a, b) => String(b.updatedAt || '').localeCompare(String(a.updatedAt || '')));
}

async function getSavedTrip({ ownerId, tripId }) {
  if (useMemoryStore()) {
    const trip = memoryTrips.get(tripId);
    if (!trip || trip.ownerId !== ownerId) return null;
    return serializeTrip(tripId, trip);
  }
  const db = getStore();
  const snap = await db.collection(COLLECTION).doc(tripId).get();
  if (!snap.exists) return null;
  const data = snap.data();
  if (data.ownerId !== ownerId) return null;
  return serializeTrip(snap.id, data);
}

async function updateSavedTrip({ ownerId, tripId, payload }) {
  const existing = await getSavedTrip({ ownerId, tripId });
  if (!existing) return null;
  const update = {
    ...normalizeSavedTripInput({ ...existing, ...payload }),
    ownerId,
    createdAt: existing.createdAt || nowIso(),
    updatedAt: nowIso(),
  };
  if (useMemoryStore()) {
    memoryTrips.set(tripId, update);
    return serializeTrip(tripId, update);
  }
  const db = getStore();
  await db.collection(COLLECTION).doc(tripId).set(update, { merge: false });
  return serializeTrip(tripId, update);
}

async function deleteSavedTrip({ ownerId, tripId }) {
  const existing = await getSavedTrip({ ownerId, tripId });
  if (!existing) return false;
  if (useMemoryStore()) {
    memoryTrips.delete(tripId);
    return true;
  }
  const db = getStore();
  await db.collection(COLLECTION).doc(tripId).delete();
  return true;
}

module.exports = {
  createSavedTrip,
  deleteSavedTrip,
  getSavedTrip,
  listSavedTrips,
  normalizeSavedTripInput,
  serializeTrip,
  updateSavedTrip,
};
