const crypto = require('crypto');
const { requireFirestoreDb } = require('../../config/firebaseAdmin');
const { loadPOIs } = require('../../services/poiDataService');
const { serializePoi } = require('./serializers');
const { buildTripPreview } = require('./tripPreview');
const { validateTripPreviewRequest } = require('./tripPreviewValidation');

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
    needsReplan: Boolean(input.needsReplan),
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
    needsReplan: Boolean(data.needsReplan),
    createdAt: timestampToIso(data.createdAt),
    updatedAt: timestampToIso(data.updatedAt),
  };
}

function canonicalPoiId(poi = {}) {
  return cleanString(poi.globalId || poi.id, 240);
}

function stopPoiId(stop = {}) {
  return cleanString(stop.poi?.globalId || stop.poi?.id || stop.poiId, 240);
}

function normalizeStopArray(value) {
  return Array.isArray(value) ? value.filter(Boolean) : [];
}

function maxStopOrder(stops) {
  return stops.reduce((max, stop) => Math.max(max, Number(stop.order) || 0), 0);
}

function renumberStops(stops) {
  const daySorted = [...stops].sort((a, b) => {
    return (
      Number(a.dayNumber || 1) - Number(b.dayNumber || 1) ||
      Number(a.order || 0) - Number(b.order || 0) ||
      String(a.stopId || '').localeCompare(String(b.stopId || ''), 'en')
    );
  });
  return daySorted.map((stop, index) => ({
    ...stop,
    order: index + 1,
    stopId: stop.stopId || `stop_${index + 1}`,
  }));
}

function daySummaryFromStops(stops, existingDays = [], trip = {}) {
  const maxDay = Math.max(
    Number(trip.dayCount) || 1,
    ...stops.map((stop) => Number(stop.dayNumber) || 1),
    ...existingDays.map((day) => Number(day.dayNumber) || 1),
  );
  return Array.from({ length: maxDay }, (_, index) => {
    const dayNumber = index + 1;
    const existing = existingDays.find((day) => Number(day.dayNumber) === dayNumber) || {};
    const dayStops = stops.filter((stop) => Number(stop.dayNumber || 1) === dayNumber);
    return {
      ...existing,
      dayNumber,
      stops: dayStops.map((stop) => stop.stopId),
      stopCount: dayStops.length,
    };
  });
}

function syncTripPreviewStops(trip, stops) {
  const preview = trip.preview && typeof trip.preview === 'object' ? trip.preview : {};
  return {
    ...trip,
    itinerary: stops,
    preview: {
      ...preview,
      stops,
      days: daySummaryFromStops(stops, Array.isArray(preview.days) ? preview.days : [], trip),
      persisted: true,
      preview: false,
      tripId: trip.tripId || preview.tripId || null,
    },
  };
}

function lifecycleError(status, code, message, details = []) {
  const error = new Error(message);
  error.status = status;
  error.code = code;
  error.details = details;
  return error;
}

async function findCanonicalPoi({ cityId, poiId }) {
  const pois = await loadPOIs({ cityId });
  const cleanPoiId = cleanString(poiId, 240);
  return pois.find((poi) => canonicalPoiId(poi) === cleanPoiId) || null;
}

function buildManualStop({ poi, dayNumber, order }) {
  const publicPoi = serializePoi(poi);
  return {
    stopId: `manual_${canonicalPoiId(poi)}`,
    order,
    dayNumber,
    poi: publicPoi,
    arrivalTime: null,
    departureTime: null,
    durationMinutes: 60,
    durationSource: 'manual_pending_replan',
    travelFromPrevious: {
      distanceKm: null,
      estimatedMinutes: null,
      distanceKnown: false,
      travelTimeKnown: false,
      calculationSource: 'manual_pending_replan',
      warnings: [{ code: 'REPLAN_REQUIRED', scope: 'leg' }],
    },
    reason: 'Added manually by the traveler; replan is required before treating this as optimized.',
    reasonCodes: ['manual_add', 'replan_required'],
    warnings: ['REPLAN_REQUIRED'],
  };
}

function buildPreviewRequestFromTrip(trip) {
  const request = trip.request && typeof trip.request === 'object' ? trip.request : {};
  const requestTrip = request.trip && typeof request.trip === 'object' ? request.trip : {};
  const constraints = request.constraints && typeof request.constraints === 'object' ? request.constraints : {};
  return {
    ...request,
    cityId: trip.cityId || request.cityId || 'da-nang',
    query: trip.query || request.query || 'Da Nang itinerary',
    trip: {
      ...requestTrip,
      date: trip.startDate || requestTrip.date || requestTrip.startDate,
      dayCount: trip.dayCount || requestTrip.dayCount || 1,
      dailyWindow: trip.dailyWindow || requestTrip.dailyWindow,
      dayWindows: trip.dayWindows || requestTrip.dayWindows || [],
      pace: trip.pace || requestTrip.pace || 'balanced',
      transport: trip.transport || requestTrip.transport || 'motorbike',
    },
    preferences: trip.preferences || request.preferences || {},
    constraints: {
      ...constraints,
      mustIncludePoiIds: trip.includedPoiIds || constraints.mustIncludePoiIds || [],
      excludePoiIds: trip.excludedPoiIds || constraints.excludePoiIds || [],
    },
  };
}

async function persistTrip({ tripId, trip }) {
  const update = {
    ...normalizeSavedTripInput(trip),
    ownerId: trip.ownerId,
    createdAt: trip.createdAt || nowIso(),
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

async function addStopToSavedTrip({ ownerId, tripId, payload = {} }) {
  const existing = await getSavedTrip({ ownerId, tripId });
  if (!existing) return null;
  const poiId = cleanString(payload.poiId || payload.globalId || payload.id, 240);
  if (!poiId) throw lifecycleError(400, 'VALIDATION_ERROR', 'poiId is required', [{ field: 'poiId', rule: 'required' }]);
  const stops = normalizeStopArray(existing.itinerary);
  if (stops.some((stop) => stopPoiId(stop) === poiId) || existing.includedPoiIds.includes(poiId)) {
    throw lifecycleError(409, 'DUPLICATE_STOP', 'POI is already included in this saved trip.', [{ field: 'poiId', value: poiId }]);
  }
  if (existing.excludedPoiIds.includes(poiId)) {
    throw lifecycleError(409, 'EXCLUDED_POI', 'POI is excluded from this saved trip.', [{ field: 'poiId', value: poiId }]);
  }
  const poi = await findCanonicalPoi({ cityId: existing.cityId, poiId });
  if (!poi) throw lifecycleError(422, 'INVALID_POI_ID', 'POI is not in the approved canonical dataset.', [{ field: 'poiId', value: poiId }]);
  const dayNumber = Math.min(Math.max(Number(payload.dayNumber) || 1, 1), existing.dayCount || 1);
  const nextStop = buildManualStop({
    poi,
    dayNumber,
    order: maxStopOrder(stops) + 1,
  });
  const nextTrip = syncTripPreviewStops({
    ...existing,
    includedPoiIds: [...new Set([...existing.includedPoiIds, poiId])],
    needsReplan: true,
  }, renumberStops([...stops, nextStop]));
  return persistTrip({ tripId, trip: { ...nextTrip, ownerId } });
}

async function removeStopFromSavedTrip({ ownerId, tripId, stopId }) {
  const existing = await getSavedTrip({ ownerId, tripId });
  if (!existing) return null;
  const stops = normalizeStopArray(existing.itinerary);
  const target = stops.find((stop) => stop.stopId === stopId);
  if (!target) throw lifecycleError(404, 'STOP_NOT_FOUND', 'Stop was not found in this saved trip.', [{ field: 'stopId', value: stopId }]);
  const poiId = stopPoiId(target);
  const nextTrip = syncTripPreviewStops({
    ...existing,
    includedPoiIds: existing.includedPoiIds.filter((id) => id !== poiId),
    excludedPoiIds: poiId ? [...new Set([...existing.excludedPoiIds, poiId])] : existing.excludedPoiIds,
    needsReplan: true,
  }, renumberStops(stops.filter((stop) => stop.stopId !== stopId)));
  return persistTrip({ tripId, trip: { ...nextTrip, ownerId } });
}

async function reorderSavedTripStops({ ownerId, tripId, payload = {} }) {
  const existing = await getSavedTrip({ ownerId, tripId });
  if (!existing) return null;
  const dayNumber = Number(payload.dayNumber);
  const stopIds = Array.isArray(payload.stopIds) ? payload.stopIds.map((id) => cleanString(id, 240)).filter(Boolean) : [];
  if (!Number.isInteger(dayNumber) || dayNumber < 1 || dayNumber > (existing.dayCount || 1)) {
    throw lifecycleError(400, 'VALIDATION_ERROR', 'dayNumber is invalid', [{ field: 'dayNumber', rule: 'integer_within_trip' }]);
  }
  if (!stopIds.length || new Set(stopIds).size !== stopIds.length) {
    throw lifecycleError(400, 'VALIDATION_ERROR', 'stopIds must be a non-empty unique array', [{ field: 'stopIds', rule: 'unique_non_empty_array' }]);
  }
  const stops = normalizeStopArray(existing.itinerary);
  const dayStops = stops.filter((stop) => Number(stop.dayNumber || 1) === dayNumber);
  const existingIds = dayStops.map((stop) => stop.stopId);
  if (stopIds.length !== existingIds.length || stopIds.some((id) => !existingIds.includes(id))) {
    throw lifecycleError(422, 'INVALID_STOP_ORDER', 'stopIds must include every stop for the selected day exactly once.', [{ field: 'stopIds', rule: 'same_day_complete_order' }]);
  }
  const orderMap = new Map(stopIds.map((id, index) => [id, index + 1]));
  const nextStops = stops.map((stop) => Number(stop.dayNumber || 1) === dayNumber
    ? { ...stop, order: orderMap.get(stop.stopId) || stop.order }
    : stop);
  const nextTrip = syncTripPreviewStops({
    ...existing,
    needsReplan: true,
  }, renumberStops(nextStops));
  return persistTrip({ tripId, trip: { ...nextTrip, ownerId } });
}

async function replanSavedTrip({ ownerId, tripId }) {
  const existing = await getSavedTrip({ ownerId, tripId });
  if (!existing) return null;
  const previewRequest = buildPreviewRequestFromTrip(existing);
  const validation = validateTripPreviewRequest(previewRequest);
  if (validation.errors) {
    throw lifecycleError(400, 'VALIDATION_ERROR', 'Saved trip cannot be replanned because its trip configuration is invalid.', validation.errors);
  }
  const result = await buildTripPreview(validation.value);
  if (result.error) throw lifecycleError(result.error.status, result.error.code, result.error.message, result.error.details);
  const calculated = {
    ...result.trip,
    tripId,
    preview: false,
    persisted: true,
    authenticated: true,
    saveEligible: true,
  };
  const nextTrip = {
    ...existing,
    request: validation.value,
    preview: calculated,
    itinerary: calculated.stops || [],
    warnings: calculated.warnings || [],
    needsReplan: false,
  };
  return persistTrip({ tripId, trip: { ...nextTrip, ownerId } });
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
  addStopToSavedTrip,
  createSavedTrip,
  deleteSavedTrip,
  getSavedTrip,
  listSavedTrips,
  normalizeSavedTripInput,
  removeStopFromSavedTrip,
  reorderSavedTripStops,
  replanSavedTrip,
  serializeTrip,
  updateSavedTrip,
};
