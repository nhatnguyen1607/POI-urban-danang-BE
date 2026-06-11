import { FieldValue, Firestore, Transaction } from "firebase-admin/firestore";

type PreferenceMap = Record<string, number>;

export interface OnlineEvent {
  source: "review" | "user_analytics";
  eventId: string;
  userId: string;
  poiId?: string | null;
  poiName?: string | null;
  poiCategory?: string | null;
  rating?: number;
  dwellMinutes?: number;
  durationMinutes?: number;
  distanceKm?: number;
  avgSpeedKmh?: number;
  inferredTransport?: string | null;
  visitPurpose?: string | null;
  visitMood?: string | null;
  eventType?: string | null;
  context?: {
    dayOfWeek?: string | null;
    timeOfDay?: string | null;
    weather?: unknown;
  } | null;
  createdAt?: unknown;
}

const MAX_MAP_KEYS = 24;
const BASE_ALPHA = 0.18;
const LONG_DWELL_MINUTES = 90;

function cleanKey(value: unknown, fallback = "unknown") {
  const cleaned = String(value || fallback)
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
  return cleaned || fallback;
}

function weekPart(dayOfWeek: unknown) {
  const day = cleanKey(dayOfWeek);
  return day === "saturday" || day === "sunday" ? "weekend" : "weekday";
}

function dwellStyle(minutes: number) {
  if (minutes < 25) return "quick_stop";
  if (minutes < 75) return "standard_visit";
  return "long_stay";
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function eventAlpha(event: OnlineEvent) {
  const dwellBoost = event.dwellMinutes ? clamp(event.dwellMinutes / LONG_DWELL_MINUTES, 0.4, 1.8) : 1;
  const ratingBoost = event.rating ? clamp(event.rating / 4, 0.75, 1.35) : 1;
  return clamp(BASE_ALPHA * dwellBoost * ratingBoost, 0.08, 0.34);
}

function topMap(map: PreferenceMap, maxKeys = MAX_MAP_KEYS) {
  return Object.fromEntries(
    Object.entries(map)
      .filter(([, value]) => Number.isFinite(value) && value > 0.001)
      .sort((a, b) => b[1] - a[1])
      .slice(0, maxKeys),
  );
}

function blendMap(current: PreferenceMap = {}, deltas: PreferenceMap = {}, alpha: number, maxKeys = MAX_MAP_KEYS) {
  const next: PreferenceMap = {};
  Object.entries(current).forEach(([key, value]) => {
    next[key] = Number(value || 0) * (1 - alpha);
  });
  Object.entries(deltas).forEach(([key, value]) => {
    next[key] = Number(next[key] || 0) + Number(value || 0) * alpha;
  });
  return topMap(next, maxKeys);
}

function nestedBlend(
  current: Record<string, { categories?: PreferenceMap; purposes?: PreferenceMap; moods?: PreferenceMap }> = {},
  contextKey: string,
  deltas: { categories?: PreferenceMap; purposes?: PreferenceMap; moods?: PreferenceMap },
  alpha: number,
) {
  const next = { ...current };
  const bucket = next[contextKey] || { categories: {}, purposes: {}, moods: {} };
  next[contextKey] = {
    categories: blendMap(bucket.categories, deltas.categories, alpha),
    purposes: blendMap(bucket.purposes, deltas.purposes, alpha),
    moods: blendMap(bucket.moods, deltas.moods, alpha),
  };
  return Object.fromEntries(Object.entries(next).slice(-16));
}

function purposeByCategoryBlend(
  current: Record<string, PreferenceMap> = {},
  category: string,
  purpose: string,
  alpha: number,
) {
  if (!purpose) return current;
  return {
    ...current,
    [category]: blendMap(current[category], { [purpose]: 1 }, alpha, 8),
  };
}

function categoryFromEvent(event: OnlineEvent) {
  return cleanKey(event.poiCategory || event.poiName || event.poiId || "unknown");
}

function eventDeltas(event: OnlineEvent) {
  const category = categoryFromEvent(event);
  const purpose = cleanKey(event.visitPurpose, "");
  const mood = cleanKey(event.visitMood, "");
  const context = event.context || {};
  const contextKey = `${cleanKey(context.timeOfDay)}.${weekPart(context.dayOfWeek)}`;
  const quality = event.source === "review" ? clamp(Number(event.rating || 3) / 5, 0.2, 1.2) : 1;
  const dwellWeight = event.dwellMinutes ? clamp(Number(event.dwellMinutes) / 45, 0.5, 3) : 1;
  const weight = quality * dwellWeight;
  return {
    category,
    purpose,
    mood,
    contextKey,
    categoryDelta: { [category]: weight },
    purposeDelta: purpose ? { [purpose]: weight } : {},
    moodDelta: mood ? { [mood]: weight } : {},
  };
}

function transportDelta(event: OnlineEvent) {
  const transport = cleanKey(event.inferredTransport, "");
  return transport ? { [transport]: 1 } : {};
}

function updateDwellPreference(current: any, event: OnlineEvent, alpha: number) {
  const sample = Number(event.dwellMinutes || 0);
  if (!sample || sample <= 0) return current || { targetMinutes: 45, style: "standard_visit", sampleCount: 0 };
  const previousTarget = Number(current?.targetMinutes || 45);
  const targetMinutes = Math.round(previousTarget * (1 - alpha) + sample * alpha);
  return {
    targetMinutes,
    style: dwellStyle(targetMinutes),
    sampleCount: Number(current?.sampleCount || 0) + 1,
  };
}

function normalizeEvent(raw: FirebaseFirestore.DocumentData, source: "review" | "user_analytics", eventId: string): OnlineEvent | null {
  const userId = String(raw.userId || "");
  if (!userId) return null;
  return {
    source,
    eventId,
    userId,
    poiId: raw.poiId || raw.toPoiId || null,
    poiName: raw.poiName || raw.toPoiName || null,
    poiCategory: raw.poiCategory || raw.category || null,
    rating: Number(raw.rating || 0),
    dwellMinutes: Number(raw.dwellMinutes || 0),
    durationMinutes: Number(raw.durationMinutes || 0),
    distanceKm: Number(raw.distanceKm || 0),
    avgSpeedKmh: Number(raw.avgSpeedKmh || 0),
    inferredTransport: raw.inferredTransport || null,
    visitPurpose: raw.visitPurpose || null,
    visitMood: raw.visitMood || null,
    eventType: raw.eventType || null,
    context: raw.context || null,
    createdAt: raw.createdAt || null,
  };
}

async function applyUpdateInTransaction(tx: Transaction, db: Firestore, event: OnlineEvent, eventKey: string) {
  const profileRef = db.collection("user_preferences").doc(event.userId);
  const markerRef = db.collection("_online_learning_events").doc(eventKey);
  const [profileSnap, markerSnap] = await Promise.all([tx.get(profileRef), tx.get(markerRef)]);
  if (markerSnap.exists) return { skipped: true };

  const profile = profileSnap.exists ? profileSnap.data() || {} : {};
  const alpha = eventAlpha(event);
  const deltas = eventDeltas(event);
  const sampleCounts = profile.sampleCounts || {};
  const nextSampleCounts = {
    ...sampleCounts,
    onlineEvents: Number(sampleCounts.onlineEvents || 0) + 1,
    reviews: Number(sampleCounts.reviews || 0) + (event.source === "review" ? 1 : 0),
    analytics: Number(sampleCounts.analytics || 0) + (event.source === "user_analytics" ? 1 : 0),
    visits: Number(sampleCounts.visits || 0) + (event.eventType === "poi_visit" ? 1 : 0),
    routeSegments: Number(sampleCounts.routeSegments || 0) + (event.eventType === "route_segment" ? 1 : 0),
  };

  const update = {
    userId: event.userId,
    version: "v1-online",
    learningMode: "online_ema",
    dwellPreference: updateDwellPreference(profile.dwellPreference, event, alpha),
    categoryWeights: blendMap(profile.categoryWeights, deltas.categoryDelta, alpha),
    moodWeights: blendMap(profile.moodWeights, deltas.moodDelta, alpha),
    purposeWeights: blendMap(profile.purposeWeights, deltas.purposeDelta, alpha),
    contextAffinities: nestedBlend(profile.contextAffinities, deltas.contextKey, {
      categories: deltas.categoryDelta,
      purposes: deltas.purposeDelta,
      moods: deltas.moodDelta,
    }, alpha),
    purposeByCategory: purposeByCategoryBlend(profile.purposeByCategory, deltas.category, deltas.purpose, alpha),
    transportWeights: blendMap(profile.transportWeights, transportDelta(event), alpha, 8),
    sampleCounts: nextSampleCounts,
    lastOnlineEvent: {
      source: event.source,
      eventId: event.eventId,
      eventType: event.eventType || null,
      alpha,
      poiId: event.poiId || null,
      contextKey: deltas.contextKey,
    },
    updatedAt: FieldValue.serverTimestamp(),
  };

  tx.set(profileRef, update, { merge: true });
  tx.set(markerRef, {
    userId: event.userId,
    source: event.source,
    eventId: event.eventId,
    processedAt: FieldValue.serverTimestamp(),
  });
  return { skipped: false };
}

export async function learnFromFirestoreDocument(
  db: Firestore,
  raw: FirebaseFirestore.DocumentData,
  source: "review" | "user_analytics",
  eventId: string,
) {
  const event = normalizeEvent(raw, source, eventId);
  if (!event) return { skipped: true, reason: "missing_user_id" };
  const eventKey = `${source}_${eventId}`;
  return db.runTransaction((tx) => applyUpdateInTransaction(tx, db, event, eventKey));
}
