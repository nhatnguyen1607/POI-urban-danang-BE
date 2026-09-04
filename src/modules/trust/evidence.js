const SOURCE_TYPES = Object.freeze({
  OFFICIAL_PARTNER: 'official_partner',
  MERCHANT_VERIFIED: 'merchant_verified',
  COMMERCIAL_POI: 'commercial_poi',
  COMMUNITY: 'community',
  URBANAGENT_USER_EVIDENCE: 'urbanagent_user_evidence',
  CANONICAL_DATASET: 'canonical_dataset',
});

const EVIDENCE_LEVELS = Object.freeze({
  LIVE_VERIFIED: 'LIVE_VERIFIED',
  AUTHORITATIVE: 'AUTHORITATIVE',
  RECENT_EVIDENCE: 'RECENT_EVIDENCE',
  STALE_UNVERIFIED: 'STALE_UNVERIFIED',
});

const EVIDENCE_STATUSES = Object.freeze({
  OPEN: 'OPEN',
  CLOSED: 'CLOSED',
  UNKNOWN: 'UNKNOWN',
  TEMPORARILY_CLOSED: 'TEMPORARILY_CLOSED',
  AVAILABLE: 'AVAILABLE',
  SOLD_OUT: 'SOLD_OUT',
  CONFLICT: 'CONFLICT',
  PROVIDER_UNAVAILABLE: 'PROVIDER_UNAVAILABLE',
});

const FRESHNESS_STATES = Object.freeze({
  FRESH: 'FRESH',
  STALE: 'STALE',
  EXPIRED: 'EXPIRED',
  UNKNOWN: 'UNKNOWN',
});

const DEFAULT_TTL_SECONDS = Object.freeze({
  hotelAvailability: 300,
  dynamicPrice: 300,
  restaurantOpenNow: 1800,
  recurringOpeningHours: 86_400,
  attractionOperationalStatus: 21_600,
  weather: 1200,
  route: 600,
  userVisitReport: 86_400,
  canonicalMetadata: 604_800,
});

const LEVEL_PRIORITY = Object.freeze({
  [EVIDENCE_LEVELS.LIVE_VERIFIED]: 4,
  [EVIDENCE_LEVELS.AUTHORITATIVE]: 3,
  [EVIDENCE_LEVELS.RECENT_EVIDENCE]: 2,
  [EVIDENCE_LEVELS.STALE_UNVERIFIED]: 1,
});

function cleanString(value, max = 1200) {
  return typeof value === 'string' ? value.trim().slice(0, max) : '';
}

function isoDate(value) {
  if (!value) return null;
  const date = value instanceof Date ? value : new Date(value);
  return Number.isFinite(date.getTime()) ? date.toISOString() : null;
}

function positiveSeconds(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? Math.round(parsed) : fallback;
}

function getFreshnessPolicy(env = process.env) {
  return {
    hotelAvailability: positiveSeconds(env.URBANAGENT_TTL_HOTEL_AVAILABILITY_SECONDS, DEFAULT_TTL_SECONDS.hotelAvailability),
    dynamicPrice: positiveSeconds(env.URBANAGENT_TTL_DYNAMIC_PRICE_SECONDS, DEFAULT_TTL_SECONDS.dynamicPrice),
    restaurantOpenNow: positiveSeconds(env.URBANAGENT_TTL_RESTAURANT_OPEN_NOW_SECONDS, DEFAULT_TTL_SECONDS.restaurantOpenNow),
    recurringOpeningHours: positiveSeconds(env.URBANAGENT_TTL_RECURRING_HOURS_SECONDS, DEFAULT_TTL_SECONDS.recurringOpeningHours),
    attractionOperationalStatus: positiveSeconds(env.URBANAGENT_TTL_ATTRACTION_STATUS_SECONDS, DEFAULT_TTL_SECONDS.attractionOperationalStatus),
    weather: positiveSeconds(env.URBANAGENT_TTL_WEATHER_SECONDS, DEFAULT_TTL_SECONDS.weather),
    route: positiveSeconds(env.URBANAGENT_TTL_ROUTE_SECONDS, DEFAULT_TTL_SECONDS.route),
    userVisitReport: positiveSeconds(env.URBANAGENT_TTL_USER_VISIT_REPORT_SECONDS, DEFAULT_TTL_SECONDS.userVisitReport),
    canonicalMetadata: positiveSeconds(env.URBANAGENT_TTL_CANONICAL_METADATA_SECONDS, DEFAULT_TTL_SECONDS.canonicalMetadata),
  };
}

function createEvidence(input = {}) {
  const sourceType = Object.values(SOURCE_TYPES).includes(input.sourceType)
    ? input.sourceType
    : SOURCE_TYPES.COMMUNITY;
  const status = Object.values(EVIDENCE_STATUSES).includes(input.status)
    ? input.status
    : EVIDENCE_STATUSES.UNKNOWN;
  const observedAt = isoDate(input.observedAt);
  const lastVerifiedAt = isoDate(input.lastVerifiedAt);
  const ttlSeconds = positiveSeconds(input.ttlSeconds, null);
  const referenceTime = lastVerifiedAt || observedAt;
  const validUntil = isoDate(input.validUntil)
    || (referenceTime && ttlSeconds
      ? new Date(new Date(referenceTime).getTime() + ttlSeconds * 1000).toISOString()
      : null);
  return {
    sourceId: cleanString(input.sourceId, 220) || null,
    sourceName: cleanString(input.sourceName, 220) || 'Unknown source',
    sourceType,
    sourceUrl: cleanString(input.sourceUrl, 1200) || null,
    dataClass: cleanString(input.dataClass, 120) || 'canonicalMetadata',
    observedAt,
    lastVerifiedAt,
    validUntil,
    ttlSeconds,
    verificationMethod: cleanString(input.verificationMethod, 160) || 'unverified',
    status,
    evidenceLevel: Object.values(EVIDENCE_LEVELS).includes(input.evidenceLevel)
      ? input.evidenceLevel
      : null,
    confidenceReason: cleanString(input.confidenceReason, 600) || null,
    conflict: Boolean(input.conflict),
    conflictSources: Array.isArray(input.conflictSources)
      ? [...new Set(input.conflictSources.map((item) => cleanString(item, 220)).filter(Boolean))]
      : [],
    provenance: input.provenance && typeof input.provenance === 'object' ? { ...input.provenance } : {},
  };
}

function getFreshnessState(evidence, now = new Date()) {
  const normalized = createEvidence(evidence);
  const currentTime = now instanceof Date ? now.getTime() : new Date(now).getTime();
  const reference = normalized.lastVerifiedAt || normalized.observedAt;
  if (!reference || !Number.isFinite(currentTime)) return FRESHNESS_STATES.UNKNOWN;
  const referenceTime = new Date(reference).getTime();
  const validUntil = normalized.validUntil ? new Date(normalized.validUntil).getTime() : null;
  if (Number.isFinite(validUntil) && currentTime > validUntil) return FRESHNESS_STATES.EXPIRED;
  if (!normalized.ttlSeconds) return FRESHNESS_STATES.UNKNOWN;
  return currentTime - referenceTime <= normalized.ttlSeconds * 1000
    ? FRESHNESS_STATES.FRESH
    : FRESHNESS_STATES.STALE;
}

function getEvidenceLevel(evidence, now = new Date()) {
  const normalized = createEvidence(evidence);
  const freshness = getFreshnessState(normalized, now);
  if (freshness !== FRESHNESS_STATES.FRESH) return EVIDENCE_LEVELS.STALE_UNVERIFIED;
  if (normalized.sourceType === SOURCE_TYPES.OFFICIAL_PARTNER) return EVIDENCE_LEVELS.LIVE_VERIFIED;
  if (normalized.sourceType === SOURCE_TYPES.MERCHANT_VERIFIED) return EVIDENCE_LEVELS.AUTHORITATIVE;
  if (
    normalized.sourceType === SOURCE_TYPES.URBANAGENT_USER_EVIDENCE
    || normalized.sourceType === SOURCE_TYPES.COMMERCIAL_POI
  ) return EVIDENCE_LEVELS.RECENT_EVIDENCE;
  return normalized.evidenceLevel || EVIDENCE_LEVELS.STALE_UNVERIFIED;
}

function statusesConflict(statuses) {
  const openLike = statuses.has(EVIDENCE_STATUSES.OPEN) || statuses.has(EVIDENCE_STATUSES.AVAILABLE);
  const closedLike = statuses.has(EVIDENCE_STATUSES.CLOSED)
    || statuses.has(EVIDENCE_STATUSES.TEMPORARILY_CLOSED)
    || statuses.has(EVIDENCE_STATUSES.SOLD_OUT);
  return openLike && closedLike;
}

function resolveEvidenceConflict(items = [], now = new Date()) {
  const evidence = items.map((item) => {
    const normalized = createEvidence(item);
    return { ...normalized, evidenceLevel: getEvidenceLevel(normalized, now), freshnessState: getFreshnessState(normalized, now) };
  });
  const usable = evidence.filter((item) => ![
    EVIDENCE_STATUSES.UNKNOWN,
    EVIDENCE_STATUSES.PROVIDER_UNAVAILABLE,
  ].includes(item.status));
  if (!usable.length) return { status: EVIDENCE_STATUSES.UNKNOWN, conflict: false, selected: null, evidence };
  const highestPriority = Math.max(...usable.map((item) => LEVEL_PRIORITY[item.evidenceLevel] || 0));
  const comparable = usable.filter((item) => (LEVEL_PRIORITY[item.evidenceLevel] || 0) >= highestPriority - 1);
  const statuses = new Set(comparable.map((item) => item.status));
  if (statusesConflict(statuses)) {
    const conflictSources = [...new Set(comparable.map((item) => item.sourceName))];
    return {
      status: EVIDENCE_STATUSES.CONFLICT,
      conflict: true,
      conflictSources,
      selected: null,
      evidence: evidence.map((item) => comparable.includes(item)
        ? { ...item, conflict: true, conflictSources }
        : item),
    };
  }
  const selected = usable.slice().sort((left, right) => {
    const priority = (LEVEL_PRIORITY[right.evidenceLevel] || 0) - (LEVEL_PRIORITY[left.evidenceLevel] || 0);
    if (priority) return priority;
    return String(right.lastVerifiedAt || right.observedAt || '').localeCompare(String(left.lastVerifiedAt || left.observedAt || ''));
  })[0];
  return { status: selected.status, conflict: false, conflictSources: [], selected, evidence };
}

function evidenceFromVerifiedUserReport(input = {}, policy = getFreshnessPolicy()) {
  if (!input.verifiedArrival) return null;
  const status = Object.values(EVIDENCE_STATUSES).includes(input.status)
    ? input.status
    : EVIDENCE_STATUSES.UNKNOWN;
  return createEvidence({
    sourceId: input.eventId || null,
    sourceName: 'UrbanAgent traveler report',
    sourceType: SOURCE_TYPES.URBANAGENT_USER_EVIDENCE,
    dataClass: 'userVisitReport',
    observedAt: input.observedAt || new Date(),
    lastVerifiedAt: input.observedAt || new Date(),
    ttlSeconds: policy.userVisitReport,
    verificationMethod: 'gps_arrival_and_post_visit_feedback',
    status,
    confidenceReason: 'A traveler report was submitted after a verified arrival. It is evidence, not a canonical decision.',
    provenance: { poiId: cleanString(input.poiId, 220) || null, canonicalMutation: false },
  });
}

module.exports = {
  DEFAULT_TTL_SECONDS,
  EVIDENCE_LEVELS,
  EVIDENCE_STATUSES,
  FRESHNESS_STATES,
  SOURCE_TYPES,
  createEvidence,
  evidenceFromVerifiedUserReport,
  getEvidenceLevel,
  getFreshnessPolicy,
  getFreshnessState,
  resolveEvidenceConflict,
};
