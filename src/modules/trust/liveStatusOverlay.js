const {
  EVIDENCE_LEVELS,
  EVIDENCE_STATUSES,
  FRESHNESS_STATES,
  SOURCE_TYPES,
  createEvidence,
  getEvidenceLevel,
  getFreshnessPolicy,
  getFreshnessState,
  resolveEvidenceConflict,
} = require('./evidence');
const { providerEvidence, merchantEvidence } = require('../partners/operationalStateResolver');
const { PROVIDER_RESULT_STATUSES } = require('../partners/providerContract');

function normalizedCategory(poi = {}) {
  return String(poi.categoryNormalized || poi.category || '').toLowerCase();
}

function isAccommodation(poi) {
  return /(hotel|hostel|homestay|resort|accommodation|lodging|khach san|nha nghi)/.test(normalizedCategory(poi));
}

function canonicalEvidence(poi, policy) {
  const hasHours = Boolean(String(poi.openingHoursRaw || '').trim());
  return createEvidence({
    sourceId: poi.globalId || poi.id || null,
    sourceName: 'UrbanAgent canonical dataset',
    sourceType: SOURCE_TYPES.CANONICAL_DATASET,
    dataClass: hasHours ? 'recurringOpeningHours' : 'canonicalMetadata',
    observedAt: null,
    lastVerifiedAt: null,
    ttlSeconds: hasHours ? policy.recurringOpeningHours : policy.canonicalMetadata,
    verificationMethod: hasHours ? 'imported_recurring_schedule' : 'canonical_import',
    status: EVIDENCE_STATUSES.UNKNOWN,
    confidenceReason: hasHours
      ? 'Recurring hours exist, but no current-status observation is available.'
      : 'Stable identity metadata is available; current operating status is unverified.',
    provenance: {
      source: poi.source || null,
      canonical: poi.canonical !== false,
      canonicalMutation: false,
    },
  });
}

function trustMessage({ resolved, hasHours, accommodation }) {
  if (resolved.conflict) return 'Các nguồn hiện đang không thống nhất. Hãy xác minh trước khi quyết định.';
  if (accommodation) return 'Chưa xác minh tình trạng phòng trống.';
  if (resolved.status === EVIDENCE_STATUSES.OPEN && resolved.selected?.freshnessState === FRESHNESS_STATES.FRESH) {
    return 'Trạng thái hiện tại đã được xác minh gần đây.';
  }
  if (hasHours) return 'Giờ mở cửa tham khảo. Vui lòng kiểm tra trước khi đến.';
  return 'Chưa xác minh trạng thái hiện tại.';
}

function buildLiveStatusOverlay(poi = {}, {
  evidence = [], providerResults = [], merchant = null, handoff = null,
  now = new Date(), policy = getFreshnessPolicy(),
} = {}) {
  const baseEvidence = canonicalEvidence(poi, policy);
  const partnerEvidence = providerResults.map(providerEvidence);
  const verifiedMerchantEvidence = merchantEvidence(merchant);
  const resolved = resolveEvidenceConflict([
    baseEvidence,
    ...partnerEvidence,
    ...(verifiedMerchantEvidence ? [verifiedMerchantEvidence] : []),
    ...evidence,
  ], now);
  const hasHours = Boolean(String(poi.openingHoursRaw || '').trim());
  const accommodation = isAccommodation(poi);
  const selected = resolved.selected;
  const freshnessState = selected ? getFreshnessState(selected, now) : FRESHNESS_STATES.UNKNOWN;
  const evidenceLevel = selected ? getEvidenceLevel(selected, now) : EVIDENCE_LEVELS.STALE_UNVERIFIED;
  const currentStatusVerified = freshnessState === FRESHNESS_STATES.FRESH
    && !resolved.conflict
    && selected?.status !== EVIDENCE_STATUSES.UNKNOWN;
  const availabilityResult = providerResults.find((item) => [
    PROVIDER_RESULT_STATUSES.AVAILABLE,
    PROVIDER_RESULT_STATUSES.SOLD_OUT,
    PROVIDER_RESULT_STATUSES.UNAVAILABLE,
  ].includes(item.status));
  const availabilityFreshness = availabilityResult?.validUntil
    && new Date(availabilityResult.validUntil).getTime() >= new Date(now).getTime()
    ? FRESHNESS_STATES.FRESH
    : availabilityResult ? FRESHNESS_STATES.STALE : FRESHNESS_STATES.UNKNOWN;
  const availabilityState = !accommodation
    ? 'NOT_APPLICABLE'
    : resolved.conflict
      ? PROVIDER_RESULT_STATUSES.CONFLICT
      : availabilityResult && availabilityFreshness === FRESHNESS_STATES.FRESH
        ? availabilityResult.status
        : availabilityResult
          ? PROVIDER_RESULT_STATUSES.STALE
          : 'UNVERIFIED';
  return {
    status: resolved.status,
    freshnessState,
    evidenceLevel,
    currentStatusVerified,
    conflict: resolved.conflict,
    conflictSources: resolved.conflictSources || [],
    message: trustMessage({ resolved, hasHours, accommodation }),
    openingHours: {
      state: hasHours ? 'REFERENCE_ONLY' : 'UNKNOWN',
      currentStatusClaimAllowed: currentStatusVerified && resolved.status === EVIDENCE_STATUSES.OPEN,
    },
    availability: {
      state: availabilityState,
      handoffRequired: accommodation,
      providerStatus: providerResults[0]?.status || PROVIDER_RESULT_STATUSES.PROVIDER_UNAVAILABLE,
      providerName: availabilityResult?.providerName || null,
      observedAt: availabilityResult?.observedAt || null,
      validUntil: availabilityResult?.validUntil || null,
      price: availabilityResult?.price ?? null,
      currency: availabilityResult?.currency || null,
    },
    handoff,
    decision: {
      eligible: ![
        EVIDENCE_STATUSES.CLOSED,
        EVIDENCE_STATUSES.TEMPORARILY_CLOSED,
        EVIDENCE_STATUSES.SOLD_OUT,
      ].includes(resolved.status),
      revalidationRequired: !currentStatusVerified || resolved.conflict,
      canonicalMutation: false,
    },
    evidence: resolved.evidence.map((item) => ({
      sourceName: item.sourceName,
      sourceType: item.sourceType,
      observedAt: item.observedAt,
      lastVerifiedAt: item.lastVerifiedAt,
      validUntil: item.validUntil,
      verificationMethod: item.verificationMethod,
      status: item.status,
      evidenceLevel: item.evidenceLevel,
      freshnessState: item.freshnessState,
      confidenceReason: item.confidenceReason,
      conflict: item.conflict,
    })),
  };
}

module.exports = { buildLiveStatusOverlay, canonicalEvidence, isAccommodation };
