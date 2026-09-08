const {
  EVIDENCE_STATUSES,
  FRESHNESS_STATES,
  SOURCE_TYPES,
  createEvidence,
  resolveEvidenceConflict,
} = require('../trust/evidence');
const { PROVIDER_RESULT_STATUSES } = require('./providerContract');

function providerEvidence(result) {
  const mapped = {
    [PROVIDER_RESULT_STATUSES.AVAILABLE]: EVIDENCE_STATUSES.AVAILABLE,
    [PROVIDER_RESULT_STATUSES.UNAVAILABLE]: EVIDENCE_STATUSES.CLOSED,
    [PROVIDER_RESULT_STATUSES.SOLD_OUT]: EVIDENCE_STATUSES.SOLD_OUT,
    [PROVIDER_RESULT_STATUSES.CLOSED]: EVIDENCE_STATUSES.CLOSED,
  }[result?.status] || EVIDENCE_STATUSES.UNKNOWN;
  return createEvidence({
    sourceId: result?.providerId,
    sourceName: result?.providerName,
    sourceType: SOURCE_TYPES.OFFICIAL_PARTNER,
    sourceUrl: result?.sourceUrl,
    dataClass: result?.capability === 'AVAILABILITY' ? 'hotelAvailability' : 'restaurantOpenNow',
    status: mapped,
    observedAt: result?.observedAt,
    validUntil: result?.validUntil,
    ttlSeconds: result?.evidence?.ttlSeconds || 300,
    verificationMethod: 'official_partner_api',
    confidenceReason: result?.evidence?.confidenceReason || 'Official partner response.',
    provenance: { providerId: result?.providerId, rawPayloadExposed: false, canonicalMutation: false },
  });
}

function merchantEvidence(record) {
  if (!record || record.claimStatus !== 'VERIFIED') return null;
  return createEvidence({
    sourceId: record.merchantId,
    sourceName: 'Verified merchant',
    sourceType: SOURCE_TYPES.MERCHANT_VERIFIED,
    dataClass: 'restaurantOpenNow',
    status: record.temporaryClosure ? EVIDENCE_STATUSES.TEMPORARILY_CLOSED : record.submittedStatus,
    observedAt: record.verifiedAt,
    lastVerifiedAt: record.verifiedAt,
    ttlSeconds: 1800,
    verificationMethod: 'admin_verified_merchant_claim',
    confidenceReason: 'Merchant evidence passed manual verification.',
    provenance: { merchantId: record.merchantId, canonicalMutation: false },
  });
}

function resolvePlaceOperationalState(place = {}, context = {}) {
  const providerItems = (context.providerResults || []).map(providerEvidence);
  const merchantItem = merchantEvidence(context.merchant);
  const evidence = [...providerItems, ...(merchantItem ? [merchantItem] : []), ...(context.userEvidence || [])];
  const resolved = resolveEvidenceConflict(evidence, context.now || new Date());
  const freshClosed = !resolved.conflict
    && [EVIDENCE_STATUSES.CLOSED, EVIDENCE_STATUSES.TEMPORARILY_CLOSED, EVIDENCE_STATUSES.SOLD_OUT].includes(resolved.status)
    && resolved.selected?.freshnessState === FRESHNESS_STATES.FRESH;
  const stale = resolved.selected && resolved.selected.freshnessState !== FRESHNESS_STATES.FRESH;
  return {
    placeId: String(place.globalId || place.id || ''),
    canonical: place.canonical !== false,
    status: resolved.status,
    evidenceLevel: resolved.selected?.evidenceLevel || 'STALE_UNVERIFIED',
    freshnessState: resolved.selected?.freshnessState || FRESHNESS_STATES.UNKNOWN,
    conflict: resolved.conflict,
    conflictSources: resolved.conflictSources || [],
    action: freshClosed ? 'AVOID_AUTOMATICALLY' : resolved.conflict ? 'ALLOW_WITH_WARNING' : stale ? 'ALLOW_WITH_STALE_WARNING' : 'ALLOW_UNVERIFIED',
    revalidateBeforeNavigation: !resolved.selected || stale || resolved.conflict,
    navigationBlocked: freshClosed,
    canonicalMutation: false,
    evidence: resolved.evidence,
  };
}

function aggregateVerifiedUserEvidence(items = []) {
  const verified = items.filter((item) => item?.sourceType === SOURCE_TYPES.URBANAGENT_USER_EVIDENCE);
  const closedReports = verified.filter((item) => [EVIDENCE_STATUSES.CLOSED, EVIDENCE_STATUSES.TEMPORARILY_CLOSED].includes(item.status));
  return {
    verifiedReportCount: verified.length,
    closedReportCount: closedReports.length,
    verificationFlagRequired: closedReports.length >= 2,
    canonicalMutation: false,
  };
}

module.exports = { aggregateVerifiedUserEvidence, merchantEvidence, providerEvidence, resolvePlaceOperationalState };
