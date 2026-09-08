const PROVIDER_CAPABILITIES = Object.freeze({
  LIVE_STATUS: 'LIVE_STATUS',
  AVAILABILITY: 'AVAILABILITY',
  PRICE: 'PRICE',
  BOOKING_HANDOFF: 'BOOKING_HANDOFF',
  FOOD_HANDOFF: 'FOOD_HANDOFF',
  RIDE_HANDOFF: 'RIDE_HANDOFF',
});

const PROVIDER_RESULT_STATUSES = Object.freeze({
  AVAILABLE: 'AVAILABLE',
  UNAVAILABLE: 'UNAVAILABLE',
  UNKNOWN: 'UNKNOWN',
  SOLD_OUT: 'SOLD_OUT',
  CLOSED: 'CLOSED',
  PROVIDER_UNAVAILABLE: 'PROVIDER_UNAVAILABLE',
  AUTH_REQUIRED: 'AUTH_REQUIRED',
  NOT_SUPPORTED: 'NOT_SUPPORTED',
  NOT_CONFIGURED: 'NOT_CONFIGURED',
  STALE: 'STALE',
  CONFLICT: 'CONFLICT',
});

const PROVIDER_HEALTH = Object.freeze({
  ENABLED: 'ENABLED',
  DISABLED: 'DISABLED',
  DEGRADED: 'DEGRADED',
  OPEN_CIRCUIT: 'OPEN_CIRCUIT',
});

function cleanString(value, max = 1000) {
  return typeof value === 'string' ? value.trim().slice(0, max) : '';
}

function isoDate(value) {
  if (!value) return null;
  const date = value instanceof Date ? value : new Date(value);
  return Number.isFinite(date.getTime()) ? date.toISOString() : null;
}

function createProviderResult(input = {}) {
  const status = Object.values(PROVIDER_RESULT_STATUSES).includes(input.status)
    ? input.status
    : PROVIDER_RESULT_STATUSES.UNKNOWN;
  const capability = Object.values(PROVIDER_CAPABILITIES).includes(input.capability)
    ? input.capability
    : null;
  return {
    providerId: cleanString(input.providerId, 120) || 'unknown_provider',
    providerName: cleanString(input.providerName, 180) || 'Unknown provider',
    capability,
    status,
    observedAt: isoDate(input.observedAt),
    validUntil: isoDate(input.validUntil),
    sourceUrl: cleanString(input.sourceUrl) || null,
    price: Number.isFinite(Number(input.price)) ? Number(input.price) : null,
    currency: cleanString(input.currency, 12) || null,
    availability: input.availability === true ? true : input.availability === false ? false : null,
    handoffUrl: cleanString(input.handoffUrl) || null,
    evidence: input.evidence && typeof input.evidence === 'object' ? { ...input.evidence } : {},
    rawProviderId: cleanString(input.rawProviderId, 240) || null,
  };
}

function publicProviderResult(input = {}) {
  const result = createProviderResult(input);
  const { rawProviderId, ...publicResult } = result;
  return publicResult;
}

module.exports = {
  PROVIDER_CAPABILITIES,
  PROVIDER_HEALTH,
  PROVIDER_RESULT_STATUSES,
  createProviderResult,
  publicProviderResult,
};
