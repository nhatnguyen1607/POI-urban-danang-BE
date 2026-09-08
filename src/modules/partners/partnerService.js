const { ProviderRegistry } = require('./providerRegistry');
const { createOfficialProviderShells } = require('./officialProviders');
const { PROVIDER_CAPABILITIES, PROVIDER_RESULT_STATUSES, publicProviderResult } = require('./providerContract');
const { createHandoff } = require('./handoff');
const { createMerchantVerificationStore } = require('./merchantVerification');
const { resolvePlaceOperationalState } = require('./operationalStateResolver');

function isAccommodation(place = {}) {
  return /(hotel|hostel|homestay|resort|accommodation|lodging|khach san|nha nghi)/i.test(String(place.categoryNormalized || place.category || ''));
}

function isFoodPlace(place = {}) {
  return /(restaurant|food|cafe|coffee|bakery|bar|quan an|nha hang)/i.test(String(place.categoryNormalized || place.category || ''));
}

function resolveAvailability(providerResults, operational, now = new Date()) {
  const result = providerResults.find((item) => [
    PROVIDER_RESULT_STATUSES.AVAILABLE,
    PROVIDER_RESULT_STATUSES.SOLD_OUT,
    PROVIDER_RESULT_STATUSES.UNAVAILABLE,
  ].includes(item.status));
  if (operational.conflict) return { state: PROVIDER_RESULT_STATUSES.CONFLICT, result };
  if (!result) return { state: 'UNVERIFIED', result: null };
  const validUntil = result.validUntil ? new Date(result.validUntil).getTime() : NaN;
  const currentTime = now instanceof Date ? now.getTime() : new Date(now).getTime();
  return {
    state: Number.isFinite(validUntil) && Number.isFinite(currentTime) && validUntil >= currentTime
      ? result.status
      : PROVIDER_RESULT_STATUSES.STALE,
    result,
  };
}

function createPartnerService({ env = process.env, providers, merchantRecords = [] } = {}) {
  const registry = new ProviderRegistry();
  for (const provider of providers || createOfficialProviderShells(env)) registry.register(provider);
  const merchants = createMerchantVerificationStore(merchantRecords);

  async function resolve(place, context = {}) {
    const capability = isAccommodation(place)
      ? PROVIDER_CAPABILITIES.AVAILABILITY
      : isFoodPlace(place)
        ? PROVIDER_CAPABILITIES.LIVE_STATUS
        : PROVIDER_CAPABILITIES.LIVE_STATUS;
    const providerResults = await registry.resolve(place, capability, context);
    const handoffCapabilities = isAccommodation(place)
      ? [PROVIDER_CAPABILITIES.BOOKING_HANDOFF]
      : isFoodPlace(place)
        ? [PROVIDER_CAPABILITIES.FOOD_HANDOFF]
        : [];
    const handoffResults = (await Promise.all(handoffCapabilities.map((item) => registry.resolve(place, item, context)))).flat();
    const handoff = handoffResults.map((item) => createHandoff(item, {
      placeId: place.globalId || place.id,
      destination: context.destination,
      requestedDates: context.requestedDates,
      guestCount: context.guestCount,
    })).find(Boolean) || null;
    const merchant = merchants.findVerified(String(place.globalId || place.id || ''));
    const operational = resolvePlaceOperationalState(place, { providerResults, merchant, userEvidence: context.userEvidence, now: context.now });
    const resolvedAvailability = isAccommodation(place)
      ? resolveAvailability(providerResults, operational, context.now || new Date())
      : { state: 'NOT_APPLICABLE', result: null };
    const availabilityResult = resolvedAvailability.result;
    return {
      placeId: String(place.globalId || place.id || ''),
      canonical: place.canonical !== false,
      providerStatus: providerResults.some((item) => item.status === PROVIDER_RESULT_STATUSES.PROVIDER_UNAVAILABLE)
        ? PROVIDER_RESULT_STATUSES.PROVIDER_UNAVAILABLE
        : providerResults[0]?.status || PROVIDER_RESULT_STATUSES.UNKNOWN,
      availability: {
        state: resolvedAvailability.state,
        observedAt: availabilityResult?.observedAt || null,
        validUntil: availabilityResult?.validUntil || null,
        providerName: availabilityResult?.providerName || null,
        price: availabilityResult?.price ?? null,
        currency: availabilityResult?.currency || null,
      },
      handoff,
      operational,
      providers: providerResults.map(publicProviderResult),
    };
  }

  return { registry, merchants, resolve };
}

const partnerService = createPartnerService();

module.exports = { createPartnerService, isAccommodation, isFoodPlace, partnerService, resolveAvailability };
