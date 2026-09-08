const assert = require('node:assert/strict');
const test = require('node:test');

const { createEvidence, EVIDENCE_STATUSES, SOURCE_TYPES } = require('../../src/modules/trust/evidence');
const { buildLiveStatusOverlay } = require('../../src/modules/trust/liveStatusOverlay');
const { validateHandoffUrl, createHandoff } = require('../../src/modules/partners/handoff');
const { CLAIM_STATES, createMerchantVerificationStore, merchantClaimResponse } = require('../../src/modules/partners/merchantVerification');
const { OfficialPartnerProvider, createOfficialProviderShells } = require('../../src/modules/partners/officialProviders');
const { resolvePlaceOperationalState, aggregateVerifiedUserEvidence } = require('../../src/modules/partners/operationalStateResolver');
const { PROVIDER_CAPABILITIES, PROVIDER_RESULT_STATUSES } = require('../../src/modules/partners/providerContract');
const { ProviderRegistry } = require('../../src/modules/partners/providerRegistry');
const { createPartnerService } = require('../../src/modules/partners/partnerService');

const NOW = new Date('2026-09-08T08:00:00.000Z');
const HOTEL = { id: 'hotel-1', category: 'Hotel', canonical: true };

test('provider registry reports disabled official adapters without fabricating availability', async () => {
  const shells = createOfficialProviderShells({});
  const registry = new ProviderRegistry();
  shells.forEach((provider) => registry.register(provider));
  assert.equal(registry.list().every((provider) => provider.enabled === false), true);
  const results = await registry.resolve(HOTEL, PROVIDER_CAPABILITIES.AVAILABILITY);
  assert.equal(results.every((result) => result.status === PROVIDER_RESULT_STATUSES.PROVIDER_UNAVAILABLE), true);
  assert.equal(results.some((result) => result.availability === true), false);
});

test('official adapter requires enable flag, credential, and request implementation', () => {
  assert.equal(createOfficialProviderShells({ URBANAGENT_BOOKING_PARTNER_ENABLED: 'true' })[0].enabled, false);
  assert.equal(createOfficialProviderShells({ URBANAGENT_BOOKING_PARTNER_API_KEY: 'present' })[0].enabled, false);
  const configured = new OfficialPartnerProvider({
    id: 'test-official', name: 'Test official', capabilities: [PROVIDER_CAPABILITIES.AVAILABILITY],
    enabled: true, credentialPresent: true, request: async () => ({ status: PROVIDER_RESULT_STATUSES.AVAILABLE }),
  });
  assert.equal(configured.enabled, true);
});

test('handoff rejects non-https and unknown domains while preserving approved context', () => {
  for (const url of ['javascript:alert(1)', 'data:text/plain,no', 'https://attacker.example/redirect']) {
    assert.equal(validateHandoffUrl(url).valid, false);
  }
  const handoff = createHandoff({
    providerId: 'booking', providerName: 'Booking partner',
    capability: PROVIDER_CAPABILITIES.BOOKING_HANDOFF,
    status: PROVIDER_RESULT_STATUSES.AVAILABLE,
    handoffUrl: 'https://secure.booking.com/hotel?id=verified',
  }, { placeId: 'hotel-1', requestedDates: { checkIn: '2026-10-01', checkOut: '2026-10-03' }, guestCount: 2 });
  assert.equal(handoff.url.startsWith('https://secure.booking.com/'), true);
  assert.equal(handoff.placeId, 'hotel-1');
});

test('fresh, stale, and conflicting accommodation evidence remain explicit', () => {
  const fresh = {
    providerId: 'official', providerName: 'Official test provider',
    capability: PROVIDER_CAPABILITIES.AVAILABILITY,
    status: PROVIDER_RESULT_STATUSES.AVAILABLE,
    observedAt: '2026-09-08T07:59:00.000Z',
    validUntil: '2026-09-08T08:04:00.000Z',
  };
  const live = buildLiveStatusOverlay(HOTEL, { providerResults: [fresh], now: NOW });
  assert.equal(live.availability.state, PROVIDER_RESULT_STATUSES.AVAILABLE);
  assert.equal(live.currentStatusVerified, true);
  const stale = buildLiveStatusOverlay(HOTEL, {
    providerResults: [{ ...fresh, validUntil: '2026-09-08T07:59:30.000Z' }],
    now: NOW,
  });
  assert.equal(stale.availability.state, PROVIDER_RESULT_STATUSES.STALE);
  const conflict = buildLiveStatusOverlay(HOTEL, {
    providerResults: [fresh],
    merchant: {
      merchantId: 'merchant-1', placeId: 'hotel-1', claimStatus: CLAIM_STATES.VERIFIED,
      verifiedAt: '2026-09-08T07:59:30.000Z', submittedStatus: EVIDENCE_STATUSES.CLOSED,
    },
    now: NOW,
  });
  assert.equal(conflict.status, EVIDENCE_STATUSES.CONFLICT);
  assert.equal(conflict.availability.state, PROVIDER_RESULT_STATUSES.CONFLICT);
});

test('merchant claims require manual admin verification and cannot self-assign', () => {
  const store = createMerchantVerificationStore([{
    merchantId: 'm1', placeId: 'p1', claimStatus: CLAIM_STATES.PENDING,
    verifiedAt: NOW.toISOString(), verifiedBy: 'self',
  }]);
  assert.equal(store.findVerified('p1'), null);
  assert.deepEqual(merchantClaimResponse(), {
    status: 'MERCHANT_CLAIM_MANUAL_VERIFICATION_REQUIRED',
    autoVerification: false,
    selfAssignmentAllowed: false,
  });
});

test('operational resolver preserves temporary places and uses warning/avoid decisions', () => {
  const closed = {
    providerId: 'official', providerName: 'Official',
    capability: PROVIDER_CAPABILITIES.LIVE_STATUS,
    status: PROVIDER_RESULT_STATUSES.CLOSED,
    observedAt: NOW.toISOString(), validUntil: '2026-09-08T08:05:00.000Z',
  };
  const resolved = resolvePlaceOperationalState({ id: 'temp-1', canonical: false }, { providerResults: [closed], now: NOW });
  assert.equal(resolved.canonical, false);
  assert.equal(resolved.action, 'AVOID_AUTOMATICALLY');
  assert.equal(resolved.canonicalMutation, false);
});

test('verified GPS user reports aggregate into a verification flag without canonical mutation', () => {
  const reports = [1, 2].map((index) => createEvidence({
    sourceId: `report-${index}`, sourceType: SOURCE_TYPES.URBANAGENT_USER_EVIDENCE,
    sourceName: 'Verified traveler', status: EVIDENCE_STATUSES.CLOSED,
    observedAt: NOW, lastVerifiedAt: NOW, ttlSeconds: 3600,
  }));
  assert.deepEqual(aggregateVerifiedUserEvidence(reports), {
    verifiedReportCount: 2, closedReportCount: 2, verificationFlagRequired: true, canonicalMutation: false,
  });
});

test('provider outage is contained by existing resilience and returns unavailable state', async () => {
  const provider = new OfficialPartnerProvider({
    id: 'outage', name: 'Outage provider', capabilities: [PROVIDER_CAPABILITIES.AVAILABILITY],
    enabled: true, credentialPresent: true,
    resilience: require('../../src/infrastructure/external/providerResilience').createProviderResilience({
      provider: 'outage', timeoutMs: 50, maxRetries: 0, circuitFailureThreshold: 1,
    }),
    request: ({ signal }) => new Promise((resolve, reject) => signal.addEventListener('abort', () => reject(new Error('aborted')))),
  });
  const result = await provider.query(HOTEL, PROVIDER_CAPABILITIES.AVAILABILITY);
  assert.equal(result.status, PROVIDER_RESULT_STATUSES.PROVIDER_UNAVAILABLE);
  assert.equal(provider.health().status, 'OPEN_CIRCUIT');
});

test('default partner service keeps hotel availability unverified and all adapters disabled', async () => {
  const service = createPartnerService({ env: {} });
  const state = await service.resolve(HOTEL);
  assert.equal(state.availability.state, 'UNVERIFIED');
  assert.equal(state.handoff, null);
  assert.equal(service.registry.list().every((provider) => provider.enabled === false), true);
});

test('partner service never presents expired availability as live', async () => {
  const provider = {
    id: 'test-availability', name: 'Test availability', enabled: true,
    capabilities: new Set([PROVIDER_CAPABILITIES.AVAILABILITY]),
    supports: (_place, capability) => capability === PROVIDER_CAPABILITIES.AVAILABILITY,
    health: () => ({ status: 'ENABLED', circuitState: 'CLOSED' }),
    query: async () => ({
      providerId: 'test-availability', providerName: 'Test availability',
      capability: PROVIDER_CAPABILITIES.AVAILABILITY,
      status: PROVIDER_RESULT_STATUSES.AVAILABLE,
      observedAt: '2026-09-08T07:50:00.000Z', validUntil: '2026-09-08T07:55:00.000Z',
    }),
  };
  const state = await createPartnerService({ providers: [provider] }).resolve(HOTEL, { now: NOW });
  assert.equal(state.availability.state, PROVIDER_RESULT_STATUSES.STALE);
});

test('AUTO_CREATE_NEW remains false and canonical objects are not mutated', () => {
  const before = JSON.stringify(HOTEL);
  assert.notEqual(process.env.AUTO_CREATE_NEW, 'true');
  resolvePlaceOperationalState(HOTEL, { now: NOW });
  assert.equal(JSON.stringify(HOTEL), before);
});
