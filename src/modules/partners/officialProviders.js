const { createProviderResilience } = require('../../infrastructure/external/providerResilience');
const {
  PROVIDER_CAPABILITIES,
  PROVIDER_HEALTH,
  PROVIDER_RESULT_STATUSES,
  createProviderResult,
} = require('./providerContract');

class OfficialPartnerProvider {
  constructor({ id, name, capabilities, enabled, credentialPresent, request, resilience } = {}) {
    this.id = id;
    this.name = name;
    this.capabilities = new Set(capabilities || []);
    this.enabled = enabled === true && credentialPresent === true && typeof request === 'function';
    this.request = request;
    this.lastSuccessAt = null;
    this.lastErrorCategory = null;
    this.guard = resilience || createProviderResilience({
      provider: id,
      maxConcurrency: 3,
      timeoutMs: 5_000,
      maxRetries: 1,
      circuitFailureThreshold: 3,
      cacheTtlMs: 60_000,
    });
  }

  supports(_place, capability) {
    return this.capabilities.has(capability);
  }

  health() {
    const snapshot = this.guard.snapshot();
    return {
      status: !this.enabled
        ? PROVIDER_HEALTH.DISABLED
        : snapshot.circuitState === 'OPEN'
          ? PROVIDER_HEALTH.OPEN_CIRCUIT
          : this.lastErrorCategory
            ? PROVIDER_HEALTH.DEGRADED
            : PROVIDER_HEALTH.ENABLED,
      lastSuccessAt: this.lastSuccessAt,
      lastErrorCategory: this.lastErrorCategory,
      circuitState: snapshot.circuitState,
    };
  }

  async query(place, capability, context = {}) {
    if (!this.enabled) {
      return createProviderResult({ providerId: this.id, providerName: this.name, capability, status: PROVIDER_RESULT_STATUSES.NOT_CONFIGURED });
    }
    try {
      const response = await this.guard.execute(`${place?.id || place?.globalId || 'place'}:${capability}`, ({ signal }) => (
        this.request({ place, capability, context, signal })
      ));
      this.lastSuccessAt = new Date().toISOString();
      this.lastErrorCategory = null;
      return createProviderResult({ ...response, providerId: this.id, providerName: this.name, capability });
    } catch (error) {
      this.lastErrorCategory = error?.code || 'PROVIDER_FAILED';
      return createProviderResult({
        providerId: this.id,
        providerName: this.name,
        capability,
        status: PROVIDER_RESULT_STATUSES.PROVIDER_UNAVAILABLE,
        evidence: { errorCategory: this.lastErrorCategory },
      });
    }
  }
}

function envGate(env, prefix) {
  return env[`${prefix}_ENABLED`] === 'true' && Boolean(String(env[`${prefix}_API_KEY`] || '').trim());
}

function createOfficialProviderShells(env = process.env) {
  return [
    new OfficialPartnerProvider({
      id: 'booking_availability', name: 'Booking availability partner',
      capabilities: [PROVIDER_CAPABILITIES.AVAILABILITY, PROVIDER_CAPABILITIES.PRICE, PROVIDER_CAPABILITIES.BOOKING_HANDOFF],
      enabled: envGate(env, 'URBANAGENT_BOOKING_PARTNER'),
      credentialPresent: Boolean(env.URBANAGENT_BOOKING_PARTNER_API_KEY),
    }),
    new OfficialPartnerProvider({
      id: 'traveloka_availability', name: 'Traveloka availability partner',
      capabilities: [PROVIDER_CAPABILITIES.AVAILABILITY, PROVIDER_CAPABILITIES.PRICE, PROVIDER_CAPABILITIES.BOOKING_HANDOFF],
      enabled: envGate(env, 'URBANAGENT_TRAVELOKA_PARTNER'),
      credentialPresent: Boolean(env.URBANAGENT_TRAVELOKA_PARTNER_API_KEY),
    }),
    new OfficialPartnerProvider({
      id: 'merchant_food_status', name: 'Merchant food service partner',
      capabilities: [PROVIDER_CAPABILITIES.LIVE_STATUS, PROVIDER_CAPABILITIES.FOOD_HANDOFF],
      enabled: envGate(env, 'URBANAGENT_MERCHANT_FOOD_PARTNER'),
      credentialPresent: Boolean(env.URBANAGENT_MERCHANT_FOOD_PARTNER_API_KEY),
    }),
  ];
}

module.exports = { OfficialPartnerProvider, createOfficialProviderShells, envGate };
