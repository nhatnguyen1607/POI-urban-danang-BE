const { PROVIDER_HEALTH, PROVIDER_RESULT_STATUSES, createProviderResult } = require('./providerContract');

class ProviderRegistry {
  constructor() {
    this.providers = new Map();
  }

  register(provider) {
    if (!provider?.id || typeof provider.supports !== 'function') {
      throw new TypeError('Provider must expose id and supports().');
    }
    if (this.providers.has(provider.id)) throw new Error(`Provider already registered: ${provider.id}`);
    this.providers.set(provider.id, provider);
    return this;
  }

  list() {
    return [...this.providers.values()].map((provider) => {
      const snapshot = typeof provider.health === 'function' ? provider.health() : {};
      return {
        providerId: provider.id,
        providerName: provider.name || provider.id,
        enabled: provider.enabled === true,
        capabilities: [...(provider.capabilities || [])],
        status: provider.enabled === true ? (snapshot.status || PROVIDER_HEALTH.ENABLED) : PROVIDER_HEALTH.DISABLED,
        lastSuccessAt: snapshot.lastSuccessAt || null,
        lastErrorCategory: snapshot.lastErrorCategory || null,
        circuitState: snapshot.circuitState || 'CLOSED',
      };
    });
  }

  supporting(place, capability) {
    return [...this.providers.values()].filter((provider) => provider.supports(place, capability));
  }

  async resolve(place, capability, context = {}) {
    const supported = this.supporting(place, capability);
    if (!supported.length) {
      return [createProviderResult({
        capability,
        status: PROVIDER_RESULT_STATUSES.NOT_SUPPORTED,
        evidence: { reason: 'no_registered_provider_supports_place' },
      })];
    }
    const enabled = supported.filter((provider) => provider.enabled === true);
    if (!enabled.length) {
      return supported.map((provider) => createProviderResult({
        providerId: provider.id,
        providerName: provider.name,
        capability,
        status: PROVIDER_RESULT_STATUSES.PROVIDER_UNAVAILABLE,
        evidence: { reason: 'provider_disabled_or_not_configured' },
      }));
    }
    return Promise.all(enabled.map((provider) => provider.query(place, capability, context)));
  }
}

module.exports = { ProviderRegistry };
