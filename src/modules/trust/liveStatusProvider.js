const { EVIDENCE_STATUSES } = require('./evidence');

class LiveStatusProvider {
  supports() {
    return false;
  }

  async getStatus() {
    return { status: EVIDENCE_STATUSES.PROVIDER_UNAVAILABLE, evidence: [] };
  }

  async getAvailability() {
    return { status: EVIDENCE_STATUSES.PROVIDER_UNAVAILABLE, evidence: [] };
  }
}

class UnavailableLiveStatusProvider extends LiveStatusProvider {
  constructor(name = 'unavailable') {
    super();
    this.name = name;
  }
}

module.exports = { LiveStatusProvider, UnavailableLiveStatusProvider };
