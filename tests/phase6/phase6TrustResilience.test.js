const assert = require('node:assert/strict');
const test = require('node:test');
const { EventEmitter } = require('node:events');

const {
  EVIDENCE_LEVELS,
  EVIDENCE_STATUSES,
  FRESHNESS_STATES,
  SOURCE_TYPES,
  createEvidence,
  evidenceFromVerifiedUserReport,
  getEvidenceLevel,
  getFreshnessState,
  resolveEvidenceConflict,
} = require('../../src/modules/trust/evidence');
const { buildLiveStatusOverlay } = require('../../src/modules/trust/liveStatusOverlay');
const { createProviderResilience } = require('../../src/infrastructure/external/providerResilience');
const { createEndpointRateLimit } = require('../../src/middleware/endpointRateLimit');
const { createEndpointConcurrencyLimit } = require('../../src/middleware/endpointConcurrencyLimit');
const { resolveDestinationSearch } = require('../../src/services/destinationGeocodingService');
const { evidenceFromFeedback, sanitizeFeedback } = require('../../src/services/feedbackService');
const { scoreBusinessLocations } = require('../../src/services/businessLocationScorer');
const { generateLocalInsight } = require('../../src/services/businessInsightGenerator');

const NOW = new Date('2026-09-04T08:00:00.000Z');

function evidence(status, ageSeconds = 30, sourceType = SOURCE_TYPES.OFFICIAL_PARTNER) {
  return createEvidence({
    sourceName: sourceType,
    sourceType,
    status,
    observedAt: new Date(NOW.getTime() - ageSeconds * 1000),
    lastVerifiedAt: new Date(NOW.getTime() - ageSeconds * 1000),
    ttlSeconds: 60,
    verificationMethod: 'test',
  });
}

test('freshness and evidence levels do not turn stale observations into live claims', () => {
  const fresh = evidence(EVIDENCE_STATUSES.OPEN);
  const stale = evidence(EVIDENCE_STATUSES.OPEN, 120);
  assert.equal(getFreshnessState(fresh, NOW), FRESHNESS_STATES.FRESH);
  assert.equal(getEvidenceLevel(fresh, NOW), EVIDENCE_LEVELS.LIVE_VERIFIED);
  assert.equal(getFreshnessState(stale, NOW), FRESHNESS_STATES.EXPIRED);
  assert.equal(getEvidenceLevel(stale, NOW), EVIDENCE_LEVELS.STALE_UNVERIFIED);
});

test('conflicting recent evidence is explicit and blocks a current-status claim', () => {
  const resolved = resolveEvidenceConflict([
    evidence(EVIDENCE_STATUSES.OPEN),
    evidence(EVIDENCE_STATUSES.CLOSED, 20, SOURCE_TYPES.MERCHANT_VERIFIED),
  ], NOW);
  assert.equal(resolved.status, EVIDENCE_STATUSES.CONFLICT);
  const overlay = buildLiveStatusOverlay({ openingHoursRaw: '08:00-18:00' }, {
    evidence: resolved.evidence,
    now: NOW,
  });
  assert.equal(overlay.conflict, true);
  assert.equal(overlay.currentStatusVerified, false);
});

test('canonical hours and hotel metadata remain reference-only', () => {
  const hours = buildLiveStatusOverlay({ category: 'Cafe', openingHoursRaw: '08:00-18:00' }, { now: NOW });
  assert.equal(hours.openingHours.state, 'REFERENCE_ONLY');
  assert.equal(hours.openingHours.currentStatusClaimAllowed, false);
  const hotel = buildLiveStatusOverlay({ category: 'Hotel' }, { now: NOW });
  assert.equal(hotel.availability.state, 'UNVERIFIED');
  assert.equal(hotel.availability.handoffRequired, true);
});

test('only GPS-verified traveler status feedback becomes non-canonical evidence', () => {
  const unverified = evidenceFromVerifiedUserReport({ poiId: 'p1', status: 'OPEN', observedAt: NOW });
  assert.equal(unverified, null);
  const event = sanitizeFeedback({
    userId: 'u1',
    eventType: 'poi_status_report',
    poiId: 'p1',
    payload: { status: 'OPEN', observedAt: NOW.toISOString(), verifiedGps: true },
  });
  const verified = evidenceFromFeedback(event);
  assert.equal(verified.sourceType, SOURCE_TYPES.URBANAGENT_USER_EVIDENCE);
  assert.equal(verified.provenance.canonicalMutation, false);
});

test('provider guard coalesces and caches identical requests', async () => {
  let calls = 0;
  const events = [];
  const guard = createProviderResilience({
    provider: 'test',
    cacheTtlMs: 1000,
    maxRetries: 0,
    logger: (event) => events.push(event),
  });
  const operation = async () => {
    calls += 1;
    await new Promise((resolve) => setTimeout(resolve, 15));
    return { ok: true };
  };
  const identical = await Promise.all(Array.from({ length: 20 }, () => guard.execute(
    'private:16.0544,108.2022:auth-token-must-not-leak',
    operation,
  )));
  const followUp = await guard.execute('private:16.0544,108.2022:auth-token-must-not-leak', operation);
  assert.ok(identical.every((value) => value.ok));
  assert.deepEqual(followUp, identical[0]);
  assert.equal(calls, 1);
  assert.equal(guard.snapshot().coalesced, 19);
  assert.equal(guard.snapshot().cacheHits, 1);
  const operationalLog = JSON.stringify(events);
  assert.doesNotMatch(operationalLog, /16\.0544|108\.2022|auth-token-must-not-leak|private:/);
  assert.match(operationalLog, /keyHash/);
});

test('provider guard retries transient failures and opens its circuit', async () => {
  let attempts = 0;
  const retrying = createProviderResilience({
    provider: 'retry',
    maxRetries: 1,
    retryBaseMs: 1,
    random: () => 0,
  });
  const value = await retrying.execute('retry', async () => {
    attempts += 1;
    if (attempts === 1) throw new Error('temporary');
    return 'ok';
  });
  assert.equal(value, 'ok');
  assert.equal(retrying.snapshot().retries, 1);

  const circuit = createProviderResilience({
    provider: 'circuit',
    maxRetries: 0,
    circuitFailureThreshold: 1,
    circuitResetMs: 10_000,
  });
  await assert.rejects(circuit.execute('one', async () => { throw new Error('down'); }));
  await assert.rejects(circuit.execute('two', async () => 'never'), { code: 'PROVIDER_CIRCUIT_OPEN' });
});

test('provider guard enforces timeout and bounded concurrency', async () => {
  const timeout = createProviderResilience({ provider: 'timeout', timeoutMs: 50, maxRetries: 0 });
  await assert.rejects(
    timeout.execute('slow', ({ signal }) => new Promise((resolve, reject) => {
      signal.addEventListener('abort', () => reject(Object.assign(new Error('aborted'), { code: 'ABORTED' })));
    })),
    { code: 'PROVIDER_TIMEOUT' },
  );
  const bulkhead = createProviderResilience({ provider: 'bulkhead', maxConcurrency: 1, maxRetries: 0 });
  let release;
  const held = bulkhead.execute('held', () => new Promise((resolve) => { release = resolve; }));
  await assert.rejects(bulkhead.execute('other', async () => 'no'), { code: 'PROVIDER_CONCURRENCY_LIMIT' });
  release('yes');
  await held;
});

test('provider circuit allows only one half-open probe', async () => {
  let currentTime = 1_000;
  const guard = createProviderResilience({
    provider: 'half-open',
    maxRetries: 0,
    circuitFailureThreshold: 1,
    circuitResetMs: 100,
    now: () => currentTime,
  });
  await assert.rejects(guard.execute('failure', async () => { throw new Error('down'); }));
  currentTime += 101;
  let releaseProbe;
  const probe = guard.execute('probe', () => new Promise((resolve) => { releaseProbe = resolve; }));
  await assert.rejects(guard.execute('second-probe', async () => 'not-run'), { code: 'PROVIDER_CIRCUIT_OPEN' });
  releaseProbe('healthy');
  assert.equal(await probe, 'healthy');
  assert.equal(guard.snapshot().circuitState, 'CLOSED');
});

test('endpoint admission has bounded active work, no pending queue, and fails fast with 503', () => {
  const limiter = createEndpointConcurrencyLimit({ maxActive: 2, retryAfterSeconds: 3, logger: () => {} });
  const response = () => {
    const res = new EventEmitter();
    res.headers = {};
    res.set = (key, value) => { res.headers[key] = value; return res; };
    res.status = (status) => { res.statusCode = status; return res; };
    res.json = (payload) => { res.payload = payload; return res; };
    return res;
  };
  const first = response();
  const second = response();
  assert.equal(limiter({}, first, () => 'first'), 'first');
  assert.equal(limiter({}, second, () => 'second'), 'second');
  const rejected = response();
  limiter({}, rejected, () => 'must-not-run');
  assert.equal(rejected.statusCode, 503);
  assert.equal(rejected.headers['Retry-After'], '3');
  assert.equal(rejected.payload.error, 'CAPACITY_EXHAUSTED');
  assert.deepEqual(limiter.snapshot(), {
    name: 'endpoint', maxActive: 2, maxPending: 0, active: 2, pending: 0, peakActive: 2, rejected: 1,
  });
  first.emit('finish');
  second.emit('close');
  assert.equal(limiter.snapshot().active, 0);
});

test('normal destination search without Google credentials calls Photon only', async () => {
  const urls = [];
  const result = await resolveDestinationSearch({
    query: 'Cầu Rồng',
    googleApiKey: '',
    googleMapCompliant: false,
    fetchImpl: async (url) => {
      urls.push(String(url));
      return {
        ok: true,
        json: async () => ({
          features: [{
            geometry: { coordinates: [108.227, 16.061] },
            properties: {
              osm_type: 'W', osm_id: 1, name: 'Cầu Rồng', city: 'Đà Nẵng', country: 'Việt Nam',
            },
          }],
        }),
      };
    },
  });
  assert.equal(result.meta.googleConfigured, false);
  assert.equal(result.meta.source, 'photon');
  assert.equal(urls.length, 1);
  assert.match(urls[0], /photon\.komoot\.io/);
  assert.doesNotMatch(urls[0], /(?:places|maps)\.googleapis\.com/);
});

test('endpoint limiter returns 429 and Retry-After without exposing client identity', () => {
  let time = 1000;
  const middleware = createEndpointRateLimit({ maxRequests: 1, windowMs: 10_000, now: () => time, key: () => 'secret-user' });
  const req = {};
  const result = { status: null, headers: {}, payload: null };
  const res = {
    set: (key, value) => { result.headers[key] = value; },
    status: (status) => { result.status = status; return res; },
    json: (payload) => { result.payload = payload; return res; },
  };
  assert.equal(middleware(req, res, () => 'allowed'), 'allowed');
  middleware(req, res, () => 'unexpected');
  assert.equal(result.status, 429);
  assert.equal(result.headers['Retry-After'], '10');
  assert.equal(JSON.stringify(result.payload).includes('secret-user'), false);
});

test('business output uses neutral evidence ranking and a verification checklist', async () => {
  const result = await scoreBusinessLocations({ concept: 'study cafe', limit: 1 });
  assert.equal(result.areas.length, 1);
  assert.equal(result.areas[0].rankingPosition, 1);
  assert.equal('score' in result.areas[0], false);
  assert.equal('scoreRaw' in result.areas[0], false);
  const area = {
    ...result.areas[0],
    evidence: {
      rawCounts: { poiTotalInArea: result.areas[0].totalPOIs },
      signals: result.areas[0].signals,
      topCategories: [],
      complementaryPOIs: [],
      competitors: [],
      routeWarnings: [],
      samplePOIs: [],
    },
  };
  const insight = generateLocalInsight(area, 'en');
  assert.equal(Array.isArray(insight.verification_checklist), true);
  assert.equal('recommended_actions' in insight, false);
  assert.doesNotMatch(JSON.stringify(insight), /\bscore(?:d|s)?\s+\d+\/100/i);
  assert.doesNotMatch(
    JSON.stringify(insight),
    /investment score|nên đầu tư|nên mở|best place to invest|recommended investment/i,
  );
});
