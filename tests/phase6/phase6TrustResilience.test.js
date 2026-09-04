const assert = require('node:assert/strict');
const test = require('node:test');

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
  const guard = createProviderResilience({ provider: 'test', cacheTtlMs: 1000, maxRetries: 0 });
  const operation = async () => {
    calls += 1;
    await new Promise((resolve) => setTimeout(resolve, 15));
    return { ok: true };
  };
  const [first, second] = await Promise.all([guard.execute('same', operation), guard.execute('same', operation)]);
  const third = await guard.execute('same', operation);
  assert.deepEqual(first, second);
  assert.deepEqual(second, third);
  assert.equal(calls, 1);
  assert.equal(guard.snapshot().coalesced, 1);
  assert.equal(guard.snapshot().cacheHits, 1);
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
});
