const crypto = require('node:crypto');

function operationalError(message, code, status = 503, retryAfterSeconds = null) {
  const error = new Error(message);
  error.code = code;
  error.status = status;
  error.retryAfterSeconds = retryAfterSeconds;
  return error;
}

function safeInteger(value, fallback, minimum = 0, maximum = Number.MAX_SAFE_INTEGER) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? Math.min(Math.max(Math.round(parsed), minimum), maximum) : fallback;
}

function defaultTransient(error) {
  const status = Number(error?.status || error?.statusCode || 0);
  if (status >= 400 && status < 500) return false;
  return !['INVALID_GEOCODE_QUERY', 'UNSUPPORTED_GEOCODER_CITY', 'UNAUTHORIZED', 'FORBIDDEN'].includes(error?.code);
}

function hashKey(value) {
  return crypto.createHash('sha256').update(String(value || 'default')).digest('hex').slice(0, 16);
}

function wait(ms, setTimer = setTimeout) {
  return new Promise((resolve) => setTimer(resolve, ms));
}

function createProviderResilience({
  provider,
  maxConcurrency = 6,
  timeoutMs = 8_000,
  maxRetries = 1,
  retryBaseMs = 120,
  circuitFailureThreshold = 4,
  circuitResetMs = 30_000,
  cacheTtlMs = 0,
  maxCacheEntries = 250,
  now = Date.now,
  random = Math.random,
  logger = () => {},
  isTransient = defaultTransient,
} = {}) {
  const name = String(provider || 'external_provider').replace(/[^a-z0-9_-]/gi, '_');
  const settings = {
    maxConcurrency: safeInteger(maxConcurrency, 6, 1, 100),
    timeoutMs: safeInteger(timeoutMs, 8_000, 50, 120_000),
    maxRetries: safeInteger(maxRetries, 1, 0, 3),
    retryBaseMs: safeInteger(retryBaseMs, 120, 1, 10_000),
    circuitFailureThreshold: safeInteger(circuitFailureThreshold, 4, 1, 100),
    circuitResetMs: safeInteger(circuitResetMs, 30_000, 100, 600_000),
    cacheTtlMs: safeInteger(cacheTtlMs, 0, 0, 86_400_000),
    maxCacheEntries: safeInteger(maxCacheEntries, 250, 1, 10_000),
  };
  const inFlight = new Map();
  const cache = new Map();
  let active = 0;
  let consecutiveFailures = 0;
  let circuitState = 'CLOSED';
  let openedAt = 0;
  const stats = {
    calls: 0, successes: 0, failures: 0, timeouts: 0, cacheHits: 0,
    coalesced: 0, concurrencyRejected: 0, circuitRejected: 0, retries: 0, peakConcurrency: 0,
  };

  function emit(result, startedAt, extra = {}) {
    logger({
      event: 'provider_operation', provider: name, result,
      durationMs: Math.max(0, now() - startedAt), circuitState, active, ...extra,
    });
  }

  function pruneCache(currentTime) {
    for (const [key, entry] of cache) if (entry.expiresAt <= currentTime) cache.delete(key);
    while (cache.size >= settings.maxCacheEntries) cache.delete(cache.keys().next().value);
  }

  function checkCircuit() {
    if (circuitState !== 'OPEN') return;
    if (now() - openedAt >= settings.circuitResetMs) {
      circuitState = 'HALF_OPEN';
      return;
    }
    stats.circuitRejected += 1;
    throw operationalError(`${name} is temporarily unavailable.`, 'PROVIDER_CIRCUIT_OPEN', 503, Math.max(1, Math.ceil((settings.circuitResetMs - (now() - openedAt)) / 1000)));
  }

  async function attempt(operation, attemptIndex) {
    const controller = new AbortController();
    let timer;
    const timeoutPromise = new Promise((_, reject) => {
      timer = setTimeout(() => {
        reject(operationalError(`${name} timed out.`, 'PROVIDER_TIMEOUT', 504));
        controller.abort();
      }, settings.timeoutMs);
    });
    try {
      return await Promise.race([operation({ signal: controller.signal, attempt: attemptIndex }), timeoutPromise]);
    } finally {
      clearTimeout(timer);
    }
  }

  async function run(operation, startedAt, keyHash) {
    active += 1;
    stats.peakConcurrency = Math.max(stats.peakConcurrency, active);
    try {
      for (let attemptIndex = 0; attemptIndex <= settings.maxRetries; attemptIndex += 1) {
        try {
          const result = await attempt(operation, attemptIndex);
          consecutiveFailures = 0;
          circuitState = 'CLOSED';
          stats.successes += 1;
          emit('success', startedAt, { attempts: attemptIndex + 1, keyHash });
          return result;
        } catch (error) {
          if (error?.code === 'PROVIDER_TIMEOUT') stats.timeouts += 1;
          if (attemptIndex >= settings.maxRetries || !isTransient(error)) throw error;
          stats.retries += 1;
          const jitter = Math.floor(settings.retryBaseMs * 0.35 * random());
          await wait(settings.retryBaseMs * (2 ** attemptIndex) + jitter);
        }
      }
      throw operationalError(`${name} failed.`, 'PROVIDER_FAILED');
    } catch (error) {
      stats.failures += 1;
      consecutiveFailures += 1;
      if (consecutiveFailures >= settings.circuitFailureThreshold) {
        circuitState = 'OPEN';
        openedAt = now();
      }
      emit(error?.code === 'PROVIDER_TIMEOUT' ? 'timeout' : 'failure', startedAt, {
        code: error?.code || 'PROVIDER_FAILED',
        keyHash,
      });
      throw error;
    } finally {
      active -= 1;
    }
  }

  async function execute(key, operation, options = {}) {
    const startedAt = now();
    const keyHash = hashKey(key);
    const ttlMs = safeInteger(options.cacheTtlMs, settings.cacheTtlMs, 0, 86_400_000);
    const currentTime = now();
    pruneCache(currentTime);
    const cached = cache.get(keyHash);
    if (cached && cached.expiresAt > currentTime) {
      stats.cacheHits += 1;
      emit('cache_hit', startedAt, { keyHash });
      return cached.value;
    }
    if (inFlight.has(keyHash)) {
      stats.coalesced += 1;
      emit('coalesced', startedAt, { keyHash });
      return inFlight.get(keyHash);
    }
    checkCircuit();
    if (active >= settings.maxConcurrency) {
      stats.concurrencyRejected += 1;
      emit('concurrency_rejected', startedAt, { keyHash });
      throw operationalError(`${name} is busy. Please retry shortly.`, 'PROVIDER_CONCURRENCY_LIMIT', 503, 1);
    }
    stats.calls += 1;
    const promise = run(operation, startedAt, keyHash)
      .then((value) => {
        if (ttlMs > 0) cache.set(keyHash, { value, expiresAt: now() + ttlMs });
        return value;
      })
      .finally(() => inFlight.delete(keyHash));
    inFlight.set(keyHash, promise);
    return promise;
  }

  return {
    execute,
    snapshot: () => ({ provider: name, ...settings, ...stats, active, inFlight: inFlight.size, cacheEntries: cache.size, circuitState }),
    reset: () => {
      inFlight.clear(); cache.clear(); active = 0; consecutiveFailures = 0; circuitState = 'CLOSED'; openedAt = 0;
      Object.keys(stats).forEach((key) => { stats[key] = 0; });
    },
  };
}

module.exports = { createProviderResilience, operationalError };
