const { createProviderResilience } = require('./providerResilience');

function envNumber(name, fallback) {
  const value = Number(process.env[name]);
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

function envNonNegative(name, fallback) {
  const value = Number(process.env[name]);
  return Number.isFinite(value) && value >= 0 ? value : fallback;
}

function structuredLogger(event) {
  if (process.env.URBANAGENT_OPERATIONAL_LOGS === 'false') return;
  console.info(JSON.stringify({ level: 'info', ...event }));
}

const photonGuard = createProviderResilience({
  provider: 'photon',
  maxConcurrency: envNumber('URBANAGENT_PHOTON_MAX_CONCURRENCY', 4),
  timeoutMs: envNumber('URBANAGENT_PHOTON_TIMEOUT_MS', 8_000),
  maxRetries: envNonNegative('URBANAGENT_PHOTON_MAX_RETRIES', 1),
  cacheTtlMs: envNumber('URBANAGENT_PHOTON_CACHE_TTL_MS', 300_000),
  logger: structuredLogger,
});

const googlePlacesGuard = createProviderResilience({
  provider: 'google_places',
  maxConcurrency: envNumber('URBANAGENT_GOOGLE_PLACES_MAX_CONCURRENCY', 4),
  timeoutMs: envNumber('URBANAGENT_GOOGLE_PLACES_TIMEOUT_MS', 8_000),
  maxRetries: envNonNegative('URBANAGENT_GOOGLE_PLACES_MAX_RETRIES', 1),
  cacheTtlMs: envNumber('URBANAGENT_GOOGLE_PLACES_CACHE_TTL_MS', 120_000),
  logger: structuredLogger,
});

const osrmGuard = createProviderResilience({
  provider: 'osrm',
  maxConcurrency: envNumber('URBANAGENT_OSRM_MAX_CONCURRENCY', 6),
  timeoutMs: envNumber('URBANAGENT_OSRM_TIMEOUT_MS', 10_000),
  maxRetries: envNonNegative('URBANAGENT_OSRM_MAX_RETRIES', 1),
  cacheTtlMs: envNumber('URBANAGENT_OSRM_CACHE_TTL_MS', 600_000),
  logger: structuredLogger,
});

const weatherGuard = createProviderResilience({
  provider: 'weather',
  maxConcurrency: envNumber('URBANAGENT_WEATHER_MAX_CONCURRENCY', 3),
  timeoutMs: envNumber('URBANAGENT_WEATHER_TIMEOUT_MS', 8_000),
  maxRetries: envNonNegative('URBANAGENT_WEATHER_MAX_RETRIES', 1),
  cacheTtlMs: envNumber('URBANAGENT_WEATHER_CACHE_TTL_MS', 1_200_000),
  logger: structuredLogger,
});

module.exports = { googlePlacesGuard, osrmGuard, photonGuard, weatherGuard };
