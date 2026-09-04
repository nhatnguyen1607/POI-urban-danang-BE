const { performance } = require('node:perf_hooks');
const express = require('express');
const { createEndpointConcurrencyLimit } = require('../src/middleware/endpointConcurrencyLimit');

let baseUrl = process.env.URBANAGENT_LOAD_BASE_URL || 'http://127.0.0.1:7860';
const profileName = process.argv[2] || 'smoke';
const profiles = {
  smoke: { requests: 20, concurrency: 2 },
  ramp: { requests: 120, concurrency: 12 },
  spike: { requests: 240, concurrency: 30 },
  'controlled-spike': { requests: 240, concurrency: 30, controlled: true },
};
const profile = profiles[profileName];

if (!profile) throw new Error('Profile must be smoke, ramp, spike, or controlled-spike.');

function assertLocalTarget(url) {
  const parsed = new URL(url);
  if (!['localhost', '127.0.0.1', '::1'].includes(parsed.hostname)
    && process.env.URBANAGENT_ALLOW_REMOTE_LOAD_TEST !== 'true') {
    throw new Error('Remote load is disabled. Use an explicit isolated staging opt-in.');
  }
}

async function startControlledServer() {
  const app = express();
  const admission = createEndpointConcurrencyLimit({
    name: 'controlled_spike',
    maxActive: 8,
    retryAfterSeconds: 1,
    logger: () => {},
  });
  app.get('/work', admission, async (req, res) => {
    await new Promise((resolve) => setTimeout(resolve, 100));
    res.json({ ok: true });
  });
  app.get('/health', (req, res) => res.json({ ok: true }));
  const server = await new Promise((resolve) => {
    const listening = app.listen(0, '127.0.0.1', () => resolve(listening));
  });
  const address = server.address();
  return {
    admission,
    server,
    url: `http://127.0.0.1:${address.port}`,
  };
}

async function main() {
  let controlled = null;
  if (profile.controlled) {
    controlled = await startControlledServer();
    baseUrl = controlled.url;
  }
  assertLocalTarget(baseUrl);

  const latencies = [];
  let next = 0;
  let completed = 0;
  let errors = 0;
  let rateLimited = 0;
  let serviceUnavailable = 0;
  let timeouts = 0;
  let cacheHits = 0;
  let observedMaxActive = 0;
  let observedMaxPending = 0;
  const started = performance.now();
  async function worker() {
    while (next < profile.requests) {
      const index = next++;
      const requestStarted = performance.now();
      try {
        const path = profile.controlled
          ? '/work'
          : `/api/v2/pois/search?cityId=da-nang&q=cafe&limit=3&offset=${index % 5}`;
        const response = await fetch(`${baseUrl}${path}`, { signal: AbortSignal.timeout(10_000) });
        observedMaxActive = Math.max(observedMaxActive, Number(response.headers.get('X-UrbanAgent-Admission-Active') || 0));
        observedMaxPending = Math.max(observedMaxPending, Number(response.headers.get('X-UrbanAgent-Admission-Pending') || 0));
        if (response.status === 429) rateLimited += 1;
        if (response.status === 503) serviceUnavailable += 1;
        if (response.headers.get('X-Cache') === 'HIT') cacheHits += 1;
        if (response.ok) completed += 1;
        else errors += 1;
        await response.arrayBuffer();
      } catch (error) {
        errors += 1;
        if (error?.name === 'TimeoutError') timeouts += 1;
      } finally {
        latencies.push(performance.now() - requestStarted);
      }
    }
  }

  await Promise.all(Array.from({ length: profile.concurrency }, worker));
  const durationMs = performance.now() - started;
  const sorted = latencies.sort((a, b) => a - b);
  const percentile = (p) => Number(sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))].toFixed(1));
  let serverResponsiveAfterSpike = null;
  if (controlled) {
    const health = await fetch(`${baseUrl}/health`, { signal: AbortSignal.timeout(2_000) });
    serverResponsiveAfterSpike = health.ok;
    const snapshot = controlled.admission.snapshot();
    observedMaxActive = Math.max(observedMaxActive, snapshot.peakActive);
    observedMaxPending = Math.max(observedMaxPending, snapshot.pending);
    await new Promise((resolve, reject) => controlled.server.close((error) => (error ? reject(error) : resolve())));
  }

  console.log(JSON.stringify({
    label: profile.controlled ? 'LOCAL_CONTROLLED_ADMISSION' : 'LOCAL_BASELINE',
    profile: profileName,
    requests: profile.requests,
    concurrency: profile.concurrency,
    completed,
    requestsPerSecond: Number((profile.requests / (durationMs / 1000)).toFixed(1)),
    latencyMs: { p50: percentile(0.5), p95: percentile(0.95), p99: percentile(0.99) },
    errors,
    rateLimited,
    serviceUnavailable,
    timeouts,
    cacheHits,
    maxActiveConcurrency: observedMaxActive,
    maxPendingQueue: observedMaxPending,
    serverResponsiveAfterSpike,
  }));
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
