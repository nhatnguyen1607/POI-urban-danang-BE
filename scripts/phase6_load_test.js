const { performance } = require('node:perf_hooks');

const baseUrl = process.env.URBANAGENT_LOAD_BASE_URL || 'http://127.0.0.1:7860';
const profileName = process.argv[2] || 'smoke';
const profiles = {
  smoke: { requests: 20, concurrency: 2 },
  ramp: { requests: 120, concurrency: 12 },
  spike: { requests: 240, concurrency: 30 },
};
const profile = profiles[profileName];

if (!profile) throw new Error('Profile must be smoke, ramp, or spike.');
const parsed = new URL(baseUrl);
if (!['localhost', '127.0.0.1', '::1'].includes(parsed.hostname)
  && process.env.URBANAGENT_ALLOW_REMOTE_LOAD_TEST !== 'true') {
  throw new Error('Remote load is disabled. Use an explicit isolated staging opt-in.');
}

async function main() {
  const latencies = [];
  let next = 0;
  let errors = 0;
  let rateLimited = 0;
  let timeouts = 0;
  const started = performance.now();
  async function worker() {
    while (next < profile.requests) {
      const index = next++;
      const requestStarted = performance.now();
      try {
        const response = await fetch(`${baseUrl}/api/v2/pois/search?cityId=da-nang&q=cafe&limit=3&offset=${index % 5}`, {
          signal: AbortSignal.timeout(10_000),
        });
        if (response.status === 429) rateLimited += 1;
        if (!response.ok) errors += 1;
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
  console.log(JSON.stringify({
    label: 'LOCAL_BASELINE',
    profile: profileName,
    requests: profile.requests,
    concurrency: profile.concurrency,
    requestsPerSecond: Number((profile.requests / (durationMs / 1000)).toFixed(1)),
    latencyMs: { p50: percentile(0.5), p95: percentile(0.95), p99: percentile(0.99) },
    errors,
    rateLimited,
    timeouts,
  }));
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
