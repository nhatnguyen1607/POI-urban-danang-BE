const assert = require('node:assert/strict');
const { test, beforeEach, afterEach } = require('node:test');
process.env.URBANAGENT_OPERATIONAL_LOGS = 'false';
const { osrmGuard } = require('../../src/infrastructure/external/providerGuards');
const { fetchOSRMRoute } = require('../../src/expert_system/expert_system');
const { createProviderResilience } = require('../../src/infrastructure/external/providerResilience');
const originalFetch = global.fetch;
const coordinates = [16, 108, 16.1, 108.1];
const payload = { code: 'Ok', routes: [{ distance: 1200, duration: 180,
  geometry: { type: 'LineString', coordinates: [[108, 16], [108.1, 16.1]] }, legs: [{ steps: [] }] }] };
beforeEach(() => osrmGuard.reset());
afterEach(() => { global.fetch = originalFetch; osrmGuard.reset(); });
function respond(data = payload, status = 200) {
  return new Response(JSON.stringify(data), { status });
}

test('valid OSRM route preserves geometry, distance, duration and steps', async () => {
  global.fetch = async () => respond();
  assert.deepEqual(await fetchOSRMRoute(...coordinates), [{ geometry: payload.routes[0].geometry, distance: 1200, duration: 180, steps: [] }]);
});

test('cached request and concurrent duplicate consume a real Response body exactly once', async () => {
  let calls = 0; let reads = 0;
  global.fetch = async () => {
    calls++;
    const response = respond();
    const read = response.text.bind(response);
    response.text = () => { reads++; return read(); };
    return response;
  };
  const [a, b] = await Promise.all([fetchOSRMRoute(...coordinates), fetchOSRMRoute(...coordinates)]);
  const c = await fetchOSRMRoute(...coordinates);
  assert.deepEqual(a, b); assert.deepEqual(b, c);
  assert.equal(calls, 1); assert.equal(reads, 1);
});

test('two distinct concurrent legs both retain their route results', async () => {
  let calls = 0;
  global.fetch = async () => { calls++; return respond(); };
  const results = await Promise.all([fetchOSRMRoute(...coordinates), fetchOSRMRoute(16.1, 108.1, 16.2, 108.2)]);
  assert.equal(calls, 2);
  results.forEach(result => assert.equal(result[0].distance, 1200));
});

test('one failed leg does not discard the other; failed response is not cached', async () => {
  global.fetch = async url => url.includes('/108,16;') ? respond({ code: 'NoRoute', message: 'No path' }) : respond();
  const results = await Promise.allSettled([fetchOSRMRoute(...coordinates), fetchOSRMRoute(16.1, 108.1, 16.2, 108.2)]);
  assert.equal(results[0].status, 'rejected'); assert.equal(results[0].reason.code, 'OSRM_NO_ROUTE');
  assert.equal(results[1].status, 'fulfilled'); assert.equal(results[1].value[0].duration, 180);
  global.fetch = async () => respond();
  assert.equal((await fetchOSRMRoute(...coordinates))[0].distance, 1200);
});

for (const status of [400, 429, 500, 503]) {
  test(`OSRM HTTP ${status} retains upstream status/code/message`, async () => {
    global.fetch = async () => respond({ code: 'InvalidQuery', message: 'upstream diagnostic' }, status);
    await assert.rejects(fetchOSRMRoute(...coordinates), error => error.code === 'OSRM_HTTP'
      && error.message.includes(String(status)) && error.message.includes('upstream diagnostic'));
  });
}
test('non-JSON HTTP failure remains HTTP failure', async () => {
  global.fetch = async () => new Response('<html>Unavailable</html>', { status: 502 });
  await assert.rejects(fetchOSRMRoute(...coordinates), { code: 'OSRM_HTTP', message: 'OSRM HTTP 502' });
});
for (const data of ['not json', JSON.stringify({ code: 'Ok', routes: [{}] })]) {
  test(`malformed success response: ${data}`, async () => {
    global.fetch = async () => new Response(data);
    await assert.rejects(fetchOSRMRoute(...coordinates), { code: 'OSRM_PARSE' });
  });
}
test('network error retains cause', async () => {
  const cause = new TypeError('fetch failed');
  global.fetch = async () => { throw cause; };
  await assert.rejects(fetchOSRMRoute(...coordinates), error => error.code === 'OSRM_NETWORK' && error.cause === cause);
});
test('invalid coordinates rejected before fetch; zero coordinates accepted', async () => {
  let calls = 0;
  global.fetch = async () => { calls++; return respond(); };
  for (const point of [[NaN, 108, 16, 108], [91, 108, 16, 108], [16, 181, 16, 108], [null, 108, 16, 108], ['16', 108, 16, 108]]) {
    await assert.rejects(fetchOSRMRoute(...point), { code: 'INVALID_ROUTE_COORDINATES' });
  }
  assert.equal(calls, 0);
  await fetchOSRMRoute(0, 0, 1, 1); assert.equal(calls, 1);
});
test('guard timeout covers body consumption as well as headers', async () => {
  const originalExecute = osrmGuard.execute;
  const guard = createProviderResilience({ provider: 'osrm', timeoutMs: 50, maxRetries: 0 });
  osrmGuard.execute = guard.execute;
  global.fetch = async (_url, { signal }) => new Response(new ReadableStream({ start(controller) {
    signal.addEventListener('abort', () => controller.error(new Error('aborted')), { once: true });
  } }));
  try { await assert.rejects(fetchOSRMRoute(...coordinates), { code: 'PROVIDER_TIMEOUT' }); }
  finally { osrmGuard.execute = originalExecute; }
});

test('POST /api/route preserves contract, concurrent leg isolation and coordinate errors', async () => {
  const { initExpertSystem } = require('../../src/expert_system/expert_system');
  await initExpertSystem();
  const { app } = require('../../src/server');
  const server = app.listen(0, '127.0.0.1');
  await new Promise(resolve => server.once('listening', resolve));
  global.fetch = async url => {
    if (url.includes('openweathermap')) return respond({ weather: [{ main: 'Clear' }] });
    return url.includes('/108,16;') ? respond({ code: 'NoRoute', message: 'No path' }) : respond();
  };
  const post = body => originalFetch(`http://127.0.0.1:${server.address().port}/api/route`, {
    method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: 'Bearer regression-local' },
    body: JSON.stringify(body),
  });
  const oldNodeEnv = process.env.NODE_ENV;
  const oldFallback = process.env.DISABLE_DEV_AUTH_FALLBACK;
  process.env.NODE_ENV = 'test'; process.env.DISABLE_DEV_AUTH_FALLBACK = 'false';
  try {
    const [bad, good] = await Promise.all([
      post({ origin: { lat: 16, lng: 108 }, destination: { lat: 16.1, lng: 108.1 } }),
      post({ origin: { lat: 0, lng: 0 }, destination: { lat: 1, lng: 1 } }),
    ]);
    assert.equal(bad.status, 500);
    assert.deepEqual(await bad.json(), { error: 'Failed to find route', details: 'OSRM NoRoute: No path' });
    assert.equal(good.status, 200);
    const result = await good.json();
    assert.equal(result.routes[0].distance, 1200);
    assert.equal(result.routes[0].duration, 180);
    assert.deepEqual(result.routes[0].route, payload.routes[0].geometry);
    const invalid = await post({ origin: { lat: 91, lng: 108 }, destination: { lat: 16, lng: 108 } });
    assert.equal(invalid.status, 400);
    const missing = await post({}); assert.equal(missing.status, 400);
  } finally {
    if (oldNodeEnv === undefined) delete process.env.NODE_ENV; else process.env.NODE_ENV = oldNodeEnv;
    if (oldFallback === undefined) delete process.env.DISABLE_DEV_AUTH_FALLBACK; else process.env.DISABLE_DEV_AUTH_FALLBACK = oldFallback;
    await new Promise(resolve => server.close(resolve));
  }
});

for (const [label, code, status, expectedAttempts] of [
  ['NoRoute', 'NoRoute', 200, 1], ['HTTP400', 'InvalidQuery', 400, 1],
  ['HTTP429', 'TooManyRequests', 429, 1], ['HTTP503', 'Unavailable', 503, 2],
]) {
  test(`retry policy ${label}: bounded appropriate fetch count`, async () => {
    let calls = 0;
    global.fetch = async () => { calls++; return respond({ code }, status); };
    await assert.rejects(fetchOSRMRoute(...coordinates));
    assert.equal(calls, expectedAttempts);
    assert.equal(osrmGuard.snapshot().cacheEntries, 0);
    assert.equal(osrmGuard.snapshot().inFlight, 0);
  });
}
test('parse failure is not retried', async () => {
  let calls = 0;
  global.fetch = async () => { calls++; return new Response('bad JSON'); };
  await assert.rejects(fetchOSRMRoute(...coordinates), { code: 'OSRM_PARSE' });
  assert.equal(calls, 1);
});
test('repeated NoRoute does not open circuit for an unrelated valid leg', async () => {
  let calls = 0;
  global.fetch = async () => { calls++; return respond({ code: 'NoRoute' }); };
  for (let i = 0; i < 6; i++) await assert.rejects(fetchOSRMRoute(...coordinates), { code: 'OSRM_NO_ROUTE' });
  assert.equal(calls, 6);
  assert.equal(osrmGuard.snapshot().circuitState, 'CLOSED');
  global.fetch = async () => respond();
  assert.equal((await fetchOSRMRoute(16.2, 108.2, 16.3, 108.3))[0].distance, 1200);
});
test('cache distinguishes direction and destination without cross-leg contamination', async () => {
  let calls = 0;
  global.fetch = async () => {
    calls++;
    return respond({ ...payload, routes: [{ ...payload.routes[0], distance: calls * 100 }] });
  };
  const a = await fetchOSRMRoute(...coordinates);
  const b = await fetchOSRMRoute(16.1, 108.1, 16, 108);
  const c = await fetchOSRMRoute(16, 108, 16.2, 108.2);
  assert.deepEqual([a[0].distance, b[0].distance, c[0].distance], [100, 200, 300]);
  assert.equal((await fetchOSRMRoute(...coordinates))[0].distance, 100);
  assert.equal(calls, 3);
});
