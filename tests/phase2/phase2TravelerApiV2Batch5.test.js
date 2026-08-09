const assert = require('node:assert/strict');
const { spawn } = require('node:child_process');
const http = require('node:http');
const path = require('node:path');
const test = require('node:test');

const fixture = require('../fixtures/phase2/tripPreviewQueries.json');

const DEFAULT_CITY_ID = 'da-nang';

function requestJson({ port, method = 'GET', path: requestPath, headers = {}, body }) {
  return new Promise((resolve, reject) => {
    const payload = body === undefined ? null : JSON.stringify(body);
    const req = http.request({
      hostname: '127.0.0.1',
      port,
      method,
      path: requestPath,
      headers: {
        ...headers,
        ...(payload ? { 'content-type': 'application/json', 'content-length': Buffer.byteLength(payload) } : {}),
      },
    }, (res) => {
      const chunks = [];
      res.on('data', (chunk) => chunks.push(chunk));
      res.on('end', () => {
        const text = Buffer.concat(chunks).toString('utf8');
        let parsed = null;
        try {
          parsed = text ? JSON.parse(text) : null;
        } catch (_) {
          parsed = null;
        }
        resolve({ statusCode: res.statusCode, body: parsed, text });
      });
    });
    req.on('error', reject);
    if (payload) req.write(payload);
    req.end();
  });
}

async function waitForTravelerApi(port, child) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < 45000) {
    if (child.exitCode !== null) {
      throw new Error(`Server exited before readiness check completed with code ${child.exitCode}`);
    }
    try {
      const response = await requestJson({ port, path: '/api/v2/cities' });
      if (response.statusCode === 200) return;
    } catch (_) {
      // Keep polling until the child server is ready.
    }
    await new Promise((resolve) => setTimeout(resolve, 750));
  }
  throw new Error('Timed out waiting for traveler API readiness');
}

async function stopServer(child) {
  if (!child || child.exitCode !== null) return;
  child.kill();
  const exited = await new Promise((resolve) => {
    const timeout = setTimeout(() => resolve(false), 5000);
    child.once('exit', () => {
      clearTimeout(timeout);
      resolve(true);
    });
  });
  if (exited) return;
  child.kill('SIGKILL');
  await new Promise((resolve) => child.once('exit', resolve));
}

function unsignedDevToken(uid) {
  const encode = (value) => Buffer.from(JSON.stringify(value)).toString('base64url');
  return `${encode({ alg: 'none', typ: 'JWT' })}.${encode({
    sub: uid,
    user_id: uid,
    email: `${uid}@urbanagent.local`,
    role: 'customer',
  })}.dev-signature`;
}

function authHeader(uid) {
  return { authorization: `Bearer ${unsignedDevToken(uid)}` };
}

function savedTripPayload(title = 'Hai ngày ở Đà Nẵng') {
  const input = fixture.cases.find((item) => item.id === 'two-day-balanced-trip')?.input ||
    fixture.cases.find((item) => item.id === 'one-day-balanced-trip').input;
  return {
    title,
    cityId: DEFAULT_CITY_ID,
    query: input.query,
    startDate: input.trip?.date,
    dayCount: input.trip?.dayCount || 2,
    dailyWindow: input.trip?.dailyWindow,
    dayWindows: input.trip?.dayWindows || [],
    pace: input.trip?.pace,
    transport: input.trip?.transport,
    includedPoiIds: input.constraints?.mustIncludePoiIds || [],
    excludedPoiIds: input.constraints?.excludePoiIds || [],
    request: input,
    preview: {
      tripId: null,
      persisted: false,
      preview: true,
      dayCount: input.trip?.dayCount || 2,
      days: [],
      stops: [],
      warnings: [],
    },
  };
}

test('Phase 2 Batch 5 saved trips require auth, preserve ownership, and keep preview explicit', {
  timeout: 120000,
}, async () => {
  const port = 25000 + Math.floor(Math.random() * 1000);
  const child = spawn(process.execPath, ['src/server.js'], {
    cwd: path.join(__dirname, '..', '..'),
    env: {
      ...process.env,
      NODE_ENV: 'test',
      PORT: String(port),
      URBANAGENT_SAVED_TRIPS_STORE: 'memory',
      URBANAGENT_POI_REPOSITORY: '',
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  const logs = [];
  child.stdout.on('data', (chunk) => logs.push(chunk.toString('utf8')));
  child.stderr.on('data', (chunk) => logs.push(chunk.toString('utf8')));

  try {
    await waitForTravelerApi(port, child);

    const unauthenticated = await requestJson({
      port,
      method: 'POST',
      path: '/api/v2/trips',
      body: savedTripPayload(),
    });
    assert.equal(unauthenticated.statusCode, 401);

    const preview = await requestJson({
      port,
      method: 'POST',
      path: '/api/v2/trips/preview',
      body: fixture.cases.find((item) => item.id === 'one-day-balanced-trip').input,
    });
    assert.equal(preview.statusCode, 200, logs.join('').slice(-1000));
    assert.equal(preview.body.data.trip.persisted, false);

    const emptyList = await requestJson({
      port,
      path: '/api/v2/trips',
      headers: authHeader('user-a'),
    });
    assert.equal(emptyList.statusCode, 200);
    assert.equal(emptyList.body.data.total, 0);

    const created = await requestJson({
      port,
      method: 'POST',
      path: '/api/v2/trips',
      headers: authHeader('user-a'),
      body: savedTripPayload(),
    });
    assert.equal(created.statusCode, 201);
    assert.equal(created.body.data.trip.title, 'Hai ngày ở Đà Nẵng');
    const tripId = created.body.data.trip.tripId;
    assert.ok(tripId);

    const ownList = await requestJson({
      port,
      path: '/api/v2/trips',
      headers: authHeader('user-a'),
    });
    assert.equal(ownList.statusCode, 200);
    assert.equal(ownList.body.data.trips.length, 1);
    assert.equal(ownList.body.data.trips[0].tripId, tripId);

    const ownGet = await requestJson({
      port,
      path: `/api/v2/trips/${tripId}`,
      headers: authHeader('user-a'),
    });
    assert.equal(ownGet.statusCode, 200);
    assert.equal(ownGet.body.data.trip.tripId, tripId);

    const userBGet = await requestJson({
      port,
      path: `/api/v2/trips/${tripId}`,
      headers: authHeader('user-b'),
    });
    assert.equal(userBGet.statusCode, 404);

    const userBPatch = await requestJson({
      port,
      method: 'PATCH',
      path: `/api/v2/trips/${tripId}`,
      headers: authHeader('user-b'),
      body: { title: 'Không được sửa' },
    });
    assert.equal(userBPatch.statusCode, 404);

    const updated = await requestJson({
      port,
      method: 'PATCH',
      path: `/api/v2/trips/${tripId}`,
      headers: authHeader('user-a'),
      body: { ...savedTripPayload('Lịch trình đã cập nhật'), excludedPoiIds: ['google_maps_1'] },
    });
    assert.equal(updated.statusCode, 200);
    assert.equal(updated.body.data.trip.title, 'Lịch trình đã cập nhật');
    assert.deepEqual(updated.body.data.trip.excludedPoiIds, ['google_maps_1']);

    const userBDelete = await requestJson({
      port,
      method: 'DELETE',
      path: `/api/v2/trips/${tripId}`,
      headers: authHeader('user-b'),
    });
    assert.equal(userBDelete.statusCode, 404);

    const deleted = await requestJson({
      port,
      method: 'DELETE',
      path: `/api/v2/trips/${tripId}`,
      headers: authHeader('user-a'),
    });
    assert.equal(deleted.statusCode, 200);
    assert.equal(deleted.body.data.deleted, true);

    const afterDelete = await requestJson({
      port,
      path: '/api/v2/trips',
      headers: authHeader('user-a'),
    });
    assert.equal(afterDelete.statusCode, 200);
    assert.equal(afterDelete.body.data.total, 0);

    const recommendation = await requestJson({
      port,
      method: 'POST',
      path: '/api/v2/recommendations',
      body: {
        cityId: DEFAULT_CITY_ID,
        query: 'quan cafe yen tinh',
        limit: 3,
      },
    });
    assert.equal(recommendation.statusCode, 200);
    assert.ok(recommendation.body.data.recommendations.length > 0);
  } finally {
    await stopServer(child);
  }
});
