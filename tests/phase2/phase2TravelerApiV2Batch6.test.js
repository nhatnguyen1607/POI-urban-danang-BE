const assert = require('node:assert/strict');
const { spawn } = require('node:child_process');
const http = require('node:http');
const path = require('node:path');
const test = require('node:test');

const fixture = require('../fixtures/phase2/tripPreviewQueries.json');

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

function savedTripPayloadFromPreview(previewTrip) {
  return {
    title: 'Batch 6 lifecycle trip',
    cityId: previewTrip.cityId,
    query: previewTrip.query,
    startDate: previewTrip.date,
    dayCount: previewTrip.dayCount,
    dailyWindow: previewTrip.dailyWindow,
    dayWindows: [],
    pace: previewTrip.pace,
    transport: previewTrip.transport,
    includedPoiIds: [],
    excludedPoiIds: [],
    request: fixture.cases.find((item) => item.id === 'one-day-balanced-trip').input,
    preview: previewTrip,
    itinerary: previewTrip.stops,
    warnings: previewTrip.warnings,
  };
}

test('Phase 2 Batch 6 saved-trip lifecycle routes preserve ownership and same-trip replan semantics', {
  timeout: 160000,
}, async () => {
  const port = 26000 + Math.floor(Math.random() * 1000);
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

    const previewInput = fixture.cases.find((item) => item.id === 'one-day-balanced-trip').input;
    const preview = await requestJson({
      port,
      method: 'POST',
      path: '/api/v2/trips/preview',
      body: previewInput,
    });
    assert.equal(preview.statusCode, 200, logs.join('').slice(-1000));
    assert.equal(preview.body.data.trip.persisted, false);
    assert.ok(preview.body.data.trip.stops.length >= 2);

    const unauthenticatedReplan = await requestJson({
      port,
      method: 'POST',
      path: '/api/v2/trips/trip_missing/replan',
    });
    assert.equal(unauthenticatedReplan.statusCode, 401);

    const created = await requestJson({
      port,
      method: 'POST',
      path: '/api/v2/trips',
      headers: authHeader('user-a'),
      body: savedTripPayloadFromPreview(preview.body.data.trip),
    });
    assert.equal(created.statusCode, 201);
    const tripId = created.body.data.trip.tripId;
    const originalStops = created.body.data.trip.itinerary;
    assert.ok(tripId);

    const userBRemove = await requestJson({
      port,
      method: 'DELETE',
      path: `/api/v2/trips/${tripId}/stops/${originalStops[0].stopId}`,
      headers: authHeader('user-b'),
    });
    assert.equal(userBRemove.statusCode, 404);

    const duplicateAdd = await requestJson({
      port,
      method: 'POST',
      path: `/api/v2/trips/${tripId}/stops`,
      headers: authHeader('user-a'),
      body: { poiId: originalStops[0].poi.globalId, dayNumber: 1 },
    });
    assert.equal(duplicateAdd.statusCode, 409);
    assert.equal(duplicateAdd.body.error.code, 'DUPLICATE_STOP');

    const search = await requestJson({
      port,
      path: '/api/v2/pois/search?cityId=da-nang&limit=25&q=cafe',
    });
    assert.equal(search.statusCode, 200);
    const scheduledIds = new Set(originalStops.map((stop) => stop.poi.globalId));
    const addablePoi = search.body.data.pois.find((poi) => !scheduledIds.has(poi.globalId));
    assert.ok(addablePoi.globalId);

    const excludedPatch = await requestJson({
      port,
      method: 'PATCH',
      path: `/api/v2/trips/${tripId}`,
      headers: authHeader('user-a'),
      body: { ...created.body.data.trip, excludedPoiIds: [addablePoi.globalId] },
    });
    assert.equal(excludedPatch.statusCode, 200);

    const excludedAdd = await requestJson({
      port,
      method: 'POST',
      path: `/api/v2/trips/${tripId}/stops`,
      headers: authHeader('user-a'),
      body: { poiId: addablePoi.globalId, dayNumber: 1 },
    });
    assert.equal(excludedAdd.statusCode, 409);
    assert.equal(excludedAdd.body.error.code, 'EXCLUDED_POI');

    const clearExcluded = await requestJson({
      port,
      method: 'PATCH',
      path: `/api/v2/trips/${tripId}`,
      headers: authHeader('user-a'),
      body: { ...excludedPatch.body.data.trip, excludedPoiIds: [] },
    });
    assert.equal(clearExcluded.statusCode, 200);

    const added = await requestJson({
      port,
      method: 'POST',
      path: `/api/v2/trips/${tripId}/stops`,
      headers: authHeader('user-a'),
      body: { poiId: addablePoi.globalId, dayNumber: 1 },
    });
    assert.equal(added.statusCode, 200, added.text);
    assert.equal(added.body.data.trip.tripId, tripId);
    assert.equal(added.body.data.trip.needsReplan, true);
    assert.ok(added.body.data.trip.itinerary.some((stop) => stop.poi.globalId === addablePoi.globalId));

    const dayOneStopIds = added.body.data.trip.itinerary
      .filter((stop) => stop.dayNumber === 1)
      .map((stop) => stop.stopId);
    const invalidReorder = await requestJson({
      port,
      method: 'PATCH',
      path: `/api/v2/trips/${tripId}/stops/reorder`,
      headers: authHeader('user-a'),
      body: { dayNumber: 1, stopIds: [dayOneStopIds[0], dayOneStopIds[0]] },
    });
    assert.equal(invalidReorder.statusCode, 400);

    const reordered = await requestJson({
      port,
      method: 'PATCH',
      path: `/api/v2/trips/${tripId}/stops/reorder`,
      headers: authHeader('user-a'),
      body: { dayNumber: 1, stopIds: [...dayOneStopIds].reverse() },
    });
    assert.equal(reordered.statusCode, 200, reordered.text);
    assert.equal(reordered.body.data.trip.needsReplan, true);
    assert.deepEqual(
      reordered.body.data.trip.itinerary.filter((stop) => stop.dayNumber === 1).map((stop) => stop.stopId),
      [...dayOneStopIds].reverse(),
    );

    const removedStopId = reordered.body.data.trip.itinerary[0].stopId;
    const removedPoiId = reordered.body.data.trip.itinerary[0].poi.globalId;
    const removed = await requestJson({
      port,
      method: 'DELETE',
      path: `/api/v2/trips/${tripId}/stops/${removedStopId}`,
      headers: authHeader('user-a'),
    });
    assert.equal(removed.statusCode, 200, removed.text);
    assert.equal(removed.body.data.trip.needsReplan, true);
    assert.ok(!removed.body.data.trip.itinerary.some((stop) => stop.stopId === removedStopId));
    assert.ok(removed.body.data.trip.excludedPoiIds.includes(removedPoiId));

    const userBReplan = await requestJson({
      port,
      method: 'POST',
      path: `/api/v2/trips/${tripId}/replan`,
      headers: authHeader('user-b'),
    });
    assert.equal(userBReplan.statusCode, 404);

    const replanned = await requestJson({
      port,
      method: 'POST',
      path: `/api/v2/trips/${tripId}/replan`,
      headers: authHeader('user-a'),
    });
    assert.equal(replanned.statusCode, 200, replanned.text);
    assert.equal(replanned.body.data.trip.tripId, tripId);
    assert.equal(replanned.body.data.trip.needsReplan, false);
    assert.equal(replanned.body.data.trip.preview.persisted, true);
    assert.equal(replanned.body.data.trip.preview.preview, false);
    assert.ok(replanned.body.data.trip.itinerary.length > 0);

    const fetched = await requestJson({
      port,
      path: `/api/v2/trips/${tripId}`,
      headers: authHeader('user-a'),
    });
    assert.equal(fetched.statusCode, 200);
    assert.equal(fetched.body.data.trip.tripId, tripId);
    assert.equal(fetched.body.data.trip.needsReplan, false);
    assert.deepEqual(
      fetched.body.data.trip.itinerary.map((stop) => stop.stopId),
      replanned.body.data.trip.itinerary.map((stop) => stop.stopId),
    );

    const previewAfterLifecycle = await requestJson({
      port,
      method: 'POST',
      path: '/api/v2/trips/preview',
      body: previewInput,
    });
    assert.equal(previewAfterLifecycle.statusCode, 200);
    assert.equal(previewAfterLifecycle.body.data.trip.persisted, false);
    assert.equal(previewAfterLifecycle.body.data.trip.tripId, null);
  } finally {
    await stopServer(child);
  }
});
