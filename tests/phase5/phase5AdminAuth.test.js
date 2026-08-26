const assert = require('node:assert/strict');
const test = require('node:test');
const express = require('express');

const { requireAdmin } = require('../../src/middleware/adminAuth');
const { createVerifiedFirebaseAuth } = require('../../src/middleware/firebaseAuth');
const { createAdminRouter } = require('../../src/modules/admin/adminRouter');

function fakeDecodedToken(token) {
  const tokens = {
    traveler: { uid: 'traveler-1', email: 'traveler@example.test' },
    'admin-false': { uid: 'user-2', email: 'user2@example.test', admin: false },
    admin: { uid: 'admin-1', email: 'admin@example.test', admin: true },
  };
  if (!tokens[token]) throw new Error('Sensitive Firebase verifier detail');
  return tokens[token];
}

function createFakeAuth() {
  return {
    verifyIdToken: async (token) => fakeDecodedToken(token),
    listUsers: async (limit, pageToken) => ({
      users: [{
        uid: 'admin-1',
        email: 'admin@example.test',
        displayName: 'Admin One',
        disabled: false,
        emailVerified: true,
        metadata: {
          creationTime: '2026-01-01T00:00:00.000Z',
          lastSignInTime: '2026-08-25T00:00:00.000Z',
        },
        customClaims: {
          admin: true,
          privateKey: 'must-not-leak',
          refreshToken: 'must-not-leak',
        },
      }].slice(0, limit),
      pageToken: pageToken ? null : 'next-page-token',
    }),
  };
}

function createTestApp() {
  const auth = createFakeAuth();
  const authenticate = createVerifiedFirebaseAuth({
    authProvider: () => auth,
    readinessProvider: () => true,
  });
  const app = express();
  app.get('/api/traveler-probe', authenticate, (req, res) => {
    res.json({ uid: req.user.uid });
  });
  app.use('/api/admin', createAdminRouter({
    authenticate,
    authorize: requireAdmin,
    rateLimit: (req, res, next) => next(),
    authProvider: () => auth,
    qualityProvider: async () => ({
      cityId: 'da-nang',
      dataset: { path: 'data/canonical/runtime.csv', sha256: 'safe-hash' },
      totals: {
        rows: 4173,
        applicationPois: 4173,
        invalidRows: 0,
        missingCoordinates: 0,
        outsideDaNangBBox: 0,
      },
      headerMatchesExpected: true,
    }),
    firebaseReadyProvider: () => true,
    firestoreProvider: () => ({}),
    overviewProvider: async () => ({
      counts: {
        users: { value: 3, exact: true },
        trips: { value: 2, exact: true },
        feedback: { value: 4, exact: true },
      },
      recentActivity: [{ id: 'trip:trip-1', type: 'trip', label: 'Beach day', ownerId: 'traveler-1', occurredAt: '2026-08-25T01:00:00.000Z' }],
    }),
    poiProvider: async () => [{
      id: 'poi-1',
      globalId: 'poi-1',
      name: 'Safe Cafe',
      category: 'Cafe',
      address: 'Da Nang',
      source: 'canonical',
      rating: null,
      reviewCount: null,
      lat: 16.05,
      lon: 108.2,
      raw: { privatePayload: 'must-not-leak' },
    }],
    tripListProvider: async () => ({ trips: [{ tripId: 'trip-1', ownerId: 'traveler-1', title: 'Beach day', dayCount: 2, stopCount: 4 }] }),
    tripDetailProvider: async ({ tripId }) => tripId === 'trip-1'
      ? { tripId, ownerId: 'traveler-1', title: 'Beach day', dayCount: 2, stopCount: 4, stops: [] }
      : null,
    feedbackProvider: async () => ({ feedback: [{ eventId: 'event-1', userId: 'traveler-1', eventType: 'poi_useful', rating: 5, message: 'Helpful' }] }),
  }));
  return app;
}

async function withServer(run) {
  const server = createTestApp().listen(0, '127.0.0.1');
  await new Promise((resolve) => server.once('listening', resolve));
  const { port } = server.address();
  try {
    await run(`http://127.0.0.1:${port}`);
  } finally {
    await new Promise((resolve, reject) => server.close((error) => (error ? reject(error) : resolve())));
  }
}

async function get(baseUrl, path, token) {
  return fetch(`${baseUrl}${path}`, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
}

test('Admin API returns 401 when Authorization is missing', async () => {
  await withServer(async (baseUrl) => {
    const response = await get(baseUrl, '/api/admin/me');
    assert.equal(response.status, 401);
    assert.deepEqual(await response.json(), { error: 'authentication_required' });
  });
});

test('Admin API returns sanitized 401 for an invalid Firebase token', async () => {
  await withServer(async (baseUrl) => {
    const response = await get(baseUrl, '/api/admin/me', 'invalid');
    const body = await response.json();
    assert.equal(response.status, 401);
    assert.equal(body.error, 'invalid_firebase_token');
    assert.ok(!JSON.stringify(body).includes('Sensitive Firebase verifier detail'));
  });
});

test('valid Firebase user without admin claim receives 403', async () => {
  await withServer(async (baseUrl) => {
    const response = await get(baseUrl, '/api/admin/me', 'traveler');
    assert.equal(response.status, 403);
    assert.deepEqual(await response.json(), { error: 'admin_access_required' });
  });
});

test('valid Firebase user with admin=false receives 403', async () => {
  await withServer(async (baseUrl) => {
    const response = await get(baseUrl, '/api/admin/me', 'admin-false');
    assert.equal(response.status, 403);
  });
});

test('valid Firebase user with admin=true can read safe identity', async () => {
  await withServer(async (baseUrl) => {
    const response = await get(baseUrl, '/api/admin/me', 'admin');
    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), {
      uid: 'admin-1',
      email: 'admin@example.test',
      admin: true,
    });
  });
});

test('non-admin Firebase user still passes normal authenticated traveler middleware', async () => {
  await withServer(async (baseUrl) => {
    const response = await get(baseUrl, '/api/traveler-probe', 'traveler');
    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), { uid: 'traveler-1' });
  });
});

test('Admin users endpoint rejects non-admin and returns safe pagination for admin', async () => {
  await withServer(async (baseUrl) => {
    const forbidden = await get(baseUrl, '/api/admin/users', 'traveler');
    assert.equal(forbidden.status, 403);

    const allowed = await get(baseUrl, '/api/admin/users?limit=500', 'admin');
    const body = await allowed.json();
    assert.equal(allowed.status, 200);
    assert.equal(body.users.length, 1);
    assert.equal(body.nextPageToken, 'next-page-token');
    assert.equal(body.users[0].admin, true);
    assert.equal(body.users[0].privateKey, undefined);
    assert.equal(body.users[0].refreshToken, undefined);
    assert.equal(body.users[0].customClaims, undefined);
  });
});

test('Admin capability, POI summary, and health endpoints expose only read-only safe state', async () => {
  await withServer(async (baseUrl) => {
    const [capabilities, pois, health] = await Promise.all([
      get(baseUrl, '/api/admin/capabilities', 'admin').then((response) => response.json()),
      get(baseUrl, '/api/admin/pois/summary', 'admin').then((response) => response.json()),
      get(baseUrl, '/api/admin/health', 'admin').then((response) => response.json()),
    ]);
    assert.equal(capabilities.capabilities.users.read, true);
    assert.equal(capabilities.capabilities.users.write, false);
    assert.equal(capabilities.capabilities.trips.read, true);
    assert.equal(capabilities.capabilities.feedback.read, true);
    assert.equal(pois.canonicalCount, 4173);
    assert.equal(health.services.googleMaps.status, 'GOOGLE_LIVE_CONFIGURATION_PENDING');
  });
});

test('Admin demo read endpoints expose real safe summaries without raw provider payloads', async () => {
  await withServer(async (baseUrl) => {
    const responses = await Promise.all([
      get(baseUrl, '/api/admin/overview', 'admin'),
      get(baseUrl, '/api/admin/pois?query=cafe', 'admin'),
      get(baseUrl, '/api/admin/pois/poi-1', 'admin'),
      get(baseUrl, '/api/admin/trips', 'admin'),
      get(baseUrl, '/api/admin/trips/trip-1', 'admin'),
      get(baseUrl, '/api/admin/feedback', 'admin'),
    ]);
    assert.deepEqual(responses.map((response) => response.status), [200, 200, 200, 200, 200, 200]);
    const [overview, pois, poi, trips, trip, feedback] = await Promise.all(responses.map((response) => response.json()));
    assert.equal(overview.counts.users.value, 3);
    assert.equal(pois.total, 1);
    assert.equal(pois.pois[0].poiId, 'poi-1');
    assert.equal(poi.poi.rating, null);
    assert.equal(trips.trips[0].ownerId, 'traveler-1');
    assert.equal(trip.trip.tripId, 'trip-1');
    assert.equal(feedback.feedback[0].rating, 5);
    assert.ok(!JSON.stringify({ overview, pois, poi, trips, trip, feedback }).includes('privatePayload'));
  });
});

test('Admin detail endpoints return sanitized 404 responses', async () => {
  await withServer(async (baseUrl) => {
    const response = await get(baseUrl, '/api/admin/trips/missing', 'admin');
    assert.equal(response.status, 404);
    assert.deepEqual(await response.json(), { error: 'admin_trip_not_found' });
  });
});

test('Admin namespace blocks writes and has no self-elevation route', async () => {
  await withServer(async (baseUrl) => {
    const write = await fetch(`${baseUrl}/api/admin/users/admin-1/status`, {
      method: 'POST',
      headers: { Authorization: 'Bearer admin', 'Content-Type': 'application/json' },
      body: JSON.stringify({ admin: true }),
    });
    const elevate = await fetch(`${baseUrl}/api/admin/make-me-admin`, {
      method: 'POST',
      headers: { Authorization: 'Bearer admin', 'Content-Type': 'application/json' },
      body: '{}',
    });
    assert.equal(write.status, 405);
    assert.equal(elevate.status, 405);
  });
});

test('local development token and URL manipulation cannot unlock Admin', async () => {
  await withServer(async (baseUrl) => {
    const response = await get(
      baseUrl,
      '/api/admin/me?role=admin&admin=true',
      'local-admin-dev-token',
    );
    assert.equal(response.status, 401);
  });
});
