const express = require('express');
const { getFirebaseAuth, getFirestoreDb, isFirebaseAdminReady } = require('../../config/firebaseAdmin');
const { requireAdmin } = require('../../middleware/adminAuth');
const { createAdminRateLimit } = require('../../middleware/adminRateLimit');
const { requireVerifiedFirebaseAuth } = require('../../middleware/firebaseAuth');
const { getPoiDataQualityReport, loadPOIs } = require('../../services/poiDataService');
const {
  getAdminOverview,
  getAdminPoi,
  getAdminTrip,
  listAdminFeedback,
  listAdminPois,
  listAdminTrips,
} = require('./adminReadService');

const ADMIN_CAPABILITIES = Object.freeze({
  identity: { read: true },
  users: { read: true, write: false },
  pois: { read: true, write: false },
  trips: { read: true, write: false },
  feedback: { read: true, write: false },
  analytics: { read: true },
  agentTelemetry: { read: false },
  integrations: { read: true, write: false },
  health: { read: true },
  logs: { read: false },
});

function parseUserLimit(value) {
  const parsed = Number.parseInt(String(value || ''), 10);
  if (!Number.isFinite(parsed)) return 50;
  return Math.min(Math.max(parsed, 1), 100);
}

function parsePageToken(value) {
  const token = String(value || '').trim();
  return token ? token.slice(0, 4096) : undefined;
}

function serializeAdminUser(user = {}) {
  return {
    uid: String(user.uid || ''),
    email: user.email || null,
    displayName: user.displayName || null,
    disabled: user.disabled === true,
    emailVerified: user.emailVerified === true,
    creationTime: user.metadata?.creationTime || null,
    lastSignInTime: user.metadata?.lastSignInTime || null,
    admin: user.customClaims?.admin === true,
  };
}

function runtimeRepositoryMode() {
  return process.env.URBANAGENT_POI_REPOSITORY === 'postgres' ? 'postgres' : 'csv';
}

function createAdminRouter({
  authenticate = requireVerifiedFirebaseAuth,
  authorize = requireAdmin,
  rateLimit = createAdminRateLimit({
    maxRequests: process.env.URBANAGENT_ADMIN_RATE_LIMIT_PER_MINUTE,
  }),
  authProvider = getFirebaseAuth,
  qualityProvider = getPoiDataQualityReport,
  poiProvider = loadPOIs,
  firebaseReadyProvider = isFirebaseAdminReady,
  firestoreProvider = getFirestoreDb,
  overviewProvider = getAdminOverview,
  poiListProvider = listAdminPois,
  poiDetailProvider = getAdminPoi,
  tripListProvider = listAdminTrips,
  tripDetailProvider = getAdminTrip,
  feedbackProvider = listAdminFeedback,
} = {}) {
  const router = express.Router();

  router.use(authenticate, authorize, rateLimit);

  router.get('/me', (req, res) => {
    res.json({
      uid: req.firebaseUser.uid,
      email: req.firebaseUser.email || null,
      admin: true,
    });
  });

  router.get('/capabilities', (req, res) => {
    res.json({ capabilities: ADMIN_CAPABILITIES });
  });

  router.get('/users', async (req, res) => {
    try {
      const auth = authProvider();
      if (!auth) return res.status(503).json({ error: 'admin_user_directory_unavailable' });
      const result = await auth.listUsers(
        parseUserLimit(req.query.limit),
        parsePageToken(req.query.pageToken),
      );
      return res.json({
        users: (result.users || []).map(serializeAdminUser),
        nextPageToken: result.pageToken || null,
      });
    } catch {
      return res.status(503).json({ error: 'admin_user_directory_unavailable' });
    }
  });

  router.get('/overview', async (req, res) => {
    try {
      const overview = await overviewProvider({
        auth: authProvider(),
        db: firestoreProvider(),
      });
      return res.json(overview);
    } catch {
      return res.status(503).json({ error: 'admin_overview_unavailable' });
    }
  });

  router.get('/pois/summary', async (req, res) => {
    try {
      const quality = await qualityProvider();
      return res.json({
        cityId: quality.cityId || null,
        canonicalCount: quality.totals?.applicationPois ?? null,
        runtimeRepository: runtimeRepositoryMode(),
        dataset: {
          path: quality.dataset?.path || null,
          sha256: quality.dataset?.sha256 || null,
        },
        quality: {
          rows: quality.totals?.rows ?? null,
          invalidRows: quality.totals?.invalidRows ?? null,
          missingCoordinates: quality.totals?.missingCoordinates ?? null,
          outsideCityBounds: quality.totals?.outsideDaNangBBox ?? null,
          headerMatchesExpected: quality.headerMatchesExpected === true,
        },
      });
    } catch {
      return res.status(503).json({ error: 'admin_poi_summary_unavailable' });
    }
  });

  router.get('/pois', async (req, res) => {
    try {
      const pois = await poiProvider();
      return res.json(poiListProvider({
        pois,
        query: req.query.query,
        category: req.query.category,
        source: req.query.source,
        limit: req.query.limit,
        offset: req.query.offset,
      }));
    } catch {
      return res.status(503).json({ error: 'admin_poi_list_unavailable' });
    }
  });

  router.get('/pois/:poiId', async (req, res) => {
    try {
      const pois = await poiProvider();
      const poi = poiDetailProvider({ pois, poiId: req.params.poiId });
      if (!poi) return res.status(404).json({ error: 'admin_poi_not_found' });
      return res.json({ poi });
    } catch {
      return res.status(503).json({ error: 'admin_poi_detail_unavailable' });
    }
  });

  router.get('/trips', async (req, res) => {
    try {
      return res.json(await tripListProvider({
        db: firestoreProvider(),
        limit: req.query.limit,
      }));
    } catch {
      return res.status(503).json({ error: 'admin_trip_list_unavailable' });
    }
  });

  router.get('/trips/:tripId', async (req, res) => {
    try {
      const trip = await tripDetailProvider({
        db: firestoreProvider(),
        tripId: req.params.tripId,
      });
      if (!trip) return res.status(404).json({ error: 'admin_trip_not_found' });
      return res.json({ trip });
    } catch {
      return res.status(503).json({ error: 'admin_trip_detail_unavailable' });
    }
  });

  router.get('/feedback', async (req, res) => {
    try {
      return res.json(await feedbackProvider({
        db: firestoreProvider(),
        limit: req.query.limit,
      }));
    } catch {
      return res.status(503).json({ error: 'admin_feedback_unavailable' });
    }
  });

  router.get('/health', async (req, res) => {
    try {
      const quality = await qualityProvider();
      const canonicalReady = (
        Number(quality.totals?.applicationPois) > 0
        && quality.totals?.invalidRows === 0
        && quality.headerMatchesExpected === true
      );
      return res.json({
        services: {
          backend: { status: 'active' },
          canonicalPoiRuntime: {
            status: canonicalReady ? 'active' : 'degraded',
            applicationPois: quality.totals?.applicationPois ?? null,
            repository: runtimeRepositoryMode(),
          },
          googleMaps: {
            status: 'GOOGLE_LIVE_CONFIGURATION_PENDING',
            liveAcceptanceComplete: false,
          },
          photon: { status: 'configured', liveChecked: false },
          osrm: { status: 'unknown', liveChecked: false },
          firebase: {
            status: firebaseReadyProvider() ? 'active' : 'unavailable',
            firestoreReady: Boolean(firestoreProvider()),
          },
        },
      });
    } catch {
      return res.status(503).json({ error: 'admin_health_unavailable' });
    }
  });

  router.use((req, res) => {
    if (req.method !== 'GET') {
      return res.status(405).json({ error: 'admin_api_read_only' });
    }
    return res.status(404).json({ error: 'admin_endpoint_not_found' });
  });

  return router;
}

module.exports = {
  ADMIN_CAPABILITIES,
  createAdminRouter,
  parseUserLimit,
  serializeAdminUser,
};
