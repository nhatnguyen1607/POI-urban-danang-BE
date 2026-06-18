const express = require('express');
const cors = require('cors');
const multer = require('multer');
const { spawn } = require('child_process');
const fs = require('fs');
const csv = require('csv-parser');
const path = require('path');

function isClosedPipeError(error) {
  return ['EPIPE', 'EOF', 'ECONNRESET', 'ERR_STREAM_DESTROYED'].includes(error?.code);
}

function ignoreBrokenPipe(stream) {
  stream.on('error', (error) => {
    if (!isClosedPipeError(error)) throw error;
  });
}

ignoreBrokenPipe(process.stdout);
ignoreBrokenPipe(process.stderr);

process.on('uncaughtException', (error) => {
  if (isClosedPipeError(error)) return;
  throw error;
});

process.on('unhandledRejection', (reason) => {
  if (isClosedPipeError(reason)) return;
  throw reason;
});

const { initExpertSystem, findOptimalRoute } = require('./expert_system/expert_system');
const { recommendPOIs } = require('./services/poiRetrievalService');
const { createItinerary } = require('./services/itineraryPlannerService');
const { scoreBusinessLocations } = require('./services/businessLocationScorer');
const { getForecast } = require('./services/weatherService');
const { estimateMatrix } = require('./services/routeMatrixService');
const { resolvePythonExecutable } = require('./services/semanticModelService');
const { recordFeedback } = require('./services/feedbackService');
const { recommendContextualPOIs } = require('./services/contextualPoiRecommenderService');
const { rebuildUserPreferences } = require('./services/userPreferenceService');
const { generateBusinessInsights } = require('./services/businessInsightGenerator');
const { getAgentTrainingStatus } = require('./services/trainingStatusService');
const { loadPOIs } = require('./services/poiDataService');
const {
  ensureUserDocument,
  getCustomerProfile,
  getAgentMemory,
  listAdminReviews,
  listBusinessAnalyses,
  listUsers,
  listPois,
  listSellerConcepts,
  listItineraries,
  listSellerBusinesses,
  saveBusinessAnalysis,
  saveAdminReview,
  saveAgentMemory,
  saveCustomerProfile,
  saveItinerary,
  saveSellerConcept,
  saveSellerBusiness,
  updatePoiStatus,
  updateUserStatus,
  upsertPoi,
  updateUserRole,
} = require('./services/firestorePersistenceService');
const { getFirebaseAdminDiagnostics, getFirestoreDb, isFirebaseAdminReady } = require('./config/firebaseAdmin');
const { optionalFirebaseAuth, requireFirebaseAuth } = require('./middleware/firebaseAuth');

const ROOT_DIR = path.resolve(__dirname, '..');
const DATA_DIR = path.join(ROOT_DIR, 'data');
const MODEL_DIR = path.join(ROOT_DIR, 'artifacts', 'model');
const LEGACY_MODEL_DIR = path.join(ROOT_DIR, 'model');
const STORAGE_DIR = path.join(ROOT_DIR, 'storage');
const UPLOAD_DIR = path.join(STORAGE_DIR, 'uploads');

fs.mkdirSync(UPLOAD_DIR, { recursive: true });

const app = express();
app.use(cors());

function errorStatus(error) {
  return error?.status || (error?.code === 'FIRESTORE_NOT_CONFIGURED' ? 503 : 500);
}

// Parse JSON but skip multipart/form-data (handled by multer)
app.use(express.json({
  skip: (req) => req.is('multipart/form-data')
}));

const upload = multer({ dest: UPLOAD_DIR });

app.get('/api/health/firebase', (req, res) => {
  const db = getFirestoreDb();
  res.json({
    firebaseAdminReady: isFirebaseAdminReady(),
    firestoreReady: Boolean(db),
    projectId: process.env.FIREBASE_PROJECT_ID || process.env.GCLOUD_PROJECT || null,
    diagnostics: getFirebaseAdminDiagnostics(),
  });
});

// Endpoint to serve figures dynamically by version
app.get('/api/figures/:version/:filename', (req, res) => {
  const { version, filename } = req.params;
  if (!['v1', 'v2', 'v3', 'v4'].includes(version)) {
    return res.status(400).send('Invalid version');
  }
  const filePath = path.join(MODEL_DIR, version, 'reports', 'figures', filename);
  const legacyFilePath = path.join(LEGACY_MODEL_DIR, version, 'reports', 'figures', filename);
  if (fs.existsSync(filePath)) {
    res.sendFile(filePath);
  } else if (fs.existsSync(legacyFilePath)) {
    res.sendFile(legacyFilePath);
  } else {
    res.status(404).send('Not found');
  }
});

// Helper to read CSV
const readCSV = (filePath) => {
  return new Promise((resolve, reject) => {
    const results = [];
    fs.createReadStream(filePath)
      .pipe(csv())
      .on('data', (data) => results.push(data))
      .on('end', () => resolve(results))
      .on('error', (err) => reject(err));
  });
};

app.get('/api/eda', async (req, res) => {
  try {
    const source = req.query.source || 'ggmap';
    const fileName = source === 'foody' ? 'poi_data_foody.csv' : 'poi_data_ggmap.csv';
    const filePath = path.join(DATA_DIR, fileName);
    
    if (!fs.existsSync(filePath)) {
      return res.status(404).json({ error: 'Data file not found' });
    }

    const data = await readCSV(filePath);
    
    // Calculate EDA Metrics
    const totalPOIs = data.length;
    
    const categories = new Set();
    let totalRating = 0;
    let ratingCount = 0;

    data.forEach(row => {
      const category = row.category || row.Category;
      const rating = row.rating || row['Overall Rating'];
      
      if (category) categories.add(category);
      if (rating && !isNaN(parseFloat(rating))) {
        totalRating += parseFloat(rating);
        ratingCount++;
      }
    });

    const numCategories = categories.size;
    const avgRating = ratingCount > 0 ? (totalRating / ratingCount).toFixed(1) : 0;
    // We mock districts if it's not in the CSV directly (from address)
    const numDistricts = 7; 

    // Return first 50 items for table to avoid huge payload
    const sampleData = data.slice(0, 50).map(row => ({
      id: row.place_id || row.RestaurantID || Math.random().toString(36).substring(7),
      name: row.name || row['Restaurant Name'],
      address: row.address || row.Address,
      category: row.category || row.Category,
      rating: row.rating || row['Overall Rating'],
      price: row.price_range || row.Price || 'Chưa cập nhật',
      lat: row.lat || row.Lat,
      lng: row.lng || row.Lon
    }));

    res.json({
      metrics: {
        totalPOIs,
        numDistricts,
        numCategories,
        avgRating
      },
      sampleData
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to process EDA' });
  }
});

app.get('/api/metrics/training-loss', async (req, res) => {
  try {
    const version = req.query.version || 'v1';
    if (!['v1', 'v2', 'v3', 'v4'].includes(version)) {
      return res.status(400).json({ error: 'Invalid model version' });
    }
    const filePath = path.join(MODEL_DIR, version, 'reports', 'metrics', `training_loss_${version}.csv`);
    const legacyFilePath = path.join(LEGACY_MODEL_DIR, version, 'reports', 'metrics', `training_loss_${version}.csv`);
    const resolvedPath = fs.existsSync(filePath) ? filePath : legacyFilePath;
    if (!fs.existsSync(resolvedPath)) {
      return res.status(404).json({ error: 'Metrics file not found' });
    }
    const data = await readCSV(resolvedPath);
    res.json(data);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to read metrics' });
  }
});

// ============================================================================
//  URBAN AGENT MVP: Traveler + Business role endpoints
// ============================================================================
app.post('/api/auth/ensure-user', requireFirebaseAuth, async (req, res) => {
  try {
    const result = await ensureUserDocument({
      uid: req.user.uid,
      email: req.user.email,
      displayName: req.body.displayName || req.user.name,
      photoURL: req.body.photoURL || req.user.picture,
      phone: req.body.phone,
      role: req.body.role || req.user.role,
      language: req.body.language,
    });
    res.json(result);
  } catch (error) {
    res.status(errorStatus(error)).json({ error: 'Failed to ensure user document', details: error.message });
  }
});

app.post('/api/auth/role', requireFirebaseAuth, async (req, res) => {
  try {
    const result = await updateUserRole({
      uid: req.user.uid,
      role: req.body.role,
      language: req.body.language,
    });
    res.json(result);
  } catch (error) {
    res.status(errorStatus(error)).json({ error: 'Failed to update user role', details: error.message });
  }
});

app.get('/api/customer/profile', requireFirebaseAuth, async (req, res) => {
  try {
    const profile = await getCustomerProfile(req.user.uid);
    res.json({ profile });
  } catch (error) {
    res.status(errorStatus(error)).json({ error: 'Failed to get customer profile', details: error.message });
  }
});

app.post('/api/customer/profile', requireFirebaseAuth, async (req, res) => {
  try {
    const result = await saveCustomerProfile({
      ...req.body,
      userId: req.user.uid,
    });
    res.json(result);
  } catch (error) {
    res.status(errorStatus(error)).json({ error: 'Failed to save customer profile', details: error.message });
  }
});

app.get('/api/agent/memory', requireFirebaseAuth, async (req, res) => {
  try {
    const memory = await getAgentMemory(req.user.uid);
    res.json({ memory });
  } catch (error) {
    res.status(errorStatus(error)).json({ error: 'Failed to get agent memory', details: error.message });
  }
});

app.post('/api/agent/memory', requireFirebaseAuth, async (req, res) => {
  try {
    const result = await saveAgentMemory({
      ...req.body,
      userId: req.user.uid,
    });
    res.json(result);
  } catch (error) {
    res.status(errorStatus(error)).json({ error: 'Failed to save agent memory', details: error.message });
  }
});

app.get('/api/pois', requireFirebaseAuth, async (req, res) => {
  try {
    const pois = await listPois({ limit: req.query.limit, status: req.query.status || 'active' });
    res.json({ pois });
  } catch (error) {
    res.status(errorStatus(error)).json({ error: 'Failed to list POIs', details: error.message });
  }
});

app.post('/api/pois', requireFirebaseAuth, async (req, res) => {
  try {
    const result = await upsertPoi({
      ...req.body,
      ownerId: req.body.ownerId || req.user.uid,
      source: req.body.source || 'manual',
      status: req.body.status || 'pending',
    });
    res.json(result);
  } catch (error) {
    res.status(errorStatus(error)).json({ error: 'Failed to save POI', details: error.message });
  }
});

app.post('/api/pois/sync-local', requireFirebaseAuth, async (req, res) => {
  try {
    const limit = Math.min(Number(req.body.limit || 200), 1000);
    const localPois = (await loadPOIs()).slice(0, limit);
    const results = [];
    for (const poi of localPois) {
      results.push(await upsertPoi({
        ...poi,
        poiId: poi.id,
        semanticText: poi.text || poi.normalized,
        location: { lat: poi.lat, lng: poi.lon, district: poi.district, address: poi.district },
        status: 'active',
        verified: true,
      }));
    }
    res.json({ synced: results.length, poiIds: results.map((item) => item.poiId) });
  } catch (error) {
    res.status(errorStatus(error)).json({ error: 'Failed to sync local POIs', details: error.message });
  }
});

app.post('/api/agent/recommend-poi', optionalFirebaseAuth, async (req, res) => {
  try {
    const { query, context, limit } = req.body;
    if (!query || !String(query).trim()) {
      return res.status(400).json({ error: 'Missing query' });
    }
    const agentMemory = req.user?.uid ? await getAgentMemory(req.user.uid).catch(() => null) : null;
    const result = await recommendPOIs({
      query,
      context: { ...(context || {}), userId: req.user?.uid || context?.userId || null, agentMemory },
      limit,
    });
    res.json(result);
  } catch (error) {
    console.error('[Agent Recommend Error]', error);
    res.status(500).json({ error: 'Failed to recommend POIs', details: error.message });
  }
});

app.post('/api/recommend-pois', optionalFirebaseAuth, async (req, res) => {
  try {
    const userId = req.body.userId || req.user?.uid || null;
    const currentLocation = req.body.currentLocation || req.body.location;
    const currentContext = req.body.currentContext || {};
    const result = await recommendContextualPOIs({
      userId,
      currentLocation,
      currentContext,
      limit: req.body.limit,
    });
    res.json(result);
  } catch (error) {
    console.error('[Contextual Recommend Error]', error);
    res.status(errorStatus(error)).json({ error: 'Failed to recommend contextual POIs', details: error.message });
  }
});

app.post('/api/user-preferences/rebuild', requireFirebaseAuth, async (req, res) => {
  try {
    const userId = req.body.userId || req.user.uid;
    if (userId !== req.user.uid && req.user.role !== 'admin') {
      return res.status(403).json({ error: 'Cannot rebuild another user profile' });
    }
    const profile = await rebuildUserPreferences(userId);
    res.json({ saved: true, profile });
  } catch (error) {
    res.status(errorStatus(error)).json({ error: 'Failed to rebuild user preferences', details: error.message });
  }
});

app.post('/api/agent/create-itinerary', requireFirebaseAuth, async (req, res) => {
  try {
    const body = req.body && typeof req.body === 'object' ? req.body : {};
    const { query, context, transport, limit, durationMinutes } = body;
    if (!query || !String(query).trim()) {
      return res.status(400).json({ error: 'Missing query' });
    }
    const agentMemory = await getAgentMemory(req.user.uid).catch(() => null);
    const result = await createItinerary({
      query,
      context: { ...(context || {}), userId: req.user.uid, agentMemory },
      transport,
      limit,
      durationMinutes,
    });
    result.userId = req.user.uid;
    res.json(result);
  } catch (error) {
    console.error('[Agent Itinerary Error]', error);
    res.status(500).json({ error: 'Failed to create itinerary', details: error.message });
  }
});

app.post('/api/agent/update-itinerary', requireFirebaseAuth, async (req, res) => {
  try {
    const { itinerary = [], action, poi } = req.body;
    let updated = Array.isArray(itinerary) ? [...itinerary] : [];
    if (action === 'add_poi' && poi) {
      updated.push({
        order: updated.length + 1,
        poi,
        reason: 'Người dùng thêm thủ công vào lịch trình.',
      });
    }
    if (action === 'remove_poi' && poi?.id) {
      updated = updated.filter((item) => item.poi?.id !== poi.id);
    }
    updated = updated.map((item, index) => ({ ...item, order: index + 1 }));
    res.json({ itinerary: updated, userId: req.user.uid, warnings: [], note: 'MVP cập nhật lịch trình ở backend.' });
  } catch (error) {
    res.status(500).json({ error: 'Failed to update itinerary', details: error.message });
  }
});

app.get('/api/agent/itineraries', requireFirebaseAuth, async (req, res) => {
  try {
    const itineraries = await listItineraries(req.user.uid);
    res.json({ itineraries });
  } catch (error) {
    res.status(errorStatus(error)).json({ error: 'Failed to list itineraries', details: error.message });
  }
});

app.post('/api/agent/itineraries', requireFirebaseAuth, async (req, res) => {
  try {
    const result = await saveItinerary({
      ...req.body,
      userId: req.user.uid,
    });
    await recordFeedback({
      role: 'customer',
      eventType: 'itinerary_saved',
      query: req.body.query,
      userId: req.user.uid,
      payload: { itineraryId: result.itineraryId, stopCount: req.body.itinerary?.length || req.body.stops?.length || 0 },
    });
    res.json(result);
  } catch (error) {
    res.status(errorStatus(error)).json({ error: 'Failed to save itinerary', details: error.message });
  }
});

app.post('/api/agent/business-location', requireFirebaseAuth, async (req, res) => {
  try {
    const { concept, limit } = req.body;
    if (!concept || !String(concept).trim()) {
      return res.status(400).json({ error: 'Missing business concept' });
    }
    const result = await scoreBusinessLocations({ concept, limit });
    result.sellerId = req.user.uid;
    res.json(result);
  } catch (error) {
    console.error('[Business Agent Error]', error);
    res.status(500).json({ error: 'Failed to score business locations', details: error.message });
  }
});

app.post('/api/agent/business-insight', requireFirebaseAuth, async (req, res) => {
  try {
    const { concept, limit, language } = req.body;
    if (!concept || !String(concept).trim()) {
      return res.status(400).json({ error: 'Missing business concept' });
    }
    const result = await generateBusinessInsights({ concept, limit, language });
    result.sellerId = req.user.uid;
    res.json(result);
  } catch (error) {
    console.error('[Business Insight Error]', error);
    res.status(errorStatus(error)).json({ error: 'Failed to generate business insights', details: error.message });
  }
});

app.get('/api/seller/concepts', requireFirebaseAuth, async (req, res) => {
  try {
    const concepts = await listSellerConcepts(req.user.uid);
    res.json({ concepts });
  } catch (error) {
    res.status(errorStatus(error)).json({ error: 'Failed to list seller concepts', details: error.message });
  }
});

app.post('/api/seller/concepts', requireFirebaseAuth, async (req, res) => {
  try {
    const result = await saveSellerConcept({
      ...req.body,
      sellerId: req.user.uid,
    });
    if (req.body.analysis || req.body.rawResult) {
      const savedAnalysis = await saveBusinessAnalysis({
        sellerId: req.user.uid,
        concept: req.body.query || req.body.concept,
        areas: req.body.analysis?.areas || req.body.rawResult?.areas || [],
        rawResult: req.body.analysis || req.body.rawResult,
      });
      result.analysisId = savedAnalysis.analysisId;
    }
    res.json(result);
  } catch (error) {
    res.status(errorStatus(error)).json({ error: 'Failed to save seller concept', details: error.message });
  }
});

app.get('/api/seller/business-analyses', requireFirebaseAuth, async (req, res) => {
  try {
    const analyses = await listBusinessAnalyses(req.user.uid);
    res.json({ analyses });
  } catch (error) {
    res.status(errorStatus(error)).json({ error: 'Failed to list business analyses', details: error.message });
  }
});

app.get('/api/seller/businesses', requireFirebaseAuth, async (req, res) => {
  try {
    const businesses = await listSellerBusinesses(req.user.uid);
    res.json({ businesses });
  } catch (error) {
    res.status(errorStatus(error)).json({ error: 'Failed to list seller businesses', details: error.message });
  }
});

app.post('/api/seller/businesses', requireFirebaseAuth, async (req, res) => {
  try {
    const { name, category, address } = req.body;
    if (!name || !category || !address) {
      return res.status(400).json({ error: 'Missing seller business fields' });
    }
    const result = await saveSellerBusiness({
      ...req.body,
      userId: req.user.uid,
    });
    res.json(result);
  } catch (error) {
    res.status(errorStatus(error)).json({ error: 'Failed to save seller business', details: error.message });
  }
});

app.post('/api/agent/feedback', requireFirebaseAuth, async (req, res) => {
  try {
    const result = await recordFeedback({
      ...req.body,
      userId: req.user.uid,
      authEmail: req.user.email,
    });
    res.json(result);
  } catch (error) {
    console.error('[Agent Feedback Error]', error);
    res.status(errorStatus(error)).json({ error: 'Failed to record feedback', details: error.message });
  }
});

app.get('/api/admin/reviews', requireFirebaseAuth, async (req, res) => {
  try {
    const reviews = await listAdminReviews({ status: req.query.status, limit: req.query.limit });
    res.json({ reviews });
  } catch (error) {
    res.status(errorStatus(error)).json({ error: 'Failed to list admin reviews', details: error.message });
  }
});

app.post('/api/admin/reviews', requireFirebaseAuth, async (req, res) => {
  try {
    const result = await saveAdminReview({
      ...req.body,
      reviewerId: req.user.uid,
    });
    res.json(result);
  } catch (error) {
    res.status(errorStatus(error)).json({ error: 'Failed to save admin review', details: error.message });
  }
});

app.get('/api/admin/users', requireFirebaseAuth, async (req, res) => {
  try {
    const users = await listUsers({ limit: req.query.limit, role: req.query.role, status: req.query.status });
    res.json({ users });
  } catch (error) {
    res.status(errorStatus(error)).json({ error: 'Failed to list users', details: error.message });
  }
});

app.post('/api/admin/users/:uid/status', requireFirebaseAuth, async (req, res) => {
  try {
    const result = await updateUserStatus({ uid: req.params.uid, status: req.body.status });
    res.json(result);
  } catch (error) {
    res.status(errorStatus(error)).json({ error: 'Failed to update user status', details: error.message });
  }
});

app.post('/api/admin/pois/:poiId/status', requireFirebaseAuth, async (req, res) => {
  try {
    const result = await updatePoiStatus({ poiId: req.params.poiId, status: req.body.status, verified: req.body.verified });
    res.json(result);
  } catch (error) {
    res.status(errorStatus(error)).json({ error: 'Failed to update POI status', details: error.message });
  }
});

app.get('/api/agent/training-status', (req, res) => {
  try {
    res.json(getAgentTrainingStatus());
  } catch (error) {
    res.status(500).json({ error: 'Failed to read agent training status', details: error.message });
  }
});

app.post('/api/recommend', upload.single('image'), (req, res) => {
  const concept = req.body.concept;
  const version = req.body.modelVersion || 'v4';
  const imagePath = req.file ? path.resolve(req.file.path) : '';

  // Ensure version is safe
  if (!['v1', 'v2', 'v3', 'v4'].includes(version)) {
    return res.status(400).json({ error: 'Invalid model version' });
  }

  const scriptPath = path.join(__dirname, 'inference.py');

  const pythonExecutable = resolvePythonExecutable();

  const pythonProcess = spawn(pythonExecutable, [
    scriptPath,
    '--version', version,
    '--text', concept,
    ...(imagePath ? ['--image', imagePath] : [])
  ], {
    env: { ...process.env, PYTHONIOENCODING: 'utf-8' }
  });

  let resultData = '';
  let errorData = '';

  pythonProcess.stdout.on('data', (data) => {
    resultData += data.toString();
  });

  pythonProcess.stderr.on('data', (data) => {
    errorData += data.toString();
  });

  pythonProcess.on('close', (code) => {
    if (req.file) fs.unlinkSync(req.file.path); // cleanup uploaded image

    if (code !== 0) {
      console.error('Python script error:', errorData);
      return res.status(500).json({ error: 'AI processing failed', details: errorData });
    }

    try {
      // Find JSON block in python output (to ignore warnings/logs from PyTorch)
      const match = resultData.match(/\{.*\}|\[.*\]/s);
      if (match) {
        const jsonResult = JSON.parse(match[0]);
        res.json(jsonResult);
      } else {
        res.status(500).json({ error: 'Invalid output from AI model', raw: resultData });
      }
    } catch (e) {
      res.status(500).json({ error: 'Failed to parse AI results', details: e.message, raw: resultData });
    }
  });
});

// ============================================================================
//  EXPERT SYSTEM: Route Finding + Validation
// ============================================================================
app.post('/api/route', requireFirebaseAuth, async (req, res) => {
  try {
    const { origin, destination } = req.body;

    if (!origin || !destination || !origin.lat || !origin.lng || !destination.lat || !destination.lng) {
      return res.status(400).json({ error: 'Missing origin or destination coordinates' });
    }

    const result = await findOptimalRoute(
      origin.lat, origin.lng,
      destination.lat, destination.lng
    );

    res.json(result);
  } catch (error) {
    console.error('[ES Route Error]', error);
    res.status(500).json({ error: 'Failed to find route', details: error.message });
  }
});

app.post('/api/route/matrix', requireFirebaseAuth, (req, res) => {
  try {
    const { origin, destinations, transport } = req.body;
    if (!origin || !Array.isArray(destinations)) {
      return res.status(400).json({ error: 'Missing origin or destinations' });
    }
    res.json({ routes: estimateMatrix({ origin, destinations, transport }) });
  } catch (error) {
    res.status(500).json({ error: 'Failed to estimate route matrix', details: error.message });
  }
});

app.get('/api/weather/forecast', async (req, res) => {
  try {
    const lat = Number.parseFloat(req.query.lat);
    const lon = Number.parseFloat(req.query.lon);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
      return res.status(400).json({ error: 'Missing lat/lon' });
    }
    const result = await getForecast({ lat, lon });
    res.json(result);
  } catch (error) {
    res.status(502).json({
      error: 'Failed to fetch weather',
      details: error.message,
      fallback: 'Weather score skipped; continue with local POI and route signals.',
    });
  }
});

const PORT = process.env.PORT || 7860;
app.listen(PORT,'0.0.0.0', async () => {
  console.log(`Backend API running on http://localhost:${PORT}`);
  
  // Initialize Expert System at startup
  try {
    await initExpertSystem();
    console.log('[ES] Expert System initialized successfully');
  } catch (err) {
    console.error('[ES] Failed to initialize Expert System:', err.message);
  }
});
