const express = require('express');
const cors = require('cors');
const multer = require('multer');
const { spawn } = require('child_process');
const fs = require('fs');
const csv = require('csv-parser');
const path = require('path');
const { initExpertSystem, findOptimalRoute } = require('./expert_system/expert_system');
const { recommendPOIs } = require('./services/poiRetrievalService');
const { createItinerary } = require('./services/itineraryPlannerService');
const { scoreBusinessLocations } = require('./services/businessLocationScorer');
const { getForecast } = require('./services/weatherService');
const { estimateMatrix } = require('./services/routeMatrixService');
const { recordFeedback } = require('./services/feedbackService');
const { generateBusinessInsights } = require('./services/businessInsightGenerator');
const { getAgentTrainingStatus } = require('./services/trainingStatusService');

const ROOT_DIR = path.resolve(__dirname, '..');
const DATA_DIR = path.join(ROOT_DIR, 'data');
const MODEL_DIR = path.join(ROOT_DIR, 'artifacts', 'model');
const STORAGE_DIR = path.join(ROOT_DIR, 'storage');
const UPLOAD_DIR = path.join(STORAGE_DIR, 'uploads');

fs.mkdirSync(UPLOAD_DIR, { recursive: true });

const app = express();
app.use(cors());

// Parse JSON but skip multipart/form-data (handled by multer)
app.use(express.json({
  skip: (req) => req.is('multipart/form-data')
}));

const upload = multer({ dest: UPLOAD_DIR });

// Endpoint to serve figures dynamically by version
app.get('/api/figures/:version/:filename', (req, res) => {
  const { version, filename } = req.params;
  if (!['v1', 'v2', 'v3', 'v4'].includes(version)) {
    return res.status(400).send('Invalid version');
  }
  const filePath = path.join(MODEL_DIR, version, 'reports', 'figures', filename);
  if (fs.existsSync(filePath)) {
    res.sendFile(filePath);
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
    if (!fs.existsSync(filePath)) {
      return res.status(404).json({ error: 'Metrics file not found' });
    }
    const data = await readCSV(filePath);
    res.json(data);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to read metrics' });
  }
});

// ============================================================================
//  URBAN AGENT MVP: Traveler + Business role endpoints
// ============================================================================
app.post('/api/agent/recommend-poi', async (req, res) => {
  try {
    const { query, context, limit } = req.body;
    if (!query || !String(query).trim()) {
      return res.status(400).json({ error: 'Missing query' });
    }
    const result = await recommendPOIs({ query, context, limit });
    res.json(result);
  } catch (error) {
    console.error('[Agent Recommend Error]', error);
    res.status(500).json({ error: 'Failed to recommend POIs', details: error.message });
  }
});

app.post('/api/agent/create-itinerary', async (req, res) => {
  try {
    const { query, context, transport, limit, durationMinutes } = req.body;
    if (!query || !String(query).trim()) {
      return res.status(400).json({ error: 'Missing query' });
    }
    const result = await createItinerary({ query, context, transport, limit, durationMinutes });
    res.json(result);
  } catch (error) {
    console.error('[Agent Itinerary Error]', error);
    res.status(500).json({ error: 'Failed to create itinerary', details: error.message });
  }
});

app.post('/api/agent/update-itinerary', async (req, res) => {
  try {
    const { itinerary = [], action, poi } = req.body;
    let updated = Array.isArray(itinerary) ? [...itinerary] : [];
    if (action === 'add_poi' && poi) {
      updated.push({
        order: updated.length + 1,
        poi,
        reason: 'Nguoi dung them thu cong vao lich trinh.',
      });
    }
    if (action === 'remove_poi' && poi?.id) {
      updated = updated.filter((item) => item.poi?.id !== poi.id);
    }
    updated = updated.map((item, index) => ({ ...item, order: index + 1 }));
    res.json({ itinerary: updated, warnings: [], note: 'MVP update itinerary locally on backend.' });
  } catch (error) {
    res.status(500).json({ error: 'Failed to update itinerary', details: error.message });
  }
});

app.post('/api/agent/business-location', async (req, res) => {
  try {
    const { concept, limit } = req.body;
    if (!concept || !String(concept).trim()) {
      return res.status(400).json({ error: 'Missing business concept' });
    }
    const result = await scoreBusinessLocations({ concept, limit });
    res.json(result);
  } catch (error) {
    console.error('[Business Agent Error]', error);
    res.status(500).json({ error: 'Failed to score business locations', details: error.message });
  }
});

app.post('/api/agent/business-insight', async (req, res) => {
  try {
    const { concept, limit, language } = req.body;
    if (!concept || !String(concept).trim()) {
      return res.status(400).json({ error: 'Missing business concept' });
    }
    const result = await generateBusinessInsights({ concept, limit, language });
    res.json(result);
  } catch (error) {
    console.error('[Business Insight Error]', error);
    res.status(500).json({ error: 'Failed to generate business insights', details: error.message });
  }
});

app.post('/api/agent/feedback', async (req, res) => {
  try {
    const result = await recordFeedback(req.body);
    res.json(result);
  } catch (error) {
    console.error('[Agent Feedback Error]', error);
    res.status(500).json({ error: 'Failed to record feedback', details: error.message });
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

  const pythonExecutable = fs.existsSync(path.resolve(ROOT_DIR, '../poi-urban-danang/.venv/Scripts/python.exe'))
    ? path.resolve(ROOT_DIR, '../poi-urban-danang/.venv/Scripts/python.exe')
    : 'python';

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
app.post('/api/route', async (req, res) => {
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

app.post('/api/route/matrix', (req, res) => {
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
