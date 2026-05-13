const express = require('express');
const cors = require('cors');
const multer = require('multer');
const { spawn } = require('child_process');
const fs = require('fs');
const csv = require('csv-parser');
const path = require('path');
const { initExpertSystem, findOptimalRoute } = require('./ES-system/expert_system');

const app = express();
app.use(cors());

// Parse JSON but skip multipart/form-data (handled by multer)
app.use(express.json({
  skip: (req) => req.is('multipart/form-data')
}));

const upload = multer({ dest: 'uploads/' });

// Endpoint to serve figures dynamically by version
app.get('/api/figures/:version/:filename', (req, res) => {
  const { version, filename } = req.params;
  if (!['v1', 'v2', 'v3', 'v4'].includes(version)) {
    return res.status(400).send('Invalid version');
  }
  const filePath = path.join(__dirname, 'model', version, 'reports', 'figures', filename);
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
    const filePath = path.join(__dirname, 'data', fileName);
    
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
    const filePath = path.join(__dirname, 'model', version, 'reports', 'metrics', `training_loss_${version}.csv`);
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

app.post('/api/recommend', upload.single('image'), (req, res) => {
  const concept = req.body.concept;
  const version = req.body.modelVersion || 'v4';
  const imagePath = req.file ? path.resolve(req.file.path) : '';

  // Ensure version is safe
  if (!['v1', 'v2', 'v3', 'v4'].includes(version)) {
    return res.status(400).json({ error: 'Invalid model version' });
  }

  const scriptPath = path.join(__dirname, 'inference.py');

  const pythonExecutable = fs.existsSync(path.resolve(__dirname, '../poi-urban-danang/.venv/Scripts/python.exe'))
    ? path.resolve(__dirname, '../poi-urban-danang/.venv/Scripts/python.exe')
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
