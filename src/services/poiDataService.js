const fs = require('fs');
const path = require('path');
const csv = require('csv-parser');

const DATA_DIR = path.resolve(__dirname, '..', '..', 'data');

let cache = null;

function normalizeText(value) {
  return String(value || '')
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/đ/g, 'd')
    .replace(/Đ/g, 'D')
    .toLowerCase();
}

function toNumber(value, fallback = 0) {
  const n = Number.parseFloat(value);
  return Number.isFinite(n) ? n : fallback;
}

function readCSV(filePath) {
  return new Promise((resolve, reject) => {
    const rows = [];
    fs.createReadStream(filePath)
      .pipe(csv())
      .on('data', (row) => rows.push(row))
      .on('end', () => resolve(rows))
      .on('error', reject);
  });
}

function getName(row) {
  return row.name || row['Restaurant Name'] || row.RestaurantName || 'Unknown POI';
}

function getCategory(row) {
  return row.category || row.Category || 'Khac';
}

function getLat(row) {
  return toNumber(row.lat || row.Lat, 16.0544);
}

function getLon(row) {
  return toNumber(row.lng || row.lon || row.Lon, 108.2022);
}

function getRating(row) {
  return toNumber(row.rating || row['Overall Rating'], 0);
}

function getText(row) {
  return row.LLM_Input_Text || row.Aggregated_Reviews || row.description || '';
}

function getPrice(row) {
  return row.price_range || row.Price || row.price || 'Chua cap nhat';
}

function canonicalize(row, source, index) {
  const name = getName(row);
  const category = getCategory(row);
  const text = getText(row);
  const rating = getRating(row);
  const lat = getLat(row);
  const lon = getLon(row);
  const reviewCount = toNumber(row.Total_Reviews_Scraped || row['User Rating Count'] || row.review_count, 0);
  const district = row.District || row.Address || row.address || 'Da Nang';

  return {
    id: String(row.RestaurantID || row.place_id || row.id || `${source}-${index}`),
    source,
    name,
    category,
    district,
    lat,
    lon,
    rating,
    reviewCount,
    price: getPrice(row),
    text,
    normalized: normalizeText(`${name} ${category} ${district} ${text}`),
    raw: row,
  };
}

async function loadPOIs() {
  if (cache) return cache;

  const sources = [
    { source: 'google_maps', file: path.join(DATA_DIR, 'poi_data_ggmap.csv') },
    { source: 'foody', file: path.join(DATA_DIR, 'poi_data_foody.csv') },
  ];

  const allRows = [];
  for (const { source, file } of sources) {
    if (!fs.existsSync(file)) continue;
    const rows = await readCSV(file);
    rows.forEach((row, index) => allRows.push(canonicalize(row, source, index)));
  }

  cache = allRows.filter((poi) => Number.isFinite(poi.lat) && Number.isFinite(poi.lon));
  return cache;
}

module.exports = {
  loadPOIs,
  normalizeText,
};
