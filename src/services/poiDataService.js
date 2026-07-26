const { DEFAULT_CITY_ID, getPoiRepository, loadPoisByCity } = require('./poiRepository');
const { normalizeText } = require('./canonicalCsvPoiRepository');

async function loadPOIs(options = {}) {
  return loadPoisByCity({ cityId: options.cityId || DEFAULT_CITY_ID });
}

function normalizeEdaSource(source) {
  const value = String(source || '').trim().toLowerCase();
  if (['all', 'canonical'].includes(value)) return 'all';
  if (value === 'foody') return 'foody';
  if (['', 'missing', 'ggmap', 'google', 'google_maps'].includes(value)) return 'google_maps';
  return 'google_maps';
}

function filterPoisForEdaSource(pois = [], source) {
  const normalizedSource = normalizeEdaSource(source);
  if (normalizedSource === 'all') return pois;
  if (normalizedSource === 'google_maps') {
    return pois.filter((poi) => poi.source === 'google_maps' || poi.source === 'google_maps+foody');
  }
  if (normalizedSource === 'foody') {
    return pois.filter((poi) => poi.source === 'foody' || poi.source === 'google_maps+foody');
  }
  return pois;
}

async function loadPOIsForEdaSource({ cityId = DEFAULT_CITY_ID, source } = {}) {
  const pois = await loadPOIs({ cityId });
  return {
    pois: filterPoisForEdaSource(pois, source),
    normalizedSource: normalizeEdaSource(source),
  };
}

async function getPoiDataQualityReport() {
  return getPoiRepository().getQualityReport();
}

function clearPoiCacheForTests() {
  getPoiRepository().clearCache();
}

module.exports = {
  DEFAULT_CITY_ID,
  clearPoiCacheForTests,
  filterPoisForEdaSource,
  getPoiDataQualityReport,
  loadPOIs,
  loadPOIsForEdaSource,
  normalizeEdaSource,
  normalizeText,
};
