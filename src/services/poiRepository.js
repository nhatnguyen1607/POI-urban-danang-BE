const { DEFAULT_CITY_ID, CanonicalCsvPoiRepository } = require('./canonicalCsvPoiRepository');

let defaultRepository = null;

function getPoiRepository() {
  if (!defaultRepository) {
    defaultRepository = new CanonicalCsvPoiRepository();
  }
  return defaultRepository;
}

function setPoiRepositoryForTests(repository) {
  defaultRepository = repository;
}

async function loadPoisByCity({ cityId = DEFAULT_CITY_ID } = {}) {
  return getPoiRepository().findByCity(cityId);
}

module.exports = {
  DEFAULT_CITY_ID,
  getPoiRepository,
  loadPoisByCity,
  setPoiRepositoryForTests,
};
