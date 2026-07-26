const { normalizeText } = require('../../services/poiDataService');
const { pageItems, parsePagination } = require('./pagination');
const { serializePoi } = require('./serializers');

function normalizedName(poi) {
  return normalizeText(poi.name || '');
}

function canonicalId(poi) {
  return String(poi.globalId || poi.id || '');
}

function compareByNameThenId(a, b) {
  return normalizedName(a).localeCompare(normalizedName(b), 'en') ||
    canonicalId(a).localeCompare(canonicalId(b), 'en');
}

function relevanceScore(poi, normalizedQuery) {
  if (!normalizedQuery) return 0;
  const tokens = normalizedQuery.split(/\s+/).filter(Boolean);
  const haystack = normalizeText([
    poi.name,
    poi.category,
    poi.categoryNormalized,
    poi.addressCurrent,
    poi.addressRaw,
    poi.district,
    poi.text,
  ].filter(Boolean).join(' '));
  return tokens.reduce((score, token) => score + (haystack.includes(token) ? 1 : 0), 0);
}

function filterByCategory(poi, category) {
  if (!category) return true;
  const needle = normalizeText(category);
  return normalizeText(`${poi.category || ''} ${poi.categoryNormalized || ''}`).includes(needle);
}

function rankPois(pois, { q, category } = {}) {
  const normalizedQuery = normalizeText(q || '').trim();
  return pois
    .filter((poi) => filterByCategory(poi, category))
    .map((poi) => ({
      poi,
      relevance: relevanceScore(poi, normalizedQuery),
    }))
    .filter((item) => !normalizedQuery || item.relevance > 0)
    .sort((a, b) => {
      if (normalizedQuery && b.relevance !== a.relevance) return b.relevance - a.relevance;
      return compareByNameThenId(a.poi, b.poi);
    })
    .map((item) => item.poi);
}

function searchTravelerPois(pois, query = {}) {
  const pagination = parsePagination(query);
  if (pagination.error) {
    return { error: pagination.error };
  }
  const ranked = rankPois(pois, { q: query.q, category: query.category });
  const page = pageItems(ranked, pagination);
  return {
    pois: page.items.map(serializePoi),
    page: page.page,
  };
}

module.exports = {
  compareByNameThenId,
  rankPois,
  relevanceScore,
  searchTravelerPois,
};
