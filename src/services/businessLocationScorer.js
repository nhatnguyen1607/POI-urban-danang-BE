const { loadPOIs, normalizeText } = require('./poiDataService');
const { detectIntent, categoryMatchScore } = require('./intentService');
const { clamp01, haversineKm, keywordScore, ratingScore, reviewSignal } = require('./scoringUtils');

const AREA_SIZE = 0.018; // roughly 1.8-2km in Da Nang latitude bands

function areaKey(poi) {
  const lat = Math.round(poi.lat / AREA_SIZE) * AREA_SIZE;
  const lon = Math.round(poi.lon / AREA_SIZE) * AREA_SIZE;
  return `${lat.toFixed(3)},${lon.toFixed(3)}`;
}

function buildAreaSummary(items) {
  const lat = items.reduce((sum, p) => sum + p.lat, 0) / items.length;
  const lon = items.reduce((sum, p) => sum + p.lon, 0) / items.length;
  const categories = new Map();
  items.forEach((poi) => categories.set(poi.category, (categories.get(poi.category) || 0) + 1));
  return {
    lat,
    lon,
    totalPOIs: items.length,
    topCategories: Array.from(categories.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([category, count]) => ({ category, count })),
  };
}

function scoreArea(items, concept) {
  const normalizedConcept = normalizeText(concept);
  const intent = detectIntent(concept);
  const summary = buildAreaSummary(items);
  const directMatches = items.filter((poi) => categoryMatchScore(poi, intent) > 0.7);
  const semanticHits = items.filter((poi) => keywordScore(normalizedConcept, poi.normalized) > 0);
  const avgRating =
    items.reduce((sum, poi) => sum + ratingScore(poi.rating), 0) / Math.max(items.length, 1);
  const review = items.reduce((sum, poi) => sum + reviewSignal(poi.reviewCount), 0) / Math.max(items.length, 1);
  const demandProxy = clamp01((semanticHits.length / Math.max(items.length, 1)) * 0.45 + avgRating * 0.3 + review * 0.25);
  const competitionPenalty = clamp01(directMatches.length / 12);
  const complementary = clamp01(Math.log10(items.length + 1) / 2);
  const accessibility = clamp01(1 - haversineKm({ lat: summary.lat, lon: summary.lon }, { lat: 16.0544, lon: 108.2022 }) / 18);
  const conceptFit = clamp01(directMatches.length / 4 + semanticHits.length / 12);
  const opportunity = clamp01(
    demandProxy * 0.30 +
      complementary * 0.20 +
      accessibility * 0.20 +
      conceptFit * 0.20 -
      competitionPenalty * 0.10,
  );

  const warnings = [];
  if (competitionPenalty > 0.65) warnings.push('Canh tranh truc tiep cao, can khac biet hoa concept.');
  if (demandProxy < 0.25) warnings.push('Tin hieu nhu cau tu du lieu hien co con yeu.');
  if (accessibility < 0.35) warnings.push('Khu vuc xa trung tam, can kiem tra lai kha nang tiep can.');

  return {
    ...summary,
    score: Math.round(opportunity * 100),
    scoreRaw: opportunity,
    signals: {
      demandProxy,
      competitionPenalty,
      complementary,
      accessibility,
      conceptFit,
      directCompetitors: directMatches.length,
      semanticHits: semanticHits.length,
    },
    warnings,
    reason: `Khu vuc co ${items.length} POI, ${semanticHits.length} tin hieu gan concept va ${directMatches.length} doi thu truc tiep trong du lieu hien co.`,
    samplePOIs: items
      .sort((a, b) => ratingScore(b.rating) - ratingScore(a.rating))
      .slice(0, 5)
      .map((poi) => ({
        id: poi.id,
        name: poi.name,
        category: poi.category,
        rating: poi.rating,
        lat: poi.lat,
        lon: poi.lon,
      })),
  };
}

async function scoreBusinessLocations({ concept, limit = 6 }) {
  const pois = await loadPOIs();
  const groups = new Map();
  pois.forEach((poi) => {
    const key = areaKey(poi);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(poi);
  });

  const areas = Array.from(groups.entries())
    .filter(([, items]) => items.length >= 4)
    .map(([key, items]) => ({ id: key, ...scoreArea(items, concept) }))
    .sort((a, b) => b.scoreRaw - a.scoreRaw)
    .slice(0, limit);

  return {
    role: 'business',
    concept,
    areas,
    note: 'Day la demand proxy tu du lieu POI/review/rating/cum dia diem, khong phai mat do khach that.',
  };
}

module.exports = {
  scoreBusinessLocations,
  areaKey,
  buildAreaSummary,
  scoreArea,
};
