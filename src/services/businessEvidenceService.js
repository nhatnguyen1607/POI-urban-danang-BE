const { loadPOIs, normalizeText } = require('./poiDataService');
const { scoreBusinessLocations, areaKey } = require('./businessLocationScorer');
const { detectIntent, categoryMatchScore } = require('./intentService');
const { haversineKm, ratingScore } = require('./scoringUtils');

function evidenceId(areaId, type, index) {
  return `${areaId}:${type}:${index + 1}`;
}

function byRating(a, b) {
  return ratingScore(b.rating) - ratingScore(a.rating) || (b.reviewCount || 0) - (a.reviewCount || 0);
}

function isComplementary(poi, intent) {
  if (!intent) return true;
  const text = normalizeText(`${poi.category} ${poi.name} ${poi.text}`);
  const direct = categoryMatchScore(poi, intent) > 0.7;
  const supportTerms = ['university', 'truong', 'dai hoc', 'khach san', 'hotel', 'bien', 'du lich', 'van phong', 'office', 'mall', 'shopping', 'quan an', 'cafe'];
  return !direct && supportTerms.some((term) => text.includes(normalizeText(term)));
}

function buildRouteWarnings(area) {
  const warnings = [];
  if (area.signals.accessibility < 0.35) {
    warnings.push('Accessibility score thap; khu vuc kha xa trung tam hoac can kiem tra lai tuyen tiep can.');
  }
  if (area.totalPOIs > 120 && area.signals.accessibility < 0.65) {
    warnings.push('Cum POI day dac nhung accessibility chua cao; can khao sat gio cao diem va bai do xe.');
  }
  if (area.signals.competitionPenalty > 0.65) {
    warnings.push('Canh tranh truc tiep cao; concept can co dinh vi khac biet.');
  }
  return warnings;
}

function buildAreaEvidence(area, areaPois, concept) {
  const intent = detectIntent(concept);
  const center = { lat: area.lat, lon: area.lon };
  const directCompetitors = areaPois
    .filter((poi) => categoryMatchScore(poi, intent) > 0.7)
    .sort((a, b) => haversineKm(center, a) - haversineKm(center, b) || byRating(a, b))
    .slice(0, 6);
  const complementaryPOIs = areaPois
    .filter((poi) => isComplementary(poi, intent))
    .sort((a, b) => byRating(a, b))
    .slice(0, 8);

  return {
    areaId: area.id,
    concept,
    center,
    score: area.score,
    signals: area.signals,
    rawCounts: {
      poiTotalInArea: area.totalPOIs,
      directCompetitorsInArea: area.signals.directCompetitors,
      semanticHitsInArea: area.signals.semanticHits,
      complementaryCandidates: complementaryPOIs.length,
    },
    topCategories: area.topCategories.map((item, index) => ({
      evidenceId: evidenceId(area.id, 'category', index),
      ...item,
    })),
    complementaryPOIs: complementaryPOIs.map((poi, index) => ({
      evidenceId: evidenceId(area.id, 'complementary_poi', index),
      poiId: poi.id,
      name: poi.name,
      category: poi.category,
      rating: poi.rating,
      distanceKm: Number(haversineKm(center, poi).toFixed(2)),
    })),
    competitors: directCompetitors.map((poi, index) => ({
      evidenceId: evidenceId(area.id, 'competitor', index),
      poiId: poi.id,
      name: poi.name,
      category: poi.category,
      rating: poi.rating,
      distanceKm: Number(haversineKm(center, poi).toFixed(2)),
    })),
    routeWarnings: buildRouteWarnings(area).map((warning, index) => ({
      evidenceId: evidenceId(area.id, 'route_warning', index),
      warning,
    })),
    samplePOIs: area.samplePOIs.map((poi, index) => ({
      evidenceId: evidenceId(area.id, 'sample_poi', index),
      ...poi,
    })),
  };
}

async function buildBusinessEvidencePack({ concept, limit = 5 }) {
  const [scored, pois] = await Promise.all([
    scoreBusinessLocations({ concept, limit }),
    loadPOIs(),
  ]);
  const groups = new Map();
  pois.forEach((poi) => {
    const key = areaKey(poi);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(poi);
  });

  const areas = scored.areas.map((area) => ({
    ...area,
    evidence: buildAreaEvidence(area, groups.get(area.id) || [], concept),
  }));

  return {
    role: 'business',
    concept,
    note: scored.note,
    areas,
    evidencePolicy: {
      demandProxy: 'Uoc luong tu category density, semantic hits, rating va review volume; khong phai mat do khach that.',
      noHallucination: 'Insight generator chi duoc dung cac field trong evidence pack.',
    },
  };
}

module.exports = {
  buildBusinessEvidencePack,
};
