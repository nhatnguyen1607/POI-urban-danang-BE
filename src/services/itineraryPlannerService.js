const { recommendPOIs } = require('./poiRetrievalService');
const { detectIntents, categoryMatchScore } = require('./intentService');
const { haversineKm } = require('./scoringUtils');

function estimateDurationMinutes(distanceKm, transport = 'motorbike') {
  const speed = transport === 'walking' ? 4.5 : transport === 'car' ? 28 : 22;
  return Math.max(3, Math.round((distanceKm / speed) * 60));
}

async function createItinerary({ query, context = {}, transport = 'motorbike', limit = 5 }) {
  const recommendation = await recommendPOIs({ query, context, limit: Math.max(limit * 5, 18) });
  const intents = detectIntents(query);
  const start = {
    lat: context.location?.lat || 16.0544,
    lon: context.location?.lon || context.location?.lng || 108.2022,
  };

  const selectedMap = new Map();
  intents.slice(0, Math.max(1, limit)).forEach((intent) => {
    const match = recommendation.results
      .filter((poi) => !selectedMap.has(poi.id))
      .sort((a, b) => categoryMatchScore(b, intent) - categoryMatchScore(a, intent) || b.score - a.score)
      .find((poi) => categoryMatchScore(poi, intent) >= 0.7);
    if (match) selectedMap.set(match.id, match);
  });

  recommendation.results.forEach((poi) => {
    if (selectedMap.size < limit && !selectedMap.has(poi.id)) {
      selectedMap.set(poi.id, poi);
    }
  });

  let cursor = start;
  const selected = [...selectedMap.values()]
    .slice(0, limit)
    .sort((a, b) => {
      const da = haversineKm(cursor, { lat: a.lat, lon: a.lon });
      const db = haversineKm(cursor, { lat: b.lat, lon: b.lon });
      return da - db || b.score - a.score;
    });

  const itinerary = selected.map((poi, index) => {
    const distanceKm = haversineKm(cursor, { lat: poi.lat, lon: poi.lon });
    const travelMinutes = estimateDurationMinutes(distanceKm, transport);
    cursor = { lat: poi.lat, lon: poi.lon };
    return {
      order: index + 1,
      poi,
      travelFromPrevious: {
        distanceKm: Number(distanceKm.toFixed(2)),
        estimatedMinutes: travelMinutes,
        transport,
      },
      suggestedStayMinutes: index === selected.length - 1 ? 75 : 55,
      reason: poi.reason,
    };
  });

  const totalTravelMinutes = itinerary.reduce(
    (sum, item) => sum + item.travelFromPrevious.estimatedMinutes,
    0,
  );

  return {
    role: 'traveler',
    query,
    itinerary,
    totalTravelMinutes,
    warnings: recommendation.warnings,
    detectedIntents: intents.map((intent) => ({ id: intent.id, label: intent.label })),
    actions: [
      {
        type: 'handoff',
        label: 'Mo ban do de xac nhan tuyen',
        note: 'MVP chuan bi tuyen va de nguoi dung xac nhan tren ung dung ban do/Grab.',
      },
    ],
  };
}

module.exports = {
  createItinerary,
};
