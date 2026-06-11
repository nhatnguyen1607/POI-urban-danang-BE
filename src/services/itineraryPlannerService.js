const { recommendPOIs } = require('./poiRetrievalService');
const { detectIntents, categoryMatchScore } = require('./intentService');
const { haversineKm } = require('./scoringUtils');

function estimateDurationMinutes(distanceKm, transport = 'motorbike') {
  const speed = transport === 'walking' ? 4.5 : transport === 'car' ? 28 : 22;
  return Math.max(3, Math.round((distanceKm / speed) * 60));
}

function limitFromDuration(durationMinutes, fallbackLimit) {
  const duration = Number.parseInt(durationMinutes, 10);
  if (!Number.isFinite(duration) || duration <= 0) return fallbackLimit;
  if (duration <= 120) return Math.min(fallbackLimit, 2);
  if (duration <= 210) return Math.min(fallbackLimit, 3);
  if (duration <= 330) return Math.min(fallbackLimit, 4);
  return Math.min(fallbackLimit, 5);
}

function stayMinutesForStop(index, totalStops, durationMinutes) {
  const duration = Number.parseInt(durationMinutes, 10);
  if (Number.isFinite(duration) && duration <= 120) return index === totalStops - 1 ? 45 : 35;
  if (Number.isFinite(duration) && duration <= 210) return index === totalStops - 1 ? 60 : 45;
  return index === totalStops - 1 ? 75 : 55;
}

async function createItinerary({ query, context = {}, transport = 'motorbike', limit = 5, durationMinutes }) {
  const effectiveLimit = limitFromDuration(durationMinutes || context.durationMinutes, limit);
  const recommendation = await recommendPOIs({ query, context, limit: Math.max(effectiveLimit * 5, 18) });
  const intents = detectIntents(query);
  const start = {
    lat: context.location?.lat || 16.0544,
    lon: context.location?.lon || context.location?.lng || 108.2022,
  };

  const selectedMap = new Map();
  intents.slice(0, Math.max(1, effectiveLimit)).forEach((intent) => {
    const match = recommendation.results
      .filter((poi) => !selectedMap.has(poi.id))
      .sort((a, b) => categoryMatchScore(b, intent) - categoryMatchScore(a, intent) || b.score - a.score)
      .find((poi) => categoryMatchScore(poi, intent) >= 0.7);
    if (match) selectedMap.set(match.id, match);
  });

  recommendation.results.forEach((poi) => {
    if (selectedMap.size < effectiveLimit && !selectedMap.has(poi.id)) {
      selectedMap.set(poi.id, poi);
    }
  });

  let cursor = start;
  const selected = [...selectedMap.values()]
    .slice(0, effectiveLimit)
    .sort((a, b) => {
      const da = haversineKm(cursor, { lat: a.lat, lon: a.lon });
      const db = haversineKm(cursor, { lat: b.lat, lon: b.lon });
      return da - db || b.score - a.score;
    });

  const itinerary = selected.slice(0, effectiveLimit).map((poi, index) => {
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
      suggestedStayMinutes: stayMinutesForStop(
        index,
        selected.slice(0, effectiveLimit).length,
        durationMinutes || context.durationMinutes,
      ),
      reason: poi.reason,
    };
  });

  const totalTravelMinutes = itinerary.reduce(
    (sum, item) => sum + item.travelFromPrevious.estimatedMinutes,
    0,
  );
  const totalStayMinutes = itinerary.reduce((sum, item) => sum + (item.suggestedStayMinutes || 0), 0);
  const totalPlanMinutes = totalTravelMinutes + totalStayMinutes;

  return {
    role: 'traveler',
    query,
    itinerary,
    totalTravelMinutes,
    totalStayMinutes,
    totalPlanMinutes,
    requestedDurationMinutes: Number.parseInt(durationMinutes || context.durationMinutes, 10) || null,
    warnings: recommendation.warnings,
    detectedIntents: intents.map((intent) => ({ id: intent.id, label: intent.label })),
    actions: [
      {
        type: 'handoff',
        label: 'Mở bản đồ để xác nhận tuyến',
        note: 'MVP chuẩn bị tuyến và để người dùng xác nhận trên ứng dụng bản đồ/Grab.',
      },
    ],
  };
}

module.exports = {
  createItinerary,
};
