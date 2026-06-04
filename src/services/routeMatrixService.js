const { haversineKm } = require('./scoringUtils');

function estimateMatrix({ origin, destinations = [], transport = 'motorbike' }) {
  const speed = transport === 'walking' ? 4.5 : transport === 'car' ? 28 : 22;
  return destinations.map((destination) => {
    const distanceKm = haversineKm(
      { lat: origin.lat, lon: origin.lon || origin.lng },
      { lat: destination.lat, lon: destination.lon || destination.lng },
    );
    return {
      destination,
      distanceKm: Number(distanceKm.toFixed(2)),
      estimatedMinutes: Math.max(3, Math.round((distanceKm / speed) * 60)),
      transport,
      source: 'local-haversine-estimate',
    };
  });
}

module.exports = {
  estimateMatrix,
};
