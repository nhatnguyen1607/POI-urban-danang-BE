const TRAVEL_POLICY_VERSION = 'phase2-batch3-travel-time-v1';
const TRAVEL_ESTIMATION_METHOD = 'local-haversine-estimate';

const MODE_ASSUMPTIONS = Object.freeze({
  walk: { speedKmH: 4.5, overheadMinutes: 0, minimumMinutes: 2 },
  motorbike: { speedKmH: 22, overheadMinutes: 3, minimumMinutes: 4 },
  car: { speedKmH: 18, overheadMinutes: 5, minimumMinutes: 5 },
  taxi: { speedKmH: 18, overheadMinutes: 7, minimumMinutes: 7 },
});

function isValidCoordinate(point) {
  if (point?.lat === null || point?.lat === undefined || point?.lon === null || point?.lon === undefined) {
    return false;
  }
  const lat = Number(point?.lat);
  const lon = Number(point?.lon);
  return Number.isFinite(lat) && Number.isFinite(lon) &&
    lat >= -90 && lat <= 90 &&
    lon >= -180 && lon <= 180;
}

function haversineMeters(a, b) {
  const earthRadiusMeters = 6371000;
  const toRad = (degrees) => degrees * Math.PI / 180;
  const lat1 = toRad(Number(a.lat));
  const lat2 = toRad(Number(b.lat));
  const deltaLat = toRad(Number(b.lat) - Number(a.lat));
  const deltaLon = toRad(Number(b.lon) - Number(a.lon));
  const sinLat = Math.sin(deltaLat / 2);
  const sinLon = Math.sin(deltaLon / 2);
  const h = sinLat * sinLat + Math.cos(lat1) * Math.cos(lat2) * sinLon * sinLon;
  return earthRadiusMeters * 2 * Math.atan2(Math.sqrt(h), Math.sqrt(1 - h));
}

function unknownLeg({ transport = 'motorbike', calculationSource = 'missing-origin', legOrder } = {}) {
  return {
    distanceMeters: null,
    travelDurationMinutes: null,
    transport,
    travelMode: transport,
    estimationMethod: null,
    estimationPolicyVersion: null,
    calculationSource,
    distanceKnown: false,
    travelTimeKnown: false,
    ...(legOrder !== undefined ? { legOrder } : {}),
  };
}

function estimateTravelLeg({ from, to, transport = 'motorbike', legOrder } = {}) {
  if (!isValidCoordinate(from) || !isValidCoordinate(to)) {
    return unknownLeg({
      transport,
      calculationSource: 'missing-coordinates',
      legOrder,
    });
  }
  const assumption = MODE_ASSUMPTIONS[transport] || MODE_ASSUMPTIONS.motorbike;
  const roundedDistanceMeters = Math.round(haversineMeters(from, to) / 10) * 10;
  const distanceKm = roundedDistanceMeters / 1000;
  const travelHours = distanceKm / assumption.speedKmH;
  const estimatedMinutes = Math.ceil(travelHours * 60 + assumption.overheadMinutes);
  const travelDurationMinutes = Math.max(assumption.minimumMinutes, estimatedMinutes);
  return {
    distanceMeters: roundedDistanceMeters,
    travelDurationMinutes,
    transport,
    travelMode: transport,
    estimationMethod: TRAVEL_ESTIMATION_METHOD,
    estimationPolicyVersion: TRAVEL_POLICY_VERSION,
    calculationSource: TRAVEL_ESTIMATION_METHOD,
    distanceKnown: true,
    travelTimeKnown: true,
    ...(legOrder !== undefined ? { legOrder } : {}),
  };
}

module.exports = {
  MODE_ASSUMPTIONS,
  TRAVEL_ESTIMATION_METHOD,
  TRAVEL_POLICY_VERSION,
  estimateTravelLeg,
  haversineMeters,
  isValidCoordinate,
  unknownLeg,
};
