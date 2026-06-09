function clamp01(value) {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(1, value));
}

function haversineKm(a, b) {
  const radius = 6371;
  const dLat = ((b.lat - a.lat) * Math.PI) / 180;
  const dLon = ((b.lon - a.lon) * Math.PI) / 180;
  const lat1 = (a.lat * Math.PI) / 180;
  const lat2 = (b.lat * Math.PI) / 180;
  const h =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLon / 2) ** 2;
  return 2 * radius * Math.asin(Math.sqrt(h));
}

function tokenize(text) {
  return String(text || '')
    .split(/[^a-z0-9]+/i)
    .map((t) => t.trim())
    .filter((t) => t.length >= 2);
}

function keywordScore(query, document) {
  const tokens = Array.from(new Set(tokenize(query)));
  if (!tokens.length) return 0;
  const matched = tokens.filter((token) => document.includes(token)).length;
  return matched / tokens.length;
}

function ratingScore(rating) {
  if (!rating) return 0.35;
  if (rating > 5) return clamp01(rating / 10);
  return clamp01(rating / 5);
}

function distanceScore(distanceKm, maxKm = 12) {
  return clamp01(1 - distanceKm / maxKm);
}

function reviewSignal(reviewCount) {
  return clamp01(Math.log10((reviewCount || 0) + 1) / 3);
}

module.exports = {
  clamp01,
  haversineKm,
  keywordScore,
  ratingScore,
  distanceScore,
  reviewSignal,
};
