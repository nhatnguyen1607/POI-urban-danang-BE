function createGeocoderRateLimit({
  maxRequests = 30,
  windowMs = 60_000,
  now = Date.now,
} = {}) {
  const safeMax = Math.min(Math.max(Number(maxRequests) || 30, 1), 120);
  const safeWindowMs = Math.max(Number(windowMs) || 60_000, 1_000);
  let windowStartedAt = now();
  let requestCount = 0;

  return function geocoderRateLimit(req, res, next) {
    const currentTime = now();
    if (currentTime - windowStartedAt >= safeWindowMs) {
      windowStartedAt = currentTime;
      requestCount = 0;
    }

    if (requestCount >= safeMax) {
      const retryAfterSeconds = Math.max(1, Math.ceil(
        (safeWindowMs - (currentTime - windowStartedAt)) / 1_000,
      ));
      res.set('Retry-After', String(retryAfterSeconds));
      return res.status(429).json({
        error: 'GEOCODER_RATE_LIMITED',
        message: 'Too many destination searches. Please try again shortly.',
      });
    }

    requestCount += 1;
    return next();
  };
}

module.exports = { createGeocoderRateLimit };
