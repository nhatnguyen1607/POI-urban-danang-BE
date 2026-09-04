function positiveInteger(value, fallback, min = 1, max = 10_000) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? Math.min(Math.max(Math.round(parsed), min), max) : fallback;
}

function requestKey(req) {
  return req.firebaseUser?.uid || req.ip || req.socket?.remoteAddress || 'unknown';
}

function createEndpointRateLimit({
  name = 'endpoint',
  maxRequests = 60,
  windowMs = 60_000,
  now = Date.now,
  key = requestKey,
} = {}) {
  const safeMax = positiveInteger(maxRequests, 60);
  const safeWindowMs = positiveInteger(windowMs, 60_000, 1_000, 3_600_000);
  const buckets = new Map();
  return function endpointRateLimit(req, res, next) {
    const currentTime = now();
    const clientKey = key(req);
    let bucket = buckets.get(clientKey);
    if (!bucket || currentTime - bucket.startedAt >= safeWindowMs) {
      bucket = { startedAt: currentTime, count: 0 };
    }
    if (bucket.count >= safeMax) {
      const retryAfterSeconds = Math.max(1, Math.ceil((safeWindowMs - (currentTime - bucket.startedAt)) / 1000));
      res.set('Retry-After', String(retryAfterSeconds));
      console.info(JSON.stringify({ level: 'warn', event: 'rate_limited', endpointClass: name, retryAfterSeconds }));
      return res.status(429).json({
        error: 'RATE_LIMITED',
        message: 'Hệ thống đang nhận nhiều yêu cầu. Vui lòng thử lại sau ít phút.',
        retryAfterSeconds,
      });
    }
    bucket.count += 1;
    buckets.set(clientKey, bucket);
    return next();
  };
}

module.exports = { createEndpointRateLimit };
