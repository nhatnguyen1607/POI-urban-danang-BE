function createAdminRateLimit({
  maxRequests = 120,
  windowMs = 60_000,
  now = Date.now,
} = {}) {
  const safeMax = Math.min(Math.max(Number(maxRequests) || 120, 10), 600);
  const safeWindowMs = Math.max(Number(windowMs) || 60_000, 1_000);
  const windows = new Map();

  return function adminRateLimit(req, res, next) {
    const key = req.firebaseUser?.uid || req.ip || 'unknown';
    const currentTime = now();
    const current = windows.get(key);
    const bucket = !current || currentTime - current.startedAt >= safeWindowMs
      ? { startedAt: currentTime, count: 0 }
      : current;

    if (bucket.count >= safeMax) {
      const retryAfterSeconds = Math.max(
        1,
        Math.ceil((safeWindowMs - (currentTime - bucket.startedAt)) / 1_000),
      );
      res.set('Retry-After', String(retryAfterSeconds));
      return res.status(429).json({ error: 'admin_rate_limited' });
    }

    bucket.count += 1;
    windows.set(key, bucket);
    return next();
  };
}

module.exports = { createAdminRateLimit };
