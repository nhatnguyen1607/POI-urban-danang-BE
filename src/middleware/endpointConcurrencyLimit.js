function positiveInteger(value, fallback, min = 1, max = 1_000) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? Math.min(Math.max(Math.round(parsed), min), max) : fallback;
}

function createEndpointConcurrencyLimit({
  name = 'endpoint',
  maxActive = 8,
  retryAfterSeconds = 1,
  logger = console.info,
} = {}) {
  const safeName = String(name || 'endpoint').replace(/[^a-z0-9_-]/gi, '_');
  const safeMaxActive = positiveInteger(maxActive, 8);
  const safeRetryAfter = positiveInteger(retryAfterSeconds, 1, 1, 300);
  let active = 0;
  let peakActive = 0;
  let rejected = 0;

  function endpointConcurrencyLimit(req, res, next) {
    if (active >= safeMaxActive) {
      rejected += 1;
      res.set('Retry-After', String(safeRetryAfter));
      res.set('X-UrbanAgent-Admission-Active', String(active));
      res.set('X-UrbanAgent-Admission-Pending', '0');
      logger(JSON.stringify({
        level: 'warn',
        event: 'capacity_rejected',
        endpointClass: safeName,
        active,
        maxActive: safeMaxActive,
        pending: 0,
      }));
      return res.status(503).json({
        error: 'CAPACITY_EXHAUSTED',
        message: 'UrbanAgent đang có nhiều yêu cầu cùng lúc. Vui lòng thử lại sau.',
        retryAfterSeconds: safeRetryAfter,
      });
    }

    active += 1;
    peakActive = Math.max(peakActive, active);
    res.set('X-UrbanAgent-Admission-Active', String(active));
    res.set('X-UrbanAgent-Admission-Pending', '0');
    let released = false;
    const release = () => {
      if (released) return;
      released = true;
      active = Math.max(0, active - 1);
    };
    res.once('finish', release);
    res.once('close', release);
    try {
      return next();
    } catch (error) {
      release();
      throw error;
    }
  }

  endpointConcurrencyLimit.snapshot = () => ({
    name: safeName,
    maxActive: safeMaxActive,
    maxPending: 0,
    active,
    pending: 0,
    peakActive,
    rejected,
  });

  return endpointConcurrencyLimit;
}

module.exports = { createEndpointConcurrencyLimit };
