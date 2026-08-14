const DEVELOPMENT_ORIGINS = Object.freeze([
  'http://localhost:5173',
  'http://127.0.0.1:5173',
]);

function parseAllowedOrigins(value) {
  return [...new Set(String(value || '')
    .split(',')
    .map((origin) => origin.trim().replace(/\/$/, ''))
    .filter(Boolean))];
}

function createCorsOptions(env = process.env) {
  const configured = parseAllowedOrigins(env.URBANAGENT_CORS_ALLOWED_ORIGINS);
  const allowedOrigins = configured.length > 0
    ? configured
    : env.NODE_ENV === 'production' ? [] : [...DEVELOPMENT_ORIGINS];

  return {
    origin(origin, callback) {
      if (!origin || allowedOrigins.includes(String(origin).replace(/\/$/, ''))) {
        callback(null, true);
        return;
      }
      const error = new Error('CORS origin is not allowed.');
      error.code = 'CORS_ORIGIN_NOT_ALLOWED';
      callback(error);
    },
  };
}

module.exports = {
  DEVELOPMENT_ORIGINS,
  createCorsOptions,
  parseAllowedOrigins,
};
