const { sendError } = require('../modules/travelerApiV2/requestContext');

function isMalformedJsonError(error) {
  return error instanceof SyntaxError
    && error?.status === 400
    && error?.type === 'entity.parse.failed';
}

function malformedJsonErrorHandler(error, req, res, next) {
  if (!isMalformedJsonError(error)) return next(error);

  if (String(req.originalUrl || '').startsWith('/api/v2/')) {
    return sendError(req, res, 400, 'MALFORMED_JSON', 'Request body contains invalid JSON.');
  }

  return res.status(400).json({ error: 'Request body contains invalid JSON.' });
}

module.exports = {
  isMalformedJsonError,
  malformedJsonErrorHandler,
};
