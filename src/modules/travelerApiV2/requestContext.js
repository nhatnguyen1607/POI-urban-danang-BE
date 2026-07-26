const crypto = require('crypto');
const { API_VERSION } = require('./constants');

const REQUEST_ID_PATTERN = /^[A-Za-z0-9._:-]{1,128}$/;

function createRequestId() {
  return `req_${crypto.randomUUID()}`;
}

function resolveRequestId(value) {
  const requestId = String(value || '').trim();
  return REQUEST_ID_PATTERN.test(requestId) ? requestId : createRequestId();
}

function travelerApiV2Context(req, _res, next) {
  req.travelerApiV2 = {
    requestId: resolveRequestId(req.get('x-request-id')),
  };
  next();
}

function buildMeta(req, { cityId, extra } = {}) {
  return {
    apiVersion: API_VERSION,
    requestId: req.travelerApiV2?.requestId || createRequestId(),
    ...(cityId ? { cityId } : {}),
    ...(extra || {}),
  };
}

function sendSuccess(req, res, data, options = {}) {
  return res.status(options.status || 200).json({
    ok: true,
    data,
    meta: buildMeta(req, options),
  });
}

function sendError(req, res, status, code, message, { details = [], cityId } = {}) {
  return res.status(status).json({
    ok: false,
    error: {
      code,
      message,
      details,
    },
    meta: buildMeta(req, { cityId }),
  });
}

module.exports = {
  REQUEST_ID_PATTERN,
  buildMeta,
  createRequestId,
  resolveRequestId,
  sendError,
  sendSuccess,
  travelerApiV2Context,
};
