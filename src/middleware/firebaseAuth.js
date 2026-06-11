const { getFirebaseAuth, isFirebaseAdminReady } = require('../config/firebaseAdmin');

function getBearerToken(req) {
  const header = req.headers.authorization || '';
  const match = header.match(/^Bearer\s+(.+)$/i);
  return match ? match[1] : '';
}

function decodeJwtPayloadUnsafe(token) {
  try {
    const payload = token.split('.')[1];
    if (!payload) return null;
    const normalized = payload.replace(/-/g, '+').replace(/_/g, '/');
    const padded = normalized.padEnd(Math.ceil(normalized.length / 4) * 4, '=');
    return JSON.parse(Buffer.from(padded, 'base64').toString('utf8'));
  } catch {
    return null;
  }
}

function allowDevAuthFallback() {
  return process.env.NODE_ENV !== 'production' && process.env.DISABLE_DEV_AUTH_FALLBACK !== 'true';
}

function attachLocalAdmin(req) {
  req.firebaseUser = { uid: 'local-admin', email: 'admin', role: 'admin', localAdmin: true };
  req.user = {
    uid: 'local-admin',
    email: 'admin',
    name: 'Admin',
    picture: null,
    role: 'admin',
    authMode: 'local-admin-dev-token',
  };
}

function attachDevUserFromToken(req, token) {
  const decoded = decodeJwtPayloadUnsafe(token) || {};
  const uid = decoded.user_id || decoded.sub || decoded.uid || 'local-dev-user';
  req.firebaseUser = { ...decoded, uid, unverifiedDevToken: true };
  req.user = {
    uid,
    email: decoded.email || 'local-dev@example.com',
    name: decoded.name || decoded.email || 'Local Dev User',
    picture: decoded.picture,
    role: decoded.role || 'customer',
    authMode: 'unverified-dev-token',
  };
}

async function verifyFirebaseToken(req, res, next, { required }) {
  const token = getBearerToken(req);
  if (!token) {
    if (!required) return next();
    return res.status(401).json({ error: 'Firebase ID token required' });
  }

  if (token === 'local-admin-dev-token' && allowDevAuthFallback()) {
    attachLocalAdmin(req);
    return next();
  }

  const auth = getFirebaseAuth();
  if (!auth || !isFirebaseAdminReady()) {
    if (token && allowDevAuthFallback()) {
      attachDevUserFromToken(req, token);
      return next();
    }
    if (!required) return next();
    return res.status(503).json({
      error: 'Firebase Admin SDK is not configured',
      details: 'Set FIREBASE_SERVICE_ACCOUNT_JSON, FIREBASE_SERVICE_ACCOUNT_BASE64, or GOOGLE_APPLICATION_CREDENTIALS.',
    });
  }

  try {
    const decoded = await auth.verifyIdToken(token);
    req.firebaseUser = decoded;
    req.user = {
      uid: decoded.uid,
      email: decoded.email,
      name: decoded.name,
      picture: decoded.picture,
      role: decoded.role || decoded.claims?.role || 'customer',
    };
    return next();
  } catch (error) {
    return res.status(401).json({ error: 'Invalid Firebase ID token', details: error.message });
  }
}

function optionalFirebaseAuth(req, res, next) {
  return verifyFirebaseToken(req, res, next, { required: false });
}

function requireFirebaseAuth(req, res, next) {
  return verifyFirebaseToken(req, res, next, { required: true });
}

module.exports = {
  optionalFirebaseAuth,
  requireFirebaseAuth,
};
