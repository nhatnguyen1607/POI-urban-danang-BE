function requireAdmin(req, res, next) {
  if (!req.firebaseUser?.uid) {
    return res.status(401).json({ error: 'authentication_required' });
  }

  if (
    req.firebaseUser.admin !== true
    || req.firebaseUser.localAdmin === true
    || req.firebaseUser.unverifiedDevToken === true
  ) {
    return res.status(403).json({ error: 'admin_access_required' });
  }

  return next();
}

module.exports = { requireAdmin };
