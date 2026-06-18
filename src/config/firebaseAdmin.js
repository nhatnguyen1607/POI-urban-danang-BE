const admin = require('firebase-admin');
const { getAuth } = require('firebase-admin/auth');
const { getFirestore } = require('firebase-admin/firestore');
const fs = require('fs');
const path = require('path');

function loadLocalEnv() {
  const envPath = path.resolve(__dirname, '..', '..', '.env');
  if (!fs.existsSync(envPath)) return;
  const lines = fs.readFileSync(envPath, 'utf8').split(/\r?\n/);
  lines.forEach((line) => {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) return;
    const index = trimmed.indexOf('=');
    if (index === -1) return;
    const key = trimmed.slice(0, index).trim();
    let value = trimmed.slice(index + 1).trim();
    if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1);
    }
    if (key && process.env[key] === undefined) process.env[key] = value;
  });
}

loadLocalEnv();

function parseServiceAccount() {
  if (process.env.FIREBASE_SERVICE_ACCOUNT_JSON) {
    return JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT_JSON);
  }

  if (process.env.FIREBASE_SERVICE_ACCOUNT_BASE64) {
    return JSON.parse(Buffer.from(process.env.FIREBASE_SERVICE_ACCOUNT_BASE64, 'base64').toString('utf8'));
  }

  const localServiceAccountPath = path.resolve(
    __dirname,
    '..',
    '..',
    'poi-urban-firebase-adminsdk-fbsvc-3613b15e7b.json',
  );
  if (fs.existsSync(localServiceAccountPath)) {
    return JSON.parse(fs.readFileSync(localServiceAccountPath, 'utf8'));
  }

  return null;
}

function initializeFirebaseAdmin() {
  if (admin.getApps().length) return admin.getApp();

  const serviceAccount = parseServiceAccount();
  if (serviceAccount) {
    return admin.initializeApp({
      credential: admin.cert(serviceAccount),
      projectId: serviceAccount.project_id || process.env.FIREBASE_PROJECT_ID,
    });
  }

  if (process.env.GOOGLE_APPLICATION_CREDENTIALS) {
    return admin.initializeApp({
      credential: admin.applicationDefault(),
      projectId: process.env.FIREBASE_PROJECT_ID,
    });
  }

  return null;
}

const firebaseAdminApp = initializeFirebaseAdmin();

function getFirebaseAdminDiagnostics() {
  return {
    hasServiceAccountJson: Boolean(process.env.FIREBASE_SERVICE_ACCOUNT_JSON),
    serviceAccountJsonLength: process.env.FIREBASE_SERVICE_ACCOUNT_JSON?.length || 0,
    hasServiceAccountBase64: Boolean(process.env.FIREBASE_SERVICE_ACCOUNT_BASE64),
    serviceAccountBase64Length: process.env.FIREBASE_SERVICE_ACCOUNT_BASE64?.length || 0,
    hasGoogleApplicationCredentials: Boolean(process.env.GOOGLE_APPLICATION_CREDENTIALS),
    firebaseProjectId: process.env.FIREBASE_PROJECT_ID || null,
    hasLocalServiceAccountFile: fs.existsSync(
      path.resolve(__dirname, '..', '..', 'poi-urban-firebase-adminsdk-fbsvc-3613b15e7b.json'),
    ),
  };
}

function isFirebaseAdminReady() {
  return Boolean(firebaseAdminApp);
}

function getFirebaseAuth() {
  if (!firebaseAdminApp) return null;
  return getAuth(firebaseAdminApp);
}

function getFirestoreDb() {
  if (!firebaseAdminApp) return null;
  return getFirestore(firebaseAdminApp);
}

function requireFirestoreDb() {
  const db = getFirestoreDb();
  if (!db) {
    const error = new Error(
      'Firestore is not configured. Set FIREBASE_SERVICE_ACCOUNT_JSON, FIREBASE_SERVICE_ACCOUNT_BASE64, or GOOGLE_APPLICATION_CREDENTIALS.',
    );
    error.code = 'FIRESTORE_NOT_CONFIGURED';
    error.status = 503;
    throw error;
  }
  return db;
}

module.exports = {
  admin,
  getFirebaseAuth,
  getFirestoreDb,
  requireFirestoreDb,
  getFirebaseAdminDiagnostics,
  isFirebaseAdminReady,
};
