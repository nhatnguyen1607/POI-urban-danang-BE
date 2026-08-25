const { getFirebaseAuth, isFirebaseAdminReady } = require('../src/config/firebaseAdmin');

function parseArguments(argv) {
  const values = {};
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === '--uid' || argument === '--admin') {
      values[argument] = argv[index + 1];
      index += 1;
      continue;
    }
    const separator = argument.indexOf('=');
    if (separator > 0) values[argument.slice(0, separator)] = argument.slice(separator + 1);
  }

  const uid = String(values['--uid'] || '').trim();
  const adminValue = String(values['--admin'] || '').trim().toLowerCase();

  if (!uid || uid.length > 128 || !['true', 'false'].includes(adminValue)) {
    throw new Error('Usage: node scripts/set-admin-claim.js --uid <FIREBASE_UID> --admin=true|false');
  }
  return { uid, admin: adminValue === 'true' };
}

async function main() {
  const { uid, admin } = parseArguments(process.argv.slice(2));
  if (!isFirebaseAdminReady()) {
    throw new Error('Firebase Admin is not configured for this operator environment.');
  }

  const auth = getFirebaseAuth();
  const user = await auth.getUser(uid);
  const claims = { ...(user.customClaims || {}) };
  if (admin) claims.admin = true;
  else delete claims.admin;

  await auth.setCustomUserClaims(uid, claims);
  console.log(`Admin claim updated for UID ${uid}: admin=${admin}`);
  console.log('The client must sign in again or explicitly refresh its Firebase ID token.');
}

main().catch((error) => {
  console.error(`Admin claim update failed: ${error.message}`);
  process.exitCode = 1;
});
