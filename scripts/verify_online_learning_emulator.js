const { initializeApp, getApps } = require('firebase-admin/app');
const { getFirestore, FieldValue } = require('firebase-admin/firestore');

const projectId = process.env.GCLOUD_PROJECT || process.env.GCLOUD_PROJECT_ID || 'demo-poi-urban';

if (!getApps().length) {
  initializeApp({ projectId });
}

const db = getFirestore();

async function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function main() {
  const userId = `emulator-user-${Date.now()}`;
  await db.collection('reviews').add({
    userId,
    poiId: 'poi-cafe-emulator',
    poiName: 'Cafe Emulator',
    poiCategory: 'Cafe',
    rating: 5,
    visitPurpose: 'work_study',
    visitMood: 'relaxed',
    context: {
      dayOfWeek: 'thursday',
      timeOfDay: 'morning',
    },
    createdAt: FieldValue.serverTimestamp(),
  });

  for (let attempt = 0; attempt < 20; attempt += 1) {
    const snap = await db.collection('user_preferences').doc(userId).get();
    if (snap.exists) {
      const profile = snap.data();
      console.log('online learning ok', {
        userId,
        learningMode: profile.learningMode,
        categoryWeights: profile.categoryWeights,
        purposeWeights: profile.purposeWeights,
        moodWeights: profile.moodWeights,
      });
      return;
    }
    await sleep(500);
  }

  throw new Error('Timed out waiting for user_preferences online update.');
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
