# Firebase Online Learning for UrbanAgent

## What Exists Now

The backend already has batch/lazy aggregation through:

- `POST /api/user-preferences/rebuild`
- `POST /api/recommend-pois`

The new `functions/` package adds true event-driven online learning:

- `learnFromReviewCreated`: triggers on `reviews/{reviewId}`
- `learnFromUserAnalyticsCreated`: triggers on `user_analytics/{eventId}`

Both update `user_preferences/{userId}` automatically in the background.

## Incremental Learning Logic

This does not retrain a large model. It updates compact preference vectors with exponential moving average (EMA):

```text
new_weight = old_weight * (1 - alpha) + event_signal * alpha
```

`alpha` is adaptive:

- base value: `0.18`
- longer dwell time increases alpha
- stronger review rating increases alpha
- clamped to `0.08..0.34`

The online profile stores only top-N keys, so the document stays small:

- `categoryWeights`
- `purposeWeights`
- `moodWeights`
- `contextAffinities`
- `purposeByCategory`
- `transportWeights`
- `dwellPreference`

The function runs inside a Firestore transaction and writes an idempotency marker to `_online_learning_events/{source_eventId}`. If Firebase retries a trigger, the marker prevents double learning.

## Local Emulator Setup

Install Firebase CLI if needed:

```powershell
npm install -g firebase-tools
```

Install Functions dependencies:

```powershell
cd D:\POI-urban-danang-BE\functions
npm install
```

Run emulators from the backend repo root:

```powershell
cd D:\POI-urban-danang-BE
npm.cmd run emulators:start
```

Emulator UI:

```text
http://127.0.0.1:22000
```

Firestore emulator:

```text
127.0.0.1:22002
```

End-to-end smoke test:

```powershell
npm.cmd run emulators:verify-learning
```

## Verify Locally

1. Open Emulator UI.
2. Create a document in `reviews`:

```js
{
  userId: "demo-user",
  poiId: "poi-cafe-1",
  poiName: "Cafe Bien Xanh",
  poiCategory: "Cafe",
  rating: 5,
  visitPurpose: "work_study",
  visitMood: "relaxed",
  context: {
    dayOfWeek: "thursday",
    timeOfDay: "morning"
  },
  createdAt: new Date()
}
```

3. Check `user_preferences/demo-user`.
4. Create a document in `user_analytics`:

```js
{
  eventType: "poi_visit",
  userId: "demo-user",
  poiId: "poi-cafe-1",
  poiName: "Cafe Bien Xanh",
  poiCategory: "Cafe",
  dwellMinutes: 64,
  context: {
    dayOfWeek: "thursday",
    timeOfDay: "morning"
  },
  createdAt: new Date()
}
```

5. Confirm `dwellPreference`, `categoryWeights`, and `contextAffinities` update without calling any manual rebuild endpoint.

## Production Deploy

After selecting the Firebase project:

```powershell
firebase deploy --only functions:online-learning
```

The same code works in emulator and production because it relies on Firebase Admin SDK default project credentials supplied by the runtime.
