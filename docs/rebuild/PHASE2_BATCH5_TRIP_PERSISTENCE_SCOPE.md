# Phase 2 Batch 5 - Authenticated Trip Persistence

Status: APPROVED FOR IMPLEMENTATION.

Batch 5 adds authenticated saved trips to the real UrbanAgent traveler product.
It does not change the canonical POI runtime, canonical CSV bytes, PostgreSQL
opt-in behavior, or trip-preview semantics.

## Approved Scope

- Save a generated trip by explicit user action.
- List the authenticated user's saved trips.
- Open a saved trip and restore it into the existing planner.
- Update an existing saved trip after edit/replan.
- Delete a saved trip after explicit confirmation.
- Enforce owner-only read, update, and delete behavior.

## APIs

- `POST /api/v2/trips`
- `GET /api/v2/trips`
- `GET /api/v2/trips/:tripId`
- `PATCH /api/v2/trips/:tripId`
- `DELETE /api/v2/trips/:tripId`

`POST /api/v2/trips/preview` remains stateless and nonpersistent. A preview
does not create a saved trip unless the user explicitly saves it.

## Ownership Rule

Every saved trip belongs to `req.user.uid` from the authenticated Firebase
request context. Request body `userId` values are ignored for authorization.
Trip IDs alone never bypass the owner check.

## Persistence Model

Production saved trips use the existing Firebase Auth plus Firestore
infrastructure for lightweight user-owned data. The POI catalog remains the
canonical CSV default runtime. Tests and local validation may use the guarded
nonproduction memory adapter with `URBANAGENT_SAVED_TRIPS_STORE=memory`.

Saved trips preserve stable POI IDs and trip-specific snapshots/request state,
not a duplicate copy of the full canonical POI catalog.

## Exclusions

- No feedback/rating system.
- No collaboration or shared trips.
- No booking or payment.
- No external routing provider.
- No external POI source.
- No offline mode.
- No second city.
- No multi-source ingestion.
- No mobile app.

## Test Result

- Backend `npm.cmd test`: PASS, 40 passed, 0 failed, 1 guarded PostGIS skip.
- Frontend `npm.cmd run build`: PASS.
- Frontend scoped lint on `src/utils/apiClient.ts`: PASS.
- Integrated local HTTP smoke with backend memory saved-trip adapter and
  frontend dev server: PASS for routes, recommendations, trip preview, save,
  list, open, update, and delete.
