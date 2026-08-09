# Phase 2 Batch 6 - Authenticated Saved-Trip Lifecycle

Status: implemented on branch `phase2/batch6-trip-lifecycle` for review.

## Approved Scope

- Authenticated saved-trip replan.
- Authenticated saved-trip stop add, remove, and reorder.
- Owner-only lifecycle operations.
- Existing `/urban-agent` integration for saved trips.

Feedback persistence is not part of Batch 6.

## Routes Implemented

- `POST /api/v2/trips/:tripId/replan`
- `POST /api/v2/trips/:tripId/stops`
- `PATCH /api/v2/trips/:tripId/stops/reorder`
- `DELETE /api/v2/trips/:tripId/stops/:stopId`

`POST /api/v2/trips/preview` remains stateless and nonpersistent.

## Ownership Rule

Every lifecycle route requires Firebase-authenticated request context. The
saved-trip owner is always derived from `req.user.uid`; request-body `userId`
values are ignored and cannot authorize access.

## Lifecycle Semantics

Manual add, remove, and reorder operations update the same saved trip and set
`needsReplan: true`. They do not claim the route has been optimized.

Saved-trip replan reuses the existing Batch 3 trip-preview engine, updates the
same `tripId`, replaces the persisted itinerary with the calculated result,
updates `updatedAt`, and clears `needsReplan`.

Removing a stop also adds its POI ID to the saved trip's exclude list so the
removed POI does not silently return during the next replan.

## Exclusions

- No feedback/rating persistence.
- No collaboration or share links.
- No booking/payment.
- No external routing provider.
- No external POI source or multi-source ingestion.
- No second city.
- No mobile app.
- No canonical CSV changes.

## Validation Result

Validation is recorded in the Batch 6 final response. Required validation:

- backend targeted owner/lifecycle tests,
- full backend test run once before push,
- frontend production build,
- scoped frontend lint on changed Batch 6 files,
- safe local smoke with memory/test persistence.
