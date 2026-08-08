# Integrated Demo Release - 2026-08

Status: `IMPLEMENTED_ON_RELEASE_BRANCH_NOT_MERGED`

Branch:

`release/integrated-demo-2026-08`

Backend clone:

`C:\tmp\urbanagent-integrated-backend-20260808-175300`

Frontend clone:

`C:\tmp\urbanagent-integrated-frontend-20260808-175300`

## Scope

This release integrates the approved Phase 2 Batch 3 traveler API flow into
the existing main web application for a local video-ready demo.

Backend scope:

- Preserve the existing `POST /api/v2/trips/preview` stateless contract.
- Preserve backward compatibility for `trip.dailyWindow`.
- Preserve per-day `trip.dayWindows[]` support from the demo branch.
- Expose actual per-day calendar dates by deriving each `trip.days[].date`
  from `trip.date + dayNumber - 1`.
- Reject impossible calendar dates instead of accepting JavaScript date
  rollover.
- Keep CSV as the default runtime.
- Keep PostgreSQL/PostGIS explicit opt-in.

Frontend scope:

- Keep the existing logged-in web application, layout, navigation, dashboard,
  Urban Agent page, Leaflet map, and existing non-v2 Urban Agent actions.
- Add a demo-only local auth fallback controlled by
  `VITE_DEMO_AUTH_MODE=true`.
- Integrate the traveler v2 recommendation and trip-preview flow into
  `/urban-agent`.
- Require a real trip start date before generating calendar days.
- Support 1-7 days, default daily windows, and per-day time overrides.
- Display itinerary stops grouped by day with durations, transfer estimates,
  reasons, reason codes, warnings, and request ID.
- Reuse the existing route modal for preview map display.
- Draw only an illustrative polyline connecting preview stops when no road
  route geometry exists, and label it as illustrative.

## Explicit Non-Scope

- No standalone `/demo` primary experience.
- No production Firebase writes.
- No production or shared database access.
- No PostgreSQL default-runtime switch.
- No external routing provider.
- No external POI source.
- No multi-source implementation.
- No second City Pack.
- No authentication production redesign.
- No trip persistence, saved-trip v2, replan, stop mutation, or feedback
  persistence.
- No mobile implementation.
- No Batch 4.

## Validation Summary

- Backend default tests: PASS, `40` total, `39` passed, `0` failed,
  `1` guarded optional PostGIS skip.
- Backend syntax checks: PASS.
- Backend production audit: PASS, `0` vulnerabilities after a narrow
  `brace-expansion` override update to `5.0.9`.
- Canonical CSV SHA-256 unchanged:
  `5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae`.
- Frontend production build: PASS.
- Frontend production audit: PASS, `0` vulnerabilities after narrow
  same-major `react-router-dom` update and `protobufjs` override.
- Frontend scoped lint: FAIL due pre-existing lint debt in large existing
  files; TypeScript production build remains PASS.

## Remaining Risks

- Full frontend lint still has pre-existing `no-explicit-any`,
  `set-state-in-effect`, and purity findings in existing large files.
- Frontend `npm ls --omit=dev --all` reports ELSPROBLEMS when run against a
  full dev install because dev/optional packages remain in `node_modules`.
  Production audit is clean and production build passes.
- The preview map polyline is illustrative and not road-network routing.
- Local demo auth is intentionally non-production and only enabled by
  `VITE_DEMO_AUTH_MODE=true`.
