# Phase 5 Stage 5C - Staging Deployment

Date: 2026-08-14

## Verdict

`STAGING_PREPARED_BLOCKED_BY_PLATFORM_ACCESS`

Repository-side staging preparation passed, but no authenticated backend
staging target is available. No staging deployment or deployed E2E claim was
made.

## Intended Architecture

- Backend: reuse the repository Docker deployment architecture documented for
  Hugging Face Spaces.
- Frontend: reuse the existing Vercel Git integration and Preview environment.
- Frontend API URL: configure `VITE_API_BASE_URL` or `VITE_API_URL` to the final
  HTTPS backend staging URL.
- Backend CORS: configure `URBANAGENT_CORS_ALLOWED_ORIGINS` to the exact Vercel
  Preview origin.

The backend Docker build now uses `npm ci --omit=dev`, executes
`npm run data:verify` during image construction, and includes a health check
against `/api/v2/cities`. Startup also retains the Stage 5B fail-closed data
verification. `.dockerignore` excludes local, raw, spike, and secret inputs
without excluding the canonical runtime dataset.

## Required Environment Variable Names

Backend core staging:

- `NODE_ENV`
- `PORT`
- `URBANAGENT_CORS_ALLOWED_ORIGINS`

Backend saved trips and authenticated flows:

- one supported Firebase Admin credential variable or workload credential
- `FIREBASE_PROJECT_ID` where required by that credential mode
- `DISABLE_DEV_AUTH_FALLBACK=true`

Frontend build:

- `VITE_API_BASE_URL` or `VITE_API_URL`
- existing `VITE_FIREBASE_*` public client configuration for the approved
  staging Firebase project
- `VITE_DEMO_AUTH_MODE=false`

No secret value is committed.

## Data And Local Pre-deploy Validation

- Runtime POIs: 4173.
- Canonical SHA-256:
  `dcb404cc8b5c7a9b5fd70df63039ab8f828c504270e22b12a671fa4ed61583f4`.
- `npm run data:verify`: PASS.
- Backend focused tests: 12 passed, 0 failed.
- Production-config startup and loader: PASS.
- Exact configured CORS origin: PASS.
- Unapproved origin: not trusted.
- Frontend production build: PASS with the existing 1.41 MB chunk warning.
- Frontend install still reports 6 known advisories; no audit fix was run.

These are local pre-deploy results, not deployed staging validation.

## Platform Discovery

Backend GitHub has no deployment, environment, workflow, deployment secret, or
deployment variable. The repository references Hugging Face Spaces, but no
Space exists for the repository owner and the available CLI is not
authenticated.

Frontend GitHub is connected to Vercel and has Preview/Production deployment
history. The locally authenticated Vercel account does not have access to the
UrbanAgent project, so its Preview environment variables cannot be configured.

No HTTPS staging backend/frontend URL was available. Firebase staging
credentials were also unavailable, so saved-trip staging remains
`SAVED_TRIP_STAGING_BLOCKED_BY_CREDENTIALS`.

## Exact Unblock Procedure

1. Grant access to an existing non-production backend Space or create and
   authorize a free staging Space using the repository Docker architecture.
2. Configure the backend environment variable names above through platform
   secrets and deploy the Stage 5C backend commit.
3. Grant access to the connected UrbanAgent Vercel project, configure its
   Preview API URL and Firebase public staging settings, then deploy a Preview.
4. Run the Stage 5C deployed HTTPS scenarios, map/mobile checks, CORS checks,
   error-state checks, logs review, latency measurements, and saved-trip tests
   when staging Firebase credentials are present.

## Rollback And Redeploy

Redeploy the prior known-good commit on the same staging service. Do not promote
the staging deployment or change production DNS. Dataset verification must pass
on every build and startup.

## Remaining Gate

Stage 5C cannot pass until authenticated backend and frontend staging targets
exist and the real deployed E2E checks succeed. Stage 5D must not start yet.
