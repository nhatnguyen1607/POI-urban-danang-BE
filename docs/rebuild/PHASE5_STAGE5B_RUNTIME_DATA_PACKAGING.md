# Phase 5 Stage 5B - Runtime Data Packaging

Date: 2026-08-14

## Verdict

PASS. The canonical traveler runtime dataset is now delivered as a normal Git
blob. A clean clone no longer needs Git LFS to obtain the runtime CSV.

## Selected Strategy

Strategy A: normal Git blob.

- Canonical size: `7220189` bytes.
- POIs: `4173`.
- SHA-256:
  `dcb404cc8b5c7a9b5fd70df63039ab8f828c504270e22b12a671fa4ed61583f4`.
- The file is comfortably below GitHub's individual-file limit.
- The migration is forward-only and does not rewrite history.
- Canonical content, ordering, encoding, and line endings are unchanged.

Before Stage 5B, `.gitattributes` applied the LFS filter to the canonical path
and a clean checkout contained a 132-byte pointer. After Stage 5B, the path has
no LFS filter and uses `-text` so Windows line-ending conversion cannot alter
the approved bytes. Other LFS rules remain unchanged.

## Runtime Contract

The versioned manifest is:

`data/canonical/runtime_dataset_manifest.json`

It pins the dataset version, runtime path, packaging mode, byte size, POI count,
and SHA-256. Verify a checkout with:

```powershell
npm.cmd run data:verify
```

The verifier is read-only and checks file presence, LFS pointer text, byte size,
SHA-256, canonical headers, row/application/unique-ID counts, duplicate IDs, and
invalid core records. Startup runs the same verifier before opening the HTTP
port and fails with an actionable `RUNTIME_DATA_*` code.

## Clean Clone Validation

Validation clone:

`D:\UrbanAgent-temp\stage5b-clean-clone-20260814-114504\backend`

The clone used `GIT_LFS_SKIP_SMUDGE=1` to prevent an existing LFS cache from
masking the result. The canonical file was still checked out as `7220189` bytes,
was absent from `git lfs ls-files`, matched the approved SHA, and passed the data
verifier. `npm ci`, backend startup, the 4173-POI loader, recommendations, trip
preview, replan, and saved-trip lifecycle checks passed.

## Deployment Configuration

Required for a production browser deployment:

- `NODE_ENV=production`
- `URBANAGENT_CORS_ALLOWED_ORIGINS`: comma-separated exact frontend origins

Required for authenticated production saved trips:

- one supported Firebase Admin credential mechanism supplied by the deployment
  secret store, plus the matching project configuration

Optional:

- `PORT`
- `FEATURE_GUEST_ITINERARY_PREVIEW`
- `PYTHON_EXECUTABLE`
- `URBANAGENT_POI_REPOSITORY=postgres`, `DATABASE_URL`, and `PGSSLMODE` only for
  the explicit PostgreSQL runtime; CSV remains the default

Development-only:

- `URBANAGENT_SAVED_TRIPS_STORE=memory`
- local development auth fallback controls

No secret value is committed. Production CORS no longer defaults to `*`.
Frontend API targeting was already configurable through `VITE_API_BASE_URL` or
`VITE_API_URL`, so no frontend change was needed.

The existing Dockerfile copies the repository and therefore includes the normal
Git canonical CSV. No new container design was introduced.

## Future Canonical Updates

Every controlled canonical PR must update the normal Git CSV and
`runtime_dataset_manifest.json` in the same PR, then run `npm run data:verify`.
The Phase 4 scheduler was not changed in Stage 5B; any automated canonical PR
must satisfy this manifest gate before merge.

## Remaining Limitations

- Staging deployment and real production-origin configuration remain Stage 5C.
- Production Firebase credentials must be supplied by the deployment platform.
- Legacy research/model/image LFS assets remain LFS-managed; they are not needed
  for the validated traveler CSV runtime, but their availability is a separate
  operational concern.
- Existing frontend dependency advisories and large-chunk warning remain outside
  Stage 5B.
