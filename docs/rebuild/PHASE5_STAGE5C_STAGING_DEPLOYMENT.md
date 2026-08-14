# Phase 5 Stage 5C - Hugging Face Deployment

Date: 2026-08-14

## Verdict

`SAFE_AUTO_DEPLOY_PREPARED_FOR_REVIEW`

The existing Hugging Face Docker Space is now known and GitHub contains the
required `HF_TOKEN` secret. Repository-side deployment preparation and local
payload validation pass. No remote deployment was executed from this branch.

## Deployment Flow

```text
GitHub backend main
  -> GitHub Actions validation
  -> deterministic runtime-only payload
  -> nhttngy/back-end Hugging Face Docker Space
  -> https://nhttngy-back-end.hf.space
```

Workflow: `.github/workflows/deploy-huggingface-space.yml`.

Triggers:

- `push` to `main` runs all validation gates.
- `workflow_dispatch` runs validation and performs the first manual sync.
- A `push` to `main` performs the sync only after the repository variable
  `HF_AUTO_DEPLOY_ENABLED` is explicitly set to `true`.

This gate prevents the first remote deployment from occurring merely because
the Stage 5C branch is merged. The workflow passes only the secret name
`HF_TOKEN` to the official `huggingface/hub-sync@v0.2.1` action and never prints
its value. The target is the existing `nhttngy/back-end` Space with
`repo_type: space` and `space_sdk: docker`.

## Runtime Payload

`deploy/huggingface/runtime-files.txt` is the allowlist.
`scripts/prepare_hf_space_payload.js` copies only tracked regular files, rejects
Git LFS pointer text, and writes a file/SHA manifest into the generated payload.

Included:

- Dockerfile, Space README, lockfiles, and Python requirements;
- `src/**` and the runtime dataset verifier;
- the 4173-POI canonical CSV and both canonical/runtime manifests;
- small agent memory and reranker artifacts;
- deterministic expert-system rule CSVs;
- small existing training-status metadata.

Excluded:

- `.git`, `.github`, `node_modules`, tests, migrations, Firebase Functions;
- Phase 4 snapshots, candidate packs, source data, local caches and output;
- legacy root CSV inputs;
- model weights and report images managed by legacy Git LFS.

No LFS-managed file is required by the default traveler runtime. Semantic model
weights remain an optional perception tool and are absent from the current
Space revision as well; recommendation has an established deterministic
non-model path. CSV remains the default repository and Postgres stays opt-in.

## Space Configuration

Verified public Space metadata:

- repository: `nhttngy/back-end`;
- SDK: `docker`;
- app port: `7860`;
- runtime URL: `https://nhttngy-back-end.hf.space`;
- pre-upgrade rollback revision:
  `240c962c170c1b6ddef333863f9692c825808e2c`.

Required Hugging Face variable names:

- `NODE_ENV=production`;
- `URBANAGENT_CORS_ALLOWED_ORIGINS=https://poi-urban-danang-fe.vercel.app`;
- `FIREBASE_PROJECT_ID`;
- `PORT=7860` when the platform does not provide it.

Existing transition variables such as `CORS_ORIGIN` and `DEMO_AUTH_MODE` are
not deleted by this workflow. Space variables and secrets remain managed in
Hugging Face Settings and are not overwritten by repository synchronization.

Saved-trip authentication uses the existing supported secret name
`FIREBASE_SERVICE_ACCOUNT_BASE64`; no credential value is stored or printed.

## Validation Gates

Before sync, the workflow requires:

1. lockfile install with `npm ci`;
2. `npm run data:verify`;
3. JavaScript syntax checks;
4. focused recommendation, trip-preview, replan, saved-trip and packaging tests;
5. deterministic runtime payload creation;
6. payload-only `npm ci --omit=dev` and data verification;
7. payload startup and exact-origin CORS smoke.

Local results:

- payload files: 77 tracked files plus generated deployment manifest;
- deterministic payload test: PASS;
- focused tests: 13 passed, 0 failed;
- payload install: PASS, 0 reported vulnerabilities;
- runtime POIs: 4173;
- canonical SHA-256:
  `dcb404cc8b5c7a9b5fd70df63039ab8f828c504270e22b12a671fa4ed61583f4`;
- payload startup, loader and exact Vercel-origin CORS: PASS;
- unapproved CORS origin: not trusted;
- canonical bytes: unchanged.

Docker image build was not run locally because no Docker daemon is available.
The repository Dockerfile remains the validated build definition and the Space
will build it only after an authorized sync.

## Rollback And Next Action

Hugging Face history must be retained. If the first upgrade fails, select or
revert the Space to revision
`240c962c170c1b6ddef333863f9692c825808e2c` without changing variables or
secrets.

Next action: review and merge the Stage 5C branch, confirm the push-to-main
validation run passes without syncing, then manually dispatch the workflow for
the first upgrade. Validate the deployed HTTPS product before setting
`HF_AUTO_DEPLOY_ENABLED=true` for later main-branch auto-deployments.

Stage 5D has not started.
