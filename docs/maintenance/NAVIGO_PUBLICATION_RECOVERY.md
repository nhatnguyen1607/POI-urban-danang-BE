# NaviGo publication recovery - 2026-11-10

Verdict: BRANCHES_PUSHED_PR_CREATION_BLOCKED. This supersedes earlier publication-blocked statements for branch pushes only. No PR exists from this recovery run.

## Repository identity and preservation

- FE: D:/CD1_finance/navigo-audit/FE; https://github.com/nhatnguyen1607/POI-urban-danang-FE.git
- BE: D:/CD1_finance/navigo-audit/BE; https://github.com/nhatnguyen1607/POI-urban-danang-BE.git
- Both branches: integration/navigo-audit-2026-11-10.
- Original integration commits verified and normally pushed: FE 9e597b7b47ad8dacbb4d056dc54e07eba4a30dd1; BE ed07e30cc6e73ad293234e44ea54fd3f7822376b.
- Fetched main unchanged: FE 3914e1c7b335deccb3d7c2c8162cac2dacb9864e; BE be6d21ae31dfc9c74f7e4b56e87a23e4df10342a. Main is ancestor in each repository; no rebase/merge needed.
- Both trees clean before tests and after restoring generated outputs. Existing patches passed reverse-check and were never applied. Original patches retained unchanged.
- No branch deleted, history rewritten, force push, main push, workflow dispatch, production deployment or merge performed. Backend fix/osrm-response-body remains at 505052a2f4793f8a8a7f79b2ad21c41595080005.

## Authentication and remote result

Configured Git Credential Manager successfully fetched, dry-ran, then normally pushed both branches. No credential/token was printed, retrieved for reuse, created or changed. GitHub CLI gh is not available in PATH or the checked standard installation locations, so gh auth status / gh PR creation cannot run. PowerShell npm.ps1 is blocked by existing execution policy; npm.cmd was used without changing policy.

Connected GitHub app metadata advertises push=true, but create_pull_request returned HTTP 403 Resource not accessible by integration for both repositories. This is a GitHub integration permission denial, not an automatic approval-review rejection. No PR URL, PR mergeability or review status exists. Install GitHub CLI and authenticate interactively with an account authorized for these repositories, then use the prepared recovery PR bodies and commands. Do not extract credentials from Git Credential Manager.

Initial published FE commit: Vercel status success and Vercel Preview Comments success. Vercel automatically created an integration-branch preview; this is not a manually triggered production deploy. Initial BE commit: no check runs and no statuses; combined status API returns pending with an empty status list, which is not evidence of an executing or passing CI run. The backend workflow runs on main pushes / workflow_dispatch, not integration pushes. Later documentation-only follow-up commits require a fresh remote status read.

## Latest local verification

- Node v24.11.1, same installed dependency trees as handoff.
- FE npm.cmd run build PASS; chunk-size warning remains. node --test tests/phase6/*.test.ts tests/reviews/*.test.ts: 19 PASS, 0 FAIL.
- FE lint: integration and exported fetched main ran the same ESLint installation/config, under the same Node/environment, without checkout. Each has 98 errors and 2 warnings. All 100 diagnostics match after normalizing checkout paths, including rule, severity, location and message. New diagnostics: 0. Existing errors: 77 @typescript-eslint/no-explicit-any; 12 react-hooks/set-state-in-effect; 8 no-irregular-whitespace; 1 react-refresh/only-export-components. Existing warnings: 2 react-hooks/exhaustive-deps. No rules disabled or unrelated code changed.
- BE phase0/1/2/5/6: 131 total, 130 PASS, 0 FAIL, 1 pre-existing guarded PostGIS SKIP.
- BE phase4: 272 total, 258 PASS, 0 FAIL, 14 pre-existing guarded historical/artifact SKIP. Combined: 403 total, 388 PASS, 0 FAIL, 15 SKIP.
- Includes 22 OSRM and 2 matrix regression cases: parsed response caching/coalescing, distinct concurrent legs, body timeout, bounded retries/concurrency, NoRoute circuit classification, zero coordinates, walk/walking matrix and local HTTP contract.
- Phase4-generated outputs were snapshotted before execution and twelve changed outputs restored to exact pre-test bytes. Test output reports zero failures; wrapper console printing subsequently raised a Windows cp1252 UnicodeEncodeError after restoration, unrelated to test results. No baseline, fixture expectation, source or test was changed during this recovery.
- npm.cmd run functions:build PASS; package expects Node22, host Node24; emulator execution unverified.
- npm.cmd run data:verify PASS: 4173 POIs, 7220189 bytes, SHA-256 dcb404cc8b5c7a9b5fd70df63039ab8f828c504270e22b12a671fa4ed61583f4, zero duplicate canonical IDs and invalid core records.
- git diff --check PASS for original integrations. Follow-up changes are publication records only.

## Technology preservation and remaining blockers

Reused the completed technology inventory and scoped main-to-integration diff; did not repeat the 70-branch/63-PR audit. FE original integration changes only seven documentation files. BE changes five runtime files, regression assertions/tests and documentation; no source paths removed. Agent orchestration, retrieval/recommendation/reranker, memory/feedback, Rete/backward chaining/fuzzy logic, OSRM/planning, Firestore/Firebase Auth/Functions, business scoring, City Pack/provenance and API/frontend flows remain present. The recovery adds documentation only.

Research representations / two-tower checkpoints are not proven runtime integrations; real model bundle/inference, browser/Firebase/PostGIS/live-provider E2E remain unverified. Existing lint debt remains. PR creation permission/authentication, actual PR checks/review and deployment authorization block merge readiness. Keep every original branch, particularly fix/osrm-response-body, until integration is reviewed and verified. No automatic merge is proposed.
