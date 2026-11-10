# NaviGo final integration and publication - 2026-11-10

## Current final status - 2026-11-10

**Verdict: BLOCKED_BY_PRODUCTION_DEPLOYMENT_GATE.**

- [FE PR #21](https://github.com/nhatnguyen1607/POI-urban-danang-FE/pull/21) and [BE PR #44](https://github.com/nhatnguyen1607/POI-urban-danang-BE/pull/44) are OPEN, Ready for Review, MERGEABLE/CLEAN. No merge performed; no merge SHA.
- GitHub CLI 2.103.3 securely authenticated as Vuz14 via Windows keyring. PRs were created through CLI after duplicate checks. Earlier connector 403 / local-only / no-PR verdicts are historical and superseded.
- FE Vercel preview success. [BE safe integration-ref CI](https://github.com/nhatnguyen1607/POI-urban-danang-BE/actions/runs/42798289516) SUCCESS: 38/38 focused tests, payload install/data/startup/CORS validation PASS; Hugging Face sync SKIPPED. No absent-checks-as-PASS claim.
- HF_AUTO_DEPLOY_ENABLED=true: backend main merge triggers production deployment. Existing FE main has a Vercel deployment labeled Production; integration deploys Preview. Production changes were not authorized, so both PRs remain unmerged.
- No branch deleted. All 70 original branch names/SHAs retained; 72 current branches including 2 integrations. Verified full-ref bundle backups and cleanup manifest under D:/CD1_finance/navigo-audit/backups-final-integration/.
- Full quality, review/protection, OSRM equivalence, technology and cleanup evidence: [final integration record](NAVIGO_FINAL_INTEGRATION.md).

## GitHub and review evidence

Both branches are integration/navigo-audit-2026-11-10. Original commits 9e597b7 (FE) and ed07e30 (BE) and publication commits aec6cd6 / 4d9a0a5 remain in history. Follow-ups change documentation and add two focused-test invocations to the existing BE workflow; deployment triggers/guards are unchanged.

Fetched main SHAs unchanged: FE 3914e1c7b335deccb3d7c2c8162cac2dacb9864e; BE be6d21ae31dfc9c74f7e4b56e87a23e4df10342a. Main remains ancestor of each integration. PR base=head targets verified as integration -> main, scoped diff inspected. No force push, reset, patch reapply, cherry-pick or history rewrite.

Both PRs were created draft then marked Ready for Review after checking metadata/diff and verification. GitHub reported MERGEABLE/CLEAN; no submitted reviews, pending review requests or unresolved review threads. This is review-ready, not permission to deploy or a claim of reviewer approval. Both main endpoints report protected=false, applicable rules arrays empty, classic protection endpoint 404. No required lint check was bypassed. Recheck before any later merge.

## Quality and CI

- FE runtime unchanged: prior same-environment build PASS and 19/19 tests PASS remain valid. ESLint FAIL on both fetched main and integration: 98 errors + 2 warnings, all diagnostics identical after checkout-path normalization, zero regression. 77 no-explicit-any, 12 set-state-in-effect, 8 no-irregular-whitespace, 1 only-export-components; 2 exhaustive-deps warnings. No rules disabled. Vite chunk-size warning remains.
- BE runtime unchanged since full rerun: 403 total, 388 PASS, 0 FAIL, 15 existing guarded SKIP. Includes 22 OSRM and 2 matrix regressions; 1 disposable PostGIS + 14 historical/artifact City Pack skips. No test disabled or fixture changed to pass.
- New focused workflow command rerun locally: 38 PASS, 0 FAIL, 0 SKIP. Remote workflow_dispatch on integration ref 4c8b7e3b819e4cc834570295af8302b5adcb2658 also SUCCESS, run 42798289516. Tested OSRM/matrix and product/packaging/saved-trip contracts, installed deterministic runtime payload, checked exact dataset, smoke-tested startup and exact-origin CORS. Existing gate resolved deploy=false; sync step SKIPPED. Main workflow_dispatch was not invoked.
- Functions build PASS on Node24.11.1; Functions package expects Node22. Real emulator execution not verified.
- Canonical verifier PASS: 4173 POIs, 7220189 bytes, SHA-256 dcb404cc8b5c7a9b5fd70df63039ab8f828c504270e22b12a671fa4ed61583f4, zero duplicate IDs/invalid core records. Historical Stage4G 4166 count/hash retained separately. Canonical and fixture bytes unchanged.
- Real browser/Firebase/PostGIS/live-provider/checkpoint E2E remain unverified; no production truth or model-quality claims from mocks. New final documentation commits need their own final remote status read; CI source/workflow is unchanged by those documentation commits.

## Technology and OSRM integration

Main already contains application orchestration, POI retrieval/recommendation/reranker, JSON memory/feedback learning, Rete/backward chaining/Fuzzy Logic, itinerary planning, Firestore/Firebase Auth/Functions, business scoring, City Pack/provenance and API/frontend flows. No source modules removed.

The PR (not main yet) restores parsed OSRM caching/coalescing, bounded body timeout/retry/concurrency/error classification and zero-coordinate validation. Matrix accepts walk/walking and preserves zero longitude. Scoped git diff against fix/osrm-response-body confirms all four OSRM runtime files and routeBodyRegression test are identical. Its missing workflow test invocation is now included, with matrix regression invocation added. Historical branch documentation remains preserved in its branch; no automatic import of old operational claims.

Two-tower/research representation checkpoints, actual multimodal model bundle/index and real checkpoint runtime inference are not verified integrations. External E2E and existing frontend lint debt remain follow-up work.

## Cleanup and backups

No cleanup executed because integration is not merged/verified in main and production safety blocks merge. Original branches: FE 21, BE 49 (70). Current: FE 22, BE 50 (72), including integrations. Deleted 0, retained 72. Current original names/SHAs exactly match NAVIGO_BRANCH_EVIDENCE.json, so no original unique commits/ref tips were lost.

Backup directory: D:/CD1_finance/navigo-audit/backups-final-integration/. FE-all-refs.bundle and BE-all-refs.bundle each pass git bundle verify and contain complete history. Ref snapshots and cleanup-manifest.json record exact branch SHAs. Final follow-up bundles can capture later documentation commits; no bundle is a license to delete uncertain branches.

Branches not ancestors of unchanged main: FE demo/2026-08-07-end-to-end and integration/navigo-audit-2026-11-10; BE clean, demo/2026-08-07-per-day-windows (content-equivalent history), fix/osrm-response-body, integration/navigo-audit-2026-11-10. Preserve clean/unrelated history, unique demo work and fix/osrm-response-body. Already-merged branches also remain because deployment/release/external usage cannot be fully proven safe.

## Production gate

BE .github/workflows/deploy-huggingface-space.yml runs on main pushes and manual dispatch. The exact non-secret Actions variable HF_AUTO_DEPLOY_ENABLED was read through gh and equals true; a main push may sync production. workflow_dispatch on main unconditionally enables deployment. Integration-ref manual dispatch is the existing safe validation path and was verified to skip sync. No guard was disabled or changed.

FE exact original main commit has GitHub deployment environment Production (legacy production_environment boolean is false, so that boolean alone is not authoritative), and integration commits have environment Preview. Treat main as production-bound until separately approved. No production deployment was triggered in this task; existing production was not modified or operationally certified. Automatic branch previews are permitted and observed.

Next action requires separate production deployment approval/change plan before merge, followed by fresh main/PR/check/review verification. No auto-merge enabled and no branch protection bypass.
