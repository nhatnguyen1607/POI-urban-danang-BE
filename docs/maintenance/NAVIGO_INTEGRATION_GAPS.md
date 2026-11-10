## Current final status - 2026-11-10

**Verdict: BLOCKED_BY_PRODUCTION_DEPLOYMENT_GATE.**

- [FE PR #21](https://github.com/nhatnguyen1607/POI-urban-danang-FE/pull/21) and [BE PR #44](https://github.com/nhatnguyen1607/POI-urban-danang-BE/pull/44) are OPEN, Ready for Review, MERGEABLE/CLEAN. No merge performed; no merge SHA.
- GitHub CLI 2.103.3 securely authenticated as Vuz14 via Windows keyring. PRs were created through CLI after duplicate checks. Earlier connector 403 / local-only / no-PR verdicts are historical and superseded.
- FE Vercel preview success. [BE safe integration-ref CI](https://github.com/nhatnguyen1607/POI-urban-danang-BE/actions/runs/42798289516) SUCCESS: 38/38 focused tests, payload install/data/startup/CORS validation PASS; Hugging Face sync SKIPPED. No absent-checks-as-PASS claim.
- HF_AUTO_DEPLOY_ENABLED=true: backend main merge triggers production deployment. Existing FE main has a Vercel deployment labeled Production; integration deploys Preview. Production changes were not authorized, so both PRs remain unmerged.
- No branch deleted. All 70 original branch names/SHAs retained; 72 current branches including 2 integrations. Verified full-ref bundle backups and cleanup manifest under D:/CD1_finance/navigo-audit/backups-final-integration/.
- Full quality, review/protection, OSRM equivalence, technology and cleanup evidence: [final integration record](NAVIGO_FINAL_INTEGRATION.md).

---

## Historical audit record (superseded where noted above)

# NaviGo integration gaps

| Component | FE implementation | BE implementation | Runtime status | Missing integration | Tests | Required action |
|---|---|---|---|---|---|---|
| OSRM body reuse | Route modal calls /api/route | Main cached Fetch Response | Fixed on integration branch | Main still needs reviewed PR | 22 regression PASS | Review bounded five-file restoration |
| Matrix transport | Planner uses walk enum | Matrix checked walking only | Fixed on integration branch | Main uses wrong speed for walk | 2 regression PASS | Merge after checks/review |
| Per-day windows | Main sends dayWindows, start/end aliases | Main contains feature and tests | Already integrated | No runtime gap | Existing batch3 tests | Keep main; demo branch is patch-equivalent |
| Old demo UI | /demo redirects to /urban-agent | v2 contracts active | Superseded | No case for restoring 796-line old screen | FE build / unit tests | Preserve branch; do not merge old App routing |
| City Pack regression baseline | No client change | Six old assertions still demand 4166 | Corrected to approved live 4173 | Phase4 not in npm test | Full Phase4 run | Keep immutable hash checks and historical summary values |
| Semantic checkpoint path | User can request V4 | inference expects artifacts/model; only legacy model LFS path tracked | NOT VERIFIED / unavailable in package | Validated model bundle + index + versioned contract absent | No real checkpoint inference | Separate artifact packaging/evaluation task; do not fabricate weights |
| Missing modality / semantic precision | Text/image query UI exists | Gate already has availability mask and keyword boost | Code present | No external zero-shot evaluation proving category accuracy | NOT VERIFIED | Evaluate retrieval by category and domain before changing model behavior |
| Firebase persistence | Auth / save / feedback UI | Firebase services and Functions | Local contracts tested | Real Firebase login/write/emulator flow not exercised | Functions build PASS; real E2E NOT VERIFIED | Isolated configured environment required |
| PostGIS | Consumes common API | Opt-in repository/migrations | Default CSV | No disposable DB available | 1 guarded SKIP | Run explicit disposable PostGIS integration before any cutover |
| Partner product | /seller pages | Registry and provider contracts | Contracts implemented | Official partner credentials/request adapters not proven | Mock contracts tested | Verify provider in approved environment; retain honest handoff wording |
| FE quality | Application builds | N/A | Build PASS | Baseline ESLint has 98 errors / 2 warnings | FAIL | Separate lint cleanup with behavioral review; rules not disabled |
| Automation / deployment | Vercel status success on main | push-main Hugging Face deploy workflow | Main merge can deploy | User forbids automatic production deployment in this task | Source workflow + GitHub checks | Leave PR unmerged; no workflow bypass or dispatch |
| Documentation drift | Shared old context mirrored | MASTER_PLAN/CURRENT_STATE headers describe old phases | Historical records mixed with current | Old 4166 facts conflict with Stage4T manifest | Verifier PASS | Add dated maintenance record; do not rewrite master context |
