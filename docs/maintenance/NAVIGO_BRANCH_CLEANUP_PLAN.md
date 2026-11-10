## Current final-integration status - 2026-11-10

Integration branches are published. GitHub CLI 2.103.3 is now securely authenticated via Windows keyring; PR creation and remote checks are in progress. Earlier local-only/permission verdicts below are historical. See [final integration record](NAVIGO_FINAL_INTEGRATION.md) for current tests, technology preservation, backups and deployment gates. No merge or cleanup performed.

---

## Historical audit record (superseded where noted above)

# NaviGo branch cleanup preparation — BE

No remote or local branch, tag, release, or repository was deleted. No history
was rewritten. SAFE_TO_DELETE: **0**. All 49 original branches retained.

Ancestry proves 19 FE / 45 BE non-main branches fully merged (64 total).
One BE demo branch is patch-equivalent (3 commits); one FE demo branch has
superseded unique UI; BE OSRM has 2 valuable unmerged commits; BE clean has
3 unrelated commits. Main branches are protected by maintenance policy.
Raw unique ancestry exists on 4 non-main branches; only OSRM has confirmed
missing runtime fixes. Counts exclude the new integration branches.

All merged/equivalent branches are only cleanup candidates. External deployment
branch bindings, webhooks, open work and owner intent are not fully observable.
The local workflow files were inspected; BE main triggers Hugging Face deploy,
and FE has Vercel status. No candidate is asserted safe while these checks are
incomplete. Existing commit refs remain the backup references; tags preserved.

| Repository | Branch | Current SHA | Merge status | Evidence | Backup reference | Proposed action | Risk |
|---|---|---|---|---|---|---|---|
| BE | `clean` | `f1db460670df6fc24036db29b20c3bd872c98bad` | UNRELATED_HISTORY | see repository audit; unresolved or protected | Existing branch at SHA | KEEP | High / protected |
| BE | `demo/2026-08-07-per-day-windows` | `76f3f0b32c5183de4642ea133b54bde97fae468f` | CONTENT_EQUIVALENT | 3 equivalent patch IDs + current code/tests | Existing branch at SHA | KEEP | Medium: deployment use unknown |
| BE | `docs/add-data-source-license-approval-gate` | `8fa583f31294b9503a5d423e6d86b1eb8afa7562` | FULLY_MERGED | ancestor of main | Existing branch at SHA | KEEP | Medium: deployment use unknown |
| BE | `docs/multi-source-poi-governance` | `0b3a93c2befffff43a170ea2640e0bc7cc9f6079` | FULLY_MERGED | ancestor of main | Existing branch at SHA | KEEP | Medium: deployment use unknown |
| BE | `docs/phase2-batch3-trip-preview-design` | `b6fc14f02d84ca17d1a1eb1dc5bad17ab0971632` | FULLY_MERGED | ancestor of main | Existing branch at SHA | KEEP | Medium: deployment use unknown |
| BE | `feat/complete-demo-web` | `039d289e8f2a13429510172d5ea5ae8d823f7bf0` | FULLY_MERGED | ancestor of main | Existing branch at SHA | KEEP | Medium: deployment use unknown |
| BE | `feat/phase6-p1-partner-readiness` | `d8bc55e0bbbb14c366a3d7cda19f77e22a7b5dd3` | FULLY_MERGED | ancestor of main | Existing branch at SHA | KEEP | Medium: deployment use unknown |
| BE | `feat/phase6-trust-freshness-resilience` | `00ea17bae1165c6f6c4d2bd01475a72b26f93cae` | FULLY_MERGED | ancestor of main | Existing branch at SHA | KEEP | Medium: deployment use unknown |
| BE | `feat/urban-agent-productization` | `ee2a18ce4f9408f5a864cabb7f25316fbd8d6b7a` | FULLY_MERGED | ancestor of main | Existing branch at SHA | KEEP | Medium: deployment use unknown |
| BE | `fix/external-place-resolution-trip-integration` | `772a40c243c85bebee3be4492731f15a7ff8949c` | FULLY_MERGED | ancestor of main | Existing branch at SHA | KEEP | Medium: deployment use unknown |
| BE | `fix/hf-auto-deploy-gate` | `f20a9cb3444f96955da0e0366cacc6015d45ddb2` | FULLY_MERGED | ancestor of main | Existing branch at SHA | KEEP | Medium: deployment use unknown |
| BE | `fix/hf-runtime-payload-admin-module` | `2021875a0a12138a113582eccd1011523bb2487f` | FULLY_MERGED | ancestor of main | Existing branch at SHA | KEEP | Medium: deployment use unknown |
| BE | `fix/osrm-response-body` | `505052a2f4793f8a8a7f79b2ad21c41595080005` | UNMERGED_VALUABLE | see repository audit; unresolved or protected | Existing branch at SHA | KEEP | High / protected |
| BE | `main` | `be6d21ae31dfc9c74f7e4b56e87a23e4df10342a` | PROTECTED | see repository audit; unresolved or protected | Existing branch at SHA | KEEP | High / protected |
| BE | `phase0/canonical-baseline` | `f4516167f773dfe5662bf3cbf13eef041510baab` | FULLY_MERGED | ancestor of main | Existing branch at SHA | KEEP | Medium: deployment use unknown |
| BE | `phase1/batch3-postgres-endpoint-smoke` | `2c34471747f6fd33d73130db2ab47df054d4f35c` | FULLY_MERGED | ancestor of main | Existing branch at SHA | KEEP | Medium: deployment use unknown |
| BE | `phase1/data-platform-foundation` | `2c34471747f6fd33d73130db2ab47df054d4f35c` | FULLY_MERGED | ancestor of main | Existing branch at SHA | KEEP | Medium: deployment use unknown |
| BE | `phase2/batch1-traveler-api-v2-foundation` | `9d5757ea526bd65fb38fca85a0c3e4f7ecfa95a5` | FULLY_MERGED | ancestor of main | Existing branch at SHA | KEEP | Medium: deployment use unknown |
| BE | `phase2/batch2-traveler-api-v2-recommendations` | `7718cd5c9e4d4d07a083f1d10aa9ad539035e14b` | FULLY_MERGED | ancestor of main | Existing branch at SHA | KEEP | Medium: deployment use unknown |
| BE | `phase2/batch3-traveler-api-v2-trip-preview` | `2c20b41865510cf3512788b60c8c464a48fe1140` | FULLY_MERGED | ancestor of main | Existing branch at SHA | KEEP | Medium: deployment use unknown |
| BE | `phase2/batch5-trip-persistence` | `6e9691898fad35f4b026774f347337bc88d91e20` | FULLY_MERGED | ancestor of main | Existing branch at SHA | KEEP | Medium: deployment use unknown |
| BE | `phase2/batch6-trip-lifecycle` | `62fdc0f27da51412a7c5ea0498e483542e83e333` | FULLY_MERGED | ancestor of main | Existing branch at SHA | KEEP | Medium: deployment use unknown |
| BE | `phase2/final-gates` | `075098166054b3d7cfa31bf4f1ce03bc83cfd077` | FULLY_MERGED | ancestor of main | Existing branch at SHA | KEEP | Medium: deployment use unknown |
| BE | `phase4/stage4a-source-discovery` | `d4c9e3aae43ccdb11e5e0d3b50e0931f45c6a417` | FULLY_MERGED | ancestor of main | Existing branch at SHA | KEEP | Medium: deployment use unknown |
| BE | `phase4/stage4b-matching-dry-run` | `39b1be496685ae2dc6b2ceced5d3a3e82dcdf83e` | FULLY_MERGED | ancestor of main | Existing branch at SHA | KEEP | Medium: deployment use unknown |
| BE | `phase4/stage4c-entity-resolution` | `2395637c55c72b2b1b898de82c6284e86b092e6b` | FULLY_MERGED | ancestor of main | Existing branch at SHA | KEEP | Medium: deployment use unknown |
| BE | `phase4/stage4d-citypack-build` | `1d60fe920fde5011ceaabc1847ae21a32f901516` | FULLY_MERGED | ancestor of main | Existing branch at SHA | KEEP | Medium: deployment use unknown |
| BE | `phase4/stage4e-real-source-validation` | `dff42be00703949f8154dc5ff0cbac8c1e099988` | FULLY_MERGED | ancestor of main | Existing branch at SHA | KEEP | Medium: deployment use unknown |
| BE | `phase4/stage4f-provenance-hardening` | `6569ff582c2b19987e49bf1402b5ab3fb236b9f5` | FULLY_MERGED | ancestor of main | Existing branch at SHA | KEEP | Medium: deployment use unknown |
| BE | `phase4/stage4g-full-danang-candidate` | `ebc58df7f522a81c252ec121c767d06cf45afec7` | FULLY_MERGED | ancestor of main | Existing branch at SHA | KEEP | Medium: deployment use unknown |
| BE | `phase4/stage4h-incremental-entity-resolution` | `573be34a37fd2636e869c68b8bad9dd1fd8823df` | FULLY_MERGED | ancestor of main | Existing branch at SHA | KEEP | Medium: deployment use unknown |
| BE | `phase4/stage4i-review-apply-preparation` | `2be867d4ff7495bc0146db89c140efa369d47322` | FULLY_MERGED | ancestor of main | Existing branch at SHA | KEEP | Medium: deployment use unknown |
| BE | `phase4/stage4j-evidence-approval-gate` | `654b946bd768e8ed6754ac2b8f638175fd18c1e3` | FULLY_MERGED | ancestor of main | Existing branch at SHA | KEEP | Medium: deployment use unknown |
| BE | `phase4/stage4k-controlled-canary-apply` | `17b22c1df0b66a07d640bcf4a11092a5ff7bee94` | FULLY_MERGED | ancestor of main | Existing branch at SHA | KEEP | Medium: deployment use unknown |
| BE | `phase4/stage4l-automated-review-policy` | `b5d601743dc744ed884a48ab61c94cf116c559b6` | FULLY_MERGED | ancestor of main | Existing branch at SHA | KEEP | Medium: deployment use unknown |
| BE | `phase4/stage4m-automated-safe-enrichment-pr` | `5429b9fb4533e71fa0b3dd6cbf0e62c506bef74f` | FULLY_MERGED | ancestor of main | Existing branch at SHA | KEEP | Medium: deployment use unknown |
| BE | `phase4/stage4n-scheduled-poi-sync` | `d34d53171d3af7f18db5a789675b88c4f3159f11` | FULLY_MERGED | ancestor of main | Existing branch at SHA | KEEP | Medium: deployment use unknown |
| BE | `phase4/stage4o-schedule-observability` | `e7a10a5dc8639c2d5e5b904459df8092682676da` | FULLY_MERGED | ancestor of main | Existing branch at SHA | KEEP | Medium: deployment use unknown |
| BE | `phase4/stage4p-create-new-policy-research` | `b0351c3f6a287c71cc36111ade36ab930be63a70` | FULLY_MERGED | ancestor of main | Existing branch at SHA | KEEP | Medium: deployment use unknown |
| BE | `phase4/stage4q-create-new-evidence-hardening` | `1a47589741da5d1e1b9cd078697d39a5d011fd1c` | FULLY_MERGED | ancestor of main | Existing branch at SHA | KEEP | Medium: deployment use unknown |
| BE | `phase4/stage4r-create-new-canary-review` | `fe7bb4626cfa37b3ba5f48327ef75aab1a87b19e` | FULLY_MERGED | ancestor of main | Existing branch at SHA | KEEP | Medium: deployment use unknown |
| BE | `phase4/stage4s-controlled-create-new-canary` | `f9703c7164ad1b2af4c449e37e954a2ef2167030` | FULLY_MERGED | ancestor of main | Existing branch at SHA | KEEP | Medium: deployment use unknown |
| BE | `phase4/stage4t-postmerge-closeout` | `56a6040c58c6d4c72d79d7734ea9ad6d52282c51` | FULLY_MERGED | ancestor of main | Existing branch at SHA | KEEP | Medium: deployment use unknown |
| BE | `phase5/admin-backend-foundation` | `8d92e9dd75fdf424fbedbe4b921e5c66e8f9b3f8` | FULLY_MERGED | ancestor of main | Existing branch at SHA | KEEP | Medium: deployment use unknown |
| BE | `phase5/stage5a-product-e2e` | `7094eb0e18ec52a2af97915eab5b9ad69be1a7ba` | FULLY_MERGED | ancestor of main | Existing branch at SHA | KEEP | Medium: deployment use unknown |
| BE | `phase5/stage5b-runtime-data-packaging` | `1b12844f7a7d2b45ff5570d3f47a3e1f97dc38ae` | FULLY_MERGED | ancestor of main | Existing branch at SHA | KEEP | Medium: deployment use unknown |
| BE | `phase5/stage5c-staging-deployment` | `ea1c6b8c6644af7ff74836b04ab9b46acfa76aea` | FULLY_MERGED | ancestor of main | Existing branch at SHA | KEEP | Medium: deployment use unknown |
| BE | `phase5/stage5d1-geocoding` | `90b9f4d2d7cc77ad291f2a97582560a2e10f72d4` | FULLY_MERGED | ancestor of main | Existing branch at SHA | KEEP | Medium: deployment use unknown |
| BE | `release/integrated-demo-2026-08` | `580b781e4c017499fb108a425a7c9547a33cc174` | FULLY_MERGED | ancestor of main | Existing branch at SHA | KEEP | Medium: deployment use unknown |

Per-branch changed-file lists, raw unique commits, patch equivalence and tags
at each tip are recorded in `NAVIGO_BRANCH_EVIDENCE.json`. This is public Git
evidence only; deployment bindings remain NOT VERIFIED and every action is KEEP.

Final remote state: original FE 21 / BE 49 branches retained. Both publication attempts were denied before any ref mutation. No remote branch deleted; SAFE_TO_DELETE remains zero. New integration branches exist only locally.
