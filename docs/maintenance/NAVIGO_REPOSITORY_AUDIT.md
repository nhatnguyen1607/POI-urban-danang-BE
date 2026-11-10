# NaviGo repository audit — BE

Audit date: 2026-11-10 (Asia/Saigon). Scope: the two explicitly named repositories only.

- Frontend: https://github.com/nhatnguyen1607/POI-urban-danang-FE (main 3914e1c7b335deccb3d7c2c8162cac2dacb9864e).
- Backend: https://github.com/nhatnguyen1607/POI-urban-danang-BE (main be6d21ae31dfc9c74f7e4b56e87a23e4df10342a).
- Default branch: main in both; connector reports pull/push true, admin/maintain false.
- Fresh full Git clones with remote branches and tags; integration branch:
  `integration/navigo-audit-2026-11-10`. Main is never checked out for edits.
- Original workspace is unrelated `Vuz14/CD1_finance`, main with existing research
  changes/untracked files. Those files were not changed or committed.
- All branch and PR pages have fewer than 100 records: FE 21 branches / 20 PRs;
  BE 49 branches / 43 PRs. All 63 historical PRs merged; zero open or closed-unmerged.
- Branch protected flags false; main retained as PROTECTED by maintenance policy.
  Repository rulesets read returned empty. Administrative protection details unavailable.
- Tags: integrated-demo-2026-08, phase-1-batch-1-2, phase-1-batch-3, phase-2-batch-1, phase-2-batch-2, phase-2-batch-3, phase0/canonical-baseline.
- GitHub releases: none on either repository. FE Vercel main status success;
  BE main validate-and-deploy check success. These are historical checks, not new PR checks.

## Branch inventory

Ahead/behind is relative to the fetched main. Unique commits are raw ancestry
counts, not proof of missing content. Patch IDs and file-level differences were
checked separately. Full branch tips, merge-base and commit evidence follow.

| Repository | Branch | SHA | Ahead/Behind | Unique commits | PR status | Classification | Action |
|---|---|---|---|---|---|---|---|
| BE | `clean` | `f1db460670df6fc24036db29b20c3bd872c98bad` | 3/171 | 3 | none | UNRELATED_HISTORY | Keep; provenance unresolved |
| BE | `demo/2026-08-07-per-day-windows` | `76f3f0b32c5183de4642ea133b54bde97fae468f` | 3/89 | 3 | none | CONTENT_EQUIVALENT | Keep; all three patch IDs already applied |
| BE | `docs/add-data-source-license-approval-gate` | `8fa583f31294b9503a5d423e6d86b1eb8afa7562` | 0/101 | 0 | #7 merged | FULLY_MERGED | Keep; deployment use not fully verified |
| BE | `docs/multi-source-poi-governance` | `0b3a93c2befffff43a170ea2640e0bc7cc9f6079` | 0/103 | 0 | #5 merged | FULLY_MERGED | Keep; deployment use not fully verified |
| BE | `docs/phase2-batch3-trip-preview-design` | `b6fc14f02d84ca17d1a1eb1dc5bad17ab0971632` | 0/99 | 0 | #8 merged | FULLY_MERGED | Keep; deployment use not fully verified |
| BE | `feat/complete-demo-web` | `039d289e8f2a13429510172d5ea5ae8d823f7bf0` | 0/9 | 0 | #40 merged | FULLY_MERGED | Keep; deployment use not fully verified |
| BE | `feat/phase6-p1-partner-readiness` | `d8bc55e0bbbb14c366a3d7cda19f77e22a7b5dd3` | 0/1 | 0 | #43 merged | FULLY_MERGED | Keep; deployment use not fully verified |
| BE | `feat/phase6-trust-freshness-resilience` | `00ea17bae1165c6f6c4d2bd01475a72b26f93cae` | 0/3 | 0 | #42 merged | FULLY_MERGED | Keep; deployment use not fully verified |
| BE | `feat/urban-agent-productization` | `ee2a18ce4f9408f5a864cabb7f25316fbd8d6b7a` | 0/84 | 0 | none | FULLY_MERGED | Keep; deployment use not fully verified |
| BE | `fix/external-place-resolution-trip-integration` | `772a40c243c85bebee3be4492731f15a7ff8949c` | 0/13 | 0 | #39 merged | FULLY_MERGED | Keep; deployment use not fully verified |
| BE | `fix/hf-auto-deploy-gate` | `f20a9cb3444f96955da0e0366cacc6015d45ddb2` | 0/17 | 0 | #38 merged | FULLY_MERGED | Keep; deployment use not fully verified |
| BE | `fix/hf-runtime-payload-admin-module` | `2021875a0a12138a113582eccd1011523bb2487f` | 0/7 | 0 | #41 merged | FULLY_MERGED | Keep; deployment use not fully verified |
| BE | `fix/osrm-response-body` | `505052a2f4793f8a8a7f79b2ad21c41595080005` | 2/0 | 2 | none | UNMERGED_VALUABLE | Integrate reviewed runtime/test paths only |
| BE | `main` | `be6d21ae31dfc9c74f7e4b56e87a23e4df10342a` | 0/0 | 0 | none | PROTECTED | Keep; deployment use not fully verified |
| BE | `phase0/canonical-baseline` | `f4516167f773dfe5662bf3cbf13eef041510baab` | 0/118 | 0 | #1 merged | FULLY_MERGED | Keep; deployment use not fully verified |
| BE | `phase1/batch3-postgres-endpoint-smoke` | `2c34471747f6fd33d73130db2ab47df054d4f35c` | 0/111 | 0 | #3 merged | FULLY_MERGED | Keep; deployment use not fully verified |
| BE | `phase1/data-platform-foundation` | `2c34471747f6fd33d73130db2ab47df054d4f35c` | 0/111 | 0 | #2 merged | FULLY_MERGED | Keep; deployment use not fully verified |
| BE | `phase2/batch1-traveler-api-v2-foundation` | `9d5757ea526bd65fb38fca85a0c3e4f7ecfa95a5` | 0/108 | 0 | #4 merged | FULLY_MERGED | Keep; deployment use not fully verified |
| BE | `phase2/batch2-traveler-api-v2-recommendations` | `7718cd5c9e4d4d07a083f1d10aa9ad539035e14b` | 0/106 | 0 | #6 merged | FULLY_MERGED | Keep; deployment use not fully verified |
| BE | `phase2/batch3-traveler-api-v2-trip-preview` | `2c20b41865510cf3512788b60c8c464a48fe1140` | 0/90 | 0 | #9 merged | FULLY_MERGED | Keep; deployment use not fully verified |
| BE | `phase2/batch5-trip-persistence` | `6e9691898fad35f4b026774f347337bc88d91e20` | 0/83 | 0 | #11 merged | FULLY_MERGED | Keep; deployment use not fully verified |
| BE | `phase2/batch6-trip-lifecycle` | `62fdc0f27da51412a7c5ea0498e483542e83e333` | 0/81 | 0 | #12 merged | FULLY_MERGED | Keep; deployment use not fully verified |
| BE | `phase2/final-gates` | `075098166054b3d7cfa31bf4f1ce03bc83cfd077` | 0/78 | 0 | #13 merged | FULLY_MERGED | Keep; deployment use not fully verified |
| BE | `phase4/stage4a-source-discovery` | `d4c9e3aae43ccdb11e5e0d3b50e0931f45c6a417` | 0/76 | 0 | #14 merged | FULLY_MERGED | Keep; deployment use not fully verified |
| BE | `phase4/stage4b-matching-dry-run` | `39b1be496685ae2dc6b2ceced5d3a3e82dcdf83e` | 0/74 | 0 | #15 merged | FULLY_MERGED | Keep; deployment use not fully verified |
| BE | `phase4/stage4c-entity-resolution` | `2395637c55c72b2b1b898de82c6284e86b092e6b` | 0/70 | 0 | #16 merged | FULLY_MERGED | Keep; deployment use not fully verified |
| BE | `phase4/stage4d-citypack-build` | `1d60fe920fde5011ceaabc1847ae21a32f901516` | 0/67 | 0 | #17 merged | FULLY_MERGED | Keep; deployment use not fully verified |
| BE | `phase4/stage4e-real-source-validation` | `dff42be00703949f8154dc5ff0cbac8c1e099988` | 0/65 | 0 | #18 merged | FULLY_MERGED | Keep; deployment use not fully verified |
| BE | `phase4/stage4f-provenance-hardening` | `6569ff582c2b19987e49bf1402b5ab3fb236b9f5` | 0/63 | 0 | #19 merged | FULLY_MERGED | Keep; deployment use not fully verified |
| BE | `phase4/stage4g-full-danang-candidate` | `ebc58df7f522a81c252ec121c767d06cf45afec7` | 0/61 | 0 | #20 merged | FULLY_MERGED | Keep; deployment use not fully verified |
| BE | `phase4/stage4h-incremental-entity-resolution` | `573be34a37fd2636e869c68b8bad9dd1fd8823df` | 0/58 | 0 | #21 merged | FULLY_MERGED | Keep; deployment use not fully verified |
| BE | `phase4/stage4i-review-apply-preparation` | `2be867d4ff7495bc0146db89c140efa369d47322` | 0/56 | 0 | #22 merged | FULLY_MERGED | Keep; deployment use not fully verified |
| BE | `phase4/stage4j-evidence-approval-gate` | `654b946bd768e8ed6754ac2b8f638175fd18c1e3` | 0/54 | 0 | #23 merged | FULLY_MERGED | Keep; deployment use not fully verified |
| BE | `phase4/stage4k-controlled-canary-apply` | `17b22c1df0b66a07d640bcf4a11092a5ff7bee94` | 0/52 | 0 | #24 merged | FULLY_MERGED | Keep; deployment use not fully verified |
| BE | `phase4/stage4l-automated-review-policy` | `b5d601743dc744ed884a48ab61c94cf116c559b6` | 0/50 | 0 | #25 merged | FULLY_MERGED | Keep; deployment use not fully verified |
| BE | `phase4/stage4m-automated-safe-enrichment-pr` | `5429b9fb4533e71fa0b3dd6cbf0e62c506bef74f` | 0/48 | 0 | #26 merged | FULLY_MERGED | Keep; deployment use not fully verified |
| BE | `phase4/stage4n-scheduled-poi-sync` | `d34d53171d3af7f18db5a789675b88c4f3159f11` | 0/46 | 0 | #27 merged | FULLY_MERGED | Keep; deployment use not fully verified |
| BE | `phase4/stage4o-schedule-observability` | `e7a10a5dc8639c2d5e5b904459df8092682676da` | 0/41 | 0 | #28 merged | FULLY_MERGED | Keep; deployment use not fully verified |
| BE | `phase4/stage4p-create-new-policy-research` | `b0351c3f6a287c71cc36111ade36ab930be63a70` | 0/39 | 0 | #29 merged | FULLY_MERGED | Keep; deployment use not fully verified |
| BE | `phase4/stage4q-create-new-evidence-hardening` | `1a47589741da5d1e1b9cd078697d39a5d011fd1c` | 0/37 | 0 | #30 merged | FULLY_MERGED | Keep; deployment use not fully verified |
| BE | `phase4/stage4r-create-new-canary-review` | `fe7bb4626cfa37b3ba5f48327ef75aab1a87b19e` | 0/35 | 0 | #31 merged | FULLY_MERGED | Keep; deployment use not fully verified |
| BE | `phase4/stage4s-controlled-create-new-canary` | `f9703c7164ad1b2af4c449e37e954a2ef2167030` | 0/33 | 0 | #32 merged | FULLY_MERGED | Keep; deployment use not fully verified |
| BE | `phase4/stage4t-postmerge-closeout` | `56a6040c58c6d4c72d79d7734ea9ad6d52282c51` | 0/31 | 0 | #33 merged | FULLY_MERGED | Keep; deployment use not fully verified |
| BE | `phase5/admin-backend-foundation` | `8d92e9dd75fdf424fbedbe4b921e5c66e8f9b3f8` | 0/12 | 0 | none | FULLY_MERGED | Keep; deployment use not fully verified |
| BE | `phase5/stage5a-product-e2e` | `7094eb0e18ec52a2af97915eab5b9ad69be1a7ba` | 0/29 | 0 | #34 merged | FULLY_MERGED | Keep; deployment use not fully verified |
| BE | `phase5/stage5b-runtime-data-packaging` | `1b12844f7a7d2b45ff5570d3f47a3e1f97dc38ae` | 0/26 | 0 | #35 merged | FULLY_MERGED | Keep; deployment use not fully verified |
| BE | `phase5/stage5c-staging-deployment` | `ea1c6b8c6644af7ff74836b04ab9b46acfa76aea` | 0/23 | 0 | #36 merged | FULLY_MERGED | Keep; deployment use not fully verified |
| BE | `phase5/stage5d1-geocoding` | `90b9f4d2d7cc77ad291f2a97582560a2e10f72d4` | 0/19 | 0 | #37 merged | FULLY_MERGED | Keep; deployment use not fully verified |
| BE | `release/integrated-demo-2026-08` | `580b781e4c017499fb108a425a7c9547a33cc174` | 0/85 | 0 | #10 merged | FULLY_MERGED | Keep; deployment use not fully verified |

## Unique-commit evidence and content decisions

### clean

Merge base: `NONE`.

- `f1db460670df6fc24036db29b20c3bd872c98bad Update EDA endpoint to return all items instead of a sample of 50`
- `8ef6c9e7ce1a68e3a94d8f2b5ecadf671af6b2f0 Refactor fuzzy logic evaluation and improve logging; remove unused files and update requirements`
- `bac756373f670aaaf6dec69a531b7ccf70a223c6 Clean repo with code, reports and v4 models`

Patch comparison (`-` means equivalent patch already on main):

- `+ bac756373f670aaaf6dec69a531b7ccf70a223c6`
- `+ 8ef6c9e7ce1a68e3a94d8f2b5ecadf671af6b2f0`
- `+ f1db460670df6fc24036db29b20c3bd872c98bad`

No common ancestor. Contains old inference/encoder/model report paths. No automatic integration or deletion; main-vs-clean file diff does not establish historical provenance.

### demo/2026-08-07-per-day-windows

Merge base: `e5f2f975b009d561dfef037932068836bde24918`.

- `76f3f0b32c5183de4642ea133b54bde97fae468f docs(demo): document per-day window extension`
- `a3e9a093f4d550ab5f6b485b4f7f1219be614fe8 test(demo): validate per-day trip windows`
- `447ac0f967cf8f196df8d91671ff7165021a886d feat(demo): support per-day trip windows`

Patch comparison (`-` means equivalent patch already on main):

- `- 447ac0f967cf8f196df8d91671ff7165021a886d`
- `- a3e9a093f4d550ab5f6b485b4f7f1219be614fe8`
- `- 76f3f0b32c5183de4642ea133b54bde97fae468f`

All three patches are equivalent on main, and current tripPreviewValidation/tripPreview plus batch3 tests retain aliases and dayWindows. Newer temporary-place and date validation must not be overwritten.

### fix/osrm-response-body

Merge base: `be6d21ae31dfc9c74f7e4b56e87a23e4df10342a`.

- `505052a2f4793f8a8a7f79b2ad21c41595080005 docs(route): record pre-merge integration and HF provenance`
- `1d35f26c6fae51b5e9e5c507a12d7034772edc90 fix(route): cache parsed OSRM routes and classify provider failures`

Patch comparison (`-` means equivalent patch already on main):

- `+ 1d35f26c6fae51b5e9e5c507a12d7034772edc90`
- `+ 505052a2f4793f8a8a7f79b2ad21c41595080005`

Nine changed paths: four runtime JS files, one regression test, deployment workflow, and three historical rebuild records. Restore only the four runtime files and test. Do not import historical append blocks or alter deploy workflow. Source implementation commit: 1d35f26c6fae51b5e9e5c507a12d7034772edc90.

## Complete PR history

| PR | Head branch | Head SHA | State | Merge commit |
|---|---|---|---|---|
| [#1](https://github.com/nhatnguyen1607/POI-urban-danang-BE/pull/1) | `phase0/canonical-baseline` | `f4516167f773dfe5662bf3cbf13eef041510baab` | MERGED | `d36e78962c65289d0b608d693280ec2ab05c3d76` |
| [#2](https://github.com/nhatnguyen1607/POI-urban-danang-BE/pull/2) | `phase1/data-platform-foundation` | `b45f4facc7f865607954d98dc517022808490128` | MERGED | `fa078052a65c0b87c3c15f749822d340b8bc1452` |
| [#3](https://github.com/nhatnguyen1607/POI-urban-danang-BE/pull/3) | `phase1/batch3-postgres-endpoint-smoke` | `2c34471747f6fd33d73130db2ab47df054d4f35c` | MERGED | `ec06a905e2f61f9f4c90dc7857bf31cf2bb78fb6` |
| [#4](https://github.com/nhatnguyen1607/POI-urban-danang-BE/pull/4) | `phase2/batch1-traveler-api-v2-foundation` | `9d5757ea526bd65fb38fca85a0c3e4f7ecfa95a5` | MERGED | `ae2b91ecb4adedf36f6980f568f05c35109bf3bb` |
| [#5](https://github.com/nhatnguyen1607/POI-urban-danang-BE/pull/5) | `docs/multi-source-poi-governance` | `0b3a93c2befffff43a170ea2640e0bc7cc9f6079` | MERGED | `75388afb4008967b8da7585b0c9db0f70e69e369` |
| [#6](https://github.com/nhatnguyen1607/POI-urban-danang-BE/pull/6) | `phase2/batch2-traveler-api-v2-recommendations` | `7718cd5c9e4d4d07a083f1d10aa9ad539035e14b` | MERGED | `707cce556cf37986d9bd78fdf25902d76850242c` |
| [#7](https://github.com/nhatnguyen1607/POI-urban-danang-BE/pull/7) | `docs/add-data-source-license-approval-gate` | `8fa583f31294b9503a5d423e6d86b1eb8afa7562` | MERGED | `35e867a3dd8f4e9dbe27705fa9a02c7f66ea901f` |
| [#8](https://github.com/nhatnguyen1607/POI-urban-danang-BE/pull/8) | `docs/phase2-batch3-trip-preview-design` | `b6fc14f02d84ca17d1a1eb1dc5bad17ab0971632` | MERGED | `03bcc2ba90be7aa2618b5381f763ac1933469deb` |
| [#9](https://github.com/nhatnguyen1607/POI-urban-danang-BE/pull/9) | `phase2/batch3-traveler-api-v2-trip-preview` | `2c20b41865510cf3512788b60c8c464a48fe1140` | MERGED | `e5f2f975b009d561dfef037932068836bde24918` |
| [#10](https://github.com/nhatnguyen1607/POI-urban-danang-BE/pull/10) | `release/integrated-demo-2026-08` | `580b781e4c017499fb108a425a7c9547a33cc174` | MERGED | `ee2a18ce4f9408f5a864cabb7f25316fbd8d6b7a` |
| [#11](https://github.com/nhatnguyen1607/POI-urban-danang-BE/pull/11) | `phase2/batch5-trip-persistence` | `6e9691898fad35f4b026774f347337bc88d91e20` | MERGED | `eefd05420f07b9392b8ae193261cf44ccee42491` |
| [#12](https://github.com/nhatnguyen1607/POI-urban-danang-BE/pull/12) | `phase2/batch6-trip-lifecycle` | `62fdc0f27da51412a7c5ea0498e483542e83e333` | MERGED | `1d75ec7ae32eae17697895dd94523017ac094d32` |
| [#13](https://github.com/nhatnguyen1607/POI-urban-danang-BE/pull/13) | `phase2/final-gates` | `075098166054b3d7cfa31bf4f1ce03bc83cfd077` | MERGED | `6ce375c69e8db2a4fec671b40303623f8c13b07d` |
| [#14](https://github.com/nhatnguyen1607/POI-urban-danang-BE/pull/14) | `phase4/stage4a-source-discovery` | `d4c9e3aae43ccdb11e5e0d3b50e0931f45c6a417` | MERGED | `519f11b1c531d12f6801f120c251ddb3489e4c96` |
| [#15](https://github.com/nhatnguyen1607/POI-urban-danang-BE/pull/15) | `phase4/stage4b-matching-dry-run` | `39b1be496685ae2dc6b2ceced5d3a3e82dcdf83e` | MERGED | `b1a73bc1e555c0c0a73c4d247f5c1adf7396389a` |
| [#16](https://github.com/nhatnguyen1607/POI-urban-danang-BE/pull/16) | `phase4/stage4c-entity-resolution` | `2395637c55c72b2b1b898de82c6284e86b092e6b` | MERGED | `b899b6f7c170ffc37fef3ac8ac7780eb65e01761` |
| [#17](https://github.com/nhatnguyen1607/POI-urban-danang-BE/pull/17) | `phase4/stage4d-citypack-build` | `1d60fe920fde5011ceaabc1847ae21a32f901516` | MERGED | `30bd6d48f3e490f3be7ab43ce2b3724a0003dca1` |
| [#18](https://github.com/nhatnguyen1607/POI-urban-danang-BE/pull/18) | `phase4/stage4e-real-source-validation` | `dff42be00703949f8154dc5ff0cbac8c1e099988` | MERGED | `ad30d747072e03bd1b4f4481f239a717b80d36e2` |
| [#19](https://github.com/nhatnguyen1607/POI-urban-danang-BE/pull/19) | `phase4/stage4f-provenance-hardening` | `6569ff582c2b19987e49bf1402b5ab3fb236b9f5` | MERGED | `526af3505f98cba5f44dde46e48090e001784fd5` |
| [#20](https://github.com/nhatnguyen1607/POI-urban-danang-BE/pull/20) | `phase4/stage4g-full-danang-candidate` | `ebc58df7f522a81c252ec121c767d06cf45afec7` | MERGED | `fa37afa291a3c85ef1cb55493f58916cbdfd4bc1` |
| [#21](https://github.com/nhatnguyen1607/POI-urban-danang-BE/pull/21) | `phase4/stage4h-incremental-entity-resolution` | `573be34a37fd2636e869c68b8bad9dd1fd8823df` | MERGED | `1a2a409aa0d32b0a0dffb8c27db2cbcef0e2a4a7` |
| [#22](https://github.com/nhatnguyen1607/POI-urban-danang-BE/pull/22) | `phase4/stage4i-review-apply-preparation` | `2be867d4ff7495bc0146db89c140efa369d47322` | MERGED | `7c6f0b6f5c3b79f6ddc23aa35db46b89e6fc9a3c` |
| [#23](https://github.com/nhatnguyen1607/POI-urban-danang-BE/pull/23) | `phase4/stage4j-evidence-approval-gate` | `654b946bd768e8ed6754ac2b8f638175fd18c1e3` | MERGED | `fe088ce9c665488b7ad832474eb9f4043ee40b09` |
| [#24](https://github.com/nhatnguyen1607/POI-urban-danang-BE/pull/24) | `phase4/stage4k-controlled-canary-apply` | `17b22c1df0b66a07d640bcf4a11092a5ff7bee94` | MERGED | `75c319425bb6603f717db312fc86157e733c9653` |
| [#25](https://github.com/nhatnguyen1607/POI-urban-danang-BE/pull/25) | `phase4/stage4l-automated-review-policy` | `b5d601743dc744ed884a48ab61c94cf116c559b6` | MERGED | `112a0110f0570a342dd13e129942c6f67464c283` |
| [#26](https://github.com/nhatnguyen1607/POI-urban-danang-BE/pull/26) | `phase4/stage4m-automated-safe-enrichment-pr` | `5429b9fb4533e71fa0b3dd6cbf0e62c506bef74f` | MERGED | `9158b4f2d975393620c466bff4464a1c4b534ca6` |
| [#27](https://github.com/nhatnguyen1607/POI-urban-danang-BE/pull/27) | `phase4/stage4n-scheduled-poi-sync` | `d34d53171d3af7f18db5a789675b88c4f3159f11` | MERGED | `85c64dfd03a26bf389b1a80a77075e7d0fd0054d` |
| [#28](https://github.com/nhatnguyen1607/POI-urban-danang-BE/pull/28) | `phase4/stage4o-schedule-observability` | `e7a10a5dc8639c2d5e5b904459df8092682676da` | MERGED | `58d4bc3f38cb72548e4041c36c2cbbcd66049cb6` |
| [#29](https://github.com/nhatnguyen1607/POI-urban-danang-BE/pull/29) | `phase4/stage4p-create-new-policy-research` | `b0351c3f6a287c71cc36111ade36ab930be63a70` | MERGED | `c4be0c69473a0a388a3bf60962c28e71ace913b3` |
| [#30](https://github.com/nhatnguyen1607/POI-urban-danang-BE/pull/30) | `phase4/stage4q-create-new-evidence-hardening` | `1a47589741da5d1e1b9cd078697d39a5d011fd1c` | MERGED | `528a9b16b826937ae9486c0f2cdca0542c5d9a0f` |
| [#31](https://github.com/nhatnguyen1607/POI-urban-danang-BE/pull/31) | `phase4/stage4r-create-new-canary-review` | `fe7bb4626cfa37b3ba5f48327ef75aab1a87b19e` | MERGED | `f16d45b475f06a074027d2540a2997e3548cd362` |
| [#32](https://github.com/nhatnguyen1607/POI-urban-danang-BE/pull/32) | `phase4/stage4s-controlled-create-new-canary` | `f9703c7164ad1b2af4c449e37e954a2ef2167030` | MERGED | `1b7054b85a4c0fe82d82c5ef5a3317d2695246ed` |
| [#33](https://github.com/nhatnguyen1607/POI-urban-danang-BE/pull/33) | `phase4/stage4t-postmerge-closeout` | `56a6040c58c6d4c72d79d7734ea9ad6d52282c51` | MERGED | `6eb4ed98a1c4b1f31b43f13f12e21bf9a160699d` |
| [#34](https://github.com/nhatnguyen1607/POI-urban-danang-BE/pull/34) | `phase5/stage5a-product-e2e` | `7094eb0e18ec52a2af97915eab5b9ad69be1a7ba` | MERGED | `5da2ee1172e75f1011aa9aa70f8b8bed93bf8ffb` |
| [#35](https://github.com/nhatnguyen1607/POI-urban-danang-BE/pull/35) | `phase5/stage5b-runtime-data-packaging` | `1b12844f7a7d2b45ff5570d3f47a3e1f97dc38ae` | MERGED | `bcbb4c0ab1cac07040393dd57679a702b63bca2c` |
| [#36](https://github.com/nhatnguyen1607/POI-urban-danang-BE/pull/36) | `phase5/stage5c-staging-deployment` | `ea1c6b8c6644af7ff74836b04ab9b46acfa76aea` | MERGED | `b6b125b8f9ef0b1fc8e2ae30c009d1167446e1cf` |
| [#37](https://github.com/nhatnguyen1607/POI-urban-danang-BE/pull/37) | `phase5/stage5d1-geocoding` | `90b9f4d2d7cc77ad291f2a97582560a2e10f72d4` | MERGED | `a4e8b02144f15bbd72b0e31075aa5a2fdc7aac40` |
| [#38](https://github.com/nhatnguyen1607/POI-urban-danang-BE/pull/38) | `fix/hf-auto-deploy-gate` | `f20a9cb3444f96955da0e0366cacc6015d45ddb2` | MERGED | `33609f6b95fc3a05be42c4505425d6881dea4a73` |
| [#39](https://github.com/nhatnguyen1607/POI-urban-danang-BE/pull/39) | `fix/external-place-resolution-trip-integration` | `40b0b4ea8cbddc1bf88888e17c54a0ee671e074c` | MERGED | `638b4a10f3804b40ac0387596fa981c3e69cb5b6` |
| [#40](https://github.com/nhatnguyen1607/POI-urban-danang-BE/pull/40) | `feat/complete-demo-web` | `039d289e8f2a13429510172d5ea5ae8d823f7bf0` | MERGED | `03e894e39c21b7dc8c6bb8c7173b797bc96b3811` |
| [#41](https://github.com/nhatnguyen1607/POI-urban-danang-BE/pull/41) | `fix/hf-runtime-payload-admin-module` | `2021875a0a12138a113582eccd1011523bb2487f` | MERGED | `5ee86c9f46fe2cbfa08eb8108796971624a00835` |
| [#42](https://github.com/nhatnguyen1607/POI-urban-danang-BE/pull/42) | `feat/phase6-trust-freshness-resilience` | `00ea17bae1165c6f6c4d2bd01475a72b26f93cae` | MERGED | `9c7b21cf695bdf88769b9ce6b9946404a56f7025` |
| [#43](https://github.com/nhatnguyen1607/POI-urban-danang-BE/pull/43) | `feat/phase6-p1-partner-readiness` | `d8bc55e0bbbb14c366a3d7cda19f77e22a7b5dd3` | MERGED | `be6d21ae31dfc9c74f7e4b56e87a23e4df10342a` |


## Remote publication blocked — verified connector response

Final verdict: **BLOCKED_BY_PERMISSIONS**.

- FE GitHub Git Trees create call returned HTTP 403: Resource not accessible by integration.
- BE GitHub Git Blobs create call returned HTTP 403: Resource not accessible by integration.
- Repository metadata advertised push=true, but the actual object-write operations
  are denied. No inference is made that repository-owner permission is missing;
  the connected integration's effective write capability is insufficient.
- No remote branch or commit was created; no push, PR creation, merge, workflow
  dispatch, production deployment or deletion occurred. No PR URL exists.
- Two local integration branches contain the reviewable changes. Patch files and
  prepared PR descriptions are available in the enclosing navigo-audit directory.
- FE main remains 3914e1c7b335deccb3d7c2c8162cac2dacb9864e;
  BE main remains be6d21ae31dfc9c74f7e4b56e87a23e4df10342a.
- To publish, restore effective Contents write and Pull requests write access for
  the GitHub integration to both named repositories. Do not bypass repository
  protection or use force push. Recheck current main before publishing.
- Even after access is restored, main merge would trigger deployment. This task
  prohibits automatic production deploy; no automatic merge is proposed.
