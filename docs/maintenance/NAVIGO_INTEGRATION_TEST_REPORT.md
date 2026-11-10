# NaviGo integration verification

| Check | Result | Scope / limitation |
|---|---|---|
| Locked FE / BE / Functions install | PASS | npm ci --ignore-scripts; Functions expects Node22, host Node24.11.1 |
| FE npm run build | PASS | tsc -b + Vite; chunk-size warning remains |
| FE phase6 + reviews tests | PASS: 19 / 19 | Node tests; no browser E2E |
| FE npm run lint baseline | FAIL: 98 errors, 2 warnings | Existing hooks/any/refresh debt; no rules disabled |
| BE npm test baseline | PASS: 106 / 107, 1 SKIP | Disposable PostGIS not configured |
| BE npm test after OSRM | PASS: 128 / 129, 1 SKIP | 22 OSRM tests added; final matrix run recorded below |
| BE OSRM regression | PASS: 22 / 22 | Real Response streams, mocked network; local HTTP endpoint, test auth fallback |
| BE Phase4 initial | FAIL: 252 PASS, 6 FAIL, 14 SKIP / 272 | Six hardcoded current count expectations stale after approved Stage4T promotion |
| BE Phase4 final | PASS in final combined run: 258 PASS / 272, 14 SKIP | No tests disabled; historical fixtures and canonical bytes preserved |
| Firebase Functions build | PASS | npm run build in functions; no deploy or emulator E2E |
| Runtime dataset verifier | PASS | 4173 POIs; 7220189 bytes; SHA dcb404cc8b5c7a9b5fd70df63039ab8f828c504270e22b12a671fa4ed61583f4 |
| Real Firebase auth / feedback write | NOT VERIFIED | No production credentials or data writes used |
| Real PostGIS runtime | NOT VERIFIED | Guarded disposable integration skipped |
| Live OSRM / weather / official providers | NOT VERIFIED | Mock regression is distinguished from live integration |
| Real multimodal checkpoint / GPU inference | NOT VERIFIED | LFS pointers and absent configured artifact bundle; no model downloads |
| Browser responsive / full FE-to-BE user journey | NOT VERIFIED | Build and API tests do not establish browser interaction |

Cross-repository flow evidence: FE sends v2 recommendation and preview requests,
BE detects intents, retrieves canonical POIs, applies JSON reranker/memory and
schedules stops; FE renders stop reasons, warnings, maps and handoffs. Local
HTTP contract tests cover these backend paths. Save/replan ownership is tested
with test auth. Feedback/Firestore and business UI require a real configured
browser/Firebase environment for complete E2E. Those flows remain NOT VERIFIED.

Restricted sandbox API tests initially failed to start child servers (exit 7).
The same baseline passed when rerun with localhost/child-process permissions;
this is recorded as an environment restriction, not a product regression.
Phase4 tests rewrite twelve tracked generated fixture outputs (including timestamp and line-ending rewrites); only these
outputs were restored from the pristine clone HEAD after testing. Canonical
CSV, weights, secrets and original workspace modifications were preserved.


## Final verification on the integration source

- Backend all suites (phase0, phase1, phase2, phase4, phase5, phase6):
  **403 total, 388 PASS, 0 FAIL, 15 SKIP**, 53.30 seconds on Node 24.11.1.
- The final set includes 22 OSRM and 2 route-matrix regressions.
- Skips: 1 disposable PostGIS integration plus 14 pre-existing guarded City Pack
  historical/artifact integration cases. No test was newly skipped or disabled.
- Exact command: `node --test tests/phase0/*.test.js tests/phase1/*.test.js tests/phase2/*.test.js tests/phase4/*.test.js tests/phase5/*.test.js tests/phase6/*.test.js`.
- Phase4 originally failed 6 current-count assertions. On the first repair rerun,
  one assertion still compared a historical Stage4G hash with the new live hash.
  It now asserts the original 4166 hash explicitly while live canonical tests
  retain the current 4173 hash. Final full suite passes all runnable tests.
- Matrix: planner `walk` now uses walking speed, legacy `walking` retained;
  zero longitude no longer falls through to a different coordinate alias.
- `git diff --check` for source/tests/docs passed. Canonical dataset verifier
  passed again after all tests; generated snapshot outputs restored separately.
- FE main source was not changed; build PASS, 19 unit tests PASS, lint baseline
  still FAIL (98 errors, 2 warnings). No lint suppression or test weakening.
- No real browser/Firebase/model/live-provider result is claimed from mocks.


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
