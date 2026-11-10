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

# NaviGo technology inventory

| Component | Implementation / evidence | Runtime classification and limit |
|---|---|---|
| React / TypeScript / Vite | FE `package.json`, `src/main.tsx`, `src/App.tsx` | Active web entry; production build verified locally |
| Express REST | BE `package.json` -> `src/server.js` | Active entry; root `server.js` is legacy, not npm start |
| API client / CORS | FE `src/utils/apiClient.ts`; BE `src/config/corsOptions.js` | Environment-selected API origin and exact allowlist; no production configuration read or changed |
| Authentication | FE `src/auth/`; BE `src/middleware/firebaseAuth.js` | Firebase token verification; local test fallback is not proof of real Firebase login |
| Authorization / admin | FE `AdminRouteGuard.tsx`; BE `adminAuth.js`, `adminRouter.js` | Admin guards implemented and tested; production claims NOT VERIFIED |
| Traveler | FE `UrbanAgentPage.tsx`; BE `modules/travelerApiV2/router.js` | Guest preview, search, recommendations, persistence and lifecycle contracts present |
| Map / itinerary UI | FE `TripPreviewDayMap.tsx`, `TravelerFullRouteModal.tsx`, `TravelerItineraryViewSwitch.tsx` | Map/timeline, route warnings, alternatives, navigation handoff implemented; browser flows NOT VERIFIED |
| Business / partner UI | FE seller route, `RolePages.tsx`; BE `businessLocationScorer.js`, partner modules | Active seller surface; `/partners` target in old context does not match current `/seller` route |
| Loading / errors / responsive | FE route Suspense, planner state handlers, CSS, view switch | Implemented; no browser/mobile accessibility or visual certification |
| Canonical POIs | BE `data/canonical/runtime_dataset_manifest.json`, verifier | PASS: 4173 POIs, zero duplicate IDs / invalid core records, SHA dcb404cc8b5c7a9b5fd70df63039ab8f828c504270e22b12a671fa4ed61583f4 |
| IDs / normalization / nulls | `canonicalCsvPoiRepository.js`, `poiDataService.js`, `serializers.js` | Global_ID, source IDs and aliases distinguished; null/missing values tested |
| Entity resolution / dedupe | `modules/cityPackPreparation/entityResolution.js`, adapters and review queue | Offline preparation pipeline; not automatically imported by product runtime |
| Provenance / freshness | canonical repository; `modules/trust/evidence.js`, live-status overlay | Structured evidence and unknown/stale/conflict states implemented; live truth requires configured providers |
| City Pack | `modules/cities/cityConfig.js`, preparation modules | Da Nang supported; no verified second-city runtime |
| Data repository | `services/poiRepository.js` | CSV default; Postgres only when URBANAGENT_POI_REPOSITORY=postgres |
| PostgreSQL / PostGIS | `migrations/`, `postgresPoiRepository.js`, `postgresClient.js` | Implemented opt-in; disposable database integration skipped, production cutover NOT VERIFIED |
| Ingestion / scheduled sync | `scripts/citypack_scheduled_runner.js`, orchestrator, config | Implemented opt-in operations; host scheduler state NOT VERIFIED; no sync executed |
| Intent detection | `intentService.js` | Rule/keyword detection, including multi-intent; not a trained LLM intent model |
| Tool selection / orchestration | `src/server.js`, recommender, planner and service calls | Explicit application orchestration; no evidence of autonomous LLM tool planning |
| POI retrieval / scoring | `poiRetrievalService.js`, v2 `recommendations.js` | Heuristic keyword/category/location/rating/review scoring with optional model scores |
| Semantic retrieval | `semanticModelService.js` -> `src/inference.py` | Optional Python subprocess with bounded candidates/timeout; not verified with real checkpoint |
| Multimodal inference | `src/encoder/multimodal_encoder.py`, `src/inference.py` | Pretrained CLIP / ResNet plus learned projection contract; weights unavailable at configured artifacts/model path |
| Gated fusion | encoder `_compute_modality_gates` | Existing semantic keyword boost + missing-modality mask + text-only path already present; quality not measured here |
| Model packaging gap | runtime allowlist excludes heavy checkpoints; repository has legacy `model/v4/...` LFS pointer | Runtime packaging is not evidence of working multimodal inference; do not claim V4 SOTA or absolute category precision |
| Reranker | `rerankerService.js`, `artifacts/reranker/agent_reranker_v1.json` | Bounded JSON weight/penalty lookup; not proof of a neural fine-tuned reranker |
| Memory | `agentMemoryService.js`, `artifacts/memory/agent_memory_v1.json` | JSON profile lookup; anonymous requests do not inherit demo global profile |
| Feedback / preference learning | `feedbackService.js`, `userPreferenceService.js`, `functions/src/` | Event-driven preference counters / persistence and Functions implementation; no online gradient training claim |
| Itinerary / replan | v2 `tripPreview.js`, `savedTrips.js`, validators | Time windows, dates, required/temporary places and ownership implemented; HTTP tests verified locally |
| Matrix / travel estimates | `routeMatrixService.js`, `tripPreviewTravelPolicy.js` | Haversine + speed/dwell heuristics, explicitly estimates; walk enum / zero longitude fixed in this branch |
| Expert facade | `src/expert_system/expert_system.js` | Active /api/route path; root ES-system implementation is legacy, rules remain shared inputs |
| Rete matching | `rete_algorithm.js`, `ES-system/rule/*.csv` | Rule lookup network used by expert facade; not a learned road model |
| Backward chaining | `backward_chaining.js` | Route-rule reasoning implementation; coverage limited to committed rules |
| Fuzzy logic / density | `fuzzy_logic.js`, `poi_density_engine.js` | Handcrafted fuzzy membership and POI density proxies; not actual traffic or customer density |
| Turn / one-way / timed rules | rule CSVs, route validation engine | Local rule coverage only; not globally complete navigation legality |
| Weather reasoning | `expert_system/weather_service.js`, `services/weatherService.js` | Provider requests / fallbacks implemented; live weather NOT VERIFIED |
| OSRM / alternatives | `fetchOSRMRoute`, `findOptimalRoute` | Parsed-route caching, coalescing, body timeout, validation and provider error classification restored from 1d35f26 |
| Provider resilience | `providerResilience.js`, `providerGuards.js` | Bounded concurrency/retry/cache/circuit; NoRoute does not open circuit; mock-provider regression verified |
| Business location analysis | scorer + `businessEvidenceService.js` | Competition, demand proxy, centrality/accessibility proxy, concept fit and complementary POIs; no observed footfall/revenue forecast |
| Business insight generation | `businessInsightGenerator.js` | Deterministic local_data_to_text with evidence IDs; prompt construction does not imply an LLM was invoked |
| Partner providers / handoff | `modules/partners/`, official provider and registry | Contracts exist; credentials + injected request required; bookings/payments not proven |
| Synthetic data / memory / reranker training | scripts named generate_synthetic_dataset, build_agent_memory, train_reranker | Offline scripts; not executed to avoid changing artifacts or benchmark claims |
| Representation / contrastive / two-tower | PLANNING + `trainingStatusService.js` references sibling poi_urban | Research contract only; checkpoint metrics reader is not runtime model invocation; external research repo not assumed accessible |
| Evaluation | agent / learning-loop scripts and reports | Existing synthetic evaluation artifacts; no new real-user Precision/Recall/F1/NDCG or zero-shot claims |
