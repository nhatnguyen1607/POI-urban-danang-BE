# PLANNING - Backend

## Vai tro cua thu muc nay

`POI-urban-danang-BE` la backend/API layer cho Danang UrbanAgent AI. Thu muc nay phu trach dieu phoi request tu frontend, goi model inference, doc du lieu POI, goi he chuyen gia, va tra ve ket qua co giai thich cho hai role:

- `traveler`: khach/nguòi di choi can goi y dia diem, lich trinh, lo trinh va action tiep theo.
- `business`: nguoi kinh doanh can goi y khu vuc/vi tri, mat do doi thu, demand proxy va ly do.

Backend khong nen chua logic UI va khong nen huan luyen model truc tiep. Backend chi nen dong vai tro service/API va orchestrator.

## Cau truc thu muc

```text
src/                 API server, Python inference, expert system, encoder
data/                CSV POI local
artifacts/model/     Metrics, figures, saved model artifacts
config/              Cau hinh backend
storage/uploads/     Upload tam thoi luc runtime
scripts/             Script tien xu ly va tao deliverables
docs/pisi_2026/      Ho so PISI, khong dat planning/code convention o day
```

## Nguyen tac kien truc

Flow chuan:

```text
Frontend -> Backend API -> Agent Orchestrator -> Services/Tools -> Response with reasons
```

Agent backend khong lam tat ca trong mot ham lon. Chia thanh cac service:

```text
agentOrchestrator.js
poiRetrievalService.js
routeExpertService.js
itineraryPlannerService.js
businessLocationScorer.js
actionConnectorService.js
weatherService.js
```

## API can co

```text
POST /api/agent/recommend-poi
POST /api/agent/create-itinerary
POST /api/agent/update-itinerary
POST /api/agent/business-location
POST /api/agent/feedback
POST /api/agent/business-insight
POST /api/route/matrix
GET  /api/weather/forecast
```

## Data contract bat buoc

Moi response recommendation phai co:

```json
{
  "score": 0.82,
  "title": "string",
  "reason": "string",
  "signals": {},
  "warnings": [],
  "actions": []
}
```

Khong tra ve score khong giai thich duoc.

## Tich hop API ngoai

Thu tu uu tien:

1. Local CSV/model hien co: fallback bat buoc.
2. Open-Meteo: weather.
3. OpenRouteService/Google Routes/Mapbox: route/duration.
4. Google Places/Foursquare: live POI, opening hours, phone, website.
5. Grab handoff: mo app/link de nguoi dung xac nhan.
6. Grab partner API: chi cam vao khi co quyen truy cap hop phap.

Khong scrape API/app trai dieu khoan.

## Pham vi MVP backend

Phase 1:
- Chuan hoa endpoint goi y POI.
- Them category/semantic rerank.
- Tra ve reason/warnings/actions.

Phase 2:
- Them route matrix va weather score.
- Tao itinerary planner co thu tu diem.

Phase 3:
- Them business location scorer.
- Tinh demand proxy, competition, complementary POI, accessibility.

Phase 4:
- Them action connector layer: map, phone, website, Grab handoff.

Phase 5:
- Ghi feedback nguoi dung vao `storage/feedback/agent-feedback.jsonl`.
- Dung feedback de train/rerank lai recommendation: POI duoc them vao lich trinh la positive signal, POI bi dislike/xoa la negative signal.
- Khong goi la online training trong MVP neu chua co pipeline fine-tune; goi dung la learning loop/feedback memory.

Phase 6:
- Tai cau truc Business Agent thanh Decision Support System.
- Them LLM data-to-text insights dua tren evidence pack, khong dung LLM de bia so lieu.
- Tao grounded synthetic data pipeline de test/fine-tune agent.
- Chuan hoa POI JSON Schema v1 cho RAG, frontend va scorer.

Tai lieu chi tiet: `docs/architecture/BUSINESS_AGENT_SYNTHETIC_DATA_POI_SCHEMA.md`.

## Quy uoc code backend

- File JS dung camelCase.
- Function nen ngan, mot service mot nhiem vu.
- Khong hard-code API key.
- Khong encode toan bo POI moi lan request neu co the precompute/cache.
- Log loi ro rang nhung khong in API key.
- Neu API ngoai fail, fallback ve local data va ghi warning.

## Commit lien quan backend

Vi du:

```text
feat(be): add agent orchestrator contract
feat(be): add business location scorer
feat(route): integrate route matrix service
fix(be): fallback to local poi when places api fails
```

## Cach chay du an BE

### Cai dat

```bash
cd D:\POI-urban-danang-BE
npm install
```

Neu can chay inference AI nang, cai Python dependencies:

```bash
pip install -r requirements.txt
```

### Chay server

```bash
npm start
```

Mac dinh backend chay tai:

```text
http://localhost:7860
```

### Endpoint MVP can test

```text
GET  /api/eda
POST /api/agent/recommend-poi
POST /api/agent/create-itinerary
POST /api/agent/update-itinerary
POST /api/agent/business-location
POST /api/agent/business-insight
POST /api/agent/feedback
POST /api/route/matrix
GET  /api/weather/forecast?lat=16.0544&lon=108.2022
```

Vi du test nhanh:

```bash
curl -X POST http://localhost:7860/api/agent/recommend-poi ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"toi muon cafe yen tinh gan bien\"}"
```

Tao synthetic dataset MVP:

```bash
npm run generate:synthetic
```

Output:

```text
data/synthetic/urbanagent_synthetic_v1.jsonl
```

Danh gia agent tren synthetic dataset:

```bash
npm run evaluate:agent
```

Output:

```text
reports/evaluation/agent_eval_latest.json
```

Train reranker tu synthetic + feedback:

```bash
npm run build:memory
npm run train:reranker
```

Output:

```text
artifacts/memory/agent_memory_v1.json
artifacts/reranker/agent_reranker_v1.json
```

Sau khi artifact nay ton tai, `recommendPOIs` se tu dong ap dung boost/penalty neu phu hop.

Export du lieu train representation/fine-tune cho repo `poi_urban`:

```bash
npm run export:representation-data
```

Output:

```text
data/training/agent_representation_pairs_v1.jsonl
data/training/agent_representation_pairs_v1.summary.json
```

File nay gom query-positive POI, query-negative POI, persona memory va feedback event.
Day la cau noi de `poi_urban` train embedding/reranker/model representation that.

Danh gia learning loop truoc/sau training:

```bash
npm run evaluate:learning
```

Quy trinh nay se:

```text
1. Tam tat reranker hien co.
2. Evaluate baseline.
3. Build memory tu synthetic + feedback.
4. Train reranker tu synthetic + feedback + memory.
5. Evaluate lai voi persona context.
6. Ghi report before/after.
```

Output:

```text
reports/evaluation/learning_loop_eval_latest.json
```

Sinh hinh truc quan sau khi evaluate:

```bash
npm run visualize:evaluation
```

Output:

```text
reports/evaluation/figures/learning_before_after.svg
reports/evaluation/figures/learning_summary_card.svg
```

Chay tron bo pipeline nghien cuu agent MVP:

```bash
npm run research:agent
```

Y nghia cac chi so chinh:

```text
exactPoiHitRate       POI ky vong co xuat hien trong ket qua hay khong
recallAtReturnedK     Ty le POI ky vong nam trong danh sach tra ve
intentCoverage        Muc do ket qua phu hop intent nguoi dung
precisionAtExpectedK  Chat luong sap xep o top dau
f1                    Can bang precision/recall
delta                 Muc thay doi sau memory + reranker training
```

Luu y nghien cuu:

```text
Backend hien chi train learning loop/reranker nhe de phuc vu agent runtime.
Fine-tune embedding hoac train representation model that nen dat trong repo poi_urban,
sau do export artifact embedding/model sang backend de phuc vu API.
```

Tai lieu chien luoc:

```text
docs/architecture/AGENT_RESEARCH_TRAINING_STRATEGY.md
```
