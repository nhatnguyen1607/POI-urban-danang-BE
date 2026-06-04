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
