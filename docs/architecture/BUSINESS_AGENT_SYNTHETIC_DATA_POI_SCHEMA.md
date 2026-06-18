# Danang UrbanAgent AI - Business Agent, Synthetic Data, POI Schema

Tai lieu nay thiet ke 3 hang muc tiep theo cho Danang UrbanAgent AI:

1. Tai cau truc Role 2 thanh Business Decision Support Agent.
2. Tao du lieu gia lap co neo du lieu that de giai quyet cold-start.
3. Chuan hoa mot POI JSON Schema dung chung cho LLM, frontend va scorer.

Nguon hoc thuat tham khao:

- Wang et al., 2023/2024, `User Behavior Simulation with Large Language Model based Agents`, arXiv:2306.02552. Tu tuong dung LLM-agent de mo phong hanh vi nguoi dung trong sandbox.
- Park et al., 2023, `Generative Agents: Interactive Simulacra of Human Behavior`, arXiv:2304.03442. Tu tuong observation, memory, reflection va planning de tao hanh vi agent dang tin.

Khong copy pipeline cua cac bai bao. Cach dung o day la bien chung thanh he sinh synthetic user + itinerary + business decision cho du lieu do thi Da Nang.

---

## 1. Role 2 - Business Agent thanh Decision Support System

### 1.1. Loai bo Text Search khoi Role 2

`Text Search` chi nen ton tai nhu legacy/debug module. Role business khong tra ve mot danh sach POI don gian theo keyword. Thay vao do, flow chuan la:

```text
Business concept
-> Concept parser
-> Candidate area scorer
-> Evidence pack builder
-> LLM data-to-text insight generator
-> BusinessInsightPanel
```

Vi du input:

```text
Mo tiem banh ngot kieu Phap gan dai hoc
```

Output UI khong phai `top POI`, ma la 3-5 khu vuc ung vien, moi khu vuc co:

- Opportunity score.
- Demand proxy.
- Competition index.
- Complementary POIs.
- Route/accessibility warnings.
- Bao cao phan tich bang ngon ngu tu nhien.
- Evidence table de nguoi dung kiem tra lai.

### 1.2. BusinessInsightPanel

Component de xuat:

```text
src/pages/urban-agent/components/BusinessInsightPanel.tsx
src/pages/urban-agent/components/BusinessAreaCard.tsx
src/pages/urban-agent/components/EvidenceTable.tsx
src/pages/urban-agent/components/InsightNarrative.tsx
```

Data contract frontend:

```json
{
  "role": "business",
  "concept": "Mo tiem banh ngot kieu Phap gan dai hoc",
  "areas": [
    {
      "area_id": "grid_16.067_108.213",
      "center": { "lat": 16.067, "lon": 108.213 },
      "score": 82,
      "signals": {
        "demand_proxy": 0.78,
        "competition_index": 0.34,
        "complementary_score": 0.72,
        "accessibility_score": 0.61,
        "concept_fit": 0.84,
        "rating_signal": 0.76,
        "review_volume_signal": 0.69
      },
      "evidence": {
        "top_categories": [
          { "category": "Cafe/Dessert", "count": 18 },
          { "category": "Truong dai hoc", "count": 3 },
          { "category": "Quan an", "count": 22 }
        ],
        "complementary_pois": [
          { "poi_id": "p_01", "name": "Dai hoc A", "category": "University", "distance_m": 420 },
          { "poi_id": "p_02", "name": "Cafe B", "category": "Cafe", "distance_m": 210 }
        ],
        "competitors": [
          { "poi_id": "p_09", "name": "Bakery C", "category": "Bakery", "distance_m": 580 }
        ],
        "route_warnings": [
          "Co nhieu duong mot chieu trong ban kinh 700m",
          "Gio cao diem co the lam giam accessibility score"
        ],
        "raw_counts": {
          "poi_total_radius_800m": 137,
          "direct_competitors_radius_800m": 4,
          "complementary_radius_800m": 31
        }
      },
      "llm_insight": {
        "summary": "Khu vuc nay phu hop voi concept banh ngot kieu Phap...",
        "potential": "...",
        "complementary_analysis": "...",
        "risks": "...",
        "recommended_actions": ["Khao sat gia thue", "Kiem tra luu luong khung 17-21h"]
      }
    }
  ]
}
```

### 1.3. Cau truc bao cao bat buoc

LLM insight phai gom 4 phan:

```text
1. Dac diem va tiem nang khu vuc
2. POI bo tro
3. Rui ro/canh bao: giao thong, route accessibility, doi thu
4. De xuat hanh dong tiep theo
```

Khong cho LLM tu quyet dinh diem so. LLM chi duoc "doc" va "dien giai" cac signal da tinh.

### 1.4. Chien luoc Data-to-Text chong hallucination

Thay vi prompt mo:

```text
Hay phan tich khu vuc nay
```

Ta dung prompt co schema va evidence pack:

```text
SYSTEM:
Ban la Business Insight Writer cho Danang UrbanAgent AI.
Nhiem vu cua ban la chuyen doi so lieu thanh bao cao ngan gon.
Ban KHONG duoc them dia diem, diem so, so luong, ten duong, ten truong, ten quan neu khong co trong EVIDENCE_JSON.
Neu thieu du lieu, phai noi "chua du bang chung" thay vi bịa.

USER:
CONCEPT:
{{business_concept}}

EVIDENCE_JSON:
{{strict_json_evidence_pack}}

OUTPUT_SCHEMA:
{
  "summary": "string",
  "area_potential": "string",
  "complementary_poi_analysis": "string",
  "risk_warnings": ["string"],
  "recommended_actions": ["string"],
  "used_evidence_ids": ["string"],
  "missing_evidence": ["string"]
}

RULES:
- Chi su dung number/name/category xuat hien trong EVIDENCE_JSON.
- Moi nhan dinh quan trong phai gan voi it nhat mot evidence id.
- Khong noi "dong khach", "mat do khach that" neu chi co demand_proxy.
- Dung cum "tin hieu nhu cau uoc luong" cho demand_proxy.
```

Ky thuat nen dung:

- Evidence Pack Builder: backend tao JSON co id cho tung bang chung.
- Constrained output: JSON schema / structured output neu provider ho tro.
- Citation by evidence id: moi insight co `used_evidence_ids`.
- Numeric guardrail: sau khi LLM tra ve, backend validate xem output co so/ten dia diem ngoai evidence khong.
- Abstention rule: neu evidence thieu, bat buoc ghi `missing_evidence`.
- Low temperature: `temperature = 0.1 - 0.3`.
- No hidden tools: LLM khong tu goi web/places neu khong qua service duoc phe duyet.

### 1.5. Backend service de xuat

```text
src/services/businessEvidenceService.js
src/services/businessInsightPromptService.js
src/services/businessInsightGenerator.js
src/services/llmProviderService.js
src/services/insightGuardrailService.js
```

Endpoint:

```text
POST /api/agent/business-insight
```

Request:

```json
{
  "concept": "Mo tiem banh ngot kieu Phap gan dai hoc",
  "areaLimit": 5,
  "language": "vi"
}
```

Response:

```json
{
  "areas": [],
  "insights": [],
  "guardrails": {
    "hallucination_checked": true,
    "unsupported_claims": []
  }
}
```

---

## 2. Grounded Synthetic Data Generation

### 2.1. Muc tieu

Ta can synthetic data de:

- Test agent khi chua co user that.
- Fine-tune intent parser/reranker.
- Tao benchmark: query nao phai ra itinerary nao.
- Sinh feedback gia lap cho learning-to-rank.

Khong tao du lieu "bay". Moi sample phai neo vao POI co that trong CSV.

### 2.2. Tu tuong tu RecAgent va Generative Agents

Tu RecAgent: dung LLM-agent de mo phong hanh vi nguoi dung va phan ung voi recommendation trong mot sandbox.

Tu Generative Agents: moi persona co memory, reflection va planning. Voi Danang UrbanAgent:

- Memory = lich su so thich/quan da thich/quan da bo qua.
- Reflection = tom tat so thich cap cao, vi du "thich quan yen tinh, gia vua, tranh noi qua dong".
- Planning = sinh query va itinerary theo thoi diem, thoi tiet, muc dich.

### 2.3. Pipeline tong the

```text
Raw CSV + reviews
-> POI schema normalization
-> Review semantic extraction
-> Persona generation
-> Grounded task generation
-> Expected output construction
-> Validator
-> Synthetic dataset JSONL
```

### 2.4. Persona Generation bang reverse engineering review

Input cho LLM khong phai toan bo review, ma la evidence pack:

```json
{
  "poi_id": "p_123",
  "name": "Cafe X",
  "category": "Cafe/Dessert",
  "district": "Hai Chau",
  "review_snippets": [
    "quan yen tinh, hop hoc bai",
    "gia vua phai, co o cam",
    "toi hoi dong vao cuoi tuan"
  ],
  "signals": {
    "rating": 4.5,
    "price_level": 2,
    "semantic_tags": ["yen tinh", "hoc bai", "gia vua", "co o cam"]
  }
}
```

Prompt:

```text
Hay suy luan nguoc persona co kha nang viet cac review tren.
Chi duoc suy luan tu evidence. Khong them dia diem moi.
Tra ve JSON theo schema.
```

Output persona:

```json
{
  "persona_id": "persona_student_quiet_cafe_001",
  "segment": "student",
  "budget_level": "medium",
  "mobility": "motorbike",
  "preferences": {
    "liked_tags": ["quiet", "study", "wifi", "socket", "moderate_price"],
    "disliked_tags": ["too_crowded", "hard_to_park"],
    "preferred_categories": ["Cafe/Dessert", "Quan an"],
    "preferred_time_slots": ["afternoon", "evening"]
  },
  "memory": [
    {
      "type": "positive_experience",
      "poi_id": "p_123",
      "evidence": "review mentioned quiet study space and sockets"
    }
  ],
  "reflection": "Persona nay uu tien quan yen tinh de hoc bai, gia vua phai va di chuyen bang xe may."
}
```

### 2.5. Grounding - ep LLM chi dung POI that

Dung 4 lop khoa:

1. Candidate set injection: prompt chi dua vao danh sach POI ung vien.
2. ID-only output: LLM chi duoc tra ve `poi_id`, khong tu viet ten dia diem moi.
3. Post-validator: backend reject output co `poi_id` khong ton tai.
4. Repair prompt: neu sai, gui lai loi va yeu cau chon lai tu candidate set.

Prompt itinerary generation:

```text
SYSTEM:
Ban la data generator cho Danang UrbanAgent AI.
Chi duoc chon POI bang poi_id trong CANDIDATE_POIS.
Khong duoc tao dia diem moi, ten moi, toa do moi.

USER:
PERSONA_JSON:
{{persona}}

TASK_CONTEXT:
{
  "city": "Da Nang",
  "time_slot": "evening",
  "weather": "light_rain",
  "transport": "motorbike",
  "goal": "di an va cafe sau gio hoc"
}

CANDIDATE_POIS:
[
  {"poi_id":"p_1","name":"...","category":"Quan an","tags":["..."]},
  {"poi_id":"p_2","name":"...","category":"Cafe/Dessert","tags":["..."]}
]

OUTPUT_SCHEMA:
{
  "user_query": "string",
  "expected_itinerary": [
    {"poi_id": "string", "reason": "string", "slot": "string"}
  ],
  "negative_pois": [
    {"poi_id": "string", "reason": "string"}
  ]
}
```

### 2.6. Synthetic record JSONL

Mot dong JSONL hoan chinh:

```json
{
  "sample_id": "synth_dn_000001",
  "version": "2026-06-urbanagent-v1",
  "source": {
    "generator": "grounded_synthetic_pipeline",
    "poi_csv_hash": "sha256:...",
    "review_source": "local_csv_reviews",
    "created_at": "2026-06-04T00:00:00Z"
  },
  "persona": {
    "persona_id": "persona_student_quiet_cafe_001",
    "segment": "student",
    "budget_level": "medium",
    "mobility": "motorbike",
    "preferences": {
      "liked_tags": ["quiet", "study", "wifi"],
      "disliked_tags": ["crowded", "expensive"],
      "preferred_categories": ["Cafe/Dessert", "Quan an"]
    },
    "memory": [
      { "poi_id": "p_123", "sentiment": "positive", "evidence": "quiet study space" }
    ],
    "reflection": "Thich quan yen tinh, gia vua, phu hop hoc bai."
  },
  "query_context": {
    "language": "vi",
    "city": "Da Nang",
    "origin": { "lat": 16.0544, "lon": 108.2022 },
    "time_slot": "evening",
    "transport": "motorbike",
    "weather": "clear",
    "group_size": 2,
    "constraints": ["co cafe", "co quan an", "khong qua xa"]
  },
  "user_query": "Toi muon toi nay di an nhe roi ghe mot quan cafe yen tinh de hoc bai, di xe may.",
  "candidate_pool": ["p_123", "p_456", "p_789"],
  "expected_output": {
    "intent_labels": ["food", "cafe"],
    "expected_itinerary": [
      {
        "order": 1,
        "poi_id": "p_456",
        "expected_role": "food",
        "reason_evidence": ["category_match", "distance_ok", "rating_good"]
      },
      {
        "order": 2,
        "poi_id": "p_123",
        "expected_role": "cafe",
        "reason_evidence": ["quiet_tag", "study_tag", "moderate_price"]
      }
    ],
    "acceptable_alternatives": {
      "food": ["p_789"],
      "cafe": ["p_124", "p_125"]
    },
    "negative_pois": [
      { "poi_id": "p_999", "reason": "wrong_category_pub" }
    ]
  },
  "evaluation": {
    "metrics": ["intent_coverage", "poi_validity", "category_precision", "route_feasibility"],
    "must_pass": {
      "all_poi_ids_exist": true,
      "covers_required_intents": true,
      "no_hallucinated_place": true
    }
  }
}
```

### 2.7. Validator bat buoc

Moi sample synthetic phai qua:

- `poi_id_exists`: tat ca POI id nam trong CSV.
- `intent_coverage`: neu query co cafe + food thi expected itinerary phai co ca hai.
- `category_precision`: POI expected phai match category/tag.
- `route_feasibility`: khong tao lich trinh qua xa neu context la walking.
- `no_name_leak`: output khong co ten POI ngoai candidate pool.
- `evidence_trace`: moi reason co evidence id.

Service/script de xuat:

```text
scripts/generate_synthetic_personas.js
scripts/generate_synthetic_itineraries.js
scripts/validate_synthetic_dataset.js
data/synthetic/urbanagent_synthetic_v1.jsonl
```

---

## 3. POI JSON Schema chuan

### 3.1. Yeu cau

Mot POI phai phuc vu 3 he:

- LLM-Friendly: de embedding/RAG/doc-to-text.
- Frontend-Ready: de render map/card/open hours/images/action.
- Scorer-Optimized: de tinh demand proxy, route, competition, category weights.

### 3.2. JSON Schema de xuat

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://danang-urbanagent.ai/schema/poi.schema.json",
  "title": "DanangUrbanAgentPOI",
  "type": "object",
  "required": ["poi_id", "name", "category", "location", "source", "scoring"],
  "properties": {
    "poi_id": { "type": "string" },
    "source": {
      "type": "object",
      "required": ["provider", "source_id", "last_seen_at"],
      "properties": {
        "provider": { "type": "string", "enum": ["ggmap", "foody", "manual", "synthetic"] },
        "source_id": { "type": "string" },
        "source_url": { "type": ["string", "null"] },
        "last_seen_at": { "type": "string", "format": "date-time" },
        "data_confidence": { "type": "number", "minimum": 0, "maximum": 1 }
      }
    },
    "name": { "type": "string" },
    "normalized_name": { "type": "string" },
    "category": {
      "type": "object",
      "required": ["primary", "normalized"],
      "properties": {
        "primary": { "type": "string" },
        "secondary": { "type": "array", "items": { "type": "string" } },
        "normalized": { "type": "string" },
        "category_weights": {
          "type": "object",
          "additionalProperties": { "type": "number", "minimum": 0, "maximum": 1 }
        }
      }
    },
    "location": {
      "type": "object",
      "required": ["lat", "lon"],
      "properties": {
        "lat": { "type": "number" },
        "lon": { "type": "number" },
        "address": { "type": ["string", "null"] },
        "street": { "type": ["string", "null"] },
        "ward": { "type": ["string", "null"] },
        "district": { "type": ["string", "null"] },
        "city": { "type": "string", "default": "Da Nang" },
        "geohash": { "type": ["string", "null"] },
        "grid_id": { "type": ["string", "null"] }
      }
    },
    "frontend": {
      "type": "object",
      "properties": {
        "display_title": { "type": "string" },
        "short_description": { "type": ["string", "null"] },
        "images": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "url": { "type": "string" },
              "caption": { "type": ["string", "null"] },
              "is_primary": { "type": "boolean" }
            }
          }
        },
        "opening_hours": {
          "type": "object",
          "properties": {
            "raw_text": { "type": ["string", "null"] },
            "weekly": { "type": "object" },
            "open_now": { "type": ["boolean", "null"] }
          }
        },
        "contact": {
          "type": "object",
          "properties": {
            "phone": { "type": ["string", "null"] },
            "website": { "type": ["string", "null"] },
            "map_url": { "type": ["string", "null"] }
          }
        },
        "price": {
          "type": "object",
          "properties": {
            "raw_text": { "type": ["string", "null"] },
            "price_level": { "type": ["integer", "null"], "minimum": 0, "maximum": 4 },
            "min_vnd": { "type": ["number", "null"] },
            "max_vnd": { "type": ["number", "null"] }
          }
        }
      }
    },
    "semantic": {
      "type": "object",
      "properties": {
        "tags": { "type": "array", "items": { "type": "string" } },
        "review_keywords": { "type": "array", "items": { "type": "string" } },
        "summary_vi": { "type": ["string", "null"] },
        "summary_en": { "type": ["string", "null"] },
        "rag_document": { "type": "string" },
        "embedding_text": { "type": "string" }
      }
    },
    "quality": {
      "type": "object",
      "properties": {
        "rating": { "type": ["number", "null"] },
        "rating_scale": { "type": "number" },
        "review_count": { "type": ["integer", "null"] },
        "sentiment_score": { "type": ["number", "null"], "minimum": -1, "maximum": 1 },
        "freshness_score": { "type": ["number", "null"], "minimum": 0, "maximum": 1 }
      }
    },
    "scoring": {
      "type": "object",
      "properties": {
        "rating_score": { "type": "number", "minimum": 0, "maximum": 1 },
        "review_volume_score": { "type": "number", "minimum": 0, "maximum": 1 },
        "price_score": { "type": ["number", "null"], "minimum": 0, "maximum": 1 },
        "demand_proxy_score": { "type": ["number", "null"], "minimum": 0, "maximum": 1 },
        "accessibility_score": { "type": ["number", "null"], "minimum": 0, "maximum": 1 },
        "competition_weight": { "type": ["number", "null"], "minimum": 0, "maximum": 1 },
        "complementary_weight": { "type": ["number", "null"], "minimum": 0, "maximum": 1 }
      }
    },
    "vectors": {
      "type": "object",
      "properties": {
        "text_embedding_id": { "type": ["string", "null"] },
        "image_embedding_id": { "type": ["string", "null"] },
        "geo_embedding_id": { "type": ["string", "null"] },
        "multimodal_embedding_id": { "type": ["string", "null"] },
        "embedding_version": { "type": ["string", "null"] }
      }
    },
    "audit": {
      "type": "object",
      "properties": {
        "created_at": { "type": "string", "format": "date-time" },
        "updated_at": { "type": "string", "format": "date-time" },
        "schema_version": { "type": "string" },
        "notes": { "type": "array", "items": { "type": "string" } }
      }
    }
  }
}
```

### 3.3. Vi du POI instance rut gon

```json
{
  "poi_id": "ggmap_abc123",
  "source": {
    "provider": "ggmap",
    "source_id": "abc123",
    "source_url": "https://maps.google.com/...",
    "last_seen_at": "2026-06-04T00:00:00Z",
    "data_confidence": 0.82
  },
  "name": "Highlands Coffee - Nguyen Van Thoai",
  "normalized_name": "highlands coffee nguyen van thoai",
  "category": {
    "primary": "Cafe/Dessert",
    "secondary": ["Coffee", "Dessert"],
    "normalized": "cafe",
    "category_weights": {
      "cafe": 0.95,
      "food": 0.25,
      "tourism": 0.05
    }
  },
  "location": {
    "lat": 16.0541,
    "lon": 108.2442,
    "address": "Nguyen Van Thoai, Da Nang",
    "street": "Nguyen Van Thoai",
    "district": "Ngu Hanh Son",
    "city": "Da Nang",
    "grid_id": "grid_16.054_108.244"
  },
  "frontend": {
    "display_title": "Highlands Coffee - Nguyen Van Thoai",
    "short_description": "Cafe phu hop gap ban, lam viec nhe.",
    "images": [],
    "opening_hours": {
      "raw_text": "07:00-22:00",
      "weekly": {},
      "open_now": null
    },
    "contact": {
      "phone": null,
      "website": null,
      "map_url": "https://maps.google.com/?q=16.0541,108.2442"
    },
    "price": {
      "raw_text": "40k-80k",
      "price_level": 2,
      "min_vnd": 40000,
      "max_vnd": 80000
    }
  },
  "semantic": {
    "tags": ["coffee", "dessert", "meeting", "near_beach"],
    "review_keywords": ["view", "coffee", "busy_evening"],
    "summary_vi": "Cafe chuoi, de tim, phu hop gap ban va nghi chan.",
    "summary_en": "Chain coffee shop suitable for casual meetings.",
    "rag_document": "Highlands Coffee - Nguyen Van Thoai is a Cafe/Dessert POI in Da Nang...",
    "embedding_text": "cafe coffee dessert meeting Nguyen Van Thoai Da Nang"
  },
  "quality": {
    "rating": 4.3,
    "rating_scale": 5,
    "review_count": 328,
    "sentiment_score": 0.62,
    "freshness_score": 0.7
  },
  "scoring": {
    "rating_score": 0.86,
    "review_volume_score": 0.78,
    "price_score": 0.65,
    "demand_proxy_score": 0.74,
    "accessibility_score": 0.68,
    "competition_weight": 0.52,
    "complementary_weight": 0.44
  },
  "vectors": {
    "text_embedding_id": "emb_text_abc123",
    "image_embedding_id": null,
    "geo_embedding_id": "emb_geo_abc123",
    "multimodal_embedding_id": "emb_mm_abc123",
    "embedding_version": "v4"
  },
  "audit": {
    "created_at": "2026-06-04T00:00:00Z",
    "updated_at": "2026-06-04T00:00:00Z",
    "schema_version": "poi_schema_v1",
    "notes": []
  }
}
```

### 3.4. Cach dung schema trong he thong

LLM/RAG:

- Dung `semantic.rag_document`.
- Dung `semantic.tags`, `category.category_weights`, `quality`.
- Khong dua raw vector vao prompt.

Frontend:

- Dung `frontend.display_title`, `location`, `frontend.images`, `frontend.opening_hours`, `frontend.contact`.

Scorer:

- Dung `scoring.*`, `category.category_weights`, `quality.review_count`, `location.grid_id`.
- Vector embedding chi luu id trong DB/vector store, khong nhet array lon vao JSON API neu khong can.

---

## Thu tu code de xuat

1. Backend: them POI normalization layer theo schema v1.
2. Backend: them `businessEvidenceService` de tao evidence pack.
3. Backend: them `businessInsightGenerator` voi prompt data-to-text va guardrail.
4. FE: tach `BusinessInsightPanel` ra component rieng, bo tu duy text search o role business.
5. Data: viet synthetic pipeline JSONL va validator.
6. Training/eval: dung synthetic data de test intent coverage, category precision va no-hallucination.
