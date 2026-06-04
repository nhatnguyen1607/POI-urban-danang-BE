# Danang UrbanAgent AI - Research and Training Strategy

## Ket luan kien truc

Agent runtime nen giu trong `POI-urban-danang-BE`.

Ly do:
- Backend can phuc vu API, session, feedback, route tool, map tool, business insight va connector hanh dong.
- Agent runtime phai gan voi san pham: nguoi dung tao lich trinh, chinh sua diem dung, mo chi duong, gui feedback.
- Backend la noi tot nhat de log du lieu hanh vi that vao `storage/feedback`.

Phan training representation/model nghien cuu nen nam trong `poi_urban`.

Ly do:
- `poi_urban` la noi phu hop de train multimodal encoder, contrastive loss, embedding, ablation va checkpoint.
- Cac ket qua trong `poi_urban` co the dung truc tiep cho bao cao khoa hoc: metric, t-SNE, confusion/error analysis, loss curve.
- Tach nhu vay giup khong bien backend thanh notebook training nang, nhung van cho backend dung artifact da train.

## Ranh gioi 2 repo

`POI-urban-danang-BE` phu trach:
- API cho frontend.
- Agent orchestration: intent -> plan -> tool -> response.
- Memory theo user/persona.
- Feedback logging.
- Reranker inference.
- Evidence pack cho Business Insight.
- Route expert va warning.
- Visualization report nhanh cho demo va PISI.

`poi_urban` phu trach:
- Multimodal representation learning.
- Fine-tune embedding/encoder.
- Semantic-aware hard negative mining.
- Supervised contrastive learning theo category/intent.
- Dataset builder nghien cuu tu POI schema + synthetic + feedback.
- Ablation giua V1/V2/V3/V4.
- Figure khoa hoc: t-SNE, UMAP, retrieval examples, confusion matrix, before/after.

Shared contract giua hai ben:
- `data/schema/poi.schema.json`
- `data/synthetic/urbanagent_synthetic_v1.jsonl`
- `data/training/agent_representation_pairs_v1.jsonl`
- `storage/feedback/agent-feedback.jsonl`
- `artifacts/reranker/agent_reranker_v1.json`
- `artifacts/memory/agent_memory_v1.json`
- `artifacts/embeddings/poi_embeddings_v*.jsonl`
- `reports/evaluation/*.json`

## Cac tang "hoc" cua Agent

### Tang A - Learning loop nhanh hien tai

Day la phan da co trong backend:
- Sinh synthetic user/persona grounded tren POI that.
- Build memory theo persona/user.
- Train reranker tu synthetic + feedback.
- Evaluate truoc/sau training.
- Tao SVG truc quan de chung minh agent co hoc tu preference.

Day khong phai neural training nang. No nhanh vi chi train profile/reranker nhe, phu hop cho demo san pham va cold-start.

Metric can nhin:
- `exactPoiHitRate`: POI mong doi co xuat hien trong ket qua hay khong.
- `recallAtReturnedK`: bao nhieu POI dung duoc tim thay trong list tra ve.
- `intentCoverage`: ket qua co phu hop nhu cau/chuyen di hay khong.
- `precisionAtExpectedK`: chat luong sap xep top dau.
- `f1`: can bang precision/recall.
- `delta`: muc tang sau khi co memory + reranker.

### Tang B - Train embedding/reranker nghiem tuc

Nen lam tiep sau Tang A:
- Tao pair/triplet tu synthetic + feedback:
  - positive: user thich, them vao lich trinh, click, xem route, dat ban.
  - negative: xoa khoi lich trinh, dislike, bo qua sau khi xem.
- Backend export du lieu bang `npm run export:representation-data`.
- Train lightweight embedding reranker:
  - input: query embedding + POI features + persona memory.
  - output: relevance score.
- Dung validation split theo persona de tranh overfit.
- Bao cao: NDCG@K, Recall@K, MRR, category accuracy, route feasibility rate.

Da co baseline trong `D:\poi_urban`:

```text
research_pipeline/train_agent_representation_reranker.py
research_pipeline/train_agent_two_tower_representation.py
```

Output hien tai:

```text
D:\poi_urban\results\agent_representation\agent_representation_metrics.png
D:\poi_urban\results\agent_representation_two_tower\two_tower_training_report.png
D:\poi_urban\results\agent_representation_two_tower\agent_two_tower_representation.pt
```

### Tang C - Fine-tune representation model trong `poi_urban`

Day la phan "train model representation that":
- Dung POI text, category, review, image, toa do, region context.
- Ap dung semantic-aware hard negative mining:
  - gan nhau ve khong gian nhung khac category thi la hard negative.
  - cung category va cung intent thi la positive semantic.
- Ket hop InfoNCE voi Supervised Contrastive Loss.
- Muc tieu: query "cafe yen tinh" khong bi keo sang "quan nhau" chi vi gan nhau hoac anh/khong gian giong.

Output cho backend:
- `artifacts/embeddings/poi_embeddings_v4.jsonl`
- `artifacts/model/model_card_v4.json`
- `reports/evaluation/retrieval_eval_v4.json`

### Tang D - Fine-tune LLM

Chi nen lam khi co du du lieu hoi-dap chat luong:
- It nhat vai nghin mau query -> expected itinerary/insight da kiem duyet.
- LLM khong nen duoc fine-tune de "bia" insight.
- Business Insight nen uu tien Data-to-Text: backend tinh so lieu, LLM chi dien giai evidence pack.

Neu chua du du lieu, dung:
- Prompt co schema chat.
- RAG tren POI/review/evidence.
- Guardrail bat buoc trich dan score tu evidence pack.

## Dong gop moi cua du an

1. Danang-first grounded urban agent:
   - Khong lam travel chatbot chung chung.
   - Moi goi y dua tren POI, review, route, category va signal dia phuong Da Nang.

2. Mot loi POI intelligence dung chung cho hai role:
   - Khach du lich tim quan va tao lich trinh.
   - Nguoi kinh doanh phan tich vi tri/mo concept.
   - Hai role khong roi rac: hanh vi cua traveler tao market signal cho business.

3. Grounded synthetic data:
   - Sinh persona va hoi thoai tu POI/review that.
   - Bat buoc neo dia diem vao CSV, khong cho LLM bia dia diem ao.

4. Agent co memory va evaluator:
   - Co hoc so thich dai han theo user/persona.
   - Co do truoc/sau training bang metric va hinh anh.

5. Expert route layer:
   - Khong chi mo Google Maps.
   - He thong tu tinh route, sap xep multi-stop, hien warning doan duong/diem can tranh.

6. Business DSS co anti-hallucination:
   - LLM chi dien giai so lieu da tinh: demand proxy, competition index, complementary POIs, accessibility.
   - Khong tu bia "mat do khach hang that" khi chua co du lieu thuc.

## Hinh anh can co khi di thi

Backend co the tao nhanh:
- Before/after learning chart.
- Summary card agent learning.
- JSON report cho evaluator.

`poi_urban` nen tao them:
- t-SNE/UMAP truoc-sau semantic hard negative mining.
- Confusion matrix category retrieval.
- Top-K retrieval examples dung/sai.
- Loss curve train/validation.
- Ablation table V1/V2/V3/V4.
- Dataset statistics: so POI, category distribution, review length, synthetic personas, feedback signals.

## Quy trinh de demo thuyet phuc

1. Chay baseline agent chua co memory/reranker.
2. Chay generate synthetic + build memory + train reranker.
3. Chay evaluate learning loop.
4. Sinh SVG before/after.
5. Mo frontend:
   - Tao lich trinh theo thoi gian.
   - Sua lộ trinh bang cach them/xoa POI.
   - Xem toan bo route da diem va warning.
   - Chuyen sang role business de xem insight tu cung loi POI intelligence.
