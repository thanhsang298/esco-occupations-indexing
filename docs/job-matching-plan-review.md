# Review: Pipeline matching TopCV job posting với ESCO Occupations

Ngày review: 2026-09-22
Phạm vi: plan `TopCV JSON → canonical → query embeddings → retrieve 3 kênh → RRF → top-10`
Đối chiếu với: `esco-occupations-indexing` (build `9ac16776cc2c`), `esco-skills-indexing`,
dữ liệu TopCV thật, model card Harrier, FastEmbed 0.8.0.

## Verdict

**Duyệt với điều kiện fix 2 must-fix (§2a, §2b)**, làm rõ §2c/§2d trong spec,
đưa §3 vào backlog calibration. Phần RRF/fusion/output schema/test plan đã chắc.

## 1. Đã verify đúng (có evidence)

| Claim trong plan | Thực tế kiểm tra |
|---|---|
| 3.203 `completed`, 2 `retry` | Đúng chính xác (`completed 3203, retry 2` / tổng 3205) |
| Top-level fields làm nguồn chính | Đủ hết: `title, description, requireCandidate, requireSummary, experience, knowledge, categories, externalId, sourceUrl, updatedAt` |
| Mọi job có category IT-257 | 3203/3203 jobs có ≥1 category `level1Id=257`; 507 jobs đa-category; 0 job thiếu category |
| `knowledge` cho lexical query | Tồn tại nhưng là **taxonomy tags** (`[{id, title}]`), **trống ở 657/3203 jobs (~20%)** → cần fallback |
| Query cần instruction, doc không cần | Đúng model card Harrier FAQ #1. Format: `Instruct: {one-sentence task}\nQuery: {text}`. Index ESCO embed không instruction → asymmetric setup khớp khuyến nghị |
| Harrier hỗ trợ VI+EN | 94 ngôn ngữ gồm Vietnamese → V1 không dịch là hợp lý |
| Không cộng cosine với BM25 | Đúng — RRF trên rank là cách chuẩn |
| `experience` mang theo | Giữ opaque — values là code `'1'..'8'` (mode `'4'` = 740 jobs), mapping chưa rõ, không diễn giải |

## 2. Must-fix

### 2a. BM25 query BẮT BUỘC dùng `query_embed`, không phải `embed`

FastEmbed 0.8.0 implement BM25 bất đối xứng (đã đọc source class `Bm25`):

- `embed()` (dùng lúc index): weights TF-IDF đầy đủ.
- `query_embed()`: binary weights `1.0`, dedup tokens — *"To emulate BM25 behaviour"*.

Qdrant field `lexical_sparse` dùng `modifier: IDF` nên query value phải là 1.0.
Dùng nhầm `embed()` cho query → score sparse sai thang (TF-weighted × IDF) → RRF fusion lệch.

Action:

- Spec ghi rõ method `query_embed` (không chỉ "đúng BM25 configuration").
- Contract test: cùng một text, `query_embed` phải trả values toàn 1.0 và khác `embed`.

### 2b. Nguồn truth cho compat check

Qdrant **không lưu `model_id`/BM25 hyperparams** (collection chỉ có vector names,
dimension, sparse modifier). Muốn "từ chối nếu không tương thích" phải đọc từ
ESCO build artifacts:

- `resolved-config.json`: `model_id`, `model_revision`, BM25 `k/b/avg_len/language`.
- `embedding-report.json:backend_metadata`: `tei_model_id/sha` thực tế đã embed.

Action:

- Thêm input `--esco-build-dir data/work/<build-id>` (hiện tại: `9ac16776cc2c`).
- Resume key = job `content_hash` + `esco_index_build_id` + config fingerprint.
  **Đổi `esco_index_build_id` → re-retrieve toàn bộ.**

### 2c. Instruction + token budget + last-token pooling

Harrier dùng **last-token pooling** → embedding bị chi phối bởi cuối text:

- Instruction ở đầu (`Instruct:...\nQuery:...`) là đúng vì right-truncation giữ
  lại instruction (khớp `_truncate_token_ids` của TEI backend).
- Thứ tự sections trong `semantic_query` ảnh hưởng kết quả → chốt thứ tự + budget
  cụ thể. Đề xuất: title → category path → requirements → description
  (description dài/nhiễu nhất nên bị cắt trước), để tập eval hiệu chỉnh.
- Một instruction chung cho `label_query` + `semantic_query` OK cho V1, nhưng để
  thành biến config để eval có thể thử tách đôi.

### 2d. Tie-break với channel thiếu

Thứ tự `(RRF desc, semantic rank, label rank, esco_id)` cần định nghĩa
`rank null → +∞`; `esco_id` sort string UUID đã deterministic. Ghi rõ trong spec,
unit test khóa lại.

## 3. Backlog calibration (sau benchmark 200 jobs)

- **Title nhiễu**: title thật chứa lương/địa điểm
  (vd `'Nhân Viên QA/QC Chuỗi Nhà Hàng Thu Nhập 12-16tr Tại Hà Nội'`).
  V1 giữ nguyên; cân nhắc title-denoise regex ở vòng hiệu chỉnh.
- **`updatedAt` là object `{"$date": "..."}"**, không phải string — parser phải xử lý.
- **Eval theo ngôn ngữ cần language labels** — data không có field ngôn ngữ.
  Auto-detect (tỉ lệ dấu tiếng Việt/ASCII) hoặc gán thủ công 200 jobs. Ghi vào plan.
- **Stratify 200 jobs / 124 level-3 categories** (~1-2 jobs/cat; top cats
  `281:449, 458:243, 329:215...` lấy theo tỉ lệ + floor 1).
- **Verification 32k candidate IDs**: scroll 1 lần lấy toàn bộ `esco_id` trong
  collection (~3k) vào set rồi check membership — nhanh hơn retrieve từng ID.
- **`matches.jsonl.zst` sort theo `job_id`** để diff được giữa các lần chạy.
- `commonInfo.level/education`, `experience`, `quantity` giữ làm metadata V1,
  là feature V2 (filter/rerank) — không lôi vào query text.
- Desc/requireCandidate không containment nhau (check 300 mẫu = 0) → dedup
  nice-to-have, ưu tiên thấp.

## 4. Quyết định kiến trúc (chốt trước khi code)

**Đặt matching pipeline ở đâu?**

| Option | Ưu | Nhược |
|---|---|---|
| Cùng package + lệnh `match-jobs` | Nhanh, reuse trực tiếp, nhất quán quyết định "fork cho nhanh" | Trộn lifecycle index (theo ESCO release) với matching (theo batch crawl); tuning matching có thể vô tình đổi index `build-id` |
| Package riêng `esco-job-matching` (cùng repo, import reuse core) | Tách lifecycle/config sạch | Thêm 1 package + wiring import |

Đề xuất: **cùng repo, package riêng, import reuse modules core**
(`embedding/tei/normalize/utils/artifacts`) — tuyệt đối không copy-paste
`qdrant_ops/embedding` lần thứ ba.

Khác: đổi `verification-report.json` của matching thành
`match-verification-report.json` để khỏi nhầm với file của index pipeline khi
grep workspace.

## 5. Thứ tự implement đề xuất (sau khi chốt §4)

1. Canonicalization + category join (`externalId`/`parentExternalId`).
2. Query templates (thứ tự sections + budgets + instruction config).
3. TEI/BM25 query encode (với `query_embed` — §2a).
4. Retrieve 3 kênh + RRF fusion + top-10.
5. CLI + artifacts (`manifest.json`, `resolved-config.json`, `canonical-jobs.jsonl.zst`,
   `query-vectors/shard-*.npz`, `matches.jsonl.zst`, `retrieval-report.json`,
   `match-verification-report.json`).
6. Unit/contract/integration/resume tests theo test plan.
7. Chạy batch 3203 jobs → lập tập eval 200 jobs → hiệu chỉnh template/limit/trọng số.
