# Review (v2): Revised plan TopCV Job → ESCO Occupation Retrieval

Ngày review: 2026-09-22
Phạm vi: bản revised — submodule `esco_occupation_indexer.matching`,
pipeline `ingest → validate → embed-queries → retrieve → verify`,
work dir `data/job-matching-work/<match-build-id>/`.
Đối chiếu với: dữ liệu TopCV thật, taxonomy TopCV thật, source FastEmbed 0.8.0.

## Verdict

**Đạt để implement** sau khi xử lý §1 (sửa số liệu), §3 (sort sparse indices —
must-fix duy nhất còn lại), §4 (4 điểm siết). Mọi lo ngại lớn ở review v1 đã
được bản revised giải quyết đúng hướng.

## 1. Claim taxonomy: kết luận đúng, con số sai

| Claim | Verify |
|---|---|
| 63 node level-3 dưới root 257 | Đúng chính xác |
| Posting dùng "80 key" | Sai — thực tế **134 distinct `key`** (71 key canonical-root 257 + 63 key root khác) |

Bằng chứng mâu thuẫn thực tế còn nặng hơn mô tả, củng cố kết luận
"taxonomy là truth, stratify sau canonicalization":

- **221 category rows** khai `level1Id=257` nhưng canonical root tra từ taxonomy
  là root khác (vd `key 1039 → root 826`, `304/303 → root 644`).
- **570 rows** có `key != level3Id`.
- Mọi `key` đều resolve được (0 missing) — vấn đề là posting tự khai path sai,
  không phải thiếu node.
- Canonical path dựng chuẩn, vd:
  `Công nghệ Thông Tin[257] > IT Project Management[265] > IT Comtor[1021]`.

Action: sửa "80" → "134 (71 IT + 63 non-IT)", đưa 2 metric 221/570 vào
validation-report spec như data-quality indicators.
`_id.$oid` và `updatedAt.$date` đã verify format.

## 2. Phản hồi các điểm revised điều chỉnh ngược

### 2a. Submodule `matching` thay vì package riêng — chấp nhận

Guardrail duy nhất: matching được import
`artifacts/utils/hashing/embedding/tei/normalize/settings`, nhưng **không import**
`ingest/validation/qdrant_ops` của index (trừ khi tách core dùng chung sau).
Stage names trùng chuỗi (`ingest/validate/verify`) OK vì manifest class riêng.

### 2b. Last-token pooling — đúng một nửa, không cần tranh luận thêm

Đúng là toàn bộ context vẫn được attention tổng hợp. Nhưng thứ tự section vẫn
phải khóa vì: (1) right-truncation quyết định section nào mất khi quá 512 tokens
— đó là information loss; (2) recency bias đo được trên decoder-only models.
Plan đã khóa order + truncation → yêu cầu coi như được đáp ứng, để eval phân xử.

## 3. Must-fix: sparse query indices có thể chưa sort

Source `Bm25.query_embed()` (fastembed 0.8.0):

```python
token_ids = np.array(list(set(self.compute_token_id(...) ...)))
```

`set → list` **không đảm bảo thứ tự tăng dần**, trong khi Qdrant yêu cầu sparse
indices sorted (cả store lẫn query). Path index-time (`embed()`) đã pass verify
nên OK; path query là code mới, chưa được chứng minh.

Action:

- Trong `query_encode()`: sort indices sau `query_embed()` (giữ values tương ứng).
- Contract test assert indices strictly ascending cho mọi query, gồm case term
  lặp (đã có trong plan) + case nhiều tokens.

## 4. Siết thêm (không blocker)

1. **Fingerprint phải bao gồm instruction text nguyên văn**, không chỉ template
   version. Đổi 1 từ trong instruction → embedding khác → query shard cũ invalid.
   Quy tắc reuse: **strict equality toàn bộ fingerprint**, không "partial compat".
2. **Alias switch giữa chừng**: resolve alias ở đầu run và snapshot collection
   name; stage `verify` resolve lại và bắt bằng đúng snapshot (kẻo ai đó promote
   index mới giữa batch 3203 jobs).
3. **`resolved-config.json` của matching** phải ghi: esco-build-dir, collection
   name đã snapshot, compat report (`configured_revision_verified` hay SHA-matched),
   query/fusion fingerprint — audit trail khi đọc lại kết quả sau này.
4. **Rủi ro cross-lingual nặng hơn plan mô tả**: terminal category names trong
   taxonomy là **tiếng Việt**, ESCO labels tiếng Anh — `label_query` gần như là
   truy vấn VI→EN thuần. Nếu Recall@1 thấp, ứng viên calibration đầu tiên nên là
   "thêm EN gloss cho 71 terminal IT categories" (làm 1 lần), trước cả
   title-denoising.
5. **Latency/ổn định hạ tầng**: 3203 jobs × (2 TEI embeds + 1 sparse + 3 Qdrant
   queries, batched). TEI forward `localhost:8787` đã chập chờn 1 lần — giữ
   checkpoint/resume theo shard đúng như plan, cộng retry quanh Qdrant batch
   (đã có tiền lệ `WriteError` ở batch 128 bên index).

## 5. Đã giải quyết đúng từ review v1 (không cần bàn thêm)

- `query_embed()` + contract test weights toàn 1.0.
- `--esco-build-dir` làm nguồn truth compat; resume key gồm job `content_hash` +
  esco build id + config fingerprint; đổi esco build → re-retrieve toàn bộ.
- Tie-break 5 keys + `null → +∞`.
- `matches.jsonl.zst` sort theo `job_id`; `match-verification-report.json` tách tên.
- Eval stratify sau canonical + manual language labels (không detector ở prod) +
  bucket unresolved/inconsistent.
- `content_hash` bao gồm version của normalization/query templates.
- TEI backend tái sử dụng từ resolved ESCO settings (kèm model identity check);
  cách xử lý nuance SHA (`configured_revision_verified` khi thiếu runtime SHA)
  là hợp lý.
