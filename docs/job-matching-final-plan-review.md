# Review (v3): Final plan TopCV Job → ESCO Occupation Retrieval

Ngày review: 2026-09-22
Phạm vi: bản final — submodule `esco_occupation_indexer.matching`,
pipeline `ingest → validate → embed_queries → retrieve → verify`,
label query 128 tokens, semantic query 1024 tokens, RRF top-50 → top-10.
Đối chiếu với: dữ liệu TopCV + taxonomy thật, qdrant-client 1.19.1 (installed),
TEI serving (`max_input_length=16384`).

## Verdict

**Approve để implement.** Bản final đã xử lý đúng toàn bộ must-fix từ review v1/v2.
Còn lại 1 hiệu chỉnh số liệu (§1), 1 ghi nhận trade-off về section order (§3),
và 2 bổ sung nhỏ cho report (§4). Không còn blocker.

## 1. Đối chiếu số liệu: 211 đúng theo định nghĩa hẹp, 221 theo định nghĩa rộng

Kết quả đếm lại trên 3710 category rows (3203 jobs completed):

| Định nghĩa | Số rows | Số jobs |
|---|---|---|
| Khai `level1Id=257` nhưng canonical root khác | **211** | 211 |
| Khai non-IT (`417/1/1042/92`) nhưng canonical root khác nữa | 10 | 10 |
| Tổng mismatch (canonical root ≠ declared level1Id) | 221 | 221 |

Bản final ghi "211" là **chính xác** theo định nghĩa của chính nó
("khai 257 nhưng canonical thuộc ngành khác") — số 221 ở review v2 của tôi dùng
định nghĩa rộng hơn. Không ai sai, chỉ khác định nghĩa.

Action duy nhất: counter spec trong validation-report nên ghi cả hai số
(`declared_it_canonical_other=211`, `cross_non_it_mismatch=10`) thay vì một số
duy nhất, để lần sau không phải tranh cãi lại. `key != level3Id = 570` khớp.

## 2. Claim sparse-sort đã được verify trong installed client

`qdrant_client/local/sparse.py` (v1.19.1) có `sort_sparse_vector()` dùng
`np.argsort` — đúng như link plan trích. Kết luận giữ nguyên và là kết luận
đúng: **sort tại application boundary**, không phụ thuộc khác biệt local/remote.
Contract test "indices strictly ascending" vẫn bắt buộc.

## 3. Query max tokens độc lập — đồng ý, kèm 1 trade-off cần ghi nhận

Về kỹ thuật, tách query max tokens khỏi document max tokens là đúng:

- Qdrant chỉ lưu vector cố định 1024-dim, không biết token limit nào đã sinh ra nó.
- TEI serving `max_input_length=16384` nên 1024 (thậm chí 2048 ở vòng calibration)
  đều hợp lệ. Compat = model/revision/dimension là đủ.

Trade-off nằm ở **section order mới của semantic query**:

```text
Title → IT category context → Job responsibilities (description) → Candidate requirements
```

Với right truncation, khi quá 1024 tokens thì **requirements bị cắt trước,
description được giữ**. Requirements (list skills, năm kinh nghiệm) thường là
tín hiệu discriminative nhất cho occupation mapping, trong khi description thường
dài và boilerplate. Đây có thể là lựa chọn có chủ ý (description mang context
công việc) hoặc ngẫu nhiên — plan không nói rõ.

Action (không đổi code V1, chỉ đổi report + eval):

- `retrieval-report.json` ghi **per-section truncation** (mỗi job: section nào bị
  cắt bao nhiêu tokens, truncation rate toàn batch theo section), thay vì chỉ
  token count/truncation chung chung. Không có số này thì calibration #2 mù.
- Coi order hiện tại là **baseline A**; calibration so sánh với **baseline B**
  (requirements trước description) **kết hợp** với vòng 512/1024/2048, vì token
  limit và section order tương tác nhau (limit càng nhỏ, order càng quyết định).

Label query 128 tokens là hợp lý (title + vài category names không bao giờ vượt;
128 còn dư cho instruction ~25 tokens).

## 4. Bổ sung nhỏ (non-blocking, đưa vào lúc implement)

1. **Retrieval shard meta** (`retrieval/shard-*.jsonl.zst` là bổ sung tốt cho
   resume) phải ghi cả query-shard checksum lẫn collection snapshot — fingerprint
   equality toàn cục đã bao hàm, nhưng ghi explicit giúp debug khi resume skip.
2. Test "đổi 1 ký tự instruction → invalidate query shards" là test hay, khóa
   đúng yêu cầu fingerprint-chứa-instruction-text. Thêm test đối xứng:
   đổi RRF constant/weights chỉ invalidate retrieval, **không** invalidate query
   vectors (đắt nhất là TEI embed — đừng bắt embed lại khi chỉ chỉnh fusion).

## 5. Đã chốt đúng, không ý kiến thêm

- Submodule + import boundary (không import `ingest/validation/qdrant_ops` của index).
- `--esco-build-dir` bắt buộc; snapshot alias đầu run + re-check trước/sau
  retrieval và tại verify (fail run nếu alias đổi).
- Fingerprint chứa giá trị thực (instruction nguyên văn, templates, order,
  limits, model, sparse config, limits, RRF, top-k, esco build + snapshot).
- `query_encode()` 6 bước + `query_embed()` + values toàn 1.
- Tie-break 5 keys, `null → +∞`, top-10, không threshold, rerank tắt.
- Matches sort theo `job_id`; `match-verification-report.json` tách tên.
- Eval: stratify sau canonical, bucket inconsistent, multi-label ESCO chấp nhận,
  manual language labels, calibration order 1→6 (gloss có guard metric, đúng).
