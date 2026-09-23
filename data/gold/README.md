# Gold set: TopCV → ESCO (200 jobs)

## Files

- `sample.json` — 200 sampled jobs (`sample_gold.py --seed 7`): floor 1/terminal,
  buckets conflict/lexical-miss/noisy-title, language top-up (VI 141 / EN 43 / mixed 16).
- `gold.jsonl` — labels (`job_id, accepted_esco_ids, language, annotator, notes, labeled_at`),
  sorted by `job_id`. Annotator `muse-spark`, human-reviewed afterwards.
- `context_part_*.json` — labeling context dumps (audit only, regenerable).

## Methodology note (pooling bias)

Labels were first assigned from fused top-10 only. A pool-miss audit
(`scripts/probe_pool.py`: re-run stored query vectors at depth 100) then found
10 jobs where the true concept exists in ESCO but ranked outside fused top-10
(mostly label/lexical miss + RRF burial, e.g. mobile application developer at
semantic rank 1 but label rank 49). Those IDs were ADDED with `pool-fix:` notes,
so metrics measure true recall instead of pooled recall:

- 4 jobs moved `none` → answerable-but-buried (honest recall failures at k≤10).
- Remaining `none` jobs are true concept gaps (no ESCO equivalent: comtor/BrSE
  liaison, loss-prevention, CCTV-operator, no-code MMO...).

## Baseline v1 (fused RRF equal weights, top-10 output, build 24bbae641cfd)

Overall R@1 0.605 / R@5 0.895 / R@10 0.950 / MRR 0.723 / zero-hit 5%.
Answerable-only (194): R@10 0.979, zero-hit 2% (= 4 buried-truth jobs).
EN slightly above VI; conflict bucket highest (declared-IT signal validated).

## v2 (semantic weight 2.0, top-20 output, build 8ddeacb204b5)

`gold_v2.jsonl` relabeled from top-20 pools with v1 accepts pre-filled, then
audited against the full local ESCO occupation set. See `gold_v2_audit.md`.
The 2026-09-23 adjudication first corrected six high-confidence rows, then
reviewed all 200 postings for merely-related labels. The reproducible override
set changes 32 additional rows. There are no empty labels; eighteen accepted
job–concept pairs are deliberately outside the v2 top-20 because the candidate
pool must not define the gold boundary.
Overall R@1 0.635 / R@5 0.910 / R@10 0.945 / R@20 0.975 / MRR 0.747 /
zero-hit 2.5%. These figures precede the adjudication and are retained as the
original v2 run record; re-score before comparing a new build.

Gold audit: `uv run python scripts/audit_gold_v2.py`

Primary-label adjudication: `uv run python scripts/adjudicate_primary_gold_v2.py`.
Each row now has `primary_esco_ids`, a strict subset of `accepted_esco_ids`.
Score the strict tier with `--label-field primary_esco_ids`; the default remains
the permissive accepted tier for backward compatibility.

The top-100 rerank review added six valid occupation gaps discovered beyond the
old top-20 pool. Current totals are 361 acceptable and 298 primary labels.

Re-score anytime: `uv run python scripts/score_gold.py [--build-dir ...] [--gold ...]`

## Rerank pilot (mxbai-rerank-large-v2 via vLLM, 200 gold jobs)

`matches_pilot.jsonl.zst` = build 8ddeacb204b5 top-20 + `rerank_score/rerank_rank`
(RRF order kept). Rerank order vs RRF order on gold v2:

| | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| RRF | 0.635 | 0.910 | 0.945 | 0.747 |
| rerank | **0.745** | **0.935** | **0.965** | **0.827** |

Caveat for threshold tuning: scores saturate (~55% of pairs score exactly 1.00;
precision@1.00 ≈ 16%). Reranker is excellent at RANKING, poor as an absolute
accept/reject threshold — prefer rank-cutoff or score-margin rules over raw
thresholds. See `scripts/rerank_pilot.py`.

## Rerank pilot v2 (Qwen3-Reranker-4B, 200 gold jobs)

`matches_pilot_qwen.jsonl.zst` (old mxbai file kept untouched). Same protocol,
`--model Qwen/Qwen3-Reranker-4B --out ..._qwen.jsonl.zst` (~242s, ~17 pairs/s).

| | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| RRF | 0.635 | 0.910 | 0.945 | 0.747 |
| mxbai rerank | 0.745 | 0.935 | 0.965 | 0.827 |
| **Qwen rerank** | **0.810** | **0.965** | **0.970** | **0.878** |

Qwen scores spread across 0.0–1.0 (only 113/4000 ≥ 0.99, nothing pinned at 1.00)
vs mxbai's saturation — threshold tuning is actually feasible with Qwen.
Precision: P@1 0.81 / P@3 0.50 / P@5 0.34 / P@10 0.18.

## Rerank doc experiment: +alternative labels — NEGATIVE, reverted

Hypothesis (confirmed on 1 case): ICT project manager scored 0.16 for a Scrum
Master job because its doc lacked the verbatim alt label "scrum master";
enriched doc scored 0.94. But pilot on 200 gold (Qwen + instruction):
R@1 0.825 → **0.795** (gained 6 jobs, lost 9 — alt-label keyword overlap blurs
fine distinctions, e.g. graphic vs UI designer, food-safety vs QC engineer).
Net wash, not significant → reverted to plain `semantic_text` docs.
Evidence kept: `matches_pilot_qwen_altdoc.jsonl.zst`, `scripts/probe_scrum_doc.py`.

## Rerank instruction (Qwen)

Server template (`qwen3_reranker.jinja`) fills `<Instruct>` from request field
`instruction` → `instruct` → web-search default. Verified live: the server
honors `instruction`, so the client sends it explicitly (`rerank.instruction`,
in fingerprint). Result on gold v2: R@1 0.810 → **0.825**, MRR 0.878 → 0.888.
Stripping the duplicated Harrier prefix from query text: no measurable change
(R@1 0.825 identical) — kept as-is for consistency with dense retrieval.
Pilot files: `matches_pilot_qwen*.jsonl.zst` (no-instruction / +instruction /
stripped variants coexist).

## Experiment: per-channel top-100 (build 2b313c1e6c93) — NEGATIVE

`candidate_limit` 50 → 100, fused top-20 + rerank unchanged. Result on gold v2:
RRF R@1 0.635 → 0.590, rerank R@1 0.825 → 0.775. Deeper pools let marginal
candidates fuse into top-20 and displace good ones (18 jobs lost accepts from
top-20). Offline RRF re-sweep on stored channel evidence: k=10 restores 0.635,
k=120 drops to 0.590 — with deep pools, sharper fusion wins. Lesson: pool depth
and fusion sharpness must be tuned together, not independently.

## Full production build with Qwen rerank (build 9d278cd83fac)

`rerank.enabled: true`, model Qwen3-Rerank-4B in `config/job-matching.yaml`.
All 3203 jobs × top-20 carry `rerank_score/rerank_rank` (64.060 pairs).
Gold v2 on full build: R@1 0.810 / R@5 0.965 / R@10 0.970 / MRR 0.878.

## Experiment: fused top-100 + rerank 100 (build 4aa45473959c) — MIXED

`candidate_limit` 100, fused `top_k` 100, rerank `top_n` 100 (320.300 pairs).
Adopted query shards, retrieve + rerank recomputed. Gold v2:

| | R@1 | R@5 | R@10 | R@50 | R@100 | MRR |
|---|---|---|---|---|---|---|
| RRF order | 0.590 | 0.860 | 0.915 | 0.985 | 0.990 | 0.704 |
| rerank order | 0.765 | 0.955 | 0.980 | 0.990 | 0.990 | 0.847 |

Reading: pool recall essentially solved (R@100 = 0.99 — only true concept gaps
remain beyond rank 100). But rerank top-1 on 100-wide pools (0.765) trails
rerank on 20-wide pools (0.825): more distractors, harder choice. Part of the
gap is measurement bias — new top-ranked faces were never judged (gold made on
top-20 pools). Next step if top-100 output is wanted: targeted adjudication of
new top-1..3 faces, then rescore.

## Threshold sweep (scripts/sweep_threshold.py, Qwen pilot)

Best F1 ≈ 0.625 at rank-cutoff k=2 (P 0.605 / R 0.645) or combined k=3+t=0.8.
Score-only thresholds underperform rank-cutoffs; margin rules have high
precision but unusable recall. Recommended default action: accept rerank top-2.
