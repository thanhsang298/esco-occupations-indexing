"""Sweep accept-thresholds on reranked matches to support action policy decisions.

Policies evaluated per job with top-20 reranked candidates:
- rank-cutoff k: accept rerank ranks <= k
- score-threshold t: accept rerank_score >= t
- margin m: accept rank-1 iff (score1 - score2) >= m, else accept nothing
- combined: accept ranks <= k AND score >= t

Usage:
    uv run python scripts/sweep_threshold.py --matches-file data/gold/matches_pilot_qwen.jsonl.zst
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _jsonl import iter_jsonl_zst

GOLD = Path("data/gold/gold_v2.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches-file", type=Path, required=True)
    parser.add_argument("--gold", type=Path, default=GOLD)
    args = parser.parse_args()

    gold = {
        json.loads(line)["job_id"]: set(json.loads(line)["accepted_esco_ids"])
        for line in args.gold.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    jobs: list[tuple[set[str], list[tuple[str, float]]]] = []
    for match in iter_jsonl_zst(args.matches_file):
        if match["job_id"] not in gold:
            continue
        ranked = sorted(
            [c for c in match["candidates"] if c.get("rerank_score") is not None],
            key=lambda c: (c["rerank_rank"], c["rank"]),
        )
        jobs.append((gold[match["job_id"]], [(c["esco_id"], c["rerank_score"]) for c in ranked]))

    def evaluate(accept_fn) -> dict[str, float]:
        tp = fp = fn = 0
        jobs_with_accept = 0
        for accepted, ranked in jobs:
            chosen = accept_fn(ranked)
            hits = len(set(chosen) & accepted)
            tp += hits
            fp += len(chosen) - hits
            fn += len(accepted - set(chosen))
            if chosen:
                jobs_with_accept += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        return {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(2 * precision * recall / (precision + recall), 3)
            if precision + recall
            else 0.0,
            "coverage": round(jobs_with_accept / len(jobs), 3),
        }

    print("== rank-cutoff (accept rerank ranks <= k) ==")
    for k in (1, 2, 3, 5, 10):
        result = evaluate(lambda ranked, k=k: [eid for eid, _ in ranked[:k]])
        print(f"  k={k:<2d}", result)

    print("== score-threshold (accept score >= t) ==")
    for threshold in (0.9, 0.8, 0.7, 0.5, 0.3):
        result = evaluate(
            lambda ranked, threshold=threshold: [eid for eid, s in ranked if s >= threshold]
        )
        print(f"  t={threshold:<4}", result)

    print("== margin (accept rank-1 iff s1-s2 >= m) ==")
    for margin in (0.2, 0.1, 0.05, 0.01):
        def accept_margin(ranked, margin=margin):
            if len(ranked) < 2:
                return [ranked[0][0]] if ranked else []
            return [ranked[0][0]] if ranked[0][1] - ranked[1][1] >= margin else []

        print(f"  m={margin:<4}", evaluate(accept_margin))

    print("== combined (rank <= 3 AND score >= t) ==")
    for threshold in (0.8, 0.5, 0.3):
        result = evaluate(
            lambda ranked, threshold=threshold: [
                eid for eid, s in ranked[:3] if s >= threshold
            ]
        )
        print(f"  k=3,t={threshold:<4}", result)


if __name__ == "__main__":
    main()
