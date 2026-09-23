"""Combine fused + rerank scores with per-job normalization (offline analysis).

Fused scores (~0.01-0.05) and rerank scores (~0-1) live on different scales,
so each is min-max normalized within its job before weighting:
    combined = w_rerank * rr_norm + (1 - w_rerank) * fused_norm

Usage:
    uv run python scripts/combine_scores.py --matches-file <matches> --gold <gold>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _jsonl import iter_jsonl_zst


def normalized(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi <= lo:
        return [0.0 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches-file", type=Path, required=True)
    parser.add_argument("--gold", type=Path, default=Path("data/gold/gold_v2.jsonl"))
    args = parser.parse_args()

    gold = {
        json.loads(line)["job_id"]: set(json.loads(line)["accepted_esco_ids"])
        for line in args.gold.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    jobs = []
    for match in iter_jsonl_zst(args.matches_file):
        if match["job_id"] not in gold:
            continue
        fused = [c["fused_score"] for c in match["candidates"]]
        reranked = [c.get("rerank_score") or 0.0 for c in match["candidates"]]
        jobs.append(
            (
                gold[match["job_id"]],
                match["candidates"],
                normalized(fused),
                normalized(reranked),
            )
        )
    print(f"jobs: {len(jobs)}")

    def metrics(order_lists: list[list[str]]) -> dict[str, float]:
        n = len(order_lists)
        out = {}
        for k in (1, 5, 10):
            out[f"R@{k}"] = sum(
                any(e in acc for e in ranked[:k]) for ranked, acc in order_lists
            ) / n
        mrr = 0.0
        for ranked, acc in order_lists:
            first = next(
                (i for i, e in enumerate(ranked, start=1) if e in acc), None
            )
            mrr += 1.0 / first if first else 0.0
        out["MRR"] = round(mrr / n, 4)
        return {k: round(v, 4) for k, v in out.items()}

    for weight in (0.5, 0.7, 0.9, 1.0):
        order_lists = []
        for accepted, candidates, fused_norm, rr_norm in jobs:
            scored = sorted(
                zip(
                    [c["esco_id"] for c in candidates],
                    [
                        weight * r + (1 - weight) * f
                        for f, r in zip(fused_norm, rr_norm, strict=True)
                    ],
                    strict=True,
                ),
                key=lambda item: -item[1],
            )
            order_lists.append(([eid for eid, _ in scored], accepted))
        print(f"w_rerank={weight}:", metrics(order_lists))


if __name__ == "__main__":
    main()
