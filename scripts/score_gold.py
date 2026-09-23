"""Score matches against gold labels: Recall@1/5/10, MRR@10, slices.

Usage:
    python scripts/score_gold.py
    python scripts/score_gold.py --build-dir data/job-matching-work/ID \
        --gold data/gold/gold_v2.jsonl

Reads a gold file + matches of a build dir. Prints overall metrics plus slices
by language, conflict bucket and answerable-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from _jsonl import iter_jsonl_zst

DEFAULT_BUILD_DIR = Path("data/job-matching-work/24bbae641cfd")
DEFAULT_GOLD = Path("data/gold/gold.jsonl")
SAMPLE = Path("data/gold/sample.json")


def load_matches(build_dir: Path) -> dict[str, dict]:
    return {
        record["job_id"]: record
        for record in iter_jsonl_zst(build_dir / "matches.jsonl.zst")
    }


def load_matches_file(path: Path) -> dict[str, dict]:
    return {record["job_id"]: record for record in iter_jsonl_zst(path)}


def order_candidates(match: dict, order: str) -> list[str]:
    if order == "rrf":
        return [c["esco_id"] for c in match["candidates"]]
    scored = [c for c in match["candidates"] if c.get("rerank_score") is not None]
    if len(scored) != len(match["candidates"]):
        raise ValueError(f"Missing rerank fields for {match['job_id']}")
    return [
        c["esco_id"]
        for c in sorted(scored, key=lambda c: (c["rerank_rank"], c["rank"]))
    ]


def metrics(pairs: list[tuple[list[str], list[str]]]) -> dict[str, float]:
    """pairs: (ranked esco_ids, accepted esco_ids)."""
    if not pairs:
        return {}
    recalls = {1: 0, 5: 0, 10: 0, 20: 0, 50: 0, 100: 0}
    firsts: list[int | None] = []
    rrs = []
    for ranked, accepted in pairs:
        accepted_set = set(accepted)
        first = next(
            (rank for rank, esco_id in enumerate(ranked, start=1) if esco_id in accepted_set),
            None,
        )
        firsts.append(first)
        rrs.append(1.0 / first if first else 0.0)
        for k in recalls:
            if any(esco_id in accepted_set for esco_id in ranked[:k]):
                recalls[k] += 1
    n = len(pairs)
    return {
        "n": n,
        **{f"recall@{k}": recalls[k] / n for k in recalls},
        "mrr@10": mean(1.0 / first if first and first <= 10 else 0.0 for first in firsts),
        "mrr@20": mean(1.0 / first if first and first <= 20 else 0.0 for first in firsts),
        "mrr@100": mean(1.0 / first if first and first <= 100 else 0.0 for first in firsts),
        "zero_hit": sum(1 for value in rrs if value == 0.0) / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--matches-file", type=Path, default=None)
    parser.add_argument("--order", choices=["rrf", "rerank"], default="rrf")
    parser.add_argument(
        "--label-field",
        choices=["accepted_esco_ids", "primary_esco_ids"],
        default="accepted_esco_ids",
    )
    args = parser.parse_args()
    gold_path: Path = args.gold
    matches_path: Path = args.matches_file or (args.build_dir / "matches.jsonl.zst")
    print(
        f"order: {args.order}, labels: {args.label_field}, "
        f"matches: {matches_path}, gold: {gold_path}"
    )
    if not gold_path.is_file():
        print("no gold labels yet — run scripts/label_gold.py first")
        return
    gold = [
        json.loads(line)
        for line in gold_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matches = (
        load_matches_file(matches_path)
        if args.matches_file
        else load_matches(args.build_dir)
    )
    sample_info = {
        entry["job_id"]: entry for entry in json.loads(SAMPLE.read_text(encoding="utf-8"))
    }
    pairs = []
    for entry in gold:
        match = matches.get(entry["job_id"])
        if match is None:
            continue
        ranked = order_candidates(match, args.order)
        enriched = {**(sample_info.get(entry["job_id"]) or {}), **entry}
        if args.label_field not in entry:
            raise ValueError(f"Missing {args.label_field} for {entry['job_id']}")
        pairs.append((entry["job_id"], ranked, entry[args.label_field], enriched))

    print(f"labeled: {len(pairs)}")
    rows = [(ranked, accepted) for _, ranked, accepted, _ in pairs]
    print("overall:", json.dumps(metrics(rows), indent=1))

    by_lang: dict[str, list] = {}
    for _, ranked, accepted, entry in pairs:
        by_lang.setdefault(entry.get("language", "?"), []).append((ranked, accepted))
    for lang, subset in sorted(by_lang.items()):
        print(f"lang={lang}:", json.dumps(metrics(subset)))

    conflict = [(r, a) for _, r, a, e in pairs if e.get("conflict")]
    if conflict:
        print("conflict-bucket:", json.dumps(metrics(conflict)))
    non_empty = [(r, a) for _, r, a, _ in pairs if a]
    print("answerable-only:", json.dumps(metrics(non_empty)))


if __name__ == "__main__":
    main()
