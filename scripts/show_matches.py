"""Inspect TopCV → ESCO match results.

Usage:
    python scripts/show_matches.py --job 1305294
    python scripts/show_matches.py --job "data analyst" --limit 5
    python scripts/show_matches.py --job "data analyst" --limit 5 --top-k 10 --full
    python scripts/show_matches.py --matches-file data/gold/matches_pilot.jsonl.zst \\
        --job 1305294 --order rerank --top-k 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _jsonl import iter_jsonl_zst

BUILD_DIR = Path("data/job-matching-work/8ddeacb204b5")


def load_titles() -> dict[str, str]:
    jobs = json.loads(
        Path("data/iviec-job-crawler.job_details.topcv.it.json").read_text(
            encoding="utf-8"
        )
    )
    return {
        f"topcv:{j['externalId']}": j.get("title") or ""
        for j in jobs
        if j.get("status") == "completed"
    }


def iter_matches(path):
    yield from iter_jsonl_zst(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True, help="externalId or title substring")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--matches-file", type=Path, default=None)
    parser.add_argument("--order", choices=["rrf", "rerank"], default="rrf")
    args = parser.parse_args()
    matches_path = args.matches_file or (BUILD_DIR / "matches.jsonl.zst")

    titles = load_titles()
    query = args.job.strip()
    wanted = {f"topcv:{query}"} if query.isdigit() else {
        job_id
        for job_id, title in titles.items()
        if query.casefold() in title.casefold()
    }
    shown = 0
    for match in iter_matches(matches_path):
        if match["job_id"] not in wanted:
            continue
        candidates = match["candidates"]
        if args.order == "rerank":
            candidates = sorted(
                [c for c in candidates if c.get("rerank_rank") is not None],
                key=lambda c: (c["rerank_rank"], c["rank"]),
            )
        print("=" * 100)
        print(f"JOB {match['job_id']}: {titles.get(match['job_id'], '?')}")
        print(f"esco_build={match['esco_build_id']} collection={match['esco_collection']}")
        for candidate in candidates[: args.top_k]:
            channels = candidate["channels"]
            evidence = ", ".join(
                f"{name.split('_')[0]}="
                f"{channels[name]['rank'] if channels[name] else '-'}"
                for name in ("label_dense", "semantic_dense", "lexical_sparse")
            )
            rerank = ""
            if candidate.get("rerank_score") is not None:
                rerank = f" rr={candidate['rerank_rank']} rs={candidate['rerank_score']:.4f}"
            print(f"  {candidate['rank']}. {candidate['preferred_label']}")
            score = candidate["fused_score"]
            print(f"     isco={candidate['isco_code']} fused={score:.4f} ({evidence}){rerank}")
            print(f"     {candidate['path_text'][:130]}")
            if args.full:
                print(f"     {candidate['esco_uri']}")
        shown += 1
        if shown >= args.limit:
            break
    if not shown:
        print(f"No match found for {args.job!r}")


if __name__ == "__main__":
    main()
