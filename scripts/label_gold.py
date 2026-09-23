"""Interactive gold labeling: pick acceptable ESCO occupations per job.

Usage:
    python scripts/label_gold.py --annotator sang --limit 20

Reads data/gold/sample.json + matches, appends to data/gold/gold.jsonl.
Resume-safe: already labeled jobs are skipped. Each line:
{job_id, accepted_esco_ids, language, annotator, notes, labeled_at}
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from _jsonl import iter_jsonl_zst

BUILD_DIR = Path("data/job-matching-work/24bbae641cfd")
SAMPLE = Path("data/gold/sample.json")
GOLD = Path("data/gold/gold.jsonl")


def load_matches() -> dict[str, dict]:
    return {
        record["job_id"]: record
        for record in iter_jsonl_zst(BUILD_DIR / "matches.jsonl.zst")
    }


def load_canonical() -> dict[str, dict]:
    return {
        record["job_id"]: record
        for record in iter_jsonl_zst(BUILD_DIR / "canonical-jobs.jsonl.zst")
    }


def load_labeled() -> dict[str, dict]:
    out = {}
    if GOLD.is_file():
        for line in GOLD.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                out[record["job_id"]] = record
    return out


def show_job(job_id: str, job: dict, match: dict) -> None:
    print("=" * 100)
    print(f"JOB {job_id}: {job.get('title')}")
    paths = [
        " > ".join(node["name"] for node in p.get("declared_path", []))
        for p in job.get("it_category_paths", [])
    ]
    print(f"CATS: {' | '.join(paths) if paths else '-'}")
    desc = (job.get("description") or "")[:700]
    req = (job.get("require_candidate") or "")[:700]
    print(f"DESC: {desc}")
    print(f"REQ:  {req}")
    print("-" * 100)
    for candidate in match.get("candidates", []):
        channels = candidate["channels"]
        evidence = ", ".join(
            f"{name.split('_')[0]}={channels[name]['rank'] if channels[name] else '-'}"
            for name in ("label_dense", "semantic_dense", "lexical_sparse")
        )
        print(f"  {candidate['rank']}. {candidate['preferred_label']}")
        print(f"     fused={candidate['fused_score']:.4f} ({evidence})")
        print(f"     {candidate['path_text'][:140]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotator", required=True)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))
    matches = load_matches()
    canonical = load_canonical()
    labeled = load_labeled()

    done = 0
    for entry in sample:
        job_id = entry["job_id"]
        if job_id in labeled:
            continue
        job = canonical.get(job_id, {})
        match = matches.get(job_id)
        if match is None:
            print(f"SKIP {job_id}: no match record")
            continue
        show_job(job_id, job, match)
        print(f"hint: language={entry.get('language_hint')} strata={entry.get('strata')}")

        while True:
            raw = input("Accept ranks (e.g. '1 3' | 'none'): ").strip().lower()
            if raw in ("none", ""):
                accepted: list[str] = []
                break
            try:
                ranks = [int(part) for part in raw.split()]
            except ValueError:
                print("  invalid input, try again")
                continue
            by_rank = {c["rank"]: c["esco_id"] for c in match["candidates"]}
            if any(rank not in by_rank for rank in ranks):
                print(f"  ranks must be in 1..{len(by_rank)}")
                continue
            accepted = [by_rank[rank] for rank in sorted(set(ranks))]
            break
        language = (
            input(f"Language [{entry.get('language_hint')}]: ").strip().lower()
            or entry.get("language_hint")
        )
        notes = input("Notes (optional): ").strip()
        record = {
            "job_id": job_id,
            "accepted_esco_ids": accepted,
            "language": language,
            "annotator": args.annotator,
            "notes": notes,
            "labeled_at": datetime.now(UTC).isoformat(),
        }
        GOLD.parent.mkdir(parents=True, exist_ok=True)
        with GOLD.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        done += 1
        print(f"saved {done} (total labeled: {len(labeled) + done})")
        if done >= args.limit:
            break


if __name__ == "__main__":
    main()
