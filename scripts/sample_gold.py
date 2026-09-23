"""Sample ~200 gold jobs stratified by terminal IT category + special buckets.

Usage:
    python scripts/sample_gold.py --seed 7 --n 200

Output: data/gold/sample.json — list of {job_id, strata, language_hint}.
Strata: proportional terminal-category base (floor 1) + conflict bucket +
lexical-miss bucket + noisy-title bucket + language top-up.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from _jsonl import iter_jsonl_zst

BUILD_DIR = Path("data/job-matching-work/24bbae641cfd")
JOBS_FILE = Path("data/iviec-job-crawler.job_details.topcv.it.json")
OUT = Path("data/gold/sample.json")

VI_MARKS = set("ăâđêôơưĂÂĐÊÔƠƯáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")
NOISE_PAT = re.compile(
    r"\[.*?\]|thu nhập|lương|\d+\s*(tr|triệu)|hà nội|hồ chí minh|đà nẵng|tuyển dụng|\bgấp\b",
    re.I,
)


def lang_hint(text: str) -> str:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "empty"
    ratio = sum(1 for c in letters if c in VI_MARKS) / len(letters)
    if ratio > 0.15:
        return "vi"
    if ratio > 0.02:
        return "mixed"
    return "en"


def load_canonical() -> dict[str, dict]:
    return {
        record["job_id"]: record
        for record in iter_jsonl_zst(BUILD_DIR / "canonical-jobs.jsonl.zst")
    }


def lexical_miss_jobs() -> set[str]:
    return {
        match["job_id"]
        for match in iter_jsonl_zst(BUILD_DIR / "matches.jsonl.zst")
        if not any(c["channels"]["lexical_sparse"] for c in match["candidates"])
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    canonical = load_canonical()
    miss = lexical_miss_jobs()

    terminal_of: dict[str, str] = {}
    conflict: set[str] = set()
    noisy: set[str] = set()
    lang_of: dict[str, str] = {}
    for job_id, record in canonical.items():
        declared_ids = [
            (p["declared_path"][-1]["external_id"] if p["declared_path"] else p["observed"]["key"])
            for p in record["it_category_paths"]
        ]
        terminal_of[job_id] = declared_ids[0] if declared_ids else "none"
        if any(p["root_conflict"] for p in record["it_category_paths"]):
            conflict.add(job_id)
        if NOISE_PAT.search(record["title"] or ""):
            noisy.add(job_id)
        lang_of[job_id] = lang_hint(
            (record["title"] or "") + " " + (record["description"] or "")
        )

    by_terminal: dict[str, list[str]] = {}
    for job_id, terminal in terminal_of.items():
        by_terminal.setdefault(terminal, []).append(job_id)

    picked: dict[str, list[str]] = {}  # job_id -> strata tags
    # 1. Floor 1 per terminal (always kept).
    for terminal, ids in by_terminal.items():
        choice = rng.choice(ids)
        picked.setdefault(choice, []).append(f"terminal:{terminal}")

    # 2. Buckets (up to N each, tagged).
    def add_bucket(candidates: set[str], tag: str, count: int) -> None:
        options = [jid for jid in candidates if jid not in picked]
        rng.shuffle(options)
        for jid in options[:count]:
            picked.setdefault(jid, []).append(tag)

    add_bucket(conflict, "conflict", 15)
    add_bucket(miss, "lexical-miss", 10)
    add_bucket(noisy, "noisy-title", 15)

    # 3. Language minimums BEFORE proportional fill (nothing removed afterwards).
    def need(wanted: str, minimum: int) -> list[str]:
        have = sum(1 for jid in picked if lang_of[jid] == wanted)
        options = [jid for jid, hint in lang_of.items() if hint == wanted and jid not in picked]
        rng.shuffle(options)
        return options[: max(0, minimum - have)]

    for jid in need("en", 40) + need("mixed", 15):
        picked.setdefault(jid, []).append("language-topup")
    assert len(picked) <= args.n, "floor+buckets+topups exceed budget"

    # 4. Proportional fill to exactly n (adds only).
    while len(picked) < args.n:
        remaining = [jid for jid in canonical if jid not in picked]
        if not remaining:
            break
        weights_fill = [len(by_terminal[terminal_of[jid]]) for jid in remaining]
        choice = rng.choices(remaining, weights=weights_fill, k=1)[0]
        picked.setdefault(choice, []).append(f"terminal:{terminal_of[choice]}")

    sample = [
        {
            "job_id": job_id,
            "strata": sorted(tags),
            "terminal": terminal_of[job_id],
            "language_hint": lang_of[job_id],
            "conflict": job_id in conflict,
        }
        for job_id, tags in sorted(picked.items())
    ]
    assert len(sample) == args.n, f"expected {args.n}, got {len(sample)}"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(sample, ensure_ascii=False, indent=1), encoding="utf-8")

    langs = Counter(item["language_hint"] for item in sample)
    terms = Counter(item["terminal"] for item in sample)
    print(f"sampled {len(sample)} jobs -> {OUT}")
    print("languages:", dict(langs))
    print("terminals covered:", len(terms))
    print("conflict jobs:", sum(1 for item in sample if item["conflict"]))


if __name__ == "__main__":
    main()
