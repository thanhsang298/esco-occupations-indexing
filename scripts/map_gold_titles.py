"""Enrich gold_v2 labels with human-readable titles for review.

ESCO ids alone are hard to review, so this adds:
  - job_title (from canonical-jobs / context_v2 fallback / raw TopCV json)
  - accepted_esco / accepted_esco_titles (preferred_label from ESCO build)
  - primary_esco / primary_esco_titles (for primary_esco_ids, if present)

Usage:
    .venv/bin/python scripts/map_gold_titles.py
    .venv/bin/python scripts/map_gold_titles.py --gold data/gold/gold_v2.jsonl \\
        --out-jsonl data/gold/gold_v2.readable.jsonl --out-md data/gold/gold_v2.readable.md

Outputs:
    data/gold/gold_v2.readable.jsonl — same rows + job_title, accepted_esco,
      accepted_esco_titles, primary_esco, primary_esco_titles
    data/gold/gold_v2.readable.md     — markdown table for quick review
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from _jsonl import iter_jsonl_zst  # noqa: E402

GOLD = ROOT / "data/gold/gold_v2.jsonl"
ESCO = ROOT / "data/work/9ac16776cc2c/canonical-occupations.jsonl.zst"
BUILD_WORK = ROOT / "data/job-matching-work"
GOLD_DIR = ROOT / "data/gold"
RAW_JOBS = ROOT / "data/iviec-job-crawler.job_details.topcv.it.json"


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_esco_titles(esco_path: Path) -> dict[str, dict]:
    """Return {esco_id: {title, code}} from canonical-occupations."""
    out: dict[str, dict] = {}
    for row in iter_jsonl_zst(esco_path):
        out[str(row["esco_id"])] = {
            "title": row.get("preferred_label") or "",
            "code": row.get("code") or "",
        }
    return out


def load_job_titles() -> dict[str, str]:
    """Merge job_id -> title from canonical-jobs builds, with fallbacks."""
    titles: dict[str, str] = {}
    # 1. Primary: canonical-jobs in every build dir (newest wins).
    for path in sorted(BUILD_WORK.glob("*/canonical-jobs.jsonl.zst")):
        try:
            for row in iter_jsonl_zst(path):
                if row.get("title"):
                    titles[str(row["job_id"])] = str(row["title"])
        except Exception as exc:  # noqa: BLE001 — one bad shard shouldn't kill the run
            print(f"warn: skip {path}: {exc}", file=sys.stderr)
    if titles:
        return titles
    # 2. Fallback: context_v2 files already carry job titles.
    for path in sorted(GOLD_DIR.glob("context_v2_part_*.json")):
        for row in json.loads(path.read_text(encoding="utf-8")):
            if row.get("job_id") and row.get("title"):
                titles.setdefault(str(row["job_id"]), str(row["title"]))
    if titles:
        return titles
    # 3. Last resort: raw crawler dump.
    if RAW_JOBS.is_file():
        for job in json.loads(RAW_JOBS.read_text(encoding="utf-8")):
            if job.get("status") == "completed" and job.get("externalId"):
                titles.setdefault(
                    f"topcv:{job['externalId']}", str(job.get("title") or "")
                )
    return titles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=GOLD)
    parser.add_argument("--esco", type=Path, default=ESCO)
    parser.add_argument("--out-jsonl", type=Path, default=GOLD_DIR / "gold_v2.readable.jsonl")
    parser.add_argument("--out-md", type=Path, default=GOLD_DIR / "gold_v2.readable.md")
    parser.add_argument("--no-md", action="store_true", help="skip markdown output")
    args = parser.parse_args()

    if not args.gold.is_file():
        raise SystemExit(f"gold file not found: {args.gold}")
    if not args.esco.is_file():
        raise SystemExit(f"ESCO file not found: {args.esco}")

    gold = load_jsonl(args.gold)
    esco = load_esco_titles(args.esco)
    job_titles = load_job_titles()

    missing_esco: set[str] = set()
    missing_job: list[str] = []
    primary_outside_accepted: list[str] = []
    enriched: list[dict] = []

    def resolve(ids: list[str]) -> list[dict]:
        details = []
        for esco_id in ids:
            info = esco.get(esco_id)
            if info is None:
                missing_esco.add(esco_id)
                details.append({"esco_id": esco_id, "title": "??MISSING??", "code": ""})
            else:
                details.append(
                    {"esco_id": esco_id, "title": info["title"], "code": info["code"]}
                )
        return details

    for row in gold:
        accepted_ids = [str(x) for x in row.get("accepted_esco_ids", [])]
        primary_ids = [str(x) for x in row.get("primary_esco_ids", [])]
        accepted = resolve(accepted_ids)
        primary = resolve(primary_ids)
        if any(pid not in set(accepted_ids) for pid in primary_ids):
            primary_outside_accepted.append(str(row.get("job_id", "")))
        job_id = str(row.get("job_id", ""))
        job_title = job_titles.get(job_id, "")
        if not job_title:
            missing_job.append(job_id)
        enriched.append(
            {
                **row,
                "job_title": job_title,
                "accepted_esco": accepted,
                "accepted_esco_titles": [d["title"] for d in accepted],
                "primary_esco": primary,
                "primary_esco_titles": [d["title"] for d in primary],
            }
        )

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as fh:
        for row in enriched:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    if not args.no_md:
        lines = [
            "# gold_v2 — readable review",
            "",
            f"source: `{args.gold.name}` | rows: {len(enriched)}",
            "",
            "| job_id | job_title | primary_esco_titles | accepted_esco_titles | notes |",
            "| --- | --- | --- | --- | --- |",
        ]
        for row in enriched:
            primary_set = set(row.get("primary_esco_ids", []) or [])
            accepted_cells = "<br>".join(
                f"{'★ ' if d['esco_id'] in primary_set else ''}"
                f"{d['title']} `[{d['esco_id'][:8]}]`"
                for d in row["accepted_esco"]
            )
            primary_cells = "<br>".join(
                f"{d['title']} `[{d['esco_id'][:8]}]`" for d in row["primary_esco"]
            )
            notes = str(row.get("notes") or "").replace("|", "\\|").replace("\n", " ")
            if len(notes) > 160:
                notes = notes[:157] + "..."
            lines.append(
                f"| {row['job_id']} "
                f"| {str(row['job_title']).replace('|', chr(92) + '|')} "
                f"| {primary_cells} | {accepted_cells} | {notes} |"
            )
        args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {len(enriched)} rows -> {args.out_jsonl}")
    if not args.no_md:
        print(f"wrote review table -> {args.out_md}")
    if missing_esco:
        print(
            f"warn: {len(missing_esco)} ESCO id(s) not in {args.esco.name}: "
            + ", ".join(sorted(missing_esco)[:10]),
            file=sys.stderr,
        )
    if missing_job:
        print(
            f"warn: {len(missing_job)} job_id(s) without title "
            f"(e.g. {', '.join(missing_job[:5])})",
            file=sys.stderr,
        )
    if primary_outside_accepted:
        print(
            f"warn: {len(primary_outside_accepted)} row(s) with primary id "
            f"outside accepted (e.g. {', '.join(primary_outside_accepted[:5])})",
            file=sys.stderr,
        )
    if missing_esco:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
