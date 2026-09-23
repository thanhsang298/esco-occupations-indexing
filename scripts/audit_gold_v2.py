"""Audit gold_v2 for structural errors and retrieval-pool leakage.

This deliberately treats ESCO as the source of truth and the v2 top-20 only as
diagnostic evidence. A correct label is allowed to be outside the retrieval
pool; such rows are recall failures, not unanswerable examples.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from _jsonl import iter_jsonl_zst

ROOT = Path(__file__).resolve().parents[1]
GOLD_DIR = ROOT / "data/gold"
GOLD = GOLD_DIR / "gold_v2.jsonl"
ESCO = ROOT / "data/work/9ac16776cc2c/canonical-occupations.jsonl.zst"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    gold = load_jsonl(GOLD)
    contexts = {
        row["job_id"]: row
        for path in sorted(GOLD_DIR.glob("context_v2_part_*.json"))
        for row in json.loads(path.read_text(encoding="utf-8"))
    }
    parts = {
        row["job_id"]: row
        for path in sorted(GOLD_DIR.glob("gold_v2_part_*.jsonl"))
        for row in load_jsonl(path)
    }
    occupations = {row["esco_id"]: row for row in iter_jsonl_zst(ESCO)}

    ids = [row["job_id"] for row in gold]
    duplicate_jobs = sorted(job_id for job_id, n in Counter(ids).items() if n > 1)
    missing_context = sorted(set(ids) - set(contexts))
    missing_parts = sorted(set(ids) - set(parts))
    different_parts = sorted(
        row["job_id"] for row in gold if parts.get(row["job_id"]) != row
    )
    duplicate_labels = sorted(
        row["job_id"]
        for row in gold
        if len(row["accepted_esco_ids"]) != len(set(row["accepted_esco_ids"]))
    )
    missing_primary_field = sorted(
        row["job_id"] for row in gold if "primary_esco_ids" not in row
    )
    primary_not_accepted = sorted(
        row["job_id"]
        for row in gold
        if not set(row.get("primary_esco_ids", [])).issubset(row["accepted_esco_ids"])
    )
    duplicate_primary_labels = sorted(
        row["job_id"]
        for row in gold
        if len(row.get("primary_esco_ids", [])) != len(set(row.get("primary_esco_ids", [])))
    )
    missing_esco = sorted(
        {
            esco_id
            for row in gold
            for esco_id in row["accepted_esco_ids"]
            if esco_id not in occupations
        }
    )

    outside_pool: list[dict] = []
    ancestor_overlaps: list[dict] = []
    for row in gold:
        context = contexts.get(row["job_id"])
        if context is None:
            continue
        candidates = {candidate["esco_id"]: candidate for candidate in context["candidates"]}
        for esco_id in row["accepted_esco_ids"]:
            if esco_id not in candidates:
                occupation = occupations[esco_id]
                outside_pool.append(
                    {
                        "job_id": row["job_id"],
                        "esco_id": esco_id,
                        "label": occupation["preferred_label"],
                    }
                )
        for ancestor_id in row["accepted_esco_ids"]:
            ancestor = candidates.get(ancestor_id)
            if ancestor is None:
                continue
            ancestor_label = ancestor["label"].casefold()
            for descendant_id in row["accepted_esco_ids"]:
                descendant = candidates.get(descendant_id)
                if descendant is None or descendant_id == ancestor_id:
                    continue
                path_labels = {part.strip().casefold() for part in descendant["path"].split(">")}
                if ancestor_label in path_labels:
                    ancestor_overlaps.append(
                        {
                            "job_id": row["job_id"],
                            "ancestor": ancestor["label"],
                            "descendant": descendant["label"],
                        }
                    )

    hard_errors = {
        "duplicate_jobs": duplicate_jobs,
        "missing_context": missing_context,
        "missing_parts": missing_parts,
        "different_parts": different_parts,
        "duplicate_labels": duplicate_labels,
        "missing_primary_field": missing_primary_field,
        "primary_not_accepted": primary_not_accepted,
        "duplicate_primary_labels": duplicate_primary_labels,
        "missing_esco": missing_esco,
    }
    result = {
        "rows": len(gold),
        "empty_rows": sum(not row["accepted_esco_ids"] for row in gold),
        "accepted_labels": sum(len(row["accepted_esco_ids"]) for row in gold),
        "primary_labels": sum(len(row.get("primary_esco_ids", [])) for row in gold),
        "empty_primary_rows": sum(not row.get("primary_esco_ids", []) for row in gold),
        "unique_esco_labels": len(
            {esco_id for row in gold for esco_id in row["accepted_esco_ids"]}
        ),
        "hard_errors": hard_errors,
        "outside_top20": outside_pool,
        "ancestor_descendant_overlaps": ancestor_overlaps,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if any(hard_errors.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
