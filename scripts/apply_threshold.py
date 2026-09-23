"""Apply frozen pair-level threshold to produce final job -> ESCO mapping.

Decision rule (see data/gold/threshold_selection_4aa45473959c.md):
    select candidate iff rerank_score >= threshold (default 0.895).
No top-k cutoff, no RRF fallback. Multi-label allowed; empty selection = unmapped.

Outputs (under --out-dir):
    results.jsonl   one record per job, one JSON object per line (compact)
    results.json    same records as a pretty-printed JSON array (human-readable)
    unmapped.jsonl  subset with mapping_status == "unmapped" (for separate reporting)
    unmapped.json   same subset as a pretty-printed JSON array
    summary.json    counts + threshold metadata

Record schema:
    job_id, job_title, esco_build_id,
    mapping_status: "mapped" | "unmapped",
    decision_rule, threshold,
    n_matches, max_rerank_score,
    matches: [selected candidates sorted by rerank_rank],
    top_candidates: [top-10 by rerank_rank, each with selected flag],
    abstain_reason (only when unmapped),
    best_candidate (only when unmapped, rank-1 for triage)

Candidate objects are slimmed (channels dict dropped; kept: retrieval rank,
fused_score, rerank_score/rank + ESCO identity fields).

Usage:
    uv run python scripts/apply_threshold.py --threshold 0.895
    uv run python scripts/apply_threshold.py --threshold 0.895 \\
        --matches-file data/job-matching-work/4aa45473959c/matches.jsonl.zst \\
        --out-dir data/job-matching-work/4aa45473959c/final-threshold-0.895
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _jsonl import iter_jsonl_zst

BUILD_DIR = Path("data/job-matching-work/4aa45473959c")
MATCHES_FILE = BUILD_DIR / "matches.jsonl.zst"
JOBS_FILE = Path("data/iviec-job-crawler.job_details.topcv.it.json")
THRESHOLD = 0.895
TOP_K = 10


def load_titles(path: Path) -> dict[str, str]:
    try:
        jobs = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    return {
        f"topcv:{j['externalId']}": j.get("title") or ""
        for j in jobs
        if j.get("status") == "completed"
    }


def slim(candidate: dict, selected: bool) -> dict:
    return {
        "esco_id": candidate["esco_id"],
        "esco_uri": candidate["esco_uri"],
        "preferred_label": candidate["preferred_label"],
        "isco_code": candidate["isco_code"],
        "path_text": candidate["path_text"],
        "retrieval_rank": candidate["rank"],
        "fused_score": candidate["fused_score"],
        "rerank_score": candidate["rerank_score"],
        "rerank_rank": candidate["rerank_rank"],
        "selected": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--matches-file", type=Path, default=MATCHES_FILE)
    parser.add_argument("--jobs-file", type=Path, default=JOBS_FILE)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or (
        BUILD_DIR / f"final-threshold-{args.threshold:.3f}".replace(".", "_")
    )
    # Keep human-friendly dirname: final-threshold-0.895
    if args.out_dir is None:
        out_dir = BUILD_DIR / f"final-threshold-{args.threshold:.3f}"
    out_dir.mkdir(parents=True, exist_ok=True)

    titles = load_titles(args.jobs_file)
    decision_rule = f"rerank_score >= {args.threshold}"

    n_mapped = n_unmapped = n_pairs = 0
    results_path = out_dir / "results.jsonl"
    unmapped_path = out_dir / "unmapped.jsonl"

    with (
        results_path.open("w", encoding="utf-8") as fres,
        unmapped_path.open("w", encoding="utf-8") as funm,
    ):
        for match in iter_jsonl_zst(args.matches_file):
            ranked = sorted(
                [c for c in match["candidates"] if c.get("rerank_score") is not None],
                key=lambda c: (c["rerank_rank"], c["rank"]),
            )
            selected_ids = {
                c["esco_id"] for c in ranked if c["rerank_score"] >= args.threshold
            }
            matches = [
                slim(c, True) for c in ranked if c["esco_id"] in selected_ids
            ]
            top = [slim(c, c["esco_id"] in selected_ids) for c in ranked[: args.top_k]]
            max_score = ranked[0]["rerank_score"] if ranked else None

            status = "mapped" if matches else "unmapped"
            if matches:
                n_mapped += 1
                n_pairs += len(matches)
            else:
                n_unmapped += 1

            record: dict = {
                "job_id": match["job_id"],
                "job_title": titles.get(match["job_id"], ""),
                "esco_build_id": match["esco_build_id"],
                "mapping_status": status,
                "decision_rule": decision_rule,
                "threshold": args.threshold,
                "n_matches": len(matches),
                "max_rerank_score": max_score,
                "matches": matches,
                "top_candidates": top,
            }
            if not matches:
                record["abstain_reason"] = (
                    f"max rerank_score {max_score:.4f} < {args.threshold}"
                    if max_score is not None
                    else "no reranked candidates"
                )
                record["best_candidate"] = top[0] if top else None

            line = json.dumps(record, ensure_ascii=False)
            fres.write(line + "\n")
            if not matches:
                funm.write(line + "\n")

    total = n_mapped + n_unmapped
    summary = {
        "match_build_id": BUILD_DIR.name,
        "matches_file": str(args.matches_file),
        "threshold": args.threshold,
        "decision_rule": decision_rule,
        "top_k_evidence": args.top_k,
        "total_jobs": total,
        "mapped_jobs": n_mapped,
        "unmapped_jobs": n_unmapped,
        "unmapped_rate": round(n_unmapped / total, 4) if total else 0.0,
        "selected_pairs": n_pairs,
        "mean_matches_per_job": round(n_pairs / total, 3) if total else 0.0,
        "mean_matches_per_mapped_job": round(n_pairs / n_mapped, 3) if n_mapped else 0.0,
        "outputs": {
            "results": str(results_path),
            "unmapped": str(unmapped_path),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    def to_pretty_array(jsonl_path: Path, json_path: Path) -> int:
        records = [
            json.loads(line)
            for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        json_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return len(records)

    n_json = to_pretty_array(results_path, out_dir / "results.json")
    n_unmapped_json = to_pretty_array(unmapped_path, out_dir / "unmapped.json")
    summary["outputs"]["results_json"] = str(out_dir / "results.json")
    summary["outputs"]["unmapped_json"] = str(out_dir / "unmapped.json")
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {n_json} records -> {out_dir / 'results.json'}")
    print(f"wrote {n_unmapped_json} records -> {out_dir / 'unmapped.json'}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
