"""Render a human-readable Markdown report from final threshold outputs.

Reads summary.json + occupation_stats.json in a final-threshold-* dir and writes
REPORT.md next to them (ranked tables, family mix, long-tail, near-miss).

Usage:
    uv run python scripts/write_final_report.py
    uv run python scripts/write_final_report.py --dir data/job-matching-work/4aa45473959c/final-threshold-0.895
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BUILD_DIR = Path("data/job-matching-work/4aa45473959c")
FINAL_DIR = BUILD_DIR / "final-threshold-0.895"
TOP_N = 20


def esc(cell: object) -> str:
    return str(cell).replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=FINAL_DIR)
    parser.add_argument("--top-n", type=int, default=TOP_N)
    args = parser.parse_args()

    summary = json.loads((args.dir / "summary.json").read_text(encoding="utf-8"))
    stats = json.loads((args.dir / "occupation_stats.json").read_text(encoding="utf-8"))
    meta, occs = stats["meta"], stats["occupations"]

    L: list[str] = []
    L.append(f"# Final mapping report — build `{summary['match_build_id']}`")
    L.append("")
    L.append(f"Decision rule: `{summary['decision_rule']}` "
             f"(frozen per `data/gold/threshold_selection_4aa45473959c.md`). "
             "Pair-level, no top-k cutoff, multi-label allowed. "
             "Empty selection = abstain (unmapped), not \"occupation absent from ESCO\".")
    L.append("")
    L.append("## Summary")
    L.append("")
    L.append("| Metric | Value |")
    L.append("|---|---|")
    L.append(f"| Total jobs | {summary['total_jobs']} |")
    L.append(f"| Mapped jobs | {summary['mapped_jobs']} |")
    L.append(f"| Unmapped jobs (abstain) | {summary['unmapped_jobs']} "
             f"({summary['unmapped_rate']:.1%}) |")
    L.append(f"| Selected pairs | {summary['selected_pairs']} |")
    L.append(f"| Mean matches / job | {summary['mean_matches_per_job']} |")
    L.append(f"| Mean matches / mapped job | {summary['mean_matches_per_mapped_job']} |")
    L.append(f"| Distinct occupations matched | {meta['distinct_occupations_matched']} "
             f"/ {meta['catalog_total']} (coverage {meta['catalog_coverage']:.1%}) |")
    L.append("")
    L.append("## Concentration & long tail")
    L.append("")
    L.append(f"- Top-1 occupation covers {meta['top1_share_mapped_jobs']:.1%} of mapped jobs.")
    L.append(f"- Top-10 cover {meta['top10_share_pairs']:.1%} of pairs; "
             f"top-20 cover {meta['top20_share_pairs']:.1%}.")
    L.append(f"- {meta['occupations_with_1_job']} occupations match exactly 1 job; "
             f"{meta['occupations_with_le3_jobs']} match ≤ 3 jobs; "
             f"median {meta['median_jobs_per_occupation']} job(s) per occupation.")
    L.append("")
    L.append(f"## Top {args.top_n} occupations by mapped jobs")
    L.append("")
    L.append("| # | Occupation | ISCO | Jobs | Share | Mean rerank | Mean rank | Example job titles |")
    L.append("|---:|---|---|---:|---:|---:|---:|---|")
    for i, r in enumerate(occs[: args.top_n], 1):
        L.append(f"| {i} | {esc(r['preferred_label'])} | {esc(r['isco_code'])} "
                 f"| {r['n_jobs']} | {r['share_mapped_jobs']:.1%} "
                 f"| {r['mean_rerank_score']:.3f} | {r['mean_rerank_rank']:.1f} "
                 f"| {esc('; '.join(r['example_job_titles'][:3]))} |")
    L.append("")
    L.append("## Family mix (by pairs)")
    L.append("")
    L.append("| Family | Pairs | Share |")
    L.append("|---|---:|---:|")
    for f in meta["family_pairs"]:
        L.append(f"| {esc(f['family'])} | {f['pairs']} | {f['share']:.1%} |")
    L.append("")
    L.append("## Near-miss: rank-1 inside unmapped jobs")
    L.append("")
    L.append("Best-scoring candidate per abstained job — audit targets if the "
             "threshold is ever re-estimated on a new adjudicated split.")
    L.append("")
    L.append("| Unmapped jobs | Candidate | Context |")
    L.append("|---:|---|---|")
    for n in meta["near_miss_top10_in_unmapped"]:
        L.append(f"| {n['n_unmapped_jobs_rank1']} | {esc(n['preferred_label'])} "
                 f"| {esc(n['context'])} |")
    L.append("")
    L.append("## Files")
    L.append("")
    L.append("| File | Content |")
    L.append("|---|---|")
    L.append("| `results.jsonl` / `results.json` | 1 record per job: `matches` (decision) + `top_candidates` (top-10 evidence with `selected` flag) |")
    L.append("| `unmapped.jsonl` / `unmapped.json` | 466 abstained jobs with `abstain_reason` + `best_candidate` |")
    L.append("| `occupation_stats.json` | 269 rows, one per matched occupation, sorted by `n_jobs` desc |")
    L.append("| `summary.json` | counts + threshold metadata |")
    L.append("")
    L.append("## Limitation")
    L.append("")
    L.append("Pair precision on Gold v2 is ~0.57 partly because gold labels are an "
             "accepted set, not exhaustive binary judgments over all candidates — "
             "measured precision is conservative. Do not re-tune the threshold on "
             "this benchmark; freeze at 0.895 and re-estimate only on a new, "
             "independently adjudicated pair-level validation split.")
    L.append("")

    out = args.dir / "REPORT.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {out} ({len(L)} lines)")


if __name__ == "__main__":
    main()
