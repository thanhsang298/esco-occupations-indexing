"""Aggregate occupation-level stats from thresholded final results.

Reads results.jsonl produced by scripts/apply_threshold.py and answers:
  - which ESCO occupations are matched the most (ranked table)?
  - what job titles map into each top occupation?
  - how concentrated / long-tail is the distribution?
  - which ISCO families (path_text top level) dominate?
  - catalog coverage: distinct matched occupations / catalog total?
  - near-miss: which occupations are rank-1 in unmapped jobs (just below threshold)?

Usage:
    uv run python scripts/occupation_stats.py
    uv run python scripts/occupation_stats.py --top-n 20

Outputs (under the final-threshold dir):
    occupation_stats.json   pretty array, one row per matched occupation, sorted
                            by n_jobs desc, plus a "meta" summary block
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

BUILD_DIR = Path("data/job-matching-work/4aa45473959c")
RESULTS_FILE = BUILD_DIR / "final-threshold-0.895" / "results.jsonl"
CATALOG_TOTAL = 3039  # distinct esco_id in data/work/9ac16776cc2c/canonical-occupations.jsonl.zst


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-file", type=Path, default=RESULTS_FILE)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out_path = args.out or (args.results_file.parent / "occupation_stats.json")

    per_occ: dict[str, dict] = {}
    family_counter: Counter[str] = Counter()
    family_pairs: Counter[str] = Counter()
    nearmiss: Counter[str] = Counter()
    nearmiss_label: dict[str, str] = {}
    n_mapped_jobs = n_pairs = n_unmapped = 0

    for line in args.results_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec["mapping_status"] == "mapped":
            n_mapped_jobs += 1
            for m in rec["matches"]:
                n_pairs += 1
                occ = per_occ.setdefault(
                    m["esco_id"],
                    {
                        "esco_id": m["esco_id"],
                        "esco_uri": m["esco_uri"],
                        "preferred_label": m["preferred_label"],
                        "isco_code": m["isco_code"],
                        "path_text": m["path_text"],
                        "family": (m["path_text"].split(">")[0].strip() or "?"),
                        "n_jobs": 0,
                        "_scores": [],
                        "_ranks": [],
                        "_retrieval_ranks": [],
                        "_titles": Counter(),
                    },
                )
                occ["n_jobs"] += 1
                occ["_scores"].append(m["rerank_score"])
                occ["_ranks"].append(m["rerank_rank"])
                occ["_retrieval_ranks"].append(m["retrieval_rank"])
                if rec.get("job_title"):
                    occ["_titles"][rec["job_title"]] += 1
                family_counter[occ["family"]] += 1  # counted per job below
                family_pairs[occ["family"]] += 1
        else:
            n_unmapped += 1
            best = rec.get("best_candidate")
            if best:
                nearmiss[best["preferred_label"]] += 1
                nearmiss_label[best["preferred_label"]] = (
                    f"{best['isco_code']} | {best['path_text'].split('>')[0].strip()}"
                )

    # family distinct-job counts need a second pass structure; approximate with pairs
    rows = []
    for occ in per_occ.values():
        scores = occ.pop("_scores")
        ranks = occ.pop("_ranks")
        ret = occ.pop("_retrieval_ranks")
        titles = occ.pop("_titles")
        rows.append(
            {
                **occ,
                "share_mapped_jobs": round(occ["n_jobs"] / n_mapped_jobs, 4),
                "share_pairs": round(occ["n_jobs"] / n_pairs, 4),
                "mean_rerank_score": round(sum(scores) / len(scores), 4),
                "min_rerank_score": round(min(scores), 4),
                "mean_rerank_rank": round(sum(ranks) / len(ranks), 2),
                "mean_retrieval_rank": round(sum(ret) / len(ret), 2),
                "example_job_titles": [t for t, _ in titles.most_common(5)],
            }
        )
    rows.sort(key=lambda r: (-r["n_jobs"], r["preferred_label"]))

    counts = [r["n_jobs"] for r in rows]
    total_distinct = len(rows)
    meta = {
        "mapped_jobs": n_mapped_jobs,
        "unmapped_jobs": n_unmapped,
        "selected_pairs": n_pairs,
        "distinct_occupations_matched": total_distinct,
        "catalog_total": CATALOG_TOTAL,
        "catalog_coverage": round(total_distinct / CATALOG_TOTAL, 4),
        "top1_share_mapped_jobs": round(counts[0] / n_mapped_jobs, 4),
        "top10_share_pairs": round(sum(counts[:10]) / n_pairs, 4),
        "top20_share_pairs": round(sum(counts[:20]) / n_pairs, 4),
        "occupations_with_1_job": sum(1 for c in counts if c == 1),
        "occupations_with_le3_jobs": sum(1 for c in counts if c <= 3),
        "median_jobs_per_occupation": sorted(counts)[len(counts) // 2],
        "family_pairs": [
            {"family": fam, "pairs": n, "share": round(n / n_pairs, 4)}
            for fam, n in family_pairs.most_common()
        ],
        "near_miss_top10_in_unmapped": [
            {
                "preferred_label": label,
                "n_unmapped_jobs_rank1": n,
                "context": nearmiss_label[label],
            }
            for label, n in nearmiss.most_common(10)
        ],
    }

    out_path.write_text(
        json.dumps({"meta": meta, "occupations": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Console: ranked table + insights
    print(f"distinct matched: {total_distinct}/{CATALOG_TOTAL} "
          f"(coverage {meta['catalog_coverage']:.1%}) | pairs: {n_pairs}")
    print(f"concentration: top1={meta['top1_share_mapped_jobs']:.1%} of mapped jobs, "
          f"top10={meta['top10_share_pairs']:.1%} of pairs, "
          f"top20={meta['top20_share_pairs']:.1%} of pairs")
    print(f"long tail: {meta['occupations_with_1_job']} occ with exactly 1 job, "
          f"{meta['occupations_with_le3_jobs']} occ with <=3 jobs, "
          f"median {meta['median_jobs_per_occupation']} job(s)/occ")
    print(f"\n== TOP {args.top_n} occupations by mapped jobs ==")
    print(f"{'rank':>4} {'jobs':>5} {'share':>6} {'mean_rs':>7} "
          f"{'mean_rr':>7}  label [isco]")
    for i, r in enumerate(rows[: args.top_n], 1):
        print(f"{i:>4} {r['n_jobs']:>5} {r['share_mapped_jobs']:>5.1%} "
              f"{r['mean_rerank_score']:>7.3f} {r['mean_rerank_rank']:>7.1f}  "
              f"{r['preferred_label']} [{r['isco_code']}]")
    print("\n== family mix (by pairs) ==")
    for f in meta["family_pairs"]:
        print(f"  {f['share']:>5.1%}  {f['pairs']:>5}  {f['family']}")
    print("\n== near-miss top10 (rank-1 inside UNMAPPED jobs) ==")
    for n in meta["near_miss_top10_in_unmapped"]:
        print(f"  {n['n_unmapped_jobs_rank1']:>4}  {n['preferred_label']} ({n['context']})")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
