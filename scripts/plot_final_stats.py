"""Plot final mapping distributions as PNG charts.

Charts (saved under <final-dir>/charts/):
    1. matches_per_job.png  bar chart of n_matches per job (0 = abstain highlighted)
    2. top_occupations.png   horizontal bars, top-N occupations by mapped jobs
    3. family_mix.png        donut chart of selected pairs by ISCO family

Usage:
    uv run python scripts/plot_final_stats.py
    uv run python scripts/plot_final_stats.py --dir <final-dir> --top-n 20
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BUILD_DIR = Path("data/job-matching-work/4aa45473959c")
FINAL_DIR = BUILD_DIR / "final-threshold-0.895"

ACCENT = "#d62728"  # abstain bar
BASE = "#1f77b4"


def chart_matches_per_job(n_matches: list[int], out: Path, threshold: float) -> None:
    dist = Counter(n_matches)
    xs = sorted(dist)
    ys = [dist[x] for x in xs]
    colors = [ACCENT if x == 0 else BASE for x in xs]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(xs, ys, color=colors, edgecolor="white", width=0.8)
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_xlabel("ESCO occupations matched per job (0 = abstain)")
    ax.set_ylabel("Number of jobs")
    ax.set_title(
        f"Matches per job @ rerank_score >= {threshold} "
        f"(n={len(n_matches)} jobs, mean={sum(n_matches)/len(n_matches):.2f})"
    )
    ax.set_xticks(xs)
    ax.legend(
        [bars[0], bars[1]],
        [f"unmapped (0 matches): {dist[0]}", "mapped"],
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def chart_top_occupations(occs: list[dict], out: Path, top_n: int) -> None:
    top = occs[:top_n][::-1]  # ascending for horizontal bars
    labels = [r["preferred_label"] for r in top]
    values = [r["n_jobs"] for r in top]
    fig, ax = plt.subplots(figsize=(10, 0.45 * top_n + 1.5))
    bars = ax.barh(labels, values, color=BASE, edgecolor="white")
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_xlabel("Mapped jobs")
    ax.set_title(f"Top {top_n} ESCO occupations by mapped jobs")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def chart_family_mix(families: list[dict], out: Path) -> None:
    labels = [f["family"] for f in families]
    sizes = [f["pairs"] for f in families]
    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, _, autotexts = ax.pie(
        sizes,
        autopct=lambda p: f"{p:.1f}%" if p >= 2 else "",
        startangle=90,
        pctdistance=0.8,
        wedgeprops={"width": 0.4, "edgecolor": "white"},
    )
    ax.legend(
        wedges,
        [f"{l} ({s})" for l, s in zip(labels, sizes)],
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        frameon=False,
    )
    for t in autotexts:
        t.set_fontsize(9)
    ax.set_title("Selected pairs by ISCO family (top level of path_text)")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=FINAL_DIR)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    summary = json.loads((args.dir / "summary.json").read_text(encoding="utf-8"))
    stats = json.loads((args.dir / "occupation_stats.json").read_text(encoding="utf-8"))

    n_matches = [
        json.loads(line)["n_matches"]
        for line in (args.dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    charts = args.dir / "charts"
    charts.mkdir(exist_ok=True)
    chart_matches_per_job(n_matches, charts / "matches_per_job.png", summary["threshold"])
    chart_top_occupations(stats["occupations"], charts / "top_occupations.png", args.top_n)
    chart_family_mix(stats["meta"]["family_pairs"], charts / "family_mix.png")
    for p in sorted(charts.glob("*.png")):
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
