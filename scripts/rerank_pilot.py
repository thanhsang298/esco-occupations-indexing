"""Pilot rerank for gold jobs only (does not mutate the match build).

Usage:
    set -a; source .env; set +a
    uv run python scripts/rerank_pilot.py \
        --build-dir data/job-matching-work/8ddeacb204b5 \
        --esco-build-dir data/work/9ac16776cc2c \
        --config config/job-matching.yaml

Output (default): data/gold/matches_pilot.jsonl.zst (RRF order kept + rerank fields).
Score with: scripts/score_gold.py --matches-file data/gold/matches_pilot.jsonl.zst --order rerank
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from _jsonl import iter_jsonl_zst

from esco_occupation_indexer.artifacts import read_jsonl_zst
from esco_occupation_indexer.documents import semantic_text as esco_semantic_text
from esco_occupation_indexer.matching.models import JobMatches
from esco_occupation_indexer.matching.rerank_client import VllmRerankBackend
from esco_occupation_indexer.matching.settings import load_match_settings
from esco_occupation_indexer.models import CanonicalOccupationRecord

OUT = Path("data/gold/matches_pilot.jsonl.zst")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--esco-build-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/job-matching.yaml"))
    parser.add_argument("--sample", type=Path, default=Path("data/gold/sample.json"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model", default=None)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument(
        "--strip-query-prefix",
        action="store_true",
        help="Remove a leading 'Instruct: ...' line from stored query texts",
    )
    args = parser.parse_args()

    settings = load_match_settings(args.config)
    cfg = settings.rerank
    if args.model:
        cfg = cfg.model_copy(update={"model": args.model})
    wanted = {
        entry["job_id"] for entry in json.loads(args.sample.read_text(encoding="utf-8"))
    }

    query_texts: dict[str, str] = {}
    for shard_path in sorted((args.build_dir / "query-vectors").glob("shard-*.npz")):
        with np.load(shard_path, allow_pickle=False) as shard:
            for job_id, text in zip(
                [str(v) for v in shard["job_ids"]],
                [str(v) for v in shard["semantic_texts"]],
                strict=True,
            ):
                if job_id in wanted:
                    query_texts[job_id] = text
    matches = [
        JobMatches(**m)
        for m in iter_jsonl_zst(args.build_dir / "matches.jsonl.zst")
        if m["job_id"] in wanted
    ]
    if args.strip_query_prefix:
        stripped = {}
        for job_id, text in query_texts.items():
            lines = text.split("\n", 1)
            if len(lines) == 2 and lines[0].startswith("Instruct:"):
                second = lines[1]
                if second.startswith("Query:"):
                    second = second[len("Query:"):].lstrip()
                stripped[job_id] = second
            else:
                stripped[job_id] = text
        query_texts = stripped
    esco_docs = {
        record.esco_id: esco_semantic_text(record)
        for record in read_jsonl_zst(
            args.esco_build_dir / "canonical-occupations.jsonl.zst",
            CanonicalOccupationRecord,
        )
    }
    print(f"pilot jobs: {len(matches)}")

    backend = VllmRerankBackend(
        base_url=cfg.base_url,
        model=cfg.model,
        timeout_seconds=cfg.timeout_seconds,
        max_retries=cfg.max_retries,
    )
    total_pairs = 0
    total_seconds = 0.0

    def rerank_one(match: JobMatches) -> JobMatches:
        nonlocal total_pairs, total_seconds
        documents = [esco_docs[c.esco_id] for c in match.candidates]
        started = time.perf_counter()
        ranked = backend.rerank(
            query_texts[match.job_id], documents, top_n=cfg.top_n,
            instruction=cfg.instruction,
        )
        total_seconds += time.perf_counter() - started
        total_pairs += len(documents)
        by_index = {index: score for index, score in ranked}
        updated = match.model_copy(deep=True)
        for position, candidate in enumerate(updated.candidates):
            candidate.rerank_score = by_index.get(position)
        ordered = sorted(
            [c for c in updated.candidates if c.rerank_score is not None],
            key=lambda c: (-c.rerank_score, c.rank),
        )
        for rerank_position, candidate in enumerate(ordered, start=1):
            candidate.rerank_rank = rerank_position
        return updated

    import zstandard as zstd

    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    compressor = zstd.ZstdCompressor(level=6)
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        updated_all = list(executor.map(rerank_one, matches))
    with temporary.open("wb") as raw, compressor.stream_writer(raw) as writer:  # noqa: SIM117
        for updated in updated_all:
            writer.write((updated.model_dump_json(exclude_none=False) + "\n").encode())
            done += 1
    import os

    os.replace(temporary, out_path)
    backend.close()
    print(f"wrote {done} jobs -> {out_path} (model={cfg.model})")
    print(f"pairs: {total_pairs}, seconds: {round(total_seconds, 1)}")


if __name__ == "__main__":
    main()
