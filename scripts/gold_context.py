"""Dump labeling context for a gold batch: job text + top-10 + ESCO descriptions.

Usage:
    uv run python scripts/gold_context.py --num-batches 4 --batch 0

Output: data/gold/context_part_{batch}.json — list of per-job dicts.
Requires Qdrant env (set -a; source .env) for ESCO descriptions.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from _jsonl import iter_jsonl_zst

BUILD_DIR = Path("data/job-matching-work/24bbae641cfd")
SAMPLE = Path("data/gold/sample.json")
COLLECTION = "esco_occupation_concepts__v1_2_1__9ac16776cc2c"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--build-dir", type=Path, default=BUILD_DIR)
    parser.add_argument("--collection", default=COLLECTION)
    parser.add_argument("--out-prefix", default="context_part")
    args = parser.parse_args()

    from qdrant_client import QdrantClient

    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))
    chunk = [s for i, s in enumerate(sample) if i % args.num_batches == args.batch]
    wanted = {s["job_id"] for s in chunk}
    meta = {s["job_id"]: s for s in chunk}

    matches = {
        m["job_id"]: m
        for m in iter_jsonl_zst(args.build_dir / "matches.jsonl.zst")
        if m["job_id"] in wanted
    }
    canonical = {
        c["job_id"]: c
        for c in iter_jsonl_zst(args.build_dir / "canonical-jobs.jsonl.zst")
        if c["job_id"] in wanted
    }

    needed_ids = sorted(
        {c["esco_id"] for m in matches.values() for c in m["candidates"]}
    )
    client = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ.get("QDRANT_API_KEY") or None,
        timeout=60,
        check_compatibility=False,
    )
    descriptions: dict[str, str] = {}
    try:
        for i in range(0, len(needed_ids), 100):
            points = client.retrieve(
                collection_name=args.collection,
                ids=needed_ids[i : i + 100],
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                content = payload.get("content") or {}
                descriptions[str(point.id)] = (
                    content.get("description")
                    or content.get("definition")
                    or content.get("scope_note")
                    or ""
                )
    finally:
        client.close()

    out = []
    for job_id in sorted(wanted):
        job = canonical.get(job_id, {})
        match = matches.get(job_id, {})
        out.append(
            {
                "job_id": job_id,
                "strata": meta[job_id].get("strata"),
                "language_hint": meta[job_id].get("language_hint"),
                "title": job.get("title"),
                "categories": [
                    " > ".join(n["name"] for n in p.get("declared_path", []))
                    for p in job.get("it_category_paths", [])
                ],
                "description": job.get("description"),
                "requirements": job.get("require_candidate"),
                "knowledge": job.get("knowledge"),
                "candidates": [
                    {
                        "rank": c["rank"],
                        "esco_id": c["esco_id"],
                        "label": c["preferred_label"],
                        "isco": c["isco_code"],
                        "path": c["path_text"],
                        "fused": round(c["fused_score"], 4),
                        "channels": {
                            k: (v["rank"] if v else None)
                            for k, v in c["channels"].items()
                        },
                        "esco_description": descriptions.get(c["esco_id"], "")[:600],
                    }
                    for c in match.get("candidates", [])
                ],
            }
        )
    out_path = Path(f"data/gold/{args.out_prefix}_{args.batch}.json")
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(out)} jobs -> {out_path}")


if __name__ == "__main__":
    main()
