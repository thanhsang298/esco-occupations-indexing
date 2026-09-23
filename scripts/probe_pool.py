"""Probe pool misses: re-run stored query vectors at limit 100 and rank suspects.

Usage:
    uv run python scripts/probe_pool.py

For each suspected (job_id, concept label): resolve the concept via label_dense,
then report its per-channel rank for that job's stored queries at depth 100.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from esco_occupation_indexer.matching.settings import CHANNELS

BUILD_DIR = Path("data/job-matching-work/24bbae641cfd")
COLLECTION = "esco_occupation_concepts__v1_2_1__9ac16776cc2c"

SUSPECTS: list[tuple[str, str]] = [
    ("topcv:2256414", "mobile application developer"),
    ("topcv:2272097", "mobile application developer"),
    ("topcv:2290732", "mobile application developer"),
    ("topcv:2185017", "database administrator"),
    ("topcv:2274506", "DevOps engineer"),
    ("topcv:2260449", "Scrum Master"),
    ("topcv:2206483", "data annotator"),
    ("topcv:2249945", "head of engineering"),
    ("topcv:2250471", "ICT helpdesk operator"),
    ("topcv:2251231", "telesales operator"),
    ("topcv:1772814", "human resources manager"),
    ("topcv:2288572", "loss prevention auditor"),
]

INSTRUCTION = (
    "Given a job posting, retrieve the ESCO occupation that best represents "
    "its work and responsibilities."
)


def main() -> None:
    import httpx
    from qdrant_client import QdrantClient, models

    client = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ.get("QDRANT_API_KEY") or None,
        timeout=120,
        check_compatibility=False,
    )
    tei = httpx.Client(timeout=120.0)
    tei_url = os.environ.get("TEI_URL", "http://localhost:8787")
    try:
        # 1. Resolve suspected labels to esco_ids via label_dense.
        resolved: dict[str, list[tuple[str, str]]] = {}
        for _, label in SUSPECTS:
            if label in resolved:
                continue
            body = {"inputs": [label], "normalize": True, "truncate": False}
            vector = tei.post(f"{tei_url}/embed", json=body).json()[0]
            hits = client.query_points(
                collection_name=COLLECTION,
                query=vector,
                using="label_dense",
                limit=5,
                with_payload=True,
            ).points
            resolved[label] = [
                (str(point.id), (point.payload or {}).get("labels", {}).get("preferred"))
                for point in hits
            ]
        print("=== label resolution ===")
        for label, hits in resolved.items():
            print(f"{label!r}: {[(esco_id[:8], pref) for esco_id, pref in hits]}")

        # 2. Load stored query vectors per job.
        queries: dict[str, dict] = {}
        for shard_path in sorted((BUILD_DIR / "query-vectors").glob("shard-*.npz")):
            with np.load(shard_path, allow_pickle=False) as shard:
                ids = [str(value) for value in shard["job_ids"]]
                for index, job_id in enumerate(ids):
                    start = int(shard["sparse_offsets"][index])
                    end = int(shard["sparse_offsets"][index + 1])
                    queries[job_id] = {
                        "label_dense": shard["label_vectors"][index].tolist(),
                        "semantic_dense": shard["semantic_vectors"][index].tolist(),
                        "lexical_sparse": models.SparseVector(
                            indices=shard["sparse_indices"][start:end].astype(int).tolist(),
                            values=shard["sparse_values"][start:end].tolist(),
                        ),
                    }

        # 3. Rank suspects at depth 100 per channel.
        print("=== suspect ranks at depth 100 (miss = absent) ===")
        for job_id, label in SUSPECTS:
            query = queries[job_id]
            ranks = {}
            for channel in CHANNELS:
                hits = client.query_points(
                    collection_name=COLLECTION,
                    query=query[channel],
                    using=channel,
                    limit=100,
                    with_payload=False,
                ).points
                order = {str(point.id): position for position, point in enumerate(hits, start=1)}
                ranks[channel] = [
                    (esco_id[:8], order.get(esco_id)) for esco_id, _ in resolved[label][:3]
                ]
            print(f"{job_id} / {label!r}:")
            for channel in CHANNELS:
                print(f"  {channel}: {ranks[channel]}")
    finally:
        client.close()
        tei.close()


if __name__ == "__main__":
    main()
