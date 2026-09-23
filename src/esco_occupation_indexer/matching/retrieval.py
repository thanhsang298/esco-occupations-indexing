from __future__ import annotations

import io
import json
import os
import time
from contextlib import suppress
from pathlib import Path

import numpy as np
import zstandard as zstd

from esco_occupation_indexer.errors import ValidationError
from esco_occupation_indexer.matching.compat import assert_alias_snapshot, esco_client
from esco_occupation_indexer.matching.models import (
    ChannelResult,
    JobMatches,
    MatchCandidate,
)
from esco_occupation_indexer.matching.settings import CHANNELS, MatchSettings
from esco_occupation_indexer.matching.stages import (
    complete_match_stage,
    fail_match_stage,
    require_match_artifact_checksums,
    require_match_completed,
    save_match_manifest,
    start_match_stage,
)
from esco_occupation_indexer.settings import IndexingSettings
from esco_occupation_indexer.utils import atomic_write_json, sha256_file

_INF = float("inf")
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


def _is_transient_qdrant_error(error: Exception) -> bool:
    import httpx

    if isinstance(error, httpx.TimeoutException | httpx.NetworkError):
        return True
    status_code = getattr(error, "status_code", None)
    return isinstance(status_code, int) and status_code in _RETRYABLE_STATUS


def _query_with_retry(client, collection_name: str, requests, max_retries: int = 4):
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return client.query_batch_points(
                collection_name=collection_name, requests=requests
            )
        except Exception as error:
            if not _is_transient_qdrant_error(error) or attempt >= max_retries:
                raise
            last_error = error
            time.sleep(min(2.0 * (2**attempt), 30.0))
    assert last_error is not None
    raise last_error


def rrf_fuse(
    channel_hits: dict[str, list[tuple[str, float]]],
    rrf_k: int,
    weights: dict[str, float],
    top_k: int,
) -> list[tuple[str, float, dict[str, tuple[int, float] | None]]]:
    ranks: dict[str, dict[str, tuple[int, float]]] = {}
    for channel, hits in channel_hits.items():
        for position, (esco_id, score) in enumerate(hits, start=1):
            ranks.setdefault(esco_id, {})[channel] = (position, score)
    fused: list[tuple[str, float, dict[str, tuple[int, float] | None]]] = []
    for esco_id, per_channel in ranks.items():
        score = sum(
            weights[channel] / (rrf_k + per_channel[channel][0])
            for channel in per_channel
        )
        evidence = {
            channel: per_channel.get(channel) for channel in CHANNELS
        }
        fused.append((esco_id, score, evidence))
    fused.sort(
        key=lambda item: (
            -item[1],
            item[2]["semantic_dense"][0] if item[2]["semantic_dense"] else _INF,
            item[2]["label_dense"][0] if item[2]["label_dense"] else _INF,
            item[2]["lexical_sparse"][0] if item[2]["lexical_sparse"] else _INF,
            item[0],
        )
    )
    return fused[:top_k]


def _channel_query(shard, index: int, channel: str):
    from qdrant_client import models

    if channel == "label_dense":
        return shard["label_vectors"][index].tolist()
    if channel == "semantic_dense":
        return shard["semantic_vectors"][index].tolist()
    start = int(shard["sparse_offsets"][index])
    end = int(shard["sparse_offsets"][index + 1])
    return models.SparseVector(
        indices=shard["sparse_indices"][start:end].astype(int).tolist(),
        values=shard["sparse_values"][start:end].tolist(),
    )


def _retrieval_shard_valid(path: Path, expected_ids: list[str]) -> bool:
    meta_path = path.with_suffix(".meta.json")
    if not path.is_file() or not meta_path.is_file():
        return False
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata.get("checksum") != sha256_file(path):
            return False
        return metadata.get("job_ids") == expected_ids
    except (ValueError, OSError, KeyError, TypeError):
        return False


def retrieve_build(
    match_dir: Path,
    esco_settings: IndexingSettings,
    esco_collection: str,
    esco_alias: str,
    require_alias: bool = True,
) -> dict[str, object]:
    from esco_occupation_indexer.matching.settings import load_match_resolved_settings

    match_dir = match_dir.resolve()
    manifest = require_match_completed(match_dir, "ingest", "validate", "embed_queries")
    require_match_artifact_checksums(match_dir, manifest, "embedding-report.json")
    settings: MatchSettings = load_match_resolved_settings(match_dir)
    retrieval_cfg = settings.retrieval

    query_shards = sorted((match_dir / "query-vectors").glob("shard-*.npz"))
    if not query_shards:
        raise ValidationError("No query shards found")

    if not any((match_dir / "retrieval").glob("shard-*.jsonl.zst")):
        from esco_occupation_indexer.matching.adopt import (
            adopt_artifact_dir,
            find_compatible_build,
        )

        donor = find_compatible_build(
            settings.match_work_dir,
            match_dir,
            "retrieval_fingerprint",
            manifest.retrieval_fingerprint,
            "retrieve",
        )
        if donor is not None:
            adopted = adopt_artifact_dir(donor, match_dir, "retrieval")
            manifest.runtime["retrieval_adopted_from"] = donor.name
            manifest.runtime["retrieval_adopted_shards"] = adopted
            save_match_manifest(match_dir, manifest)

    start_match_stage(match_dir, "retrieve")
    client = esco_client(esco_settings)
    try:
        from qdrant_client import models

        assert_alias_snapshot(client, esco_alias, esco_collection, require_alias)
        (match_dir / "retrieval").mkdir(parents=True, exist_ok=True)

        all_matches: list[dict[str, object]] = []
        jobs_retrieved_this_run = 0
        zero_result_jobs = 0
        channel_coverage = {channel: 0 for channel in CHANNELS}
        total_seconds = 0.0
        total_queries = 0

        for query_path in query_shards:
            name = query_path.stem.replace("shard-", "retrieval/shard-")
            out_path = match_dir / f"{name}.jsonl.zst"
            with np.load(query_path, allow_pickle=False) as shard:
                job_ids = [str(value) for value in shard["job_ids"]]
                if _retrieval_shard_valid(out_path, job_ids):
                    with open(out_path, "rb") as handle:
                        decompressor = zstd.ZstdDecompressor()
                        with decompressor.stream_reader(handle) as reader:
                            for line in io.TextIOWrapper(reader, encoding="utf-8"):
                                if line.strip():
                                    all_matches.append(json.loads(line))
                    continue

                started = time.perf_counter()
                shard_lines: list[str] = []
                for index, job_id in enumerate(job_ids):
                    requests = [
                        models.QueryRequest(
                            query=_channel_query(shard, index, channel),
                            using=channel,
                            limit=retrieval_cfg.candidate_limit,
                            with_payload=True,
                        )
                        for channel in CHANNELS
                    ]
                    job_started = time.perf_counter()
                    responses = _query_with_retry(
                        client, esco_collection, requests
                    )
                    total_seconds += time.perf_counter() - job_started
                    total_queries += len(requests)
                    channel_hits: dict[str, list[tuple[str, float]]] = {}
                    payload_by_id: dict[str, dict[str, object]] = {}
                    for channel, response in zip(CHANNELS, responses, strict=True):
                        hits = []
                        for point in response.points:
                            pid = str(point.id)
                            hits.append((pid, float(point.score)))
                            if pid not in payload_by_id:
                                payload_by_id[pid] = point.payload or {}
                        channel_hits[channel] = hits
                        if hits:
                            channel_coverage[channel] += 1
                    fused = rrf_fuse(
                        channel_hits,
                        retrieval_cfg.rrf_k,
                        retrieval_cfg.rrf_weights,
                        retrieval_cfg.top_k,
                    )
                    if not fused:
                        zero_result_jobs += 1
                    candidates = []
                    for rank, (esco_id, fused_score, evidence) in enumerate(fused, start=1):
                        payload = payload_by_id.get(esco_id, {})
                        labels = payload.get("labels") or {}
                        hierarchy = payload.get("hierarchy") or {}
                        candidates.append(
                            MatchCandidate(
                                rank=rank,
                                esco_id=esco_id,
                                esco_uri=str(payload.get("esco_uri") or ""),
                                preferred_label=str(labels.get("preferred") or ""),
                                isco_code=str(payload.get("isco_code") or ""),
                                path_text=str(hierarchy.get("path_text") or ""),
                                fused_score=fused_score,
                                channels={
                                    channel: (
                                        ChannelResult(
                                            rank=evidence[channel][0],
                                            score=evidence[channel][1],
                                        )
                                        if evidence[channel] is not None
                                        else None
                                    )
                                    for channel in CHANNELS
                                },
                            )
                        )
                    record = JobMatches(
                        job_id=job_id,
                        job_content_hash="",
                        esco_build_id=manifest.esco_build_id,
                        esco_collection=esco_collection,
                        match_config_fingerprint=manifest.config_fingerprint,
                        query_template_versions={
                            "label": settings.queries.label_template_version,
                            "semantic": settings.queries.semantic_template_version,
                        },
                        candidates=candidates,
                    )
                    shard_lines.append(
                        record.model_dump_json(exclude_none=False) + "\n"
                    )
                    all_matches.append(json.loads(shard_lines[-1]))
                jobs_retrieved_this_run += len(job_ids)

                temporary = out_path.with_suffix(out_path.suffix + ".tmp")
                compressor = zstd.ZstdCompressor(level=6)
                with temporary.open("wb") as raw, compressor.stream_writer(raw) as writer:
                    for line in shard_lines:
                        writer.write(line.encode("utf-8"))
                os.replace(temporary, out_path)
                atomic_write_json(
                    out_path.with_suffix(".meta.json"),
                    {
                        "filename": out_path.name,
                        "checksum": sha256_file(out_path),
                        "job_ids": job_ids,
                        "jobs": len(job_ids),
                        "seconds": time.perf_counter() - started,
                        "resumed": False,
                    },
                )

        assert_alias_snapshot(client, esco_alias, esco_collection, require_alias)

        # Fill job content hashes from canonical records.
        from esco_occupation_indexer.artifacts import read_jsonl_zst
        from esco_occupation_indexer.matching.models import CanonicalJobPosting

        hashes = {
            record.job_id: record.content_hash
            for record in read_jsonl_zst(
                match_dir / "canonical-jobs.jsonl.zst", CanonicalJobPosting
            )
        }
        for match in all_matches:
            match["job_content_hash"] = hashes.get(match["job_id"], "")
        all_matches.sort(key=lambda item: item["job_id"])

        matches_path = match_dir / "matches.jsonl.zst"
        temporary = matches_path.with_suffix(matches_path.suffix + ".tmp")
        compressor = zstd.ZstdCompressor(level=6)
        with temporary.open("wb") as raw, compressor.stream_writer(raw) as writer:
            for match in all_matches:
                writer.write((json.dumps(match, ensure_ascii=False) + "\n").encode())
        os.replace(temporary, matches_path)

        report: dict[str, object] = {
            "status": "completed",
            "job_count": len(all_matches),
            "jobs_retrieved_this_run": jobs_retrieved_this_run,
            "zero_result_jobs": zero_result_jobs,
            "channel_coverage_jobs": channel_coverage,
            "total_queries": total_queries,
            "retrieval_seconds_total": total_seconds,
            "queries_per_second": total_queries / total_seconds if total_seconds else None,
            "candidate_limit": retrieval_cfg.candidate_limit,
            "rrf_k": retrieval_cfg.rrf_k,
            "top_k": retrieval_cfg.top_k,
        }
        atomic_write_json(match_dir / "retrieval-report.json", report)
        manifest = require_match_completed(
            match_dir, "ingest", "validate", "embed_queries"
        )
        manifest.counts["matched_jobs"] = len(all_matches)
        manifest.artifact_checksums["retrieval-report.json"] = sha256_file(
            match_dir / "retrieval-report.json"
        )
        manifest.artifact_checksums["matches.jsonl.zst"] = sha256_file(matches_path)
        save_match_manifest(match_dir, manifest)
        complete_match_stage(match_dir, "retrieve")
        return report
    except Exception as error:
        with suppress(Exception):
            fail_match_stage(match_dir, "retrieve", error)
        raise
    finally:
        client.close()
