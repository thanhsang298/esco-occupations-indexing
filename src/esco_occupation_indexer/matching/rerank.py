from __future__ import annotations

import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import zstandard as zstd

from esco_occupation_indexer.artifacts import read_jsonl_zst
from esco_occupation_indexer.documents import semantic_text as esco_semantic_text
from esco_occupation_indexer.errors import ValidationError
from esco_occupation_indexer.matching.models import JobMatches
from esco_occupation_indexer.matching.rerank_client import VllmRerankBackend
from esco_occupation_indexer.matching.settings import MatchSettings
from esco_occupation_indexer.matching.stages import (
    complete_match_stage,
    fail_match_stage,
    require_match_artifact_checksums,
    require_match_completed,
    save_match_manifest,
    start_match_stage,
)
from esco_occupation_indexer.models import CanonicalOccupationRecord
from esco_occupation_indexer.utils import atomic_write_json, sha256_file


def _load_query_texts(match_dir: Path) -> dict[str, str]:
    texts: dict[str, str] = {}
    for shard_path in sorted((match_dir / "query-vectors").glob("shard-*.npz")):
        with np.load(shard_path, allow_pickle=False) as shard:
            for job_id, text in zip(
                [str(value) for value in shard["job_ids"]],
                [str(value) for value in shard["semantic_texts"]],
                strict=True,
            ):
                texts[job_id] = text
    return texts


def _load_esco_docs(esco_build_dir: Path) -> dict[str, str]:
    return {
        record.esco_id: esco_semantic_text(record)
        for record in read_jsonl_zst(
            esco_build_dir / "canonical-occupations.jsonl.zst",
            CanonicalOccupationRecord,
        )
    }


def _read_matches(path: Path) -> list[JobMatches]:
    decompressor = zstd.ZstdDecompressor()
    records: list[JobMatches] = []
    with path.open("rb") as raw, decompressor.stream_reader(raw) as reader:
        for line in io.TextIOWrapper(reader, encoding="utf-8"):
            if line.strip():
                records.append(JobMatches.model_validate_json(line))
    return records


def _reranked_shard_valid(path: Path, expected_ids: list[str]) -> bool:
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


def rerank_build(
    match_dir: Path,
    esco_build_dir: Path,
    backend=None,
) -> dict[str, object]:
    from esco_occupation_indexer.matching.settings import load_match_resolved_settings

    match_dir = match_dir.resolve()
    esco_build_dir = esco_build_dir.resolve()
    manifest = require_match_completed(match_dir, "ingest", "validate", "embed_queries", "retrieve")
    require_match_artifact_checksums(match_dir, manifest, "matches.jsonl.zst")
    settings: MatchSettings = load_match_resolved_settings(match_dir)
    cfg = settings.rerank

    start_match_stage(match_dir, "rerank")
    if not cfg.enabled:
        report: dict[str, object] = {"status": "completed", "enabled": False}
        atomic_write_json(match_dir / "rerank-report.json", report)
        manifest = require_match_completed(
            match_dir, "ingest", "validate", "embed_queries", "retrieve"
        )
        manifest.artifact_checksums["rerank-report.json"] = sha256_file(
            match_dir / "rerank-report.json"
        )
        save_match_manifest(match_dir, manifest)
        complete_match_stage(match_dir, "rerank")
        return report

    owns_backend = backend is None
    try:
        backend = backend or VllmRerankBackend(
            base_url=cfg.base_url,
            model=cfg.model,
            timeout_seconds=cfg.timeout_seconds,
            max_retries=cfg.max_retries,
            retry_backoff_seconds=cfg.retry_backoff_seconds,
        )
        query_texts = _load_query_texts(match_dir)
        esco_docs = _load_esco_docs(esco_build_dir)
        matches = _read_matches(match_dir / "matches.jsonl.zst")
        missing_queries = [m.job_id for m in matches if m.job_id not in query_texts]
        if missing_queries:
            raise ValidationError(f"Missing stored queries for {len(missing_queries)} jobs")

        rerank_dir = match_dir / "reranked"
        rerank_dir.mkdir(parents=True, exist_ok=True)
        top_n = cfg.top_n
        total_pairs = 0
        total_seconds = 0.0
        scores: list[float] = []

        def rerank_one(match: JobMatches) -> JobMatches:
            query = query_texts[match.job_id]
            documents = []
            for candidate in match.candidates:
                doc = esco_docs.get(candidate.esco_id)
                if doc is None:
                    raise ValidationError(f"Missing ESCO doc for {candidate.esco_id}")
                documents.append(doc)
            started = time.perf_counter()
            ranked = backend.rerank(
                query, documents, top_n=top_n, instruction=cfg.instruction
            )
            elapsed = time.perf_counter() - started
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
            scored = [
                c.rerank_score
                for c in updated.candidates
                if c.rerank_score is not None
            ]
            return updated, elapsed, scored

        # Group by existing retrieval shards for resumable checkpoints.
        groups: list[tuple[Path, list[JobMatches]]] = []
        query_shards = sorted((match_dir / "query-vectors").glob("shard-*.npz"))
        by_job = {match.job_id: match for match in matches}
        for query_path in query_shards:
            with np.load(query_path, allow_pickle=False) as shard:
                job_ids = [str(value) for value in shard["job_ids"]]
            out_path = rerank_dir / f"{query_path.stem}.jsonl.zst"
            groups.append((out_path, [by_job[job_id] for job_id in job_ids]))

        all_reranked: list[JobMatches] = []
        with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
            for out_path, chunk in groups:
                job_ids = [match.job_id for match in chunk]
                if _reranked_shard_valid(out_path, job_ids):
                    all_reranked.extend(_read_matches(out_path))
                    continue
                results = list(executor.map(rerank_one, chunk))
                lines: list[str] = []
                for updated, elapsed, job_scores in results:
                    total_seconds += elapsed
                    total_pairs += len(updated.candidates)
                    scores.extend(job_scores)
                    lines.append(updated.model_dump_json(exclude_none=False) + "\n")
                    all_reranked.append(updated)
                temporary = out_path.with_suffix(out_path.suffix + ".tmp")
                compressor = zstd.ZstdCompressor(level=6)
                with temporary.open("wb") as raw, compressor.stream_writer(raw) as writer:
                    for line in lines:
                        writer.write(line.encode("utf-8"))
                os.replace(temporary, out_path)
                atomic_write_json(
                    out_path.with_suffix(".meta.json"),
                    {
                        "filename": out_path.name,
                        "checksum": sha256_file(out_path),
                        "job_ids": job_ids,
                        "jobs": len(job_ids),
                        "resumed": False,
                    },
                )

        all_reranked.sort(key=lambda item: item.job_id)
        matches_path = match_dir / "matches.jsonl.zst"
        temporary = matches_path.with_suffix(matches_path.suffix + ".tmp")
        compressor = zstd.ZstdCompressor(level=6)
        with temporary.open("wb") as raw, compressor.stream_writer(raw) as writer:
            for match in all_reranked:
                writer.write(
                    (match.model_dump_json(exclude_none=False) + "\n").encode()
                )
        os.replace(temporary, matches_path)

        report = {
            "status": "completed",
            "enabled": True,
            "backend": getattr(backend, "backend_name", type(backend).__name__),
            "backend_metadata": getattr(backend, "runtime_metadata", {}),
            "job_count": len(all_reranked),
            "total_pairs": total_pairs,
            "rerank_seconds_total": total_seconds,
            "pairs_per_second": total_pairs / total_seconds if total_seconds else None,
            "score_min": min(scores) if scores else None,
            "score_max": max(scores) if scores else None,
            "score_mean": sum(scores) / len(scores) if scores else None,
        }
        atomic_write_json(match_dir / "rerank-report.json", report)
        manifest = require_match_completed(
            match_dir, "ingest", "validate", "embed_queries", "retrieve"
        )
        manifest.counts["reranked_jobs"] = len(all_reranked)
        manifest.artifact_checksums["matches.jsonl.zst"] = sha256_file(matches_path)
        manifest.artifact_checksums["rerank-report.json"] = sha256_file(
            match_dir / "rerank-report.json"
        )
        save_match_manifest(match_dir, manifest)
        complete_match_stage(match_dir, "rerank")
        return report
    except Exception as error:
        fail_match_stage(match_dir, "rerank", error)
        raise
    finally:
        if owns_backend and backend is not None:
            close = getattr(backend, "close", None)
            if callable(close):
                close()
