from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from esco_occupation_indexer.artifacts import read_jsonl_zst
from esco_occupation_indexer.errors import ValidationError
from esco_occupation_indexer.matching.models import CanonicalJobPosting
from esco_occupation_indexer.matching.queries import (
    FittedQuery,
    TokenCounter,
    fit_to_budget,
    label_sections,
    lexical_text,
    semantic_sections,
)
from esco_occupation_indexer.matching.query_encoding import (
    QuerySparseBackend,
    build_query_dense_backend,
)
from esco_occupation_indexer.matching.settings import MatchSettings
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


def _fit_all(
    counter: TokenCounter,
    instruction: str,
    jobs: list[CanonicalJobPosting],
    label_budget: int,
    semantic_budget: int,
) -> tuple[list[FittedQuery], list[FittedQuery], list[str]]:
    labels = [
        fit_to_budget(counter, instruction, label_sections(job), label_budget)
        for job in jobs
    ]
    semantics = [
        fit_to_budget(counter, instruction, semantic_sections(job), semantic_budget)
        for job in jobs
    ]
    lexicals = [lexical_text(job) for job in jobs]
    return labels, semantics, lexicals


def _write_query_shard(
    path: Path,
    jobs: list[CanonicalJobPosting],
    labels: list[FittedQuery],
    semantics: list[FittedQuery],
    lexicals: list[str],
    dense_backend,
    sparse_backend: QuerySparseBackend,
    dimension: int,
) -> dict[str, object]:
    started = time.perf_counter()
    label_vectors = dense_backend.encode(
        [item.text for item in labels], dense_backend.last_effective_batch_size
    )
    semantic_vectors = dense_backend.encode(
        [item.text for item in semantics], dense_backend.last_effective_batch_size
    )
    for name, vectors in (("label", label_vectors), ("semantic", semantic_vectors)):
        if vectors.shape != (len(jobs), dimension):
            raise ValidationError(f"Query {name} vectors have shape {vectors.shape}")
        if not np.isfinite(vectors).all():
            raise ValidationError(f"Query {name} vectors contain NaN or Inf")
    sparse = sparse_backend.query_encode(lexicals)
    offsets = [0]
    for item in sparse:
        offsets.append(offsets[-1] + len(item.indices))

    max_text = max(
        [len(item.text) for item in labels]
        + [len(item.text) for item in semantics]
        + [len(text) for text in lexicals]
        + [1]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            job_ids=np.asarray([job.job_id for job in jobs], dtype="U64"),
            label_texts=np.asarray([item.text for item in labels], dtype=f"U{max_text}"),
            semantic_texts=np.asarray(
                [item.text for item in semantics], dtype=f"U{max_text}"
            ),
            lexical_texts=np.asarray(lexicals, dtype=f"U{max_text}"),
            label_vectors=label_vectors.astype(np.float32),
            semantic_vectors=semantic_vectors.astype(np.float32),
            sparse_indices=np.concatenate([item.indices for item in sparse]).astype(
                np.uint32
            ),
            sparse_values=np.concatenate([item.values for item in sparse]).astype(
                np.float32
            ),
            sparse_offsets=np.asarray(offsets, dtype=np.int64),
            label_tok_orig=np.asarray(
                [item.original_tokens for item in labels], dtype=np.int64
            ),
            label_tok_actual=np.asarray(
                [item.actual_tokens for item in labels], dtype=np.int64
            ),
            sem_tok_orig=np.asarray(
                [item.original_tokens for item in semantics], dtype=np.int64
            ),
            sem_tok_actual=np.asarray(
                [item.actual_tokens for item in semantics], dtype=np.int64
            ),
            truncated=np.asarray(
                [
                    label.truncated or semantic.truncated
                    for label, semantic in zip(labels, semantics, strict=True)
                ],
                dtype=np.bool_,
            ),
            shortened=np.asarray(
                [
                    ",".join(label.shortened_sections + semantic.shortened_sections)
                    for label, semantic in zip(labels, semantics, strict=True)
                ],
                dtype="U64",
            ),
        )
    os.replace(temporary, path)
    return {
        "seconds": time.perf_counter() - started,
        "label_truncated": sum(item.truncated for item in labels),
        "semantic_truncated": sum(item.truncated for item in semantics),
    }


def _valid_existing_shard(path: Path, expected_ids: list[str]) -> bool:
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


def embed_query_build(
    match_dir: Path,
    esco_settings: IndexingSettings,
    dense_backend=None,
    sparse_backend: QuerySparseBackend | None = None,
) -> dict[str, object]:
    from esco_occupation_indexer.matching.settings import load_match_resolved_settings

    match_dir = match_dir.resolve()
    manifest = require_match_completed(match_dir, "ingest", "validate")
    require_match_artifact_checksums(
        match_dir, manifest, "canonical-jobs.jsonl.zst", "validation-report.json"
    )
    settings: MatchSettings = load_match_resolved_settings(match_dir)
    jobs = list(
        read_jsonl_zst(match_dir / "canonical-jobs.jsonl.zst", CanonicalJobPosting)
    )
    shard_size = settings.shard_size
    specs = [
        (match_dir / "query-vectors" / f"shard-{i:05d}.npz", jobs[start : start + shard_size])
        for i, start in enumerate(range(0, len(jobs), shard_size))
    ]

    if not any(path.is_file() for path, _ in specs):
        from esco_occupation_indexer.matching.adopt import (
            adopt_artifact_dir,
            find_compatible_build,
        )

        donor = find_compatible_build(
            settings.match_work_dir,
            match_dir,
            "query_fingerprint",
            manifest.query_fingerprint,
            "embed_queries",
        )
        if donor is not None:
            adopted = adopt_artifact_dir(donor, match_dir, "query-vectors")
            manifest.runtime["queries_adopted_from"] = donor.name
            manifest.runtime["queries_adopted_shards"] = adopted
            save_match_manifest(match_dir, manifest)

    start_match_stage(match_dir, "embed_queries")
    owns_dense = dense_backend is None
    try:
        dense_backend = dense_backend or build_query_dense_backend(
            esco_settings, settings.queries.semantic_max_tokens
        )
        if dense_backend.dimension != esco_settings.embedding.dimension:
            raise ValidationError("Query dense dimension differs from ESCO dimension")
        sparse_backend = sparse_backend or QuerySparseBackend(esco_settings.sparse)
        counter = TokenCounter(dense_backend)

        shards: list[dict[str, object]] = []
        embedded_this_run = 0
        label_truncated = 0
        semantic_truncated = 0
        total_seconds = 0.0
        for path, chunk in specs:
            expected_ids = [job.job_id for job in chunk]
            if _valid_existing_shard(path, expected_ids):
                metadata = json.loads(path.with_suffix(".meta.json").read_text())
                metadata["resumed"] = True
                label_truncated += int(metadata.get("label_truncated", 0))
                semantic_truncated += int(metadata.get("semantic_truncated", 0))
            else:
                labels, semantics, lexicals = _fit_all(
                    counter,
                    settings.queries.instruction,
                    chunk,
                    settings.queries.label_max_tokens,
                    settings.queries.semantic_max_tokens,
                )
                stats = _write_query_shard(
                    path,
                    chunk,
                    labels,
                    semantics,
                    lexicals,
                    dense_backend,
                    sparse_backend,
                    esco_settings.embedding.dimension,
                )
                embedded_this_run += len(chunk)
                total_seconds += stats["seconds"]
                label_truncated += stats["label_truncated"]
                semantic_truncated += stats["semantic_truncated"]
                metadata = {
                    "filename": path.name,
                    "checksum": sha256_file(path),
                    "job_ids": expected_ids,
                    "jobs": len(chunk),
                    "label_truncated": stats["label_truncated"],
                    "semantic_truncated": stats["semantic_truncated"],
                    "seconds": stats["seconds"],
                    "resumed": False,
                }
                atomic_write_json(path.with_suffix(".meta.json"), metadata)
            shards.append(metadata)

        report: dict[str, object] = {
            "status": "completed",
            "backend": getattr(dense_backend, "backend_name", type(dense_backend).__name__),
            "device": dense_backend.device,
            "effective_batch_size": getattr(
                dense_backend, "last_effective_batch_size", None
            ),
            "record_count": len(jobs),
            "records_embedded_this_run": embedded_this_run,
            "embedding_seconds_this_run": total_seconds,
            "label_truncated": label_truncated,
            "semantic_truncated": semantic_truncated,
            "shards": shards,
        }
        atomic_write_json(match_dir / "embedding-report.json", report)
        manifest = require_match_completed(match_dir, "ingest", "validate")
        manifest.counts["embedded_jobs"] = len(jobs)
        manifest.artifact_checksums["embedding-report.json"] = sha256_file(
            match_dir / "embedding-report.json"
        )
        save_match_manifest(match_dir, manifest)
        complete_match_stage(match_dir, "embed_queries")
        return report
    except Exception as error:
        fail_match_stage(match_dir, "embed_queries", error)
        raise
    finally:
        if owns_dense and dense_backend is not None:
            close = getattr(dense_backend, "close", None)
            if callable(close):
                close()
