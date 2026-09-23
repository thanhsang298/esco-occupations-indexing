from __future__ import annotations

import gc
import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from esco_occupation_indexer.artifacts import (
    complete_stage,
    fail_stage,
    read_jsonl_zst,
    require_artifact_checksums,
    require_completed,
    save_manifest,
    start_stage,
)
from esco_occupation_indexer.documents import dense_label_variants, lexical_text, semantic_text
from esco_occupation_indexer.errors import ValidationError
from esco_occupation_indexer.models import CanonicalOccupationRecord
from esco_occupation_indexer.settings import (
    IndexingSettings,
    SparseSettings,
    load_resolved_settings,
)
from esco_occupation_indexer.utils import atomic_write_json, sha256_file


@dataclass(frozen=True)
class SparseEncoding:
    indices: np.ndarray
    values: np.ndarray


class DenseBackend(Protocol):
    dimension: int
    device: str

    def encode(self, texts: list[str], batch_size: int) -> np.ndarray: ...

    def count_truncated(self, texts: list[str], max_tokens: int) -> int: ...


class SparseBackend(Protocol):
    def encode(self, texts: list[str]) -> list[SparseEncoding]: ...


def select_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class SentenceTransformerBackend:
    backend_name = "sentence_transformers"

    def __init__(self, settings: IndexingSettings) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise ValidationError(
                "The local dense backend requires: uv sync --extra local-embedding"
            ) from error

        self.dimension = settings.embedding.dimension
        self.device = select_device(settings.embedding.device)
        self.model = SentenceTransformer(
            settings.embedding.model_id,
            revision=settings.embedding.model_revision,
            device=self.device,
        )
        self.model.max_seq_length = settings.embedding.max_tokens
        actual_dimension = self.model.get_sentence_embedding_dimension()
        if actual_dimension != self.dimension:
            raise ValidationError(
                "Embedding dimension mismatch: "
                f"configured {self.dimension}, model {actual_dimension}"
            )

    def encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def count_truncated(self, texts: list[str], max_tokens: int) -> int:
        count = 0
        tokenizer = self.model.tokenizer
        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=True, truncation=False)
            count += len(tokens) > max_tokens
        return count


def create_dense_backend(settings: IndexingSettings) -> DenseBackend:
    if settings.embedding.backend == "tei":
        from esco_occupation_indexer.tei import TeiDenseBackend

        return TeiDenseBackend(settings)
    return SentenceTransformerBackend(settings)


class FastEmbedBm25Backend:
    def __init__(self, settings: SparseSettings) -> None:
        from fastembed import SparseTextEmbedding

        self.model = SparseTextEmbedding(
            model_name=settings.model_id,
            language=settings.language,
            k=settings.k,
            b=settings.b,
            avg_len=settings.avg_len,
            token_max_length=settings.token_max_length,
            disable_stemmer=settings.disable_stemmer,
        )

    def encode(self, texts: list[str]) -> list[SparseEncoding]:
        encoded = self.model.embed(texts)
        return [
            SparseEncoding(
                indices=np.asarray(item.indices, dtype=np.uint32),
                values=np.asarray(item.values, dtype=np.float32),
            )
            for item in encoded
        ]


def _is_oom(error: RuntimeError) -> bool:
    message = str(error).casefold()
    return "out of memory" in message or "cannot allocate memory" in message


def _clear_device_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except ImportError:
        pass


def _encode_with_backoff(
    backend: DenseBackend, texts: list[str], initial_batch_size: int
) -> tuple[np.ndarray, int]:
    batch_size = max(1, initial_batch_size)
    backend_limit = getattr(backend, "max_batch_size", None)
    if isinstance(backend_limit, int) and backend_limit > 0:
        batch_size = min(batch_size, backend_limit)
    while True:
        try:
            vectors = backend.encode(texts, batch_size)
            actual_batch = getattr(backend, "last_effective_batch_size", batch_size)
            if isinstance(actual_batch, int) and actual_batch > 0:
                batch_size = min(batch_size, actual_batch)
            return vectors, batch_size
        except RuntimeError as error:
            if not _is_oom(error) or batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            _clear_device_cache()


def _validate_dense(vectors: np.ndarray, count: int, dimension: int, name: str) -> None:
    if vectors.shape != (count, dimension):
        raise ValidationError(
            f"{name} has shape {vectors.shape}; expected {(count, dimension)}"
        )
    if not np.isfinite(vectors).all():
        raise ValidationError(f"{name} contains NaN or Inf")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValidationError(f"{name} contains non-normalized vectors")


def _write_shard(
    path: Path,
    records: list[CanonicalOccupationRecord],
    dense_backend: DenseBackend,
    sparse_backend: SparseBackend,
    settings: IndexingSettings,
    initial_batch_size: int,
) -> tuple[int, int, float]:
    started = time.perf_counter()
    label_texts: list[str] = []
    label_offsets = [0]
    semantic_texts: list[str] = []
    lexical_texts: list[str] = []
    for record in records:
        labels = [variant.text for variant in dense_label_variants(record)]
        label_texts.extend(labels)
        label_offsets.append(len(label_texts))
        semantic_texts.append(semantic_text(record))
        lexical_texts.append(lexical_text(record))

    label_vectors, label_batch = _encode_with_backoff(
        dense_backend, label_texts, initial_batch_size
    )
    semantic_vectors, semantic_batch = _encode_with_backoff(
        dense_backend, semantic_texts, initial_batch_size
    )
    effective_batch = min(label_batch, semantic_batch)
    _validate_dense(
        label_vectors, len(label_texts), settings.embedding.dimension, "label_dense"
    )
    _validate_dense(
        semantic_vectors, len(records), settings.embedding.dimension, "semantic_dense"
    )

    sparse = sparse_backend.encode(lexical_texts)
    if len(sparse) != len(records):
        raise ValidationError("Sparse backend returned the wrong number of vectors")
    sparse_offsets = [0]
    sparse_indices: list[np.ndarray] = []
    sparse_values: list[np.ndarray] = []
    for item in sparse:
        if item.indices.shape != item.values.shape or item.indices.ndim != 1:
            raise ValidationError("Sparse indices and values must be aligned 1-D arrays")
        if not len(item.indices):
            raise ValidationError("Sparse backend returned an empty vector")
        if not np.isfinite(item.values).all():
            raise ValidationError("Sparse vector contains NaN or Inf")
        sparse_indices.append(item.indices)
        sparse_values.append(item.values)
        sparse_offsets.append(sparse_offsets[-1] + len(item.indices))

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            ids=np.asarray([record.esco_id for record in records], dtype="U36"),
            semantic_vectors=semantic_vectors.astype(np.float32),
            label_vectors=label_vectors.astype(np.float32),
            label_offsets=np.asarray(label_offsets, dtype=np.int64),
            sparse_indices=np.concatenate(sparse_indices).astype(np.uint32),
            sparse_values=np.concatenate(sparse_values).astype(np.float32),
            sparse_offsets=np.asarray(sparse_offsets, dtype=np.int64),
        )
    os.replace(temporary, path)
    return effective_batch, len(label_texts), time.perf_counter() - started


def validate_shard(path: Path, dimension: int) -> dict[str, int]:
    with np.load(path, allow_pickle=False) as shard:
        ids = shard["ids"]
        semantic = shard["semantic_vectors"]
        labels = shard["label_vectors"]
        label_offsets = shard["label_offsets"]
        sparse_indices = shard["sparse_indices"]
        sparse_values = shard["sparse_values"]
        sparse_offsets = shard["sparse_offsets"]
        count = len(ids)
        _validate_dense(semantic, count, dimension, f"{path.name}:semantic_dense")
        _validate_dense(labels, len(labels), dimension, f"{path.name}:label_dense")
        if len(label_offsets) != count + 1 or label_offsets[0] != 0:
            raise ValidationError(f"Invalid label offsets in {path}")
        if label_offsets[-1] != len(labels) or np.any(np.diff(label_offsets) < 1):
            raise ValidationError(f"Invalid label vector ranges in {path}")
        if len(sparse_offsets) != count + 1 or sparse_offsets[0] != 0:
            raise ValidationError(f"Invalid sparse offsets in {path}")
        if sparse_offsets[-1] != len(sparse_indices) or len(sparse_indices) != len(
            sparse_values
        ):
            raise ValidationError(f"Invalid sparse vector ranges in {path}")
        if np.any(np.diff(sparse_offsets) < 1):
            raise ValidationError(f"Empty sparse vector in {path}")
        return {"records": count, "labels": len(labels)}


def _valid_existing_shard(path: Path, dimension: int, expected_ids: list[str]) -> bool:
    meta_path = path.with_suffix(".meta.json")
    if not path.is_file() or not meta_path.is_file():
        return False
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata["checksum"] != sha256_file(path):
            return False
        validate_shard(path, dimension)
        with np.load(path, allow_pickle=False) as shard:
            if [str(value) for value in shard["ids"]] != expected_ids:
                return False
        return True
    except (KeyError, TypeError, ValueError, OSError, ValidationError):
        return False


def _shard_specs(
    vectors_dir: Path,
    records: list[CanonicalOccupationRecord],
    shard_size: int,
) -> list[tuple[Path, list[CanonicalOccupationRecord]]]:
    return [
        (
            vectors_dir / f"shard-{shard_index:05d}.npz",
            records[start : start + shard_size],
        )
        for shard_index, start in enumerate(range(0, len(records), shard_size))
    ]


def _all_expected_shards_are_valid(
    specs: list[tuple[Path, list[CanonicalOccupationRecord]]], dimension: int
) -> bool:
    if not specs:
        return False
    expected_paths = {path for path, _ in specs}
    actual_paths = {path for path in specs[0][0].parent.glob("shard-*.npz")}
    if actual_paths != expected_paths:
        return False
    return all(
        _valid_existing_shard(
            path,
            dimension,
            [record.esco_id for record in records],
        )
        for path, records in specs
    )


def _valid_embedding_report(build_dir: Path, manifest) -> dict[str, object] | None:
    filename = "embedding-report.json"
    expected_checksum = manifest.artifact_checksums.get(filename)
    path = build_dir / filename
    if not expected_checksum or not path.is_file():
        return None
    try:
        if sha256_file(path) != expected_checksum:
            return None
    except OSError:
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return report if isinstance(report, dict) and report.get("status") == "completed" else None


def _resumed_embedding_report(report: dict[str, object]) -> dict[str, object]:
    """Return invocation-specific resume stats without rewriting a valid artifact."""

    resumed = dict(report)
    shards = report.get("shards")
    if isinstance(shards, list):
        resumed["shards"] = [
            {**shard, "resumed": True} if isinstance(shard, dict) else shard
            for shard in shards
        ]
    resumed["records_embedded_this_run"] = 0
    resumed["embedding_seconds_this_run"] = 0.0
    resumed["records_per_second_this_run"] = None
    return resumed


def embed_build(
    build_dir: Path,
    dense_backend: DenseBackend | None = None,
    sparse_backend: SparseBackend | None = None,
) -> dict[str, object]:
    build_dir = build_dir.resolve()
    manifest = require_completed(build_dir, "ingest", "validate")
    require_artifact_checksums(
        build_dir,
        manifest,
        "canonical-occupations.jsonl.zst",
        "isco-groups.jsonl.zst",
        "source-report.json",
        "validation-report.json",
    )
    settings = load_resolved_settings(build_dir)
    records = list(
        read_jsonl_zst(build_dir / "canonical-occupations.jsonl.zst", CanonicalOccupationRecord)
    )
    specs = _shard_specs(build_dir / "vectors", records, settings.embedding.shard_size)
    existing_report = _valid_embedding_report(build_dir, manifest)
    if (
        manifest.stages["embed"].status == "completed"
        and existing_report is not None
        and _all_expected_shards_are_valid(specs, settings.embedding.dimension)
    ):
        return _resumed_embedding_report(existing_report)

    start_stage(build_dir, "embed")
    owns_dense_backend = dense_backend is None
    try:
        dense_backend = dense_backend or create_dense_backend(settings)
        if dense_backend.dimension != settings.embedding.dimension:
            raise ValidationError("Dense backend dimension differs from build configuration")

        backend_name = str(
            getattr(dense_backend, "backend_name", type(dense_backend).__name__)
        )
        raw_backend_metadata = getattr(dense_backend, "runtime_metadata", {})
        backend_metadata = (
            dict(raw_backend_metadata) if isinstance(raw_backend_metadata, Mapping) else {}
        )

        semantic_texts = [semantic_text(record) for record in records]
        truncated_count = dense_backend.count_truncated(
            semantic_texts, settings.embedding.max_tokens
        )
        shards: list[dict[str, object]] = []
        effective_batch = settings.embedding.batch_size
        total_labels = 0
        total_seconds = 0.0
        total_shard_seconds = 0.0
        records_embedded = 0
        for path, chunk in specs:
            expected_ids = [record.esco_id for record in chunk]
            if _valid_existing_shard(path, settings.embedding.dimension, expected_ids):
                stats = validate_shard(path, settings.embedding.dimension)
                prior_metadata = json.loads(
                    path.with_suffix(".meta.json").read_text(encoding="utf-8")
                )
                shard_batch = int(
                    prior_metadata.get("effective_batch_size", settings.embedding.batch_size)
                )
                shard_seconds = float(prior_metadata.get("seconds", 0.0))
                total_shard_seconds += shard_seconds
                effective_batch = min(effective_batch, shard_batch)
                metadata = {
                    "filename": path.name,
                    "checksum": sha256_file(path),
                    "records": stats["records"],
                    "labels": stats["labels"],
                    "effective_batch_size": shard_batch,
                    "seconds": shard_seconds,
                    "resumed": True,
                }
            else:
                if sparse_backend is None:
                    sparse_backend = FastEmbedBm25Backend(settings.sparse)
                shard_batch, label_count, elapsed = _write_shard(
                    path,
                    chunk,
                    dense_backend,
                    sparse_backend,
                    settings,
                    effective_batch,
                )
                effective_batch = min(effective_batch, shard_batch)
                total_seconds += elapsed
                total_shard_seconds += elapsed
                records_embedded += len(chunk)
                metadata = {
                    "filename": path.name,
                    "checksum": sha256_file(path),
                    "records": len(chunk),
                    "labels": label_count,
                    "effective_batch_size": shard_batch,
                    "seconds": elapsed,
                    "resumed": False,
                }
                atomic_write_json(path.with_suffix(".meta.json"), metadata)
            total_labels += int(metadata["labels"])
            shards.append(metadata)

        report: dict[str, object] = {
            "status": "completed",
            "backend": backend_name,
            "backend_metadata": backend_metadata,
            "device": dense_backend.device,
            "effective_batch_size": effective_batch,
            "record_count": len(records),
            "label_vector_count": total_labels,
            "semantic_truncation_count": truncated_count,
            "embedding_seconds_this_run": total_seconds,
            "embedding_seconds_total": total_shard_seconds,
            "records_embedded_this_run": records_embedded,
            "records_per_second_this_run": (
                records_embedded / total_seconds if total_seconds else None
            ),
            "records_per_second_total": (
                len(records) / total_shard_seconds if total_shard_seconds else None
            ),
            "shards": shards,
        }
        atomic_write_json(build_dir / "embedding-report.json", report)

        validation_path = build_dir / "validation-report.json"
        validation_report = json.loads(validation_path.read_text(encoding="utf-8"))
        validation_report["semantic_truncation_count"] = truncated_count
        atomic_write_json(validation_path, validation_report)

        manifest = require_completed(build_dir, "ingest", "validate")
        manifest.counts["embedded_occupation_concepts"] = len(records)
        manifest.counts["label_vectors"] = total_labels
        manifest.counts["semantic_truncation_count"] = truncated_count
        manifest.artifact_checksums["validation-report.json"] = sha256_file(validation_path)
        manifest.artifact_checksums["embedding-report.json"] = sha256_file(
            build_dir / "embedding-report.json"
        )
        manifest.runtime.update(
            {
                "embedding_backend": backend_name,
                "embedding_device": dense_backend.device,
                "effective_batch_size": effective_batch,
            }
        )
        manifest.runtime.update(
            {
                key: value
                for key, value in backend_metadata.items()
                if isinstance(value, (str, int, float, bool)) or value is None
            }
        )
        save_manifest(build_dir, manifest)
        complete_stage(build_dir, "embed")
        return report
    except Exception as error:
        fail_stage(build_dir, "embed", error)
        raise
    finally:
        if owns_dense_backend and dense_backend is not None:
            close = getattr(dense_backend, "close", None)
            if callable(close):
                close()
