from __future__ import annotations

from pathlib import Path

from esco_occupation_indexer.embedding import DenseBackend, SparseBackend, embed_build
from esco_occupation_indexer.ingest import ingest_source
from esco_occupation_indexer.qdrant_ops import promote_build, upload_build, verify_build
from esco_occupation_indexer.settings import IndexingSettings
from esco_occupation_indexer.validation import validate_build


def run_build(
    settings: IndexingSettings,
    source_zip: Path,
    *,
    promote: bool = False,
    dense_backend: DenseBackend | None = None,
    sparse_backend: SparseBackend | None = None,
) -> Path:
    build_dir = ingest_source(settings, source_zip)
    validate_build(build_dir)
    embed_build(build_dir, dense_backend=dense_backend, sparse_backend=sparse_backend)
    upload_build(build_dir)
    verify_build(build_dir)
    if promote:
        promote_build(build_dir)
    return build_dir
