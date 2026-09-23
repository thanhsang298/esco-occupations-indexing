from __future__ import annotations

from pathlib import Path

from esco_occupation_indexer.matching.canonicalize import ingest_jobs
from esco_occupation_indexer.matching.compat import check_esco_compat
from esco_occupation_indexer.matching.embed_queries import embed_query_build
from esco_occupation_indexer.matching.rerank import rerank_build
from esco_occupation_indexer.matching.retrieval import retrieve_build
from esco_occupation_indexer.matching.settings import MatchSettings
from esco_occupation_indexer.matching.stages import load_match_manifest, save_match_manifest
from esco_occupation_indexer.matching.validation import validate_match_build
from esco_occupation_indexer.matching.verify import verify_match_build
from esco_occupation_indexer.settings import load_resolved_settings


def run_match(
    settings: MatchSettings,
    jobs_path: Path,
    categories_path: Path,
    esco_build_dir: Path,
    dense_backend=None,
    sparse_backend=None,
    allow_collection: bool = False,
) -> Path:
    compat = check_esco_compat(esco_build_dir, require_alias=not allow_collection)
    esco_settings = load_resolved_settings(esco_build_dir.resolve())
    dense_model = {
        "model_id": esco_settings.embedding.model_id,
        "model_revision": esco_settings.embedding.model_revision,
        "dimension": esco_settings.embedding.dimension,
    }
    sparse_config = esco_settings.sparse.model_dump(mode="json")
    match_dir = ingest_jobs(
        settings,
        jobs_path,
        categories_path,
        esco_build_dir,
        compat["esco_build_id"],
        compat["collection_name"],
        dense_model,
        sparse_config,
    )
    validate_match_build(match_dir)
    embed_query_build(
        match_dir, esco_settings, dense_backend=dense_backend, sparse_backend=sparse_backend
    )
    retrieve_build(
        match_dir,
        esco_settings,
        compat["collection_name"],
        compat["alias"],
        require_alias=not allow_collection,
    )
    rerank_build(match_dir, esco_build_dir)
    verify_match_build(match_dir, esco_build_dir, require_alias=not allow_collection)
    if compat.get("alias_bypassed"):
        manifest = load_match_manifest(match_dir)
        manifest.runtime["alias_bypassed"] = True
        manifest.runtime["alias_live_target"] = str(compat.get("alias_target"))
        save_match_manifest(match_dir, manifest)
    return match_dir
