from __future__ import annotations

import json
from pathlib import Path

from esco_occupation_indexer.artifacts import load_manifest
from esco_occupation_indexer.errors import StageError, ValidationError
from esco_occupation_indexer.settings import IndexingSettings, load_resolved_settings
from esco_occupation_indexer.utils import sha256_file

EXPECTED_PAYLOAD_SCHEMA = "esco_occupation_point/v1"


def esco_client(esco_settings: IndexingSettings):
    from qdrant_client import QdrantClient

    return QdrantClient(
        url=esco_settings.qdrant.url,
        api_key=esco_settings.qdrant.api_key,
        timeout=esco_settings.qdrant.timeout_seconds,
    )


def _alias_target(client, alias: str) -> str | None:
    return next(
        (
            item.collection_name
            for item in client.get_aliases().aliases
            if item.alias_name == alias
        ),
        None,
    )


def check_esco_compat(esco_build_dir: Path, require_alias: bool = True) -> dict[str, object]:
    """Validate ESCO build artifacts and snapshot the live alias target.

    With require_alias=False the run matches the manifest collection directly
    and records the live alias target as a warning instead of failing.
    """
    esco_build_dir = esco_build_dir.resolve()
    manifest = load_manifest(esco_build_dir)
    for stage in ("ingest", "validate", "embed", "upload", "verify"):
        if manifest.stages[stage].status != "completed":
            raise StageError(f"ESCO build stage not completed: {stage}")
    for filename in (
        "canonical-occupations.jsonl.zst",
        "isco-groups.jsonl.zst",
        "source-report.json",
        "validation-report.json",
        "embedding-report.json",
    ):
        expected = manifest.artifact_checksums.get(filename)
        if not expected or sha256_file(esco_build_dir / filename) != expected:
            raise StageError(f"ESCO artifact checksum invalid: {filename}")

    settings = load_resolved_settings(esco_build_dir)
    embedding_report = json.loads(
        (esco_build_dir / "embedding-report.json").read_text(encoding="utf-8")
    )
    if embedding_report.get("status") != "completed":
        raise StageError("ESCO embedding report is not completed")

    client = esco_client(settings)
    try:
        target = _alias_target(client, settings.qdrant.alias)
        alias_bypassed = False
        if target != manifest.collection_name:
            if require_alias:
                raise StageError(
                    f"ESCO alias {settings.qdrant.alias!r} points to {target!r}, "
                    f"expected {manifest.collection_name!r}"
                )
            alias_bypassed = True
        from qdrant_client import models

        info = client.get_collection(manifest.collection_name)
        vectors = info.config.params.vectors
        if not isinstance(vectors, dict) or set(vectors) != {
            "label_dense",
            "semantic_dense",
        }:
            raise ValidationError("ESCO collection has unexpected dense vectors")
        if vectors["label_dense"].size != settings.embedding.dimension:
            raise ValidationError("ESCO collection dimension mismatch")
        if (
            vectors["label_dense"].multivector_config is None
            or vectors["label_dense"].multivector_config.comparator
            != models.MultiVectorComparator.MAX_SIM
        ):
            raise ValidationError("ESCO label_dense must use MAX_SIM")
        sparse = info.config.params.sparse_vectors or {}
        if set(sparse) != {"lexical_sparse"}:
            raise ValidationError("ESCO collection has unexpected sparse vectors")
        if sparse["lexical_sparse"].modifier != models.Modifier.IDF:
            raise ValidationError("ESCO lexical_sparse must use IDF")

        points, _ = client.scroll(
            collection_name=manifest.collection_name,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            raise StageError("ESCO collection is empty")
        payload = points[0].payload or {}
        if payload.get("schema_version") != EXPECTED_PAYLOAD_SCHEMA:
            raise ValidationError(
                f"ESCO payload schema {payload.get('schema_version')!r} != "
                f"{EXPECTED_PAYLOAD_SCHEMA!r}"
            )
        remote_count = client.count(
            collection_name=manifest.collection_name, exact=True
        ).count
    finally:
        client.close()

    backend_meta = embedding_report.get("backend_metadata") or {}
    runtime_sha = backend_meta.get("tei_model_sha")
    status = "sha_verified" if runtime_sha else "configured_revision_verified"

    return {
        "status": status,
        "esco_build_id": manifest.build_id,
        "collection_name": manifest.collection_name,
        "alias": settings.qdrant.alias,
        "alias_target": target,
        "alias_bypassed": alias_bypassed,
        "remote_point_count": remote_count,
        "embedding_model_id": settings.embedding.model_id,
        "embedding_model_revision": settings.embedding.model_revision,
        "embedding_dimension": settings.embedding.dimension,
        "embedding_max_tokens": settings.embedding.max_tokens,
        "runtime_model_sha": runtime_sha,
    }


def assert_alias_snapshot(
    client, alias: str, snapshot: str, require_alias: bool = True
) -> None:
    if require_alias:
        target = _alias_target(client, alias)
        if target != snapshot:
            raise StageError(
                f"ESCO alias {alias!r} moved during run: {target!r} != snapshot {snapshot!r}"
            )
    elif not client.collection_exists(snapshot):
        raise StageError(f"ESCO snapshot collection no longer exists: {snapshot!r}")
