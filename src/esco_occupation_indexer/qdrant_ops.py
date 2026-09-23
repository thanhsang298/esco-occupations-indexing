from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

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
from esco_occupation_indexer.documents import dense_label_variants, qdrant_payload
from esco_occupation_indexer.embedding import validate_shard
from esco_occupation_indexer.errors import StageError, ValidationError
from esco_occupation_indexer.models import CanonicalOccupationRecord
from esco_occupation_indexer.settings import IndexingSettings, load_resolved_settings
from esco_occupation_indexer.utils import atomic_write_json, sha256_file

PAYLOAD_INDEXES = {
    "entity_type": "keyword",
    "esco_id": "uuid",
    "esco_version": "keyword",
    "isco_group": "keyword",
    "isco_code": "keyword",
    "collections": "keyword",
    "hierarchy.isco_major": "keyword",
    "hierarchy.terminal_isco_uri": "keyword",
    "labels.normalized_all": "keyword",
}


def _client(settings: IndexingSettings):
    from qdrant_client import QdrantClient

    return QdrantClient(
        url=settings.qdrant.url,
        api_key=settings.qdrant.api_key,
        timeout=settings.qdrant.timeout_seconds,
    )


def _create_collection(client, collection_name: str, settings: IndexingSettings) -> None:
    from qdrant_client import models

    created = client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "label_dense": models.VectorParams(
                size=settings.embedding.dimension,
                distance=models.Distance.COSINE,
                datatype=models.Datatype.FLOAT32,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
            ),
            "semantic_dense": models.VectorParams(
                size=settings.embedding.dimension,
                distance=models.Distance.COSINE,
                datatype=models.Datatype.FLOAT32,
            ),
        },
        sparse_vectors_config={
            "lexical_sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )
    if created is False:
        raise StageError(f"Qdrant did not create collection {collection_name!r}")


def _payload_schema_type(schema: str):
    from qdrant_client import models

    return {
        "keyword": models.PayloadSchemaType.KEYWORD,
        "uuid": models.PayloadSchemaType.UUID,
    }[schema]


def _has_payload_schema_type(current: object, expected_schema: str) -> bool:
    """Accept enum and string forms returned by compatible Qdrant servers."""

    expected = _payload_schema_type(expected_schema)
    actual = getattr(current, "data_type", None)
    return actual == expected or actual == expected.value


def _ensure_payload_indexes(client, collection_name: str) -> None:
    existing = client.get_collection(collection_name).payload_schema or {}
    for field_name, schema in PAYLOAD_INDEXES.items():
        current = existing.get(field_name)
        if current is not None:
            if not _has_payload_schema_type(current, schema):
                raise ValidationError(
                    f"Qdrant payload index {field_name} has type "
                    f"{getattr(current, 'data_type', None)}, "
                    f"expected {schema}"
                )
            continue
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=_payload_schema_type(schema),
            wait=True,
        )


def _assert_collection_schema(client, collection_name: str, dimension: int) -> None:
    from qdrant_client import models

    info = client.get_collection(collection_name)
    vectors = info.config.params.vectors
    if not isinstance(vectors, dict):
        raise ValidationError("Qdrant collection must use named dense vectors")
    if set(vectors) != {"label_dense", "semantic_dense"}:
        raise ValidationError(f"Unexpected dense vector names: {sorted(vectors)}")
    for name in ("label_dense", "semantic_dense"):
        if vectors[name].size != dimension:
            raise ValidationError(f"Qdrant {name} dimension mismatch")
        if vectors[name].distance != models.Distance.COSINE:
            raise ValidationError(f"Qdrant {name} must use cosine distance")
        if vectors[name].datatype != models.Datatype.FLOAT32:
            raise ValidationError(f"Qdrant {name} must use float32 storage")
        if vectors[name].quantization_config is not None:
            raise ValidationError(f"Qdrant {name} must not use vector quantization in V1")
        if vectors[name].hnsw_config is not None:
            raise ValidationError(f"Qdrant {name} must not use per-vector HNSW tuning in V1")
    multivector = vectors["label_dense"].multivector_config
    if (
        multivector is None
        or multivector.comparator != models.MultiVectorComparator.MAX_SIM
    ):
        raise ValidationError("Qdrant label_dense must use the MAX_SIM comparator")
    if vectors["semantic_dense"].multivector_config is not None:
        raise ValidationError("Qdrant semantic_dense must be a single vector")
    sparse = info.config.params.sparse_vectors or {}
    if set(sparse) != {"lexical_sparse"}:
        raise ValidationError(f"Unexpected sparse vector names: {sorted(sparse)}")
    if sparse["lexical_sparse"].modifier != models.Modifier.IDF:
        raise ValidationError("Qdrant lexical_sparse must use the IDF modifier")
    payload_schema = info.payload_schema or {}
    for field_name, schema in PAYLOAD_INDEXES.items():
        current = payload_schema.get(field_name)
        if current is None or not _has_payload_schema_type(current, schema):
            raise ValidationError(
                f"Missing or invalid Qdrant payload index {field_name!r} ({schema})"
            )


def _load_records(
    build_dir: Path,
) -> tuple[list[CanonicalOccupationRecord], dict[str, CanonicalOccupationRecord]]:
    records = list(
        read_jsonl_zst(build_dir / "canonical-occupations.jsonl.zst", CanonicalOccupationRecord)
    )
    record_by_id = {record.esco_id: record for record in records}
    if len(record_by_id) != len(records):
        raise ValidationError("Canonical artifact contains duplicate ESCO point IDs")
    return records, record_by_id


def _validated_shard_paths(
    build_dir: Path,
    record_by_id: Mapping[str, CanonicalOccupationRecord],
    dimension: int,
) -> list[Path]:
    """Validate the complete vector inventory before any remote mutation.

    Counting upserts alone is not sufficient: duplicate shard IDs can otherwise
    overwrite one another while still making the batch count look correct.
    """

    shard_paths = sorted((build_dir / "vectors").glob("shard-*.npz"))
    if not shard_paths:
        raise ValidationError("No vector shards found for upload")

    expected_ids = set(record_by_id)
    seen_ids: set[str] = set()
    for shard_path in shard_paths:
        metadata_path = shard_path.with_suffix(".meta.json")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValidationError(
                f"Missing or invalid shard metadata: {metadata_path}"
            ) from error
        if not isinstance(metadata, dict):
            raise ValidationError(f"Invalid shard metadata object: {metadata_path}")
        if metadata.get("checksum") != sha256_file(shard_path):
            raise ValidationError(f"Shard checksum mismatch: {shard_path}")
        validate_shard(shard_path, dimension)
        with np.load(shard_path, allow_pickle=False) as shard:
            for raw_id in shard["ids"]:
                point_id = str(raw_id)
                if point_id not in expected_ids:
                    raise ValidationError(
                        f"Vector shard references unknown record ID: {point_id}"
                    )
                if point_id in seen_ids:
                    raise ValidationError(
                        f"Vector shards contain duplicate record ID: {point_id}"
                    )
                seen_ids.add(point_id)

    if seen_ids != expected_ids:
        missing = sorted(expected_ids - seen_ids)[:10]
        extra = sorted(seen_ids - expected_ids)[:10]
        raise ValidationError(
            f"Vector shard inventory does not match canonical records; "
            f"missing={missing}, extra={extra}"
        )
    return shard_paths


def _points_from_shard(
    shard_path: Path,
    record_by_id: dict[str, CanonicalOccupationRecord],
    build_id: str,
):
    from qdrant_client import models

    with np.load(shard_path, allow_pickle=False) as shard:
        ids = shard["ids"]
        semantic = shard["semantic_vectors"]
        labels = shard["label_vectors"]
        label_offsets = shard["label_offsets"]
        sparse_indices = shard["sparse_indices"]
        sparse_values = shard["sparse_values"]
        sparse_offsets = shard["sparse_offsets"]
        for index, raw_id in enumerate(ids):
            point_id = str(raw_id)
            record = record_by_id.get(point_id)
            if record is None:
                raise ValidationError(f"Vector shard references unknown record ID: {point_id}")
            label_start, label_end = int(label_offsets[index]), int(label_offsets[index + 1])
            sparse_start = int(sparse_offsets[index])
            sparse_end = int(sparse_offsets[index + 1])
            label_matrix = labels[label_start:label_end]
            expected_labels = len(dense_label_variants(record))
            if len(label_matrix) != expected_labels:
                raise ValidationError(f"Label vector alignment failed for {point_id}")
            yield models.PointStruct(
                id=point_id,
                vector={
                    "label_dense": label_matrix.tolist(),
                    "semantic_dense": semantic[index].tolist(),
                    "lexical_sparse": models.SparseVector(
                        indices=sparse_indices[sparse_start:sparse_end].astype(int).tolist(),
                        values=sparse_values[sparse_start:sparse_end].tolist(),
                    ),
                },
                payload=qdrant_payload(record, build_id),
            )


def upload_build(build_dir: Path) -> dict[str, object]:
    build_dir = build_dir.resolve()
    manifest = require_completed(build_dir, "ingest", "validate", "embed")
    require_artifact_checksums(
        build_dir,
        manifest,
        "canonical-occupations.jsonl.zst",
        "isco-groups.jsonl.zst",
        "source-report.json",
        "validation-report.json",
        "embedding-report.json",
    )
    settings = load_resolved_settings(build_dir)
    start_stage(build_dir, "upload")
    try:
        client = _client(settings)
        collection_name = manifest.collection_name
        created = False
        if not client.collection_exists(collection_name):
            _create_collection(client, collection_name, settings)
            created = True
        _ensure_payload_indexes(client, collection_name)
        _assert_collection_schema(client, collection_name, settings.embedding.dimension)

        records, record_by_id = _load_records(build_dir)
        shard_paths = _validated_shard_paths(
            build_dir, record_by_id, settings.embedding.dimension
        )
        uploaded = 0
        batch = []
        for shard_path in shard_paths:
            for point in _points_from_shard(shard_path, record_by_id, manifest.build_id):
                batch.append(point)
                if len(batch) >= settings.qdrant.upload_batch_size:
                    client.upsert(collection_name=collection_name, points=batch, wait=True)
                    uploaded += len(batch)
                    batch = []
        if batch:
            client.upsert(collection_name=collection_name, points=batch, wait=True)
            uploaded += len(batch)
        if uploaded != len(records):
            raise ValidationError(
                f"Uploaded {uploaded} points but canonical artifact contains {len(records)}"
            )

        remote_count = client.count(collection_name=collection_name, exact=True).count
        if remote_count != len(records):
            raise ValidationError(
                f"Remote point count {remote_count} differs from expected {len(records)} "
                "after upload"
            )
        report: dict[str, object] = {
            "status": "completed",
            "collection_name": collection_name,
            "collection_created": created,
            "uploaded_this_run": uploaded,
            "remote_point_count": remote_count,
            "vector_shard_count": len(shard_paths),
        }
        atomic_write_json(build_dir / "upload-report.json", report)
        manifest = require_completed(build_dir, "ingest", "validate", "embed")
        manifest.counts["uploaded_occupation_concepts"] = uploaded
        save_manifest(build_dir, manifest)
        complete_stage(build_dir, "upload")
        return report
    except Exception as error:
        fail_stage(build_dir, "upload", error)
        raise


def _vector_dict(vector: object) -> dict[str, object]:
    if isinstance(vector, Mapping):
        return dict(vector)
    root = getattr(vector, "root", None)
    if isinstance(root, Mapping):
        return dict(root)
    model_dump = getattr(vector, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dict(dumped)
    return {}


def _sparse_parts(vector: object) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(vector, Mapping):
        indices = vector.get("indices", [])
        values = vector.get("values", [])
    else:
        indices = getattr(vector, "indices", [])
        values = getattr(vector, "values", [])
    return np.asarray(indices, dtype=np.uint32), np.asarray(values, dtype=np.float32)


def _validate_remote_vectors(
    point_id: str,
    vector: object,
    *,
    dimension: int,
    expected_label_count: int,
) -> None:
    named = _vector_dict(vector)
    if set(named) != {"label_dense", "semantic_dense", "lexical_sparse"}:
        raise ValidationError(f"Missing named vector for point {point_id}")
    label_dense = np.asarray(named["label_dense"], dtype=np.float32)
    semantic_dense = np.asarray(named["semantic_dense"], dtype=np.float32)
    _validate_dense_shape(
        point_id,
        "label_dense",
        label_dense,
        (expected_label_count, dimension),
    )
    _validate_dense_shape(
        point_id,
        "semantic_dense",
        semantic_dense,
        (dimension,),
    )
    sparse_indices, sparse_values = _sparse_parts(named["lexical_sparse"])
    if (
        sparse_indices.ndim != 1
        or sparse_values.ndim != 1
        or not len(sparse_indices)
        or sparse_indices.shape != sparse_values.shape
        or not np.isfinite(sparse_values).all()
    ):
        raise ValidationError(f"Invalid lexical_sparse vector for point {point_id}")


def _validate_dense_shape(
    point_id: str,
    name: str,
    vector: np.ndarray,
    expected_shape: tuple[int, ...],
) -> None:
    if vector.shape != expected_shape or not np.isfinite(vector).all():
        raise ValidationError(
            f"Invalid {name} shape or values for {point_id}: {vector.shape}"
        )
    norms = np.linalg.norm(vector, axis=-1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValidationError(f"Non-normalized {name} vector for point {point_id}")


def verify_build(build_dir: Path) -> dict[str, object]:
    build_dir = build_dir.resolve()
    manifest = require_completed(build_dir, "ingest", "validate", "embed", "upload")
    require_artifact_checksums(
        build_dir,
        manifest,
        "canonical-occupations.jsonl.zst",
        "isco-groups.jsonl.zst",
        "source-report.json",
        "validation-report.json",
        "embedding-report.json",
    )
    settings = load_resolved_settings(build_dir)
    start_stage(build_dir, "verify")
    try:
        client = _client(settings)
        collection_name = manifest.collection_name
        _assert_collection_schema(client, collection_name, settings.embedding.dimension)
        records, record_by_id = _load_records(build_dir)
        expected_ids = {record.esco_id for record in records}
        remote_count = client.count(collection_name=collection_name, exact=True).count
        if remote_count != len(records):
            raise ValidationError(
                f"Remote point count {remote_count} differs from expected {len(records)}"
            )

        seen_ids: set[str] = set()
        isco_groups: Counter[str] = Counter()
        isco_majors: Counter[str] = Counter()
        collections: Counter[str] = Counter()
        label_counts: list[int] = []
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=collection_name,
                limit=64,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            for point in points:
                point_id = str(point.id)
                if point_id in seen_ids:
                    raise ValidationError(f"Duplicate point ID returned by Qdrant: {point_id}")
                seen_ids.add(point_id)
                record = record_by_id.get(point_id)
                if record is None:
                    raise ValidationError(f"Unexpected Qdrant point: {point_id}")
                payload = point.payload or {}
                expected_payload = qdrant_payload(record, manifest.build_id)
                if payload != expected_payload:
                    raise ValidationError(
                        f"Qdrant payload does not exactly match canonical record for "
                        f"{point_id}"
                    )
                labels = payload.get("labels") or {}
                dense_variants = labels.get("dense_variants") or []
                expected_variants = dense_label_variants(record)
                _validate_remote_vectors(
                    point_id,
                    point.vector,
                    dimension=settings.embedding.dimension,
                    expected_label_count=len(expected_variants),
                )
                isco_groups[str(payload.get("isco_group"))] += 1
                hierarchy = payload.get("hierarchy") or {}
                isco_majors[str(hierarchy.get("isco_major"))] += 1
                collections.update(str(value) for value in payload.get("collections") or [])
                label_counts.append(len(dense_variants))
            if offset is None:
                break

        if seen_ids != expected_ids:
            missing = sorted(expected_ids - seen_ids)[:10]
            extra = sorted(seen_ids - expected_ids)[:10]
            raise ValidationError(f"Remote ID mismatch; missing={missing}, extra={extra}")

        validation_report = json.loads(
            (build_dir / "validation-report.json").read_text(encoding="utf-8")
        )
        embedding_report = json.loads(
            (build_dir / "embedding-report.json").read_text(encoding="utf-8")
        )
        report: dict[str, object] = {
            "status": "completed",
            "collection_name": collection_name,
            "expected_point_count": len(records),
            "remote_point_count": remote_count,
            "verified_point_count": len(seen_ids),
            "counts_by_isco_group": dict(sorted(isco_groups.items())),
            "counts_by_isco_major": dict(sorted(isco_majors.items())),
            "counts_by_collection": dict(sorted(collections.items())),
            "dense_labels_per_concept": {
                "min": min(label_counts) if label_counts else 0,
                "max": max(label_counts) if label_counts else 0,
                "average": sum(label_counts) / len(label_counts) if label_counts else 0.0,
            },
            "missing_optional": validation_report["missing_optional"],
            "semantic_truncation_count": embedding_report[
                "semantic_truncation_count"
            ],
            "embedding_records_per_second": embedding_report.get(
                "records_per_second_total"
            ),
            "alias_promoted": False,
        }
        atomic_write_json(build_dir / "verification-report.json", report)
        manifest = require_completed(build_dir, "ingest", "validate", "embed", "upload")
        manifest.counts["verified_occupation_concepts"] = len(seen_ids)
        manifest.artifact_checksums["verification-report.json"] = sha256_file(
            build_dir / "verification-report.json"
        )
        save_manifest(build_dir, manifest)
        complete_stage(build_dir, "verify")
        return report
    except Exception as error:
        fail_stage(build_dir, "verify", error)
        raise


def promote_build(build_dir: Path) -> dict[str, object]:
    build_dir = build_dir.resolve()
    manifest = require_completed(
        build_dir, "ingest", "validate", "embed", "upload", "verify"
    )
    require_artifact_checksums(
        build_dir,
        manifest,
        "canonical-occupations.jsonl.zst",
        "isco-groups.jsonl.zst",
        "source-report.json",
        "validation-report.json",
        "embedding-report.json",
        "verification-report.json",
    )
    settings = load_resolved_settings(build_dir)
    start_stage(build_dir, "promote")
    try:
        report_path = build_dir / "verification-report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "completed":
            raise StageError("Verification report is not completed")
        if report.get("collection_name") != manifest.collection_name:
            raise StageError("Verification report targets a different collection")
        verified_count = report.get("verified_point_count")
        expected_count = report.get("expected_point_count")
        if not isinstance(verified_count, int) or verified_count != expected_count:
            raise StageError("Verification report does not contain a complete point count")

        from qdrant_client import models

        client = _client(settings)
        if not client.collection_exists(manifest.collection_name):
            raise StageError("Verified physical collection no longer exists")
        _assert_collection_schema(client, manifest.collection_name, settings.embedding.dimension)
        current_count = client.count(
            collection_name=manifest.collection_name, exact=True
        ).count
        if current_count != verified_count:
            raise ValidationError(
                "Verified collection point count changed before promotion: "
                f"{current_count} != {verified_count}"
            )
        existing = {
            alias.alias_name: alias.collection_name for alias in client.get_aliases().aliases
        }
        actions = []
        previous_collection = existing.get(settings.qdrant.alias)
        if previous_collection and previous_collection != manifest.collection_name:
            actions.append(
                models.DeleteAliasOperation(
                    delete_alias=models.DeleteAlias(alias_name=settings.qdrant.alias)
                )
            )
        if previous_collection != manifest.collection_name:
            actions.append(
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(
                        collection_name=manifest.collection_name,
                        alias_name=settings.qdrant.alias,
                    )
                )
            )
        if actions:
            client.update_collection_aliases(change_aliases_operations=actions)

        current = {
            alias.alias_name: alias.collection_name for alias in client.get_aliases().aliases
        }
        if current.get(settings.qdrant.alias) != manifest.collection_name:
            raise ValidationError("Qdrant alias did not point to the verified collection")
        report["alias_promoted"] = True
        report["alias"] = settings.qdrant.alias
        if previous_collection != manifest.collection_name:
            report["previous_collection"] = previous_collection
            report["promotion_action"] = "switched"
        else:
            report["previous_collection"] = report.get("previous_collection")
            report["promotion_action"] = "already_current"
        atomic_write_json(report_path, report)
        manifest = require_completed(
            build_dir, "ingest", "validate", "embed", "upload", "verify"
        )
        manifest.artifact_checksums["verification-report.json"] = sha256_file(report_path)
        save_manifest(build_dir, manifest)
        complete_stage(build_dir, "promote")
        return report
    except Exception as error:
        fail_stage(build_dir, "promote", error)
        raise
