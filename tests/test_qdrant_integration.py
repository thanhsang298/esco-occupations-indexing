from __future__ import annotations

import os
from contextlib import suppress
from pathlib import Path
from uuid import uuid4

import pytest
from qdrant_client import QdrantClient, models

from esco_occupation_indexer.embedding import embed_build
from esco_occupation_indexer.errors import StageError, ValidationError
from esco_occupation_indexer.ingest import esco_id_from_uri, ingest_source
from esco_occupation_indexer.normalize import normalize_label
from esco_occupation_indexer.qdrant_ops import promote_build, upload_build, verify_build
from esco_occupation_indexer.validation import validate_build
from tests.helpers import (
    CHILD_OCC_URI,
    OBSOLETE_OCC_URI,
    PARENT_OCC_URI,
    UNIT_URI,
    FakeDenseBackend,
    FakeSparseBackend,
    make_occupation_zip,
    make_settings,
)

pytestmark = pytest.mark.integration


def _live_qdrant_client() -> QdrantClient:
    client = QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY") or None,
    )
    try:
        client.get_collections()
    except Exception as error:  # pragma: no cover - depends on local Docker state
        client.close()
        pytest.skip(f"Qdrant is not reachable: {error}")
    return client


def _alias_target(client: QdrantClient, alias_name: str) -> str | None:
    return next(
        (
            alias.collection_name
            for alias in client.get_aliases().aliases
            if alias.alias_name == alias_name
        ),
        None,
    )


def _cleanup_test_collections(
    client: QdrantClient, alias_name: str, collection_names: set[str]
) -> None:
    """Remove only collections and aliases created by this test run."""
    with suppress(Exception):
        if _alias_target(client, alias_name) in collection_names:
            client.update_collection_aliases(
                change_aliases_operations=[
                    models.DeleteAliasOperation(
                        delete_alias=models.DeleteAlias(alias_name=alias_name)
                    )
                ]
            )
    for collection_name in collection_names:
        with suppress(Exception):
            if client.collection_exists(collection_name):
                client.delete_collection(collection_name)


def _configure_qdrant(settings, *, url: str, prefix: str, alias: str) -> None:
    settings.qdrant.url = url
    settings.qdrant.collection_prefix = prefix
    settings.qdrant.alias = alias


@pytest.mark.skipif(
    os.getenv("RUN_QDRANT_INTEGRATION") != "1",
    reason="set RUN_QDRANT_INTEGRATION=1 with Qdrant running",
)
def test_v1_qdrant_acceptance_contract(tmp_path: Path) -> None:
    """Exercise the V1 storage contract against a real Qdrant instance."""
    client = _live_qdrant_client()
    run_id = uuid4().hex
    alias_name = f"esco_occ_test_current_{run_id}"
    collection_prefix = f"esco_occ_test_{run_id}"
    collection_names: set[str] = set()
    settings = make_settings(tmp_path / "work")
    _configure_qdrant(
        settings,
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        prefix=collection_prefix,
        alias=alias_name,
    )
    source = make_occupation_zip(tmp_path / "esco.zip")
    child_id = esco_id_from_uri(CHILD_OCC_URI)
    parent_id = esco_id_from_uri(PARENT_OCC_URI)

    try:
        build_dir = ingest_source(settings, source)
        assert ingest_source(settings, source) == build_dir
        validation = validate_build(build_dir)
        assert validation["record_count"] == 2

        initial_dense = FakeDenseBackend(settings.embedding.dimension)
        embedding = embed_build(build_dir, initial_dense, FakeSparseBackend())
        assert embedding["records_embedded_this_run"] == 2
        assert initial_dense.encode_calls > 0

        # A repeat uses validated shard checksums rather than recomputing vectors.
        resumed_dense = FakeDenseBackend(settings.embedding.dimension)
        resumed = embed_build(build_dir, resumed_dense, FakeSparseBackend())
        assert resumed["records_embedded_this_run"] == 0
        assert all(shard["resumed"] for shard in resumed["shards"])
        assert resumed_dense.encode_calls == 0

        upload = upload_build(build_dir)
        assert upload["remote_point_count"] == 2
        collection_name = upload["collection_name"]
        assert isinstance(collection_name, str)
        collection_names.add(collection_name)

        info = client.get_collection(collection_name)
        dense_vectors = info.config.params.vectors
        assert isinstance(dense_vectors, dict)
        assert set(dense_vectors) == {"label_dense", "semantic_dense"}
        for vector_name in ("label_dense", "semantic_dense"):
            vector = dense_vectors[vector_name]
            assert vector.size == settings.embedding.dimension
            assert vector.distance == models.Distance.COSINE
            assert vector.datatype == models.Datatype.FLOAT32
        assert (
            dense_vectors["label_dense"].multivector_config.comparator
            == models.MultiVectorComparator.MAX_SIM
        )
        assert dense_vectors["semantic_dense"].multivector_config is None

        sparse_vectors = info.config.params.sparse_vectors
        assert sparse_vectors is not None
        assert set(sparse_vectors) == {"lexical_sparse"}
        assert sparse_vectors["lexical_sparse"].modifier == models.Modifier.IDF

        expected_payload_indexes = {
            "entity_type": models.PayloadSchemaType.KEYWORD,
            "esco_id": models.PayloadSchemaType.UUID,
            "esco_version": models.PayloadSchemaType.KEYWORD,
            "isco_group": models.PayloadSchemaType.KEYWORD,
            "isco_code": models.PayloadSchemaType.KEYWORD,
            "collections": models.PayloadSchemaType.KEYWORD,
            "hierarchy.isco_major": models.PayloadSchemaType.KEYWORD,
            "hierarchy.terminal_isco_uri": models.PayloadSchemaType.KEYWORD,
            "labels.normalized_all": models.PayloadSchemaType.KEYWORD,
        }
        for field_name, expected_type in expected_payload_indexes.items():
            assert info.payload_schema[field_name].data_type == expected_type

        points, _ = client.scroll(
            collection_name=collection_name,
            limit=10,
            with_payload=True,
            with_vectors=False,
        )
        point_ids = {str(point.id) for point in points}
        assert point_ids == {child_id, parent_id}
        assert esco_id_from_uri(OBSOLETE_OCC_URI) not in point_ids
        assert {point.payload["entity_type"] for point in points} == {"occupation_concept"}
        assert {point.payload["status"] for point in points} == {"released"}
        child_point = next(point for point in points if str(point.id) == child_id)
        assert child_point.payload["skills"]["essential_count"] == 2
        assert child_point.payload["skills"]["optional_count"] == 1
        assert child_point.payload["green_share"] == 0.5
        assert child_point.payload["hierarchy"]["terminal_isco_uri"] == UNIT_URI

        # MAX_SIM must find the record from a non-preferred dense label.
        dense_query = FakeDenseBackend(settings.embedding.dimension).encode(
            ["backend engineer"], batch_size=1
        )[0]
        dense_hits = client.query_points(
            collection_name=collection_name,
            query=dense_query.tolist(),
            using="label_dense",
            limit=1,
            with_payload=True,
        ).points
        assert str(dense_hits[0].id) == child_id

        # Hidden labels are deliberately omitted from dense rows but retained in BM25.
        sparse_query = FakeSparseBackend().encode(["be dev"])[0]
        sparse_hits = client.query_points(
            collection_name=collection_name,
            query=models.SparseVector(
                indices=sparse_query.indices.astype(int).tolist(),
                values=sparse_query.values.tolist(),
            ),
            using="lexical_sparse",
            limit=1,
            with_payload=True,
        ).points
        assert str(sparse_hits[0].id) == child_id

        exact_matches, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="labels.normalized_all",
                        match=models.MatchValue(value=normalize_label(" BE dev ")),
                    )
                ]
            ),
            limit=10,
            with_payload=True,
        )
        assert {str(point.id) for point in exact_matches} == {child_id}

        verification = verify_build(build_dir)
        assert verification["verified_point_count"] == 2
        promoted = promote_build(build_dir)
        assert promoted["alias_promoted"] is True
        assert _alias_target(client, alias_name) == collection_name

        # A verification failure in a distinct staged collection cannot move the
        # already-promoted alias.
        candidate_settings = make_settings(tmp_path / "work")
        candidate_settings.embedding.model_revision = "test-revision-candidate"
        _configure_qdrant(
            candidate_settings,
            url=settings.qdrant.url,
            prefix=collection_prefix,
            alias=alias_name,
        )
        candidate_build = ingest_source(candidate_settings, source)
        validate_build(candidate_build)
        embed_build(
            candidate_build,
            FakeDenseBackend(candidate_settings.embedding.dimension),
            FakeSparseBackend(),
        )
        candidate_upload = upload_build(candidate_build)
        candidate_collection = candidate_upload["collection_name"]
        assert isinstance(candidate_collection, str)
        collection_names.add(candidate_collection)
        assert candidate_collection != collection_name
        client.delete(
            collection_name=candidate_collection,
            points_selector=models.PointIdsList(points=[child_id]),
            wait=True,
        )
        with pytest.raises(ValidationError, match="Remote point count"):
            verify_build(candidate_build)
        assert _alias_target(client, alias_name) == collection_name
        with pytest.raises(StageError):
            promote_build(candidate_build)
        assert _alias_target(client, alias_name) == collection_name
    finally:
        _cleanup_test_collections(client, alias_name, collection_names)
        client.close()
