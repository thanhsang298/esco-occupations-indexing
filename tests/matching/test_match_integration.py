from __future__ import annotations

import os
from contextlib import suppress
from pathlib import Path
from uuid import uuid4

import pytest
from qdrant_client import QdrantClient, models

from esco_occupation_indexer.embedding import embed_build
from esco_occupation_indexer.errors import StageError
from esco_occupation_indexer.ingest import ingest_source
from esco_occupation_indexer.matching.pipeline import run_match
from esco_occupation_indexer.matching.query_encoding import QuerySparseBackend
from esco_occupation_indexer.matching.settings import MatchSettings
from esco_occupation_indexer.matching.verify import verify_match_build
from esco_occupation_indexer.qdrant_ops import upload_build, verify_build
from esco_occupation_indexer.settings import SparseSettings
from esco_occupation_indexer.validation import validate_build
from tests.helpers import (
    FakeDenseBackend,
    FakeSparseBackend,
    make_occupation_zip,
    make_settings,
)
from tests.matching.helpers import (
    FakeQueryDenseBackend,
    make_jobs_file,
    make_taxonomy_file,
)

pytestmark = pytest.mark.integration


def _live_qdrant_client() -> QdrantClient:
    client = QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY") or None,
    )
    try:
        client.get_collections()
    except Exception as error:
        client.close()
        pytest.skip(f"Qdrant is not reachable: {error}")
    return client


def _build_synthetic_esco(tmp_path: Path, client: QdrantClient, run_id: str):
    alias = f"esco_match_test_current_{run_id}"
    prefix = f"esco_match_test_{run_id}"
    settings = make_settings(tmp_path / "esco-work")
    settings.qdrant.url = os.getenv("QDRANT_URL", "http://localhost:6333")
    settings.qdrant.collection_prefix = prefix
    settings.qdrant.alias = alias
    source = make_occupation_zip(tmp_path / "esco.zip")
    build_dir = ingest_source(settings, source)
    validate_match = validate_build(build_dir)
    assert validate_match["record_count"] == 2
    embed_build(
        build_dir,
        FakeDenseBackend(settings.embedding.dimension),
        FakeSparseBackend(),
    )
    upload = upload_build(build_dir)
    verify_build(build_dir)
    from esco_occupation_indexer.qdrant_ops import promote_build

    promote_build(build_dir)
    return build_dir, upload["collection_name"], alias


def _match_settings(tmp_path: Path) -> MatchSettings:
    return MatchSettings.model_validate(
        {
            "match_work_dir": str(tmp_path / "match-work"),
            "shard_size": 1,
            "retrieval": {"candidate_limit": 5, "top_k": 3},
        }
    )


def test_instruction_change_invalidates_match_build(tmp_path: Path) -> None:
    from esco_occupation_indexer.matching.canonicalize import ingest_jobs

    jobs = make_jobs_file(tmp_path / "jobs.json")
    taxonomy = make_taxonomy_file(tmp_path / "cats.json")
    first = ingest_jobs(
        _match_settings(tmp_path),
        jobs,
        taxonomy,
        tmp_path / "esco",
        "esco-x",
        "col-x",
        {"model_id": "m", "model_revision": "r", "dimension": 8},
        {"model_id": "Qdrant/bm25"},
    )
    altered = MatchSettings.model_validate(
        {
            "match_work_dir": str(tmp_path / "match-work"),
            "shard_size": 1,
            "queries": {"instruction": "Different instruction."},
        }
    )
    second = ingest_jobs(
        altered,
        jobs,
        taxonomy,
        tmp_path / "esco",
        "esco-x",
        "col-x",
        {"model_id": "m", "model_revision": "r", "dimension": 8},
        {"model_id": "Qdrant/bm25"},
    )
    assert first != second


@pytest.mark.skipif(
    os.getenv("RUN_QDRANT_INTEGRATION") != "1",
    reason="set RUN_QDRANT_INTEGRATION=1 with Qdrant running",
)
def test_full_match_pipeline_with_resume(tmp_path: Path) -> None:
    client = _live_qdrant_client()
    run_id = uuid4().hex
    collections: set[str] = set()
    alias = f"esco_match_test_current_{run_id}"
    try:
        esco_build, collection, _ = _build_synthetic_esco(tmp_path, client, run_id)
        collections.add(collection)
        jobs = make_jobs_file(tmp_path / "jobs.json")
        taxonomy = make_taxonomy_file(tmp_path / "cats.json")
        settings = _match_settings(tmp_path)
        dense = FakeQueryDenseBackend(dimension=8)
        sparse = QuerySparseBackend(SparseSettings())

        match_dir = run_match(settings, jobs, taxonomy, esco_build, dense, sparse)
        assert dense.encode_calls > 0
        import io

        import zstandard as zstd

        matches = []
        decompressor = zstd.ZstdDecompressor()
        with (match_dir / "matches.jsonl.zst").open("rb") as raw, decompressor.stream_reader(
            raw
        ) as reader:
            import json

            for line in io.TextIOWrapper(reader, encoding="utf-8"):
                if line.strip():
                    matches.append(json.loads(line))
        assert [match["job_id"] for match in matches] == ["topcv:1001", "topcv:1002"]
        for match in matches:
            assert 0 < len(match["candidates"]) <= 3
            assert [c["rank"] for c in match["candidates"]] == list(
                range(1, len(match["candidates"]) + 1)
            )
            assert len({c["esco_id"] for c in match["candidates"]}) == len(
                match["candidates"]
            )
            for candidate in match["candidates"]:
                assert set(candidate["channels"]) == {
                    "label_dense",
                    "semantic_dense",
                    "lexical_sparse",
                }

        # Second run resumes without re-embedding.
        resumed_dense = FakeQueryDenseBackend(dimension=8)
        rerun = run_match(settings, jobs, taxonomy, esco_build, resumed_dense, sparse)
        assert rerun == match_dir
        assert resumed_dense.encode_calls == 0

        # Alias moved away → verify must fail.
        with suppress(Exception):
            client.update_collection_aliases(
                change_aliases_operations=[
                    models.DeleteAliasOperation(
                        delete_alias=models.DeleteAlias(alias_name=alias)
                    )
                ]
            )
        with pytest.raises(StageError):
            verify_match_build(match_dir, esco_build)

        # ...but the bypass flag matches the snapshot collection directly.
        bypassed = run_match(
            settings, jobs, taxonomy, esco_build, FakeQueryDenseBackend(dimension=8),
            sparse, allow_collection=True,
        )
        assert bypassed == match_dir
        from esco_occupation_indexer.matching.stages import load_match_manifest

        assert load_match_manifest(match_dir).runtime.get("alias_bypassed") is True
    finally:
        with suppress(Exception):
            client.update_collection_aliases(
                change_aliases_operations=[
                    models.DeleteAliasOperation(
                        delete_alias=models.DeleteAlias(alias_name=alias)
                    )
                ]
            )
        for name in collections:
            with suppress(Exception):
                if client.collection_exists(name):
                    client.delete_collection(name)
        client.close()
