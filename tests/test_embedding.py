from pathlib import Path

from esco_occupation_indexer.embedding import embed_build, validate_shard
from esco_occupation_indexer.ingest import ingest_source
from esco_occupation_indexer.validation import validate_build
from tests.helpers import (
    FakeDenseBackend,
    FakeSparseBackend,
    make_occupation_zip,
    make_settings,
)


def test_embedding_writes_valid_resumable_shards(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "work")
    build_dir = ingest_source(settings, make_occupation_zip(tmp_path / "esco.zip"))
    validate_build(build_dir)
    backend = FakeDenseBackend(settings.embedding.dimension)
    report = embed_build(build_dir, backend, FakeSparseBackend())
    assert report["record_count"] == 2
    assert len(report["shards"]) == 2
    for shard_path in sorted((build_dir / "vectors").glob("shard-*.npz")):
        assert validate_shard(shard_path, settings.embedding.dimension)["records"] == 1

    resumed_backend = FakeDenseBackend(settings.embedding.dimension)
    resumed = embed_build(build_dir, resumed_backend, FakeSparseBackend())
    assert all(shard["resumed"] for shard in resumed["shards"])
    assert resumed["records_embedded_this_run"] == 0
    assert resumed_backend.encode_calls == 0
