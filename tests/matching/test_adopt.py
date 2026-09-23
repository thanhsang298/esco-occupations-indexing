from pathlib import Path

import pytest

from esco_occupation_indexer.matching.adopt import (
    adopt_artifact_dir,
    find_compatible_build,
)
from esco_occupation_indexer.matching.canonicalize import (
    _query_fingerprint,
    _retrieval_fingerprint,
    ingest_jobs,
)
from esco_occupation_indexer.matching.settings import MatchSettings
from esco_occupation_indexer.matching.stages import load_match_manifest
from tests.matching.helpers import make_jobs_file, make_taxonomy_file

dense_model = {"model_id": "m", "model_revision": "r", "dimension": 8}
sparse_config = {"model_id": "Qdrant/bm25"}


def _base_settings(tmp_path: Path) -> MatchSettings:
    return MatchSettings.model_validate({"match_work_dir": str(tmp_path / "m")})


def test_layered_fingerprints_split_query_and_retrieval(tmp_path: Path) -> None:
    settings = _base_settings(tmp_path)
    query_fp = _query_fingerprint(settings, "jobs", "cats", dense_model, sparse_config)
    retrieval_fp = _retrieval_fingerprint(query_fp, "esco-x", "col-x", settings)
    assert query_fp and retrieval_fp and query_fp != retrieval_fp

    altered = MatchSettings.model_validate(
        {
            "match_work_dir": str(tmp_path / "m"),
            "retrieval": {
                "rrf_weights": {
                    "label_dense": 1.0,
                    "semantic_dense": 2.0,
                    "lexical_sparse": 1.0,
                }
            },
        }
    )
    assert (
        _query_fingerprint(altered, "jobs", "cats", dense_model, sparse_config)
        == query_fp
    )
    assert (
        _retrieval_fingerprint(query_fp, "esco-x", "col-x", altered) != retrieval_fp
    )


def _write_donor(tmp_path: Path, fingerprint: str, stage: str, subdir: str) -> Path:
    donor = tmp_path / "m" / "donor12345678"
    (donor / subdir).mkdir(parents=True)
    filename = "shard-00000.npz" if subdir == "query-vectors" else "shard-00000.jsonl.zst"
    shard = donor / subdir / filename
    shard.write_bytes(b"fake-shard")
    import hashlib
    import json as json_module

    (donor / subdir / shard.with_suffix(".meta.json").name).write_text(
        json_module.dumps(
            {
                "checksum": hashlib.sha256(b"fake-shard").hexdigest(),
                "job_ids": ["topcv:1"],
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "topcv_match_build/v1",
        "match_build_id": "donor12345678",
        "jobs_file": "j",
        "categories_file": "c",
        "esco_build_dir": "e",
        "esco_build_id": "esco-x",
        "esco_collection_snapshot": "col-x",
        "source_checksums": {},
        "config_fingerprint": "x",
        "query_fingerprint": fingerprint if subdir == "query-vectors" else "other",
        "retrieval_fingerprint": fingerprint if subdir == "retrieval" else "other",
        "match_config": {},
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "stages": {
            name: {"status": "completed" if name == stage else "pending"}
            for name in ("ingest", "validate", "embed_queries", "retrieve", "verify")
        },
        "counts": {},
        "runtime": {},
        "artifact_checksums": {},
    }
    (donor / "manifest.json").write_text(json_module.dumps(manifest), encoding="utf-8")
    return donor


def test_find_compatible_build_matches_fingerprint_and_stage(tmp_path: Path) -> None:
    work = tmp_path / "m"
    work.mkdir()
    target = work / "target"
    target.mkdir()
    donor = _write_donor(tmp_path, "fp-1", "embed_queries", "query-vectors")
    assert (
        find_compatible_build(work, target, "query_fingerprint", "fp-1", "embed_queries")
        == donor
    )
    assert (
        find_compatible_build(work, target, "query_fingerprint", "fp-2", "embed_queries")
        is None
    )
    assert (
        find_compatible_build(work, target, "query_fingerprint", "fp-1", "retrieve")
        is None
    )
    assert find_compatible_build(work, target, "query_fingerprint", "", "embed_queries") is None


def test_adopt_copies_verified_shards(tmp_path: Path) -> None:
    work = tmp_path / "m"
    work.mkdir()
    donor = _write_donor(tmp_path, "fp-1", "embed_queries", "query-vectors")
    dest = work / "dest"
    dest.mkdir()
    assert adopt_artifact_dir(donor, dest, "query-vectors") == 1
    assert (dest / "query-vectors" / "shard-00000.npz").read_bytes() == b"fake-shard"


def test_adopt_rejects_tampered_source(tmp_path: Path) -> None:
    work = tmp_path / "m"
    work.mkdir()
    donor = _write_donor(tmp_path, "fp-1", "embed_queries", "query-vectors")
    (donor / "query-vectors" / "shard-00000.npz").write_bytes(b"tampered")
    dest = work / "dest"
    dest.mkdir()
    with pytest.raises(ValueError, match="invalid"):
        adopt_artifact_dir(donor, dest, "query-vectors")


def test_ingest_records_layered_fingerprints(tmp_path: Path) -> None:
    settings = _base_settings(tmp_path)
    jobs = make_jobs_file(tmp_path / "jobs.json")
    taxonomy = make_taxonomy_file(tmp_path / "cats.json")
    match_dir = ingest_jobs(
        settings, jobs, taxonomy, tmp_path / "esco", "esco-x", "col-x",
        dense_model, sparse_config,
    )
    manifest = load_match_manifest(match_dir)
    assert manifest.query_fingerprint
    assert manifest.retrieval_fingerprint
    assert manifest.query_fingerprint != manifest.retrieval_fingerprint
