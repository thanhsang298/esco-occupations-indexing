from __future__ import annotations

from pathlib import Path

import pytest

from esco_occupation_indexer.artifacts import load_manifest
from esco_occupation_indexer.embedding import embed_build
from esco_occupation_indexer.errors import StageError
from esco_occupation_indexer.ingest import ingest_source
from esco_occupation_indexer.utils import sha256_file
from esco_occupation_indexer.validation import validate_build
from tests.helpers import FakeDenseBackend, FakeSparseBackend, make_occupation_zip, make_settings


def test_ingest_rebuilds_when_completed_artifact_checksum_does_not_match(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path / "work")
    build_dir = ingest_source(settings, make_occupation_zip(tmp_path / "esco.zip"))
    canonical = build_dir / "canonical-occupations.jsonl.zst"
    original = canonical.read_bytes()

    canonical.write_bytes(b"corrupt artifact")
    assert ingest_source(settings, tmp_path / "esco.zip") == build_dir

    assert canonical.read_bytes() == original
    manifest = load_manifest(build_dir)
    assert manifest.artifact_checksums[canonical.name] == sha256_file(canonical)


def test_validate_reuses_a_verified_validation_report(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "work")
    build_dir = ingest_source(settings, make_occupation_zip(tmp_path / "esco.zip"))
    first = validate_build(build_dir)
    manifest_before = (build_dir / "manifest.json").read_bytes()

    second = validate_build(build_dir)

    assert second == first
    assert (build_dir / "manifest.json").read_bytes() == manifest_before


def test_embed_rejects_canonical_artifact_changed_after_validation(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "work")
    build_dir = ingest_source(settings, make_occupation_zip(tmp_path / "esco.zip"))
    validate_build(build_dir)
    (build_dir / "canonical-occupations.jsonl.zst").write_bytes(b"corrupt artifact")

    with pytest.raises(StageError, match="Artifact checksum"):
        embed_build(
            build_dir,
            FakeDenseBackend(settings.embedding.dimension),
            FakeSparseBackend(),
        )
