from pathlib import Path

import pytest

from esco_occupation_indexer.artifacts import read_jsonl_zst
from esco_occupation_indexer.errors import ValidationError
from esco_occupation_indexer.matching.canonicalize import (
    ingest_jobs,
    load_taxonomy,
    resolve_category,
)
from esco_occupation_indexer.matching.models import CanonicalJobPosting
from esco_occupation_indexer.matching.settings import MatchSettings
from esco_occupation_indexer.matching.validation import validate_match_build
from tests.matching.helpers import make_jobs_file, make_taxonomy_file


def _ingest(tmp_path: Path):
    settings = MatchSettings.model_validate({"match_work_dir": str(tmp_path / "m")})
    jobs = make_jobs_file(tmp_path / "jobs.json")
    taxonomy = make_taxonomy_file(tmp_path / "cats.json")
    match_dir = ingest_jobs(
        settings,
        jobs,
        taxonomy,
        tmp_path / "esco",
        "esco-build-test",
        "collection-test",
        {"model_id": "m", "model_revision": "r", "dimension": 8},
        {"model_id": "Qdrant/bm25"},
    )
    records = list(
        read_jsonl_zst(match_dir / "canonical-jobs.jsonl.zst", CanonicalJobPosting)
    )
    return match_dir, records


def test_ingest_keeps_completed_and_resolves_paths(tmp_path: Path) -> None:
    match_dir, records = _ingest(tmp_path)
    assert [record.job_id for record in records] == ["topcv:1001", "topcv:1002"]

    first = records[0]
    assert first.platform == "topcv"
    assert first.source_url == "https://topcv.vn/j1"
    assert first.updated_at is not None and first.updated_at.year == 2026
    assert first.knowledge == ["IT - Software"]
    assert len(first.it_category_paths) == 1
    path = first.it_category_paths[0]
    assert [node.external_id for node in path.declared_path] == ["257", "265", "1021"]
    assert [node.external_id for node in path.canonical_path] == ["257", "265", "1021"]
    assert path.terminal_label == "Backend"
    assert not path.root_conflict and not path.terminal_conflict
    assert first.content_hash

    # Declared IT but canonical non-IT: stays in IT signal with conflict flag.
    second = records[1]
    assert len(second.it_category_paths) == 1
    conflicted = second.it_category_paths[0]
    assert conflicted.declared_root == "257"
    assert conflicted.canonical_root == "1"
    assert conflicted.root_conflict
    assert conflicted.terminal_conflict  # key 92 != level3Id 1021


def test_ingest_counts_match_real_data_contract(tmp_path: Path) -> None:
    real_jobs = Path("data/iviec-job-crawler.job_details.topcv.it.json")
    real_cats = Path("data/iviec-job-crawler.job_categories.topcv.level_id.json")
    if not real_jobs.is_file():
        pytest.skip("real TopCV data not present")
    settings = MatchSettings.model_validate({"match_work_dir": str(tmp_path / "m")})
    match_dir = ingest_jobs(
        settings,
        real_jobs,
        real_cats,
        tmp_path / "esco",
        "esco-build-test",
        "collection-test",
        {"model_id": "m", "model_revision": "r", "dimension": 8},
        {"model_id": "Qdrant/bm25"},
    )
    report = validate_match_build(match_dir)
    assert report["record_count"] == 3203
    assert report["distinct_posting_keys"] == 134
    assert report["canonical_it_keys"] == 71
    assert report["canonical_non_it_keys"] == 63
    assert report["declared_it_canonical_other_rows"] == 211
    assert report["terminal_conflict_rows"] == 570


def test_taxonomy_rejects_cycle(tmp_path: Path) -> None:
    import json

    path = tmp_path / "cats.json"
    path.write_text(
        json.dumps(
            [
                {"externalId": "a", "name": "A", "parentExternalId": "b"},
                {"externalId": "b", "name": "B", "parentExternalId": "a"},
            ]
        ),
        encoding="utf-8",
    )
    taxonomy = load_taxonomy(path)
    with pytest.raises(ValidationError, match="cycle"):
        resolve_category(
            {"key": "a", "name": "A", "level1Id": "a", "level2Id": "b", "level3Id": "a"},
            taxonomy,
        )


def test_taxonomy_rejects_duplicate_id(tmp_path: Path) -> None:
    import json

    path = tmp_path / "cats.json"
    path.write_text(
        json.dumps(
            [
                {"externalId": "a", "name": "A", "parentExternalId": None},
                {"externalId": "a", "name": "A2", "parentExternalId": None},
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="Duplicate taxonomy ID"):
        load_taxonomy(path)
