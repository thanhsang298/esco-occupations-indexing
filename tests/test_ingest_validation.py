from pathlib import Path

import pytest

from esco_occupation_indexer.artifacts import load_manifest, read_jsonl_zst
from esco_occupation_indexer.documents import dense_label_variants, semantic_text
from esco_occupation_indexer.errors import ValidationError
from esco_occupation_indexer.ingest import _deduplicate_occupation_rows, ingest_source
from esco_occupation_indexer.models import CanonicalOccupationRecord
from esco_occupation_indexer.validation import validate_build
from tests.helpers import (
    CHILD_OCC_URI,
    PARENT_OCC_URI,
    SKILL_1,
    SKILL_2,
    SKILL_3,
    UNIT_URI,
    make_occupation_zip,
    make_settings,
)


def test_ingest_and_validate_synthetic_package(tmp_path: Path) -> None:
    source = make_occupation_zip(tmp_path / "esco.zip")
    settings = make_settings(tmp_path / "work")
    build_dir = ingest_source(settings, source)
    report = validate_build(build_dir)

    records = list(
        read_jsonl_zst(
            build_dir / "canonical-occupations.jsonl.zst", CanonicalOccupationRecord
        )
    )
    assert len(records) == 2
    assert report["record_count"] == 2
    assert report["counts_by_isco_major"] == {"2": 2}

    child = next(record for record in records if record.esco_uri == CHILD_OCC_URI)
    assert child.hierarchy.parent_type == "occupation"
    assert child.hierarchy.parent_uri == PARENT_OCC_URI
    assert child.hierarchy.terminal_isco_uri == UNIT_URI
    assert child.hierarchy.isco_major == "2"
    assert child.collections == ["research"]
    assert child.green_share == 0.5
    assert child.essential_skills == sorted([SKILL_1, SKILL_2])
    assert child.optional_skills == [SKILL_3]
    assert [item.text for item in dense_label_variants(child)] == [
        "backend developer",
        "backend engineer",
    ]
    assert "be dev" not in [item.text for item in dense_label_variants(child)]
    assert "Occupation: backend developer" in semantic_text(child)
    assert "Description: Develops REST APIs and backend services." in semantic_text(child)
    assert (
        "Context: Professionals > Software and applications developers and analysts"
        " > software developer" in semantic_text(child)
    )

    parent = next(record for record in records if record.esco_uri == PARENT_OCC_URI)
    assert parent.hierarchy.parent_type == "isco_group"
    assert parent.essential_skills == []
    assert parent.collections == []

    manifest = load_manifest(build_dir)
    assert manifest.counts["source_occupation_concepts"] == 3
    assert manifest.counts["released_occupation_concepts"] == 2
    assert manifest.counts["obsolete_occupation_concepts"] == 1
    assert manifest.counts["essential_relations"] == 2
    assert manifest.counts["optional_relations"] == 1
    assert manifest.stages["validate"].status == "completed"


def test_same_source_and_config_produce_same_build_id(tmp_path: Path) -> None:
    source = make_occupation_zip(tmp_path / "esco.zip")
    settings = make_settings(tmp_path / "work")
    first = ingest_source(settings, source)
    second = ingest_source(settings, source)
    assert first == second


def test_duplicate_occupation_rows_keep_latest_modified_record() -> None:
    older = {
        "conceptUri": CHILD_OCC_URI,
        "preferredLabel": "backend developer",
        "modifiedDate": "2025-07-31",
    }
    newer = {**older, "modifiedDate": "2025-11-26"}

    assert _deduplicate_occupation_rows((older, newer)) == [newer]


def test_duplicate_occupation_rows_reject_conflicting_content() -> None:
    first = {
        "conceptUri": CHILD_OCC_URI,
        "preferredLabel": "backend developer",
        "modifiedDate": "2025-07-31",
    }
    conflicting = {**first, "preferredLabel": "different label", "modifiedDate": "2025-11-26"}

    with pytest.raises(ValidationError, match="Conflicting duplicate ESCO URI"):
        _deduplicate_occupation_rows((first, conflicting))


def test_relations_reject_invalid_skill_uri(tmp_path: Path) -> None:
    import zipfile

    source = make_occupation_zip(tmp_path / "esco.zip")
    with zipfile.ZipFile(source) as archive:
        members = {
            info.filename: archive.read(info.filename) for info in archive.infolist()
        }
    target = next(name for name in members if name.endswith("occupationSkillRelations_en.csv"))
    members[target] = (
        b"occupationUri,occupationLabel,relationType,skillType,skillUri,skillLabel\n"
        + f"{CHILD_OCC_URI},backend developer,essential,skill/competence,not-a-uri,skill\n".encode()
    )
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)

    with pytest.raises(ValidationError, match="Invalid ESCO skill URI"):
        ingest_source(make_settings(tmp_path / "work"), source)
