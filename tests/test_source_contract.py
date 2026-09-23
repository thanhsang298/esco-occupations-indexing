from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from esco_occupation_indexer.errors import SourceDataError
from esco_occupation_indexer.source import _parse_csv, load_source_package
from tests.helpers import make_occupation_zip


def _replace_zip_member(path: Path, member_name: str, replacement: bytes) -> None:
    with zipfile.ZipFile(path) as source:
        members = {
            info.filename: source.read(info.filename)
            for info in source.infolist()
            if not info.is_dir()
        }
    target = next(name for name in members if name.endswith(member_name))
    members[target] = replacement
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as destination:
        for name, content in members.items():
            destination.writestr(name, content)


def test_csv_parser_preserves_quoted_newlines_and_optional_empty_fields() -> None:
    table = _parse_csv(
        "occupations",
        "occupations_en.csv",
        (
            b"conceptUri,preferredLabel,description,definition\n"
            b"urn:test,\"REST API\",\"First line\nSecond line\",\"\"\n"
        ),
    )

    assert table.rows == (
        {
            "conceptUri": "urn:test",
            "preferredLabel": "REST API",
            "description": "First line\nSecond line",
            "definition": "",
        },
    )


def test_source_package_rejects_missing_required_occupation_header(tmp_path: Path) -> None:
    source = make_occupation_zip(tmp_path / "esco.zip")
    _replace_zip_member(
        source,
        "occupations_en.csv",
        b"conceptUri,preferredLabel,status\n",
    )

    with pytest.raises(SourceDataError, match="conceptType"):
        load_source_package(source)


def test_source_package_accepts_zip_without_optional_collections(tmp_path: Path) -> None:
    source = make_occupation_zip(tmp_path / "esco.zip")
    with zipfile.ZipFile(source) as archive:
        members = {
            info.filename: archive.read(info.filename)
            for info in archive.infolist()
            if not info.is_dir()
            and not info.filename.endswith("researchOccupationsCollection_en.csv")
            and not info.filename.endswith("greenShareOcc_en.csv")
        }
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)

    package = load_source_package(source)

    assert set(package.tables) == {"occupations", "isco_groups", "broader", "relations"}


def test_source_package_rejects_missing_required_member(tmp_path: Path) -> None:
    source = make_occupation_zip(tmp_path / "esco.zip")
    with zipfile.ZipFile(source) as archive:
        members = {
            info.filename: archive.read(info.filename)
            for info in archive.infolist()
            if not info.is_dir()
            and not info.filename.endswith("occupationSkillRelations_en.csv")
        }
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)

    with pytest.raises(SourceDataError, match="occupationSkillRelations_en.csv"):
        load_source_package(source)
