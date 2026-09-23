from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from esco_occupation_indexer.errors import SourceDataError
from esco_occupation_indexer.utils import sha256_bytes

REQUIRED_FILES = {
    "occupations": "occupations_en.csv",
    "isco_groups": "ISCOGroups_en.csv",
    "broader": "broaderRelationsOccPillar_en.csv",
    "relations": "occupationSkillRelations_en.csv",
}

OPTIONAL_FILES = {
    "research": "researchOccupationsCollection_en.csv",
    "green": "greenShareOcc_en.csv",
}

OCCUPATION_REQUIRED_COLUMNS = {
    "conceptType",
    "conceptUri",
    "preferredLabel",
    "status",
}
ISCO_REQUIRED_COLUMNS = {"conceptUri", "code", "preferredLabel"}
BROADER_REQUIRED_COLUMNS = {"conceptUri", "broaderUri"}
RELATIONS_REQUIRED_COLUMNS = {"occupationUri", "relationType", "skillUri"}


@dataclass(frozen=True)
class CsvTable:
    logical_name: str
    filename: str
    headers: tuple[str, ...]
    rows: tuple[dict[str, str], ...]
    checksum: str


@dataclass(frozen=True)
class SourcePackage:
    tables: dict[str, CsvTable]
    zip_checksum: str


def _find_member(
    archive: zipfile.ZipFile, expected_basename: str, *, required: bool
) -> str | None:
    matches = [
        name
        for name in archive.namelist()
        if not name.endswith("/")
        and PurePosixPath(name.replace("\\", "/")).name.casefold()
        == expected_basename.casefold()
    ]
    if len(matches) > 1:
        raise SourceDataError(
            f"Expected at most one {expected_basename!r} in ZIP, found {len(matches)}"
        )
    if not matches:
        if required:
            raise SourceDataError(f"Expected {expected_basename!r} in ZIP, found none")
        return None
    return matches[0]


def _parse_csv(logical_name: str, filename: str, content: bytes) -> CsvTable:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise SourceDataError(f"{filename} is not valid UTF-8") from error

    reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    if not reader.fieldnames:
        raise SourceDataError(f"{filename} has no CSV header")
    if any(header is None for header in reader.fieldnames):
        raise SourceDataError(f"{filename} has an invalid CSV header")
    headers = tuple(header.strip() for header in reader.fieldnames)
    if not all(headers):
        raise SourceDataError(f"{filename} has an empty CSV header name")
    duplicates = sorted({header for header in headers if headers.count(header) > 1})
    if duplicates:
        raise SourceDataError(f"{filename} has duplicate CSV header names: {duplicates}")
    rows: list[dict[str, str]] = []
    try:
        for row_number, source_row in enumerate(reader, start=2):
            if None in source_row:
                raise SourceDataError(f"{filename}:{row_number} has more values than headers")
            row = {
                str(key).strip(): (value or "")
                for key, value in source_row.items()
                if key is not None
            }
            if any(value.strip() for value in row.values()):
                rows.append(row)
    except csv.Error as error:
        raise SourceDataError(f"{filename} is not valid CSV: {error}") from error
    return CsvTable(
        logical_name=logical_name,
        filename=Path(filename).name,
        headers=headers,
        rows=tuple(rows),
        checksum=sha256_bytes(content),
    )


def _require_columns(table: CsvTable, required: set[str]) -> None:
    missing = sorted(required.difference(table.headers))
    if missing:
        raise SourceDataError(f"{table.filename} is missing required columns: {missing}")


def load_source_package(source_zip: Path) -> SourcePackage:
    if not source_zip.is_file():
        raise SourceDataError(f"Source ZIP does not exist: {source_zip}")
    try:
        content = source_zip.read_bytes()
    except OSError as error:
        raise SourceDataError(f"Could not read source ZIP: {source_zip}") from error
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as error:
        raise SourceDataError(f"Source is not a valid ZIP file: {source_zip}") from error

    with archive:
        tables: dict[str, CsvTable] = {}
        for logical_name, expected_name in REQUIRED_FILES.items():
            member = _find_member(archive, expected_name, required=True)
            assert member is not None
            try:
                member_content = archive.read(member)
            except (RuntimeError, NotImplementedError, zipfile.BadZipFile) as error:
                raise SourceDataError(f"Could not read {expected_name} from source ZIP") from error
            tables[logical_name] = _parse_csv(logical_name, member, member_content)
        for logical_name, expected_name in OPTIONAL_FILES.items():
            member = _find_member(archive, expected_name, required=False)
            if member is None:
                continue
            try:
                member_content = archive.read(member)
            except (RuntimeError, NotImplementedError, zipfile.BadZipFile) as error:
                raise SourceDataError(f"Could not read {expected_name} from source ZIP") from error
            tables[logical_name] = _parse_csv(logical_name, member, member_content)

    _require_columns(tables["occupations"], OCCUPATION_REQUIRED_COLUMNS)
    _require_columns(tables["isco_groups"], ISCO_REQUIRED_COLUMNS)
    _require_columns(tables["broader"], BROADER_REQUIRED_COLUMNS)
    _require_columns(tables["relations"], RELATIONS_REQUIRED_COLUMNS)

    return SourcePackage(tables=tables, zip_checksum=sha256_bytes(content))


def collection_uris(table: CsvTable) -> set[str]:
    """URIs of the research collection (conceptUri column)."""
    if "conceptUri" not in table.headers:
        raise SourceDataError(f"{table.filename} must contain a conceptUri column")
    return {row["conceptUri"].strip() for row in table.rows if row["conceptUri"].strip()}
