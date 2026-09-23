from __future__ import annotations

import io
import os
from collections.abc import Iterable, Iterator
from pathlib import Path

import zstandard as zstd
from pydantic import BaseModel

from esco_occupation_indexer.models import BuildManifest, StageState
from esco_occupation_indexer.utils import atomic_write_json, sha256_file, utc_now

STAGE_NAMES = ("ingest", "validate", "embed", "upload", "verify", "promote")


def write_jsonl_zst(path: Path, records: Iterable[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    compressor = zstd.ZstdCompressor(level=6)
    with temporary.open("wb") as raw, compressor.stream_writer(raw) as writer:
        for record in records:
            line = record.model_dump_json(exclude_none=False) + "\n"
            writer.write(line.encode("utf-8"))
    os.replace(temporary, path)


def read_jsonl_zst[ModelT: BaseModel](path: Path, model: type[ModelT]) -> Iterator[ModelT]:
    decompressor = zstd.ZstdDecompressor()
    with path.open("rb") as raw, decompressor.stream_reader(raw) as reader:
        text_reader = io.TextIOWrapper(reader, encoding="utf-8")
        try:
            for line in text_reader:
                if line.strip():
                    yield model.model_validate_json(line)
        finally:
            text_reader.detach()


def manifest_path(build_dir: Path) -> Path:
    return build_dir / "manifest.json"


def load_manifest(build_dir: Path) -> BuildManifest:
    return BuildManifest.model_validate_json(manifest_path(build_dir).read_text(encoding="utf-8"))


def save_manifest(build_dir: Path, manifest: BuildManifest) -> None:
    manifest.updated_at = utc_now()
    atomic_write_json(manifest_path(build_dir), manifest.model_dump(mode="json"))


def new_stages() -> dict[str, StageState]:
    return {name: StageState() for name in STAGE_NAMES}


def start_stage(build_dir: Path, name: str) -> BuildManifest:
    manifest = load_manifest(build_dir)
    stage = manifest.stages[name]
    stage.status = "running"
    stage.started_at = utc_now()
    stage.finished_at = None
    stage.error = None
    save_manifest(build_dir, manifest)
    return manifest


def complete_stage(build_dir: Path, name: str) -> BuildManifest:
    manifest = load_manifest(build_dir)
    stage = manifest.stages[name]
    stage.status = "completed"
    stage.finished_at = utc_now()
    stage.error = None
    # A completed stage is a new immutable input for every following stage.
    # Preserve old artifacts for diagnosis/rollback, but require downstream
    # stages to run again rather than allowing a stale verification report to
    # authorize promotion after a rebuild or re-upload.
    stage_index = STAGE_NAMES.index(name)
    for downstream_name in STAGE_NAMES[stage_index + 1 :]:
        downstream = manifest.stages[downstream_name]
        downstream.status = "pending"
        downstream.started_at = None
        downstream.finished_at = None
        downstream.error = None
    save_manifest(build_dir, manifest)
    return manifest


def fail_stage(build_dir: Path, name: str, error: Exception) -> None:
    manifest = load_manifest(build_dir)
    stage = manifest.stages[name]
    stage.status = "failed"
    stage.finished_at = utc_now()
    stage.error = str(error)
    save_manifest(build_dir, manifest)


def require_completed(build_dir: Path, *names: str) -> BuildManifest:
    manifest = load_manifest(build_dir)
    incomplete = [name for name in names if manifest.stages[name].status != "completed"]
    if incomplete:
        from esco_occupation_indexer.errors import StageError

        raise StageError(f"Required stages are not completed: {incomplete}")
    return manifest


def artifact_checksum_matches(
    build_dir: Path, manifest: BuildManifest, filename: str
) -> bool:
    expected = manifest.artifact_checksums.get(filename)
    if not expected:
        return False
    try:
        return sha256_file(build_dir / filename) == expected
    except OSError:
        return False


def require_artifact_checksums(
    build_dir: Path, manifest: BuildManifest, *filenames: str
) -> None:
    invalid = [
        filename
        for filename in filenames
        if not artifact_checksum_matches(build_dir, manifest, filename)
    ]
    if invalid:
        from esco_occupation_indexer.errors import StageError

        raise StageError(f"Artifact checksum is missing or invalid: {invalid}")
