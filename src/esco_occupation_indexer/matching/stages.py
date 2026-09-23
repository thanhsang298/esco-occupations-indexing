from __future__ import annotations

from pathlib import Path

from esco_occupation_indexer.matching.models import MatchManifest
from esco_occupation_indexer.utils import atomic_write_json, sha256_file, utc_now

MATCH_STAGE_NAMES = ("ingest", "validate", "embed_queries", "retrieve", "rerank", "verify")


def load_match_manifest(match_dir: Path) -> MatchManifest:
    return MatchManifest.model_validate_json(
        (match_dir / "manifest.json").read_text(encoding="utf-8")
    )


def save_match_manifest(match_dir: Path, manifest: MatchManifest) -> None:
    manifest.updated_at = utc_now()
    atomic_write_json(match_dir / "manifest.json", manifest.model_dump(mode="json"))


def new_match_stages() -> dict[str, object]:
    from esco_occupation_indexer.matching.models import StageState

    return {name: StageState() for name in MATCH_STAGE_NAMES}


def start_match_stage(match_dir: Path, name: str) -> MatchManifest:
    manifest = load_match_manifest(match_dir)
    stage = manifest.stages[name]
    stage.status = "running"
    stage.started_at = utc_now()
    stage.finished_at = None
    stage.error = None
    save_match_manifest(match_dir, manifest)
    return manifest


def complete_match_stage(match_dir: Path, name: str) -> MatchManifest:
    manifest = load_match_manifest(match_dir)
    stage = manifest.stages[name]
    stage.status = "completed"
    stage.finished_at = utc_now()
    stage.error = None
    stage_index = MATCH_STAGE_NAMES.index(name)
    for downstream_name in MATCH_STAGE_NAMES[stage_index + 1 :]:
        downstream = manifest.stages[downstream_name]
        downstream.status = "pending"
        downstream.started_at = None
        downstream.finished_at = None
        downstream.error = None
    save_match_manifest(match_dir, manifest)
    return manifest


def fail_match_stage(match_dir: Path, name: str, error: Exception) -> None:
    manifest = load_match_manifest(match_dir)
    stage = manifest.stages[name]
    stage.status = "failed"
    stage.finished_at = utc_now()
    stage.error = str(error)
    save_match_manifest(match_dir, manifest)


def require_match_completed(match_dir: Path, *names: str) -> MatchManifest:
    from esco_occupation_indexer.errors import StageError

    manifest = load_match_manifest(match_dir)
    incomplete = [name for name in names if manifest.stages[name].status != "completed"]
    if incomplete:
        raise StageError(f"Required match stages are not completed: {incomplete}")
    return manifest


def match_artifact_checksum_matches(
    match_dir: Path, manifest: MatchManifest, filename: str
) -> bool:
    expected = manifest.artifact_checksums.get(filename)
    if not expected:
        return False
    try:
        return sha256_file(match_dir / filename) == expected
    except OSError:
        return False


def require_match_artifact_checksums(
    match_dir: Path, manifest: MatchManifest, *filenames: str
) -> None:
    from esco_occupation_indexer.errors import StageError

    invalid = [
        filename
        for filename in filenames
        if not match_artifact_checksum_matches(match_dir, manifest, filename)
    ]
    if invalid:
        raise StageError(f"Match artifact checksum is missing or invalid: {invalid}")
