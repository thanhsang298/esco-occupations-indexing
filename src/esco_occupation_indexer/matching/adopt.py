from __future__ import annotations

import json
import shutil
from pathlib import Path

from esco_occupation_indexer.matching.stages import load_match_manifest
from esco_occupation_indexer.utils import sha256_file


def _manifest_if_compatible(
    candidate: Path, fingerprint_field: str, fingerprint: str, stage: str
):
    manifest_file = candidate / "manifest.json"
    if not manifest_file.is_file():
        return None
    try:
        manifest = load_match_manifest(candidate)
    except (ValueError, OSError):
        return None
    if getattr(manifest, fingerprint_field, "") != fingerprint:
        return None
    if manifest.stages.get(stage) is None or manifest.stages[stage].status != "completed":
        return None
    return manifest


def find_compatible_build(
    work_dir: Path,
    exclude: Path,
    fingerprint_field: str,
    fingerprint: str,
    stage: str,
) -> Path | None:
    """Newest sibling build with equal layered fingerprint and completed stage."""
    if not fingerprint:
        return None
    candidates = sorted(
        (path for path in work_dir.iterdir() if path.is_dir() and path != exclude),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if _manifest_if_compatible(candidate, fingerprint_field, fingerprint, stage) is not None:
            return candidate
    return None


def _verify_shard_meta(shard_path: Path) -> bool:
    meta_path = shard_path.with_suffix(".meta.json")
    if not shard_path.is_file() or not meta_path.is_file():
        return False
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return False
    return metadata.get("checksum") == sha256_file(shard_path)


def adopt_artifact_dir(source: Path, dest: Path, subdir: str) -> int:
    """Copy a shard directory after verifying every shard checksum. Returns count."""
    src_dir = source / subdir
    if not src_dir.is_dir():
        raise FileNotFoundError(f"No {subdir} in {source}")
    dest_dir = dest / subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    adopted = 0
    for shard_path in sorted(src_dir.iterdir()):
        if shard_path.suffix not in {".npz", ".zst"}:
            continue
        meta_path = shard_path.with_suffix(".meta.json")
        if not _verify_shard_meta(shard_path):
            raise ValueError(f"Adoption source shard invalid: {shard_path}")
        shutil.copy2(shard_path, dest_dir / shard_path.name)
        if meta_path.is_file():
            shutil.copy2(meta_path, dest_dir / meta_path.name)
        adopted += 1
    if not adopted:
        raise ValueError(f"No shards to adopt in {src_dir}")
    return adopted
