from __future__ import annotations

import io
import math
from contextlib import suppress
from pathlib import Path

import zstandard as zstd

from esco_occupation_indexer.artifacts import read_jsonl_zst
from esco_occupation_indexer.errors import ValidationError
from esco_occupation_indexer.matching.compat import assert_alias_snapshot, esco_client
from esco_occupation_indexer.matching.models import CanonicalJobPosting, JobMatches
from esco_occupation_indexer.matching.stages import (
    complete_match_stage,
    fail_match_stage,
    require_match_artifact_checksums,
    require_match_completed,
    save_match_manifest,
    start_match_stage,
)
from esco_occupation_indexer.settings import load_resolved_settings
from esco_occupation_indexer.utils import atomic_write_json, sha256_file


def _read_matches(path: Path) -> list[JobMatches]:
    decompressor = zstd.ZstdDecompressor()
    records: list[JobMatches] = []
    with path.open("rb") as raw, decompressor.stream_reader(raw) as reader:
        for line in io.TextIOWrapper(reader, encoding="utf-8"):
            if line.strip():
                records.append(JobMatches.model_validate_json(line))
    return records


def verify_match_build(
    match_dir: Path, esco_build_dir: Path, require_alias: bool = True
) -> dict[str, object]:
    match_dir = match_dir.resolve()
    manifest = require_match_completed(match_dir, "ingest", "validate", "embed_queries", "retrieve")
    require_match_artifact_checksums(
        match_dir, manifest, "matches.jsonl.zst", "retrieval-report.json"
    )
    start_match_stage(match_dir, "verify")
    try:
        matches = _read_matches(match_dir / "matches.jsonl.zst")
        jobs = list(
            read_jsonl_zst(match_dir / "canonical-jobs.jsonl.zst", CanonicalJobPosting)
        )
        if len(matches) != len(jobs):
            raise ValidationError(
                f"Match lines {len(matches)} != canonical jobs {len(jobs)}"
            )
        job_ids = [record.job_id for record in jobs]
        if [match.job_id for match in matches] != sorted(job_ids):
            raise ValidationError("Matches are not sorted by job_id")
        hashes = {record.job_id: record.content_hash for record in jobs}
        candidate_counts: list[int] = []
        reranked_counts: list[int] = []
        for match in matches:
            if match.job_content_hash != hashes.get(match.job_id, ""):
                raise ValidationError(f"Stale content hash for {match.job_id}")
            if match.esco_build_id != manifest.esco_build_id:
                raise ValidationError(f"Wrong ESCO build for {match.job_id}")
            esco_ids = [candidate.esco_id for candidate in match.candidates]
            if len(esco_ids) != len(set(esco_ids)):
                raise ValidationError(f"Duplicate candidate for {match.job_id}")
            if [candidate.rank for candidate in match.candidates] != list(
                range(1, len(esco_ids) + 1)
            ):
                raise ValidationError(f"Non-continuous ranks for {match.job_id}")
            candidate_counts.append(len(esco_ids))
            scored = [c for c in match.candidates if c.rerank_score is not None]
            if scored:
                if any(not isinstance(c.rerank_score, float) for c in scored):
                    raise ValidationError(f"Non-float rerank score for {match.job_id}")
                if any(not math.isfinite(c.rerank_score) for c in scored):
                    raise ValidationError(f"Non-finite rerank score for {match.job_id}")
                ranks = sorted(
                    c.rerank_rank for c in scored if c.rerank_rank is not None
                )
                if ranks != list(range(1, len(scored) + 1)):
                    raise ValidationError(f"Non-continuous rerank ranks for {match.job_id}")
            reranked_counts.append(len(scored))

        esco_settings = load_resolved_settings(esco_build_dir.resolve())
        client = esco_client(esco_settings)
        try:
            assert_alias_snapshot(
                client,
                esco_settings.qdrant.alias,
                manifest.esco_collection_snapshot,
                require_alias,
            )
            seen: set[str] = set()
            offset = None
            while True:
                points, offset = client.scroll(
                    collection_name=manifest.esco_collection_snapshot,
                    limit=256,
                    offset=offset,
                    with_payload=False,
                    with_vectors=False,
                )
                seen.update(str(point.id) for point in points)
                if offset is None:
                    break
            unknown: set[str] = set()
            for match in matches:
                for candidate in match.candidates:
                    if candidate.esco_id not in seen:
                        unknown.add(candidate.esco_id)
            if unknown:
                raise ValidationError(
                    f"Candidates missing from ESCO collection: {sorted(unknown)[:5]}"
                )
        finally:
            client.close()

        report: dict[str, object] = {
            "status": "completed",
            "job_count": len(matches),
            "esco_point_count": len(seen),
            "candidates_per_job": {
                "min": min(candidate_counts) if candidate_counts else 0,
                "max": max(candidate_counts) if candidate_counts else 0,
                "average": sum(candidate_counts) / len(candidate_counts)
                if candidate_counts
                else 0.0,
            },
            "alias_snapshot": manifest.esco_collection_snapshot,
            "alias_stable": require_alias,
        }
        atomic_write_json(match_dir / "match-verification-report.json", report)
        manifest = require_match_completed(
            match_dir, "ingest", "validate", "embed_queries", "retrieve"
        )
        manifest.counts["verified_matches"] = len(matches)
        manifest.artifact_checksums["match-verification-report.json"] = sha256_file(
            match_dir / "match-verification-report.json"
        )
        save_match_manifest(match_dir, manifest)
        complete_match_stage(match_dir, "verify")
        return report
    except Exception as error:
        with suppress(Exception):
            fail_match_stage(match_dir, "verify", error)
        raise
