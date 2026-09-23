from __future__ import annotations

import json
from pathlib import Path

from esco_occupation_indexer.artifacts import read_jsonl_zst
from esco_occupation_indexer.errors import ValidationError
from esco_occupation_indexer.matching.models import CanonicalJobPosting
from esco_occupation_indexer.matching.stages import (
    complete_match_stage,
    fail_match_stage,
    match_artifact_checksum_matches,
    require_match_completed,
    save_match_manifest,
    start_match_stage,
)
from esco_occupation_indexer.normalize import normalize_label
from esco_occupation_indexer.utils import atomic_write_json, sha256_file


def validate_match_build(match_dir: Path) -> dict[str, object]:
    match_dir = match_dir.resolve()
    manifest = require_match_completed(match_dir, "ingest")
    if not match_artifact_checksum_matches(match_dir, manifest, "canonical-jobs.jsonl.zst"):
        raise ValidationError("Canonical jobs artifact checksum mismatch")
    report_filename = "validation-report.json"
    if (
        manifest.stages["validate"].status == "completed"
        and match_artifact_checksum_matches(match_dir, manifest, report_filename)
    ):
        report = json.loads((match_dir / report_filename).read_text(encoding="utf-8"))
        if isinstance(report, dict) and report.get("status") == "completed":
            return report
    start_match_stage(match_dir, "validate")
    try:
        records = list(
            read_jsonl_zst(match_dir / "canonical-jobs.jsonl.zst", CanonicalJobPosting)
        )
        if not records:
            raise ValidationError("Canonical jobs artifact is empty")
        job_ids = [record.job_id for record in records]
        if len(job_ids) != len(set(job_ids)):
            raise ValidationError("Duplicate job IDs in canonical artifact")
        if job_ids != sorted(job_ids):
            raise ValidationError("Canonical jobs are not sorted by job_id")

        category_rows = 0
        distinct_keys: set[str] = set()
        canonical_it_keys: set[str] = set()
        canonical_non_it_keys: set[str] = set()
        declared_it_canonical_other = 0
        terminal_conflicts = 0
        unresolved_key_rows = 0
        cyclic_or_missing = 0
        missing_titles = 0
        missing_descriptions = 0
        missing_requirements = 0
        empty_it_signal = 0
        for record in records:
            if not record.title.strip():
                missing_titles += 1
            if not (record.description or "").strip():
                missing_descriptions += 1
            if not (record.require_candidate or "").strip():
                missing_requirements += 1
            if not record.it_category_paths:
                empty_it_signal += 1
            for path in (
                *record.it_category_paths,
                *record.secondary_category_paths,
            ):
                category_rows += 1
                distinct_keys.add(path.observed.key)
                if path.unresolved_key:
                    unresolved_key_rows += 1
                    continue
                if path.canonical_root == "257":
                    canonical_it_keys.add(path.observed.key)
                elif path.canonical_root is not None:
                    canonical_non_it_keys.add(path.observed.key)
                else:
                    cyclic_or_missing += 1
                if path.root_conflict and path.declared_root == "257":
                    declared_it_canonical_other += 1
                if path.terminal_conflict:
                    terminal_conflicts += 1

        expected = manifest.counts.get("canonical_jobs")
        if expected is not None and expected != len(records):
            raise ValidationError("Canonical job count does not match ingest manifest")

        report: dict[str, object] = {
            "status": "completed",
            "record_count": len(records),
            "category_rows": category_rows,
            "distinct_posting_keys": len(distinct_keys),
            "canonical_it_keys": len(canonical_it_keys),
            "canonical_non_it_keys": len(canonical_non_it_keys),
            "declared_it_canonical_other_rows": declared_it_canonical_other,
            "terminal_conflict_rows": terminal_conflicts,
            "unresolved_key_rows": unresolved_key_rows,
            "cyclic_or_missing_roots": cyclic_or_missing,
            "missing": {
                "title": missing_titles,
                "description": missing_descriptions,
                "require_candidate": missing_requirements,
                "it_signal": empty_it_signal,
            },
            "normalizer_examples": {
                value: normalize_label(value)
                for value in ("C++", "C#", ".NET", "Node.js", "CI/CD", "REST API")
            },
        }
        atomic_write_json(match_dir / "validation-report.json", report)
        manifest = require_match_completed(match_dir, "ingest")
        manifest.counts["validated_jobs"] = len(records)
        manifest.artifact_checksums[report_filename] = sha256_file(
            match_dir / report_filename
        )
        save_match_manifest(match_dir, manifest)
        complete_match_stage(match_dir, "validate")
        return report
    except Exception as error:
        fail_match_stage(match_dir, "validate", error)
        raise
