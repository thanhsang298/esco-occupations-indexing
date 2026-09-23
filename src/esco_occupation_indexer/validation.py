from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from esco_occupation_indexer.artifacts import (
    complete_stage,
    fail_stage,
    read_jsonl_zst,
    require_completed,
    save_manifest,
    start_stage,
)
from esco_occupation_indexer.documents import (
    canonical_content_hash,
    dense_label_variants,
    lexical_text,
    normalized_all_labels,
    semantic_text,
)
from esco_occupation_indexer.errors import ValidationError
from esco_occupation_indexer.ingest import (
    INGEST_ARTIFACT_NAMES,
    RESEARCH_COLLECTION,
    esco_id_from_uri,
)
from esco_occupation_indexer.models import (
    BuildManifest,
    CanonicalOccupationRecord,
    IscoGroupRecord,
)
from esco_occupation_indexer.normalize import normalize_label
from esco_occupation_indexer.source import OPTIONAL_FILES, REQUIRED_FILES
from esco_occupation_indexer.utils import atomic_write_json, sha256_file

VALID_COLLECTIONS = frozenset({RESEARCH_COLLECTION})
REQUIRED_RECORD_SOURCE_FILES = frozenset(
    {
        REQUIRED_FILES["occupations"],
        REQUIRED_FILES["isco_groups"],
        REQUIRED_FILES["broader"],
        REQUIRED_FILES["relations"],
    }
)


def _validate_label_text(text: str, field: str, uri: str) -> str:
    if text != text.strip() or "\n" in text or "\r" in text:
        raise ValidationError(f"Invalid whitespace in {field} for {uri}")
    key = normalize_label(text)
    if not key:
        raise ValidationError(f"Empty {field} for {uri}")
    return key


def _validate_group(group: IscoGroupRecord) -> None:
    if not group.esco_uri.strip() or group.esco_uri != group.esco_uri.strip():
        raise ValidationError(f"Invalid ISCO group URI: {group.esco_uri!r}")
    if not group.code.strip() or group.code != group.code.strip():
        raise ValidationError(f"Invalid ISCO group code: {group.esco_uri}")
    preferred_key = _validate_label_text(
        group.preferred_label, "ISCO group preferred label", group.esco_uri
    )
    alternative_keys = [
        _validate_label_text(label, "ISCO group alternative label", group.esco_uri)
        for label in group.alternative_labels
    ]
    all_keys = [preferred_key, *alternative_keys]
    if len(all_keys) != len(set(all_keys)):
        raise ValidationError(f"Duplicate normalized ISCO group labels: {group.esco_uri}")
    if alternative_keys != sorted(alternative_keys):
        raise ValidationError(f"Unsorted ISCO group alternative labels: {group.esco_uri}")


def _validate_hierarchy(
    record: CanonicalOccupationRecord, groups_by_uri: dict[str, IscoGroupRecord]
) -> None:
    hierarchy = record.hierarchy
    path = hierarchy.path
    if not path:
        raise ValidationError(f"Empty hierarchy path for {record.esco_uri}")
    if hierarchy.parent_uri != path[-1].uri:
        raise ValidationError(f"Parent/path mismatch for {record.esco_uri}")
    if hierarchy.parent_type != path[-1].node_type:
        raise ValidationError(f"Parent type/path mismatch for {record.esco_uri}")
    if hierarchy.ancestor_uris != [node.uri for node in path]:
        raise ValidationError(f"Ancestor/path mismatch for {record.esco_uri}")
    if hierarchy.level != len(path) + 1:
        raise ValidationError(f"Hierarchy level mismatch for {record.esco_uri}")
    if record.esco_uri in hierarchy.ancestor_uris:
        raise ValidationError(f"Hierarchy path contains the concept itself: {record.esco_uri}")
    if len(hierarchy.ancestor_uris) != len(set(hierarchy.ancestor_uris)):
        raise ValidationError(f"Hierarchy path contains a cycle: {record.esco_uri}")

    root = path[0]
    if root.node_type != "isco_group":
        raise ValidationError(f"Hierarchy root is not an ISCO group: {record.esco_uri}")
    if len((root.code or "").strip()) != 1:
        raise ValidationError(f"Hierarchy root is not an ISCO major group: {record.esco_uri}")
    if (root.code or "").strip() != hierarchy.isco_major:
        raise ValidationError(f"Invalid hierarchy ISCO major for {record.esco_uri}")

    nearest_isco_uri: str | None = None
    for node in path:
        if not node.uri.strip() or node.uri != node.uri.strip():
            raise ValidationError(f"Invalid hierarchy URI for {record.esco_uri}")
        _validate_label_text(node.label, "hierarchy label", record.esco_uri)
        if node.node_type == "isco_group":
            group = groups_by_uri.get(node.uri)
            if group is None:
                raise ValidationError(
                    f"Unknown hierarchy ISCO group for {record.esco_uri}: {node.uri}"
                )
            if node.code != group.code or node.label != group.preferred_label:
                raise ValidationError(f"Hierarchy group metadata mismatch for {record.esco_uri}")
            nearest_isco_uri = node.uri
        elif node.code is not None:
            raise ValidationError(f"Occupation hierarchy node has a code: {record.esco_uri}")
    if nearest_isco_uri != hierarchy.terminal_isco_uri:
        raise ValidationError(f"Invalid terminal ISCO group for {record.esco_uri}")


def _validate_record(
    record: CanonicalOccupationRecord,
    groups_by_uri: dict[str, IscoGroupRecord],
    manifest: BuildManifest,
) -> None:
    if esco_id_from_uri(record.esco_uri) != record.esco_id:
        raise ValidationError(f"ESCO ID does not match URI for {record.esco_uri}")
    if record.esco_version != manifest.esco_version:
        raise ValidationError(f"ESCO version mismatch for {record.esco_uri}")
    if record.source_language != manifest.source_language:
        raise ValidationError(f"Source language mismatch for {record.esco_uri}")
    if not record.concept_type.strip():
        raise ValidationError(f"Empty concept type for {record.esco_uri}")
    if record.status != "released":
        raise ValidationError(f"Non-released concept in canonical artifact: {record.esco_uri}")
    if not record.isco_group.strip():
        raise ValidationError(f"Empty ISCO group for {record.esco_uri}")
    if not record.code.strip():
        raise ValidationError(f"Empty occupation code for {record.esco_uri}")
    if len(record.hierarchy.isco_major) != 1:
        raise ValidationError(f"Invalid ISCO major for {record.esco_uri}")
    _validate_hierarchy(record, groups_by_uri)

    preferred_key = _validate_label_text(record.preferred_label, "preferred label", record.esco_uri)
    alternative_keys = [
        _validate_label_text(label, "alternative label", record.esco_uri)
        for label in record.alternative_labels
    ]
    hidden_keys = [
        _validate_label_text(label, "hidden label", record.esco_uri)
        for label in record.hidden_labels
    ]
    all_label_keys = [preferred_key, *alternative_keys, *hidden_keys]
    if len(all_label_keys) != len(set(all_label_keys)):
        raise ValidationError(f"Duplicate normalized labels for {record.esco_uri}")
    if alternative_keys != sorted(alternative_keys):
        raise ValidationError(f"Unsorted alternative labels for {record.esco_uri}")
    if hidden_keys != sorted(hidden_keys):
        raise ValidationError(f"Unsorted hidden labels for {record.esco_uri}")

    variants = dense_label_variants(record)
    if not variants or variants[0].kind != "preferred":
        raise ValidationError(f"Missing preferred dense label for {record.esco_uri}")
    if [variant.text for variant in variants] != [
        record.preferred_label,
        *record.alternative_labels,
    ]:
        raise ValidationError(f"Dense label ordering mismatch for {record.esco_uri}")
    if [variant.normalized for variant in variants] != [preferred_key, *alternative_keys]:
        raise ValidationError(f"Dense label normalization mismatch for {record.esco_uri}")
    normalized_all = normalized_all_labels(record)
    if normalized_all != all_label_keys:
        raise ValidationError(f"Normalized label payload mismatch for {record.esco_uri}")
    if not semantic_text(record):
        raise ValidationError(f"Empty semantic text for {record.esco_uri}")
    if lexical_text(record).splitlines() != [
        record.preferred_label,
        *record.alternative_labels,
        *record.hidden_labels,
    ]:
        raise ValidationError(f"Lexical text mismatch for {record.esco_uri}")

    if record.essential_skills != sorted(record.essential_skills) or len(
        record.essential_skills
    ) != len(set(record.essential_skills)):
        raise ValidationError(f"Invalid essential skill ordering for {record.esco_uri}")
    if record.optional_skills != sorted(record.optional_skills) or len(
        record.optional_skills
    ) != len(set(record.optional_skills)):
        raise ValidationError(f"Invalid optional skill ordering for {record.esco_uri}")
    overlap = set(record.essential_skills) & set(record.optional_skills)
    if overlap:
        raise ValidationError(
            f"Skill listed as both essential and optional for {record.esco_uri}"
        )
    if record.green_share is not None:
        if not 0.0 <= record.green_share <= 1.0:
            raise ValidationError(f"green_share out of range for {record.esco_uri}")
        if record.green_share != round(record.green_share, 6):
            raise ValidationError(f"green_share is not rounded to 6 decimals for {record.esco_uri}")

    if record.collections != sorted(record.collections) or len(record.collections) != len(
        set(record.collections)
    ):
        raise ValidationError(f"Invalid collection ordering for {record.esco_uri}")
    unknown_collections = set(record.collections) - VALID_COLLECTIONS
    if unknown_collections:
        raise ValidationError(
            f"Unknown collection membership for {record.esco_uri}: {sorted(unknown_collections)}"
        )
    if record.source_files != sorted(record.source_files) or len(record.source_files) != len(
        set(record.source_files)
    ):
        raise ValidationError(f"Invalid source file ordering for {record.esco_uri}")
    if not REQUIRED_RECORD_SOURCE_FILES.issubset(record.source_files):
        raise ValidationError(f"Missing source provenance for {record.esco_uri}")
    if RESEARCH_COLLECTION in record.collections and (
        OPTIONAL_FILES["research"] not in record.source_files
    ):
        raise ValidationError(f"Missing research source provenance for {record.esco_uri}")
    if record.green_share is not None and OPTIONAL_FILES["green"] not in record.source_files:
        raise ValidationError(f"Missing green share source provenance for {record.esco_uri}")

    expected_hash = canonical_content_hash(record)
    if expected_hash != record.content_hash:
        raise ValidationError(f"Content hash mismatch for {record.esco_uri}")


def _checksum_matches(build_dir: Path, manifest: BuildManifest, filename: str) -> bool:
    expected = manifest.artifact_checksums.get(filename)
    if not expected:
        return False
    try:
        return sha256_file(build_dir / filename) == expected
    except OSError:
        return False


def validate_build(build_dir: Path) -> dict[str, object]:
    build_dir = build_dir.resolve()
    manifest = require_completed(build_dir, "ingest")
    if not all(
        _checksum_matches(build_dir, manifest, filename) for filename in INGEST_ARTIFACT_NAMES
    ):
        raise ValidationError("Ingest artifact checksum is missing or does not match manifest")
    report_filename = "validation-report.json"
    if (
        manifest.stages["validate"].status == "completed"
        and _checksum_matches(build_dir, manifest, report_filename)
    ):
        report = json.loads((build_dir / report_filename).read_text(encoding="utf-8"))
        if isinstance(report, dict) and report.get("status") == "completed":
            return report
    start_stage(build_dir, "validate")
    try:
        records = list(
            read_jsonl_zst(build_dir / "canonical-occupations.jsonl.zst", CanonicalOccupationRecord)
        )
        groups = list(read_jsonl_zst(build_dir / "isco-groups.jsonl.zst", IscoGroupRecord))
        if not records:
            raise ValidationError("Canonical occupation artifact is empty")

        uris = [record.esco_uri for record in records]
        ids = [record.esco_id for record in records]
        if len(uris) != len(set(uris)):
            raise ValidationError("Canonical artifact contains duplicate ESCO URIs")
        if len(ids) != len(set(ids)):
            raise ValidationError("Canonical artifact contains duplicate point IDs")
        if uris != sorted(uris):
            raise ValidationError("Canonical artifact is not sorted by ESCO URI")

        group_uris = [group.esco_uri for group in groups]
        if len(group_uris) != len(set(group_uris)):
            raise ValidationError("Group artifact contains duplicate ESCO URIs")
        if group_uris != sorted(group_uris):
            raise ValidationError("Group artifact is not sorted by ESCO URI")
        groups_by_uri = {group.esco_uri: group for group in groups}
        for group in groups:
            _validate_group(group)
        for record in records:
            _validate_record(record, groups_by_uri, manifest)

        expected_records = manifest.counts.get("released_occupation_concepts")
        if expected_records is not None and expected_records != len(records):
            raise ValidationError("Canonical record count does not match ingest manifest")
        expected_groups = manifest.counts.get("isco_groups")
        if expected_groups is not None and expected_groups != len(groups):
            raise ValidationError("Group record count does not match ingest manifest")

        majors = Counter(record.hierarchy.isco_major for record in records)
        collection_counts = Counter(
            collection for record in records for collection in record.collections
        )
        dense_label_counts = [1 + len(record.alternative_labels) for record in records]
        all_label_counts = [
            1 + len(record.alternative_labels) + len(record.hidden_labels)
            for record in records
        ]
        essential_counts = [len(record.essential_skills) for record in records]
        optional_counts = [len(record.optional_skills) for record in records]
        missing_optional = {
            field: sum(getattr(record, field) is None for record in records)
            for field in (
                "modified_at",
                "description",
                "definition",
                "scope_note",
                "regulated_profession_note",
                "nace_code",
                "green_share",
            )
        }
        report: dict[str, object] = {
            "status": "completed",
            "record_count": len(records),
            "group_count": len(groups),
            "rejected_structural_count": 0,
            "counts_by_isco_major": dict(sorted(majors.items())),
            "counts_by_collection": {
                name: collection_counts[name] for name in sorted(VALID_COLLECTIONS)
            },
            "dense_labels_per_concept": {
                "min": min(dense_label_counts),
                "max": max(dense_label_counts),
                "average": sum(dense_label_counts) / len(dense_label_counts),
            },
            "all_labels_per_concept": {
                "min": min(all_label_counts),
                "max": max(all_label_counts),
                "average": sum(all_label_counts) / len(all_label_counts),
            },
            "essential_skills_per_concept": {
                "min": min(essential_counts),
                "max": max(essential_counts),
                "average": sum(essential_counts) / len(essential_counts),
            },
            "optional_skills_per_concept": {
                "min": min(optional_counts),
                "max": max(optional_counts),
                "average": sum(optional_counts) / len(optional_counts),
            },
            "missing_optional": missing_optional,
            "semantic_truncation_count": None,
            "normalizer_examples": {
                value: normalize_label(value)
                for value in ("C++", "C#", ".NET", "Node.js", "CI/CD", "REST API")
            },
        }
        atomic_write_json(build_dir / "validation-report.json", report)
        manifest = require_completed(build_dir, "ingest")
        manifest.counts["validated_occupation_concepts"] = len(records)
        manifest.counts["rejected_structural_count"] = 0
        manifest.artifact_checksums[report_filename] = sha256_file(
            build_dir / report_filename
        )
        save_manifest(build_dir, manifest)
        complete_stage(build_dir, "validate")
        return report
    except Exception as error:
        fail_stage(build_dir, "validate", error)
        raise
