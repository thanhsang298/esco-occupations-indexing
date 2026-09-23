from __future__ import annotations

from esco_occupation_indexer.models import CanonicalOccupationRecord, LabelVariant
from esco_occupation_indexer.normalize import normalize_label
from esco_occupation_indexer.utils import sha256_bytes, stable_json_bytes


def dense_label_variants(record: CanonicalOccupationRecord) -> list[LabelVariant]:
    return [
        LabelVariant(
            kind="preferred",
            text=record.preferred_label,
            normalized=normalize_label(record.preferred_label),
        ),
        *[
            LabelVariant(
                kind="alternative",
                text=label,
                normalized=normalize_label(label),
            )
            for label in record.alternative_labels
        ],
    ]


def normalized_all_labels(record: CanonicalOccupationRecord) -> list[str]:
    values = [
        normalize_label(record.preferred_label),
        *(normalize_label(label) for label in record.alternative_labels),
        *(normalize_label(label) for label in record.hidden_labels),
    ]
    return list(dict.fromkeys(value for value in values if value))


def semantic_text(record: CanonicalOccupationRecord) -> str:
    lines = [
        f"Occupation: {record.preferred_label}",
    ]
    if record.description:
        lines.append(f"Description: {record.description}")
    if record.definition and record.definition != record.description:
        lines.append(f"Definition: {record.definition}")
    if record.scope_note:
        lines.append(f"Scope note: {record.scope_note}")
    if record.hierarchy.path:
        lines.append(f"Context: {record.hierarchy.path_text}")
    return "\n".join(lines)


def lexical_text(record: CanonicalOccupationRecord) -> str:
    labels = [record.preferred_label, *record.alternative_labels, *record.hidden_labels]
    return "\n".join(labels)


def content_hash_material(record: CanonicalOccupationRecord) -> dict[str, object]:
    """Stable content addressed material for one canonical candidate.

    Keep the embedding inputs explicit even though they are derivable from the
    canonical record.  This makes a content hash change whenever a template or
    label construction change alters the bytes sent to an embedding backend.
    """

    return {
        "canonical_record": record.model_dump(mode="json", exclude={"content_hash"}),
        "embedding_inputs": {
            "label_dense": [variant.text for variant in dense_label_variants(record)],
            "semantic_dense": semantic_text(record),
            "lexical_sparse": lexical_text(record),
        },
    }


def canonical_content_hash(record: CanonicalOccupationRecord) -> str:
    return sha256_bytes(stable_json_bytes(content_hash_material(record)))


def qdrant_payload(record: CanonicalOccupationRecord, build_id: str) -> dict[str, object]:
    variants = dense_label_variants(record)
    return {
        "schema_version": "esco_occupation_point/v1",
        "entity_type": "occupation_concept",
        "esco_uri": record.esco_uri,
        "esco_id": record.esco_id,
        "esco_version": record.esco_version,
        "isco_group": record.isco_group,
        "isco_code": record.code,
        "status": record.status,
        "labels": {
            "preferred": record.preferred_label,
            "alternative": record.alternative_labels,
            "hidden": record.hidden_labels,
            "dense_variants": [variant.model_dump(mode="json") for variant in variants],
            "normalized_all": normalized_all_labels(record),
        },
        "content": {
            "description": record.description,
            "definition": record.definition,
            "scope_note": record.scope_note,
            "regulated_profession_note": record.regulated_profession_note,
            "semantic_text": semantic_text(record),
        },
        "hierarchy": {
            **record.hierarchy.model_dump(mode="json"),
            "path_text": record.hierarchy.path_text,
        },
        "skills": {
            "essential": record.essential_skills,
            "optional": record.optional_skills,
            "essential_count": len(record.essential_skills),
            "optional_count": len(record.optional_skills),
        },
        "green_share": record.green_share,
        "collections": record.collections,
        "modified_at": record.modified_at.isoformat() if record.modified_at else None,
        "content_hash": record.content_hash,
        "index_build_id": build_id,
    }
