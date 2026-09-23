from __future__ import annotations

import re
import uuid
from collections import defaultdict
from pathlib import Path

from esco_occupation_indexer.artifacts import (
    complete_stage,
    fail_stage,
    load_manifest,
    new_stages,
    save_manifest,
    start_stage,
    write_jsonl_zst,
)
from esco_occupation_indexer.documents import canonical_content_hash
from esco_occupation_indexer.errors import SourceDataError, ValidationError
from esco_occupation_indexer.hierarchy import GraphNode, HierarchyGraph
from esco_occupation_indexer.models import (
    BuildManifest,
    CanonicalOccupationRecord,
    IscoGroupRecord,
)
from esco_occupation_indexer.normalize import (
    clean_optional,
    deduplicate_hidden,
    deduplicate_labels,
    split_labels,
)
from esco_occupation_indexer.settings import IndexingSettings, write_resolved_settings
from esco_occupation_indexer.source import (
    OPTIONAL_FILES,
    REQUIRED_FILES,
    collection_uris,
    load_source_package,
)
from esco_occupation_indexer.utils import (
    atomic_write_json,
    sha256_bytes,
    sha256_file,
    stable_json_bytes,
    utc_now,
)

OCCUPATION_URI_PATTERN = re.compile(
    r"^https?://data\.europa\.eu/esco/occupation/"
    r"(?P<id>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})$"
)

SKILL_URI_PATTERN = re.compile(
    r"^https?://data\.europa\.eu/esco/skill/"
    r"(?P<id>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})$"
)

ISCO_URI_PREFIX = "http://data.europa.eu/esco/isco/"

# Optional topical collection for occupations.  greenShareOcc_en.csv is not a
# membership collection: it carries a numeric green_share per URI instead.
RESEARCH_COLLECTION = "research"

INGEST_ARTIFACT_NAMES = (
    "canonical-occupations.jsonl.zst",
    "isco-groups.jsonl.zst",
    "source-report.json",
)
VALID_OCCUPATION_STATUSES = {"released", "obsolete"}
VALID_RELATION_TYPES = {"essential", "optional"}
VALID_RELATION_SKILL_TYPES = {"skill/competence", "knowledge", ""}


def esco_id_from_uri(uri: str) -> str:
    match = OCCUPATION_URI_PATTERN.fullmatch(uri.strip())
    if not match:
        raise ValidationError(f"Invalid ESCO occupation URI: {uri!r}")
    return str(uuid.UUID(match.group("id")))


def _check_skill_uri(uri: str) -> str:
    value = uri.strip()
    if not SKILL_URI_PATTERN.fullmatch(value):
        raise ValidationError(f"Invalid ESCO skill URI in occupation relations: {uri!r}")
    return value


def _check_isco_uri(uri: str) -> str:
    value = uri.strip()
    if not value.startswith(ISCO_URI_PREFIX) or value == ISCO_URI_PREFIX:
        raise ValidationError(f"Invalid ESCO ISCO group URI: {uri!r}")
    return value


def _config_fingerprint(settings: IndexingSettings) -> str:
    return sha256_bytes(stable_json_bytes(_index_config(settings)))


def _index_config(settings: IndexingSettings) -> dict[str, object]:
    return {
        "build_schema_version": "esco_occupation_index_build/v1",
        "canonical_schema_version": "esco_occupation_record/v1",
        "esco_version": settings.esco_version,
        "source_language": settings.source_language,
        "embedding": settings.embedding.model_dump(mode="json"),
        "sparse": settings.sparse.model_dump(mode="json"),
        "qdrant_schema": {
            "collection_prefix": settings.qdrant.collection_prefix,
            "label_dense": {"dimension": settings.embedding.dimension, "max_sim": True},
            "semantic_dense": {"dimension": settings.embedding.dimension},
            "lexical_sparse": {"modifier": "idf"},
            "payload_schema": "esco_occupation_point/v1",
        },
        "normalizer": "technical-label-v1",
        "text_preservation": "trim-only-v1",
        "label_template": "label-v1",
        "semantic_template": "occupation-semantic-v1-preferred-description-context",
        "content_hash": "canonical-and-embedding-inputs-v2-green-share-r6",
    }


def compute_build_id(source_checksum: str, settings: IndexingSettings) -> tuple[str, str]:
    config_fingerprint = _config_fingerprint(settings)
    build_id = sha256_bytes(f"{source_checksum}:{config_fingerprint}".encode())[:12]
    return build_id, config_fingerprint


def _deduplicate_occupation_rows(
    occupation_rows: tuple[dict[str, str], ...],
) -> list[dict[str, str]]:
    selected: dict[str, dict[str, str]] = {}
    for row in occupation_rows:
        uri = row.get("conceptUri", "").strip()
        if not uri:
            raise SourceDataError("occupations_en.csv contains an empty conceptUri")
        previous = selected.get(uri)
        if previous is None:
            selected[uri] = row
            continue

        comparable = {key: value for key, value in row.items() if key != "modifiedDate"}
        previous_comparable = {
            key: value for key, value in previous.items() if key != "modifiedDate"
        }
        if comparable != previous_comparable:
            raise ValidationError(f"Conflicting duplicate ESCO URI: {uri}")
        if row.get("modifiedDate", "").strip() > previous.get("modifiedDate", "").strip():
            selected[uri] = row
    return list(selected.values())


def _source_report(package) -> dict[str, object]:
    expected_columns = {
        "occupations": {
            "conceptType",
            "conceptUri",
            "iscoGroup",
            "preferredLabel",
            "altLabels",
            "hiddenLabels",
            "status",
            "modifiedDate",
            "regulatedProfessionNote",
            "scopeNote",
            "definition",
            "inScheme",
            "description",
            "code",
            "naceCode",
        },
        "isco_groups": {
            "conceptType",
            "conceptUri",
            "code",
            "preferredLabel",
            "status",
            "altLabels",
            "inScheme",
            "description",
        },
        "broader": {
            "conceptType",
            "conceptUri",
            "conceptLabel",
            "broaderType",
            "broaderUri",
            "broaderLabel",
        },
        "relations": {
            "occupationUri",
            "occupationLabel",
            "relationType",
            "skillType",
            "skillUri",
            "skillLabel",
        },
        "research": {
            "conceptType",
            "conceptUri",
            "preferredLabel",
            "status",
            "altLabels",
            "description",
            "broaderConceptUri",
            "broaderConceptPT",
        },
        "green": {
            "conceptType",
            "conceptUri",
            "code",
            "preferredLabel",
            "greenShare",
        },
    }
    return {
        "zip_checksum": package.zip_checksum,
        "tables": {
            name: {
                "filename": table.filename,
                "checksum": table.checksum,
                "rows": len(table.rows),
                "headers": list(table.headers),
                "extra_columns": (
                    sorted(set(table.headers) - expected_columns[name])
                    if name in expected_columns
                    else []
                ),
            }
            for name, table in sorted(package.tables.items())
        },
    }


def _parse_green_share(package) -> dict[str, float]:
    table = package.tables.get("green")
    if table is None:
        return {}
    if "greenShare" not in table.headers:
        raise SourceDataError(f"{table.filename} must contain a greenShare column")
    shares: dict[str, float] = {}
    for row in table.rows:
        uri = row.get("conceptUri", "").strip()
        if not uri:
            continue
        raw = row.get("greenShare", "").strip()
        if not raw:
            continue
        try:
            value = float(raw)
        except ValueError as error:
            raise ValidationError(
                f"Invalid greenShare {raw!r} for {uri} in {table.filename}"
            ) from error
        if not 0.0 <= value <= 1.0:
            raise ValidationError(f"greenShare out of range for {uri}: {raw!r}")
        # Round to 6 decimals: Qdrant does not preserve full f64 precision
        # through a payload round-trip (last-ulp drift), which would break the
        # exact payload-match verification. 6 decimals is ample for a share.
        shares[uri] = round(value, 6)
    return shares


def _build_records(package, settings: IndexingSettings):
    occupation_rows = _deduplicate_occupation_rows(package.tables["occupations"].rows)
    isco_rows = package.tables["isco_groups"].rows

    groups: dict[str, IscoGroupRecord] = {}
    nodes: dict[str, GraphNode] = {}
    for row in isco_rows:
        uri = _check_isco_uri(row["conceptUri"])
        if uri in nodes:
            raise ValidationError(f"Duplicate ESCO URI: {uri}")
        preferred, alternatives = deduplicate_labels(
            row["preferredLabel"], split_labels(row.get("altLabels"))
        )
        if not preferred:
            raise ValidationError(f"ISCO group has empty preferredLabel: {uri}")
        code = row["code"].strip()
        if not code:
            raise ValidationError(f"ISCO group has empty code: {uri}")
        groups[uri] = IscoGroupRecord(
            esco_uri=uri,
            code=code,
            preferred_label=preferred,
            alternative_labels=alternatives,
            description=clean_optional(row.get("description")),
            status=clean_optional(row.get("status")),
        )
        nodes[uri] = GraphNode(
            uri=uri,
            label=preferred,
            node_type="isco_group",
            code=code,
        )

    all_occupation_rows: dict[str, dict[str, str]] = {}
    for row in occupation_rows:
        uri = row["conceptUri"].strip()
        if not uri:
            raise SourceDataError("occupations_en.csv contains an empty conceptUri")
        esco_id_from_uri(uri)
        if uri in nodes:
            raise ValidationError(f"Duplicate ESCO URI: {uri}")
        label = row["preferredLabel"].strip()
        if not label:
            raise ValidationError(f"Occupation has empty preferredLabel: {uri}")
        if not row["conceptType"].strip():
            raise ValidationError(f"Occupation has empty conceptType: {uri}")
        status = row["status"].strip().casefold()
        if status not in VALID_OCCUPATION_STATUSES:
            raise ValidationError(f"Invalid status {status!r} for {uri}")
        if not row.get("iscoGroup", "").strip():
            raise ValidationError(f"Occupation has empty iscoGroup: {uri}")
        if not row.get("code", "").strip():
            raise ValidationError(f"Occupation has empty code: {uri}")
        nodes[uri] = GraphNode(uri=uri, label=label, node_type="occupation")
        all_occupation_rows[uri] = row

    parents: dict[str, set[str]] = defaultdict(set)
    for row in package.tables["broader"].rows:
        child = row["conceptUri"].strip()
        parent = row["broaderUri"].strip()
        if not child or not parent:
            raise ValidationError("broaderRelationsOccPillar_en.csv contains an empty URI")
        parents[child].add(parent)
    graph = HierarchyGraph(nodes, dict(parents))

    essential: dict[str, set[str]] = defaultdict(set)
    optional: dict[str, set[str]] = defaultdict(set)
    for row in package.tables["relations"].rows:
        occupation_uri = row["occupationUri"].strip()
        skill_uri = row["skillUri"].strip()
        relation_type = row["relationType"].strip().casefold()
        skill_type = row["skillType"].strip()
        if not occupation_uri or not skill_uri:
            raise ValidationError(
                "occupationSkillRelations_en.csv contains an empty occupation/skill URI"
            )
        if relation_type not in VALID_RELATION_TYPES:
            raise ValidationError(f"Invalid relationType {relation_type!r}")
        if skill_type not in VALID_RELATION_SKILL_TYPES:
            raise ValidationError(f"Invalid skillType {skill_type!r} in relations")
        # Format-only check: skill URIs are stored for payload/relation joins at
        # retrieval time, never resolved against the skill index during ingest.
        _check_skill_uri(skill_uri)
        if occupation_uri not in nodes:
            raise ValidationError(
                f"Relations reference unknown occupation URI: {occupation_uri}"
            )
        if relation_type == "essential":
            essential[occupation_uri].add(skill_uri)
        else:
            optional[occupation_uri].add(skill_uri)

    memberships: dict[str, set[str]] = defaultdict(set)
    research_table = package.tables.get("research")
    if research_table is not None:
        for uri in collection_uris(research_table):
            if uri not in nodes:
                raise ValidationError(
                    f"Research collection references unknown ESCO URI: {uri}"
                )
            if uri in all_occupation_rows:
                memberships[uri].add(RESEARCH_COLLECTION)

    green_shares = _parse_green_share(package)

    records: list[CanonicalOccupationRecord] = []
    obsolete_count = 0
    for uri, row in all_occupation_rows.items():
        status = row["status"].strip().casefold()
        if status != "released":
            obsolete_count += 1
            continue
        preferred, alternatives = deduplicate_labels(
            row["preferredLabel"], split_labels(row.get("altLabels"))
        )
        hidden = deduplicate_hidden(
            preferred, alternatives, split_labels(row.get("hiddenLabels"))
        )
        source_files = {
            REQUIRED_FILES["occupations"],
            REQUIRED_FILES["isco_groups"],
            REQUIRED_FILES["broader"],
            REQUIRED_FILES["relations"],
        }
        if RESEARCH_COLLECTION in memberships.get(uri, set()):
            source_files.add(OPTIONAL_FILES["research"])
        if uri in green_shares:
            source_files.add(OPTIONAL_FILES["green"])
        base = {
            "esco_uri": uri,
            "esco_id": esco_id_from_uri(uri),
            "esco_version": settings.esco_version,
            "concept_type": row["conceptType"].strip(),
            "isco_group": row["iscoGroup"].strip(),
            "code": row["code"].strip(),
            "modified_at": clean_optional(row.get("modifiedDate")),
            "preferred_label": preferred,
            "alternative_labels": alternatives,
            "hidden_labels": hidden,
            "description": clean_optional(row.get("description")),
            "definition": clean_optional(row.get("definition")),
            "scope_note": clean_optional(row.get("scopeNote")),
            "regulated_profession_note": clean_optional(
                row.get("regulatedProfessionNote")
            ),
            "nace_code": clean_optional(row.get("naceCode")),
            "hierarchy": graph.hierarchy_for(uri),
            "essential_skills": sorted(essential.get(uri, set())),
            "optional_skills": sorted(optional.get(uri, set())),
            "green_share": green_shares.get(uri),
            "collections": sorted(memberships.get(uri, set())),
            "source_files": sorted(source_files),
        }
        draft = CanonicalOccupationRecord(**base, content_hash="")
        content_hash = canonical_content_hash(draft)
        records.append(draft.model_copy(update={"content_hash": content_hash}))

    records.sort(key=lambda record: record.esco_uri)
    group_records = sorted(groups.values(), key=lambda group: group.esco_uri)
    return records, group_records, obsolete_count


def _ingest_artifacts_are_valid(build_dir: Path, manifest: BuildManifest) -> bool:
    checksums = manifest.artifact_checksums
    if set(checksums) < set(INGEST_ARTIFACT_NAMES):
        return False
    try:
        return all(
            sha256_file(build_dir / filename) == checksums[filename]
            for filename in INGEST_ARTIFACT_NAMES
        )
    except OSError:
        return False


def _assert_manifest_identity(
    manifest: BuildManifest,
    *,
    build_id: str,
    config_fingerprint: str,
    source_checksum: str,
) -> None:
    if manifest.build_id != build_id:
        raise ValidationError("Existing manifest has a different build ID")
    if manifest.config_fingerprint != config_fingerprint:
        raise ValidationError("Existing manifest has a different configuration fingerprint")
    if manifest.source_checksums.get("source_zip") != source_checksum:
        raise ValidationError("Existing manifest has a different source ZIP checksum")


def ingest_source(settings: IndexingSettings, source_zip: Path) -> Path:
    source_zip = source_zip.resolve()
    package = load_source_package(source_zip)
    build_id, config_fingerprint = compute_build_id(package.zip_checksum, settings)
    build_dir = settings.work_dir / build_id
    manifest_file = build_dir / "manifest.json"

    if manifest_file.exists():
        manifest = load_manifest(build_dir)
        _assert_manifest_identity(
            manifest,
            build_id=build_id,
            config_fingerprint=config_fingerprint,
            source_checksum=package.zip_checksum,
        )
        if (
            manifest.stages["ingest"].status == "completed"
            and _ingest_artifacts_are_valid(build_dir, manifest)
        ):
            return build_dir
    else:
        build_dir.mkdir(parents=True, exist_ok=True)
        now = utc_now()
        manifest = BuildManifest(
            build_id=build_id,
            esco_version=settings.esco_version,
            source_language=settings.source_language,
            source_zip=str(source_zip),
            source_checksums={
                "source_zip": package.zip_checksum,
                **{
                    table.filename: table.checksum for table in package.tables.values()
                },
            },
            config_fingerprint=config_fingerprint,
            index_config=_index_config(settings),
            collection_name=(
                f"{settings.qdrant.collection_prefix}__v"
                f"{settings.esco_version.replace('.', '_')}__{build_id}"
            ),
            created_at=now,
            updated_at=now,
            stages=new_stages(),
        )
        save_manifest(build_dir, manifest)
        write_resolved_settings(settings, build_dir / "resolved-config.json")

    start_stage(build_dir, "ingest")
    try:
        records, groups, obsolete_count = _build_records(package, settings)
        write_jsonl_zst(build_dir / "canonical-occupations.jsonl.zst", records)
        write_jsonl_zst(build_dir / "isco-groups.jsonl.zst", groups)
        atomic_write_json(build_dir / "source-report.json", _source_report(package))

        manifest = load_manifest(build_dir)
        manifest.counts.update(
            {
                "source_occupation_rows": len(package.tables["occupations"].rows),
                "source_occupation_concepts": len(
                    {row["conceptUri"].strip() for row in package.tables["occupations"].rows}
                ),
                "duplicate_occupation_rows": len(package.tables["occupations"].rows)
                - len(
                    {row["conceptUri"].strip() for row in package.tables["occupations"].rows}
                ),
                "released_occupation_concepts": len(records),
                "obsolete_occupation_concepts": obsolete_count,
                "isco_groups": len(groups),
                "essential_relations": sum(len(r.essential_skills) for r in records),
                "optional_relations": sum(len(r.optional_skills) for r in records),
            }
        )
        manifest.artifact_checksums.update(
            {
                filename: sha256_file(build_dir / filename)
                for filename in INGEST_ARTIFACT_NAMES
            }
        )
        save_manifest(build_dir, manifest)
        complete_stage(build_dir, "ingest")
        return build_dir
    except Exception as error:
        fail_stage(build_dir, "ingest", error)
        raise
