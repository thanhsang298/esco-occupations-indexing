from __future__ import annotations

import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from esco_occupation_indexer.artifacts import write_jsonl_zst
from esco_occupation_indexer.errors import SourceDataError, ValidationError
from esco_occupation_indexer.matching.models import (
    CanonicalJobPosting,
    CategoryPathNode,
    MatchManifest,
    ObservedCategory,
    ResolvedCategory,
)
from esco_occupation_indexer.matching.settings import (
    MatchSettings,
    write_match_resolved_settings,
)
from esco_occupation_indexer.matching.stages import (
    complete_match_stage,
    fail_match_stage,
    load_match_manifest,
    new_match_stages,
    save_match_manifest,
    start_match_stage,
)
from esco_occupation_indexer.utils import (
    atomic_write_json,
    sha256_bytes,
    sha256_file,
    stable_json_bytes,
    utc_now,
)

NORMALIZER_VERSION = "topcv-normalize-v1"
CANONICAL_SCHEMA_VERSION = "topcv_job_record/v1"
IT_ROOT = "257"

_HTML_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")


def clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = _HTML_TAG.sub(" ", value)
    text = _WHITESPACE.sub(" ", text).strip()
    return text or None


def clean_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [cleaned for item in value if (cleaned := clean_text(str(item)))]


def parse_mongo_date(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, dict) and isinstance(value.get("$date"), str):
        try:
            parsed = datetime.fromisoformat(value["$date"].replace("Z", "+00:00"))
        except ValueError as error:
            raise ValidationError(f"Invalid Mongo date: {value!r}") from error
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    raise ValidationError(f"Expected Mongo Extended JSON date, got: {value!r}")


def _dedup_consecutive(ids: list[str]) -> list[str]:
    result: list[str] = []
    for item in ids:
        if not result or result[-1] != item:
            result.append(item)
    return result


class Taxonomy:
    def __init__(self, nodes: dict[str, dict[str, object]]) -> None:
        self.nodes = nodes

    def name_of(self, external_id: str) -> str | None:
        node = self.nodes.get(external_id)
        if node is None:
            return None
        name = node.get("name")
        return str(name) if name else None

    def canonical_path(self, key: str) -> tuple[list[CategoryPathNode], str | None]:
        """Walk key up via parentExternalId. Returns (root-first path, root id)."""
        seen: set[str] = set()
        reversed_nodes: list[CategoryPathNode] = []
        current: str | None = key
        while current:
            if current in seen:
                raise ValidationError(f"Category cycle detected at {current!r}")
            seen.add(current)
            node = self.nodes.get(current)
            if node is None:
                raise ValidationError(f"Category parent missing: {current!r}")
            name = node.get("name")
            if not name:
                raise ValidationError(f"Category has empty name: {current!r}")
            reversed_nodes.append(
                CategoryPathNode(external_id=current, name=str(name))
            )
            parent = node.get("parentExternalId")
            current = str(parent) if parent else None
        path = list(reversed(reversed_nodes))
        return path, path[0].external_id if path else None


def load_taxonomy(path: Path) -> Taxonomy:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SourceDataError(f"Cannot read taxonomy file: {path}") from error
    if not isinstance(raw, list):
        raise SourceDataError("Taxonomy file must contain a JSON list")
    nodes: dict[str, dict[str, object]] = {}
    for entry in raw:
        if not isinstance(entry, dict):
            raise SourceDataError("Taxonomy entry must be an object")
        external_id = entry.get("externalId")
        if not external_id or not isinstance(external_id, str):
            raise SourceDataError("Taxonomy entry has empty externalId")
        if external_id in nodes:
            raise ValidationError(f"Duplicate taxonomy ID: {external_id}")
        nodes[external_id] = entry
    return Taxonomy(nodes)


def resolve_category(raw: dict[str, object], taxonomy: Taxonomy) -> ResolvedCategory:
    key = str(raw.get("key") or "").strip()
    name = str(raw.get("name") or "").strip()
    observed = ObservedCategory(
        key=key,
        name=name,
        url=str(raw.get("url") or "").strip() or None,
        group=str(raw.get("group") or "").strip() or None,
    )
    if not key:
        return ResolvedCategory(
            observed=observed, unresolved_key=True, missing_group=observed.group is None
        )

    declared_ids = _dedup_consecutive(
        [
            str(raw.get("level1Id") or "").strip(),
            str(raw.get("level2Id") or "").strip(),
            str(raw.get("level3Id") or "").strip(),
        ]
    )
    declared_ids = [item for item in declared_ids if item]
    declared_path: list[CategoryPathNode] = []
    unresolved_declared: list[str] = []
    for external_id in declared_ids:
        label = taxonomy.name_of(external_id)
        if label is None:
            unresolved_declared.append(external_id)
        else:
            declared_path.append(CategoryPathNode(external_id=external_id, name=label))
    declared_root = declared_ids[0] if declared_ids else None

    try:
        canonical_path, canonical_root = taxonomy.canonical_path(key)
        unresolved_key = False
    except ValidationError as error:
        if "cycle" in str(error):
            raise
        canonical_path, canonical_root = [], None
        unresolved_key = True

    terminal_id = declared_ids[-1] if declared_ids else None
    terminal_label = taxonomy.name_of(terminal_id) if terminal_id else None
    if terminal_label is None:
        terminal_label = name or None

    return ResolvedCategory(
        observed=observed,
        declared_path=declared_path,
        canonical_path=canonical_path,
        declared_root=declared_root,
        canonical_root=canonical_root,
        terminal_label=terminal_label,
        root_conflict=bool(
            declared_root and canonical_root and declared_root != canonical_root
        ),
        terminal_conflict=bool(raw.get("level3Id")) and key != str(raw.get("level3Id")),
        unresolved_declared_ids=unresolved_declared,
        unresolved_key=unresolved_key,
        missing_group=observed.group is None,
    )


def job_content_hash(
    material: dict[str, object], template_versions: dict[str, str]
) -> str:
    return sha256_bytes(
        stable_json_bytes(
            {
                "normalizer": NORMALIZER_VERSION,
                "templates": template_versions,
                "job": material,
            }
        )
    )


def canonicalize_job(
    raw: dict[str, object],
    taxonomy: Taxonomy,
    template_versions: dict[str, str],
) -> CanonicalJobPosting | None:
    if raw.get("status") != "completed":
        return None
    external_id = str(raw.get("externalId") or "").strip()
    if not external_id:
        raise ValidationError("Job has empty externalId")
    title = clean_text(str(raw.get("title") or ""))
    if not title:
        raise ValidationError(f"Job has empty title: {external_id}")

    it_paths: list[ResolvedCategory] = []
    secondary_paths: list[ResolvedCategory] = []
    unresolved: list[ObservedCategory] = []
    for entry in raw.get("categories") or []:
        if not isinstance(entry, dict):
            continue
        resolved = resolve_category(entry, taxonomy)
        if resolved.unresolved_key:
            unresolved.append(resolved.observed)
        elif resolved.declared_root == IT_ROOT:
            it_paths.append(resolved)
        else:
            secondary_paths.append(resolved)

    knowledge = [
        cleaned
        for item in (raw.get("knowledge") or [])
        if isinstance(item, dict) and (cleaned := clean_text(str(item.get("title") or "")))
    ]
    common_info: dict[str, str] = {}
    if isinstance(raw.get("commonInfo"), dict):
        for field, value in raw["commonInfo"].items():
            if value is None:
                continue
            common_info[str(field)] = _WHITESPACE.sub(" ", str(value)).strip()

    material = {
        "title": title,
        "description": clean_text(str(raw.get("description") or "")),
        "require_candidate": clean_text(str(raw.get("requireCandidate") or "")),
        "knowledge": knowledge,
        "it_paths": [path.model_dump(mode="json") for path in it_paths],
    }
    draft = CanonicalJobPosting(
        job_id=f"topcv:{external_id}",
        external_id=external_id,
        platform=str(raw.get("platformId") or "topcv"),
        source_url=str(raw.get("sourceUrl") or "").strip() or None,
        updated_at=parse_mongo_date(raw.get("updatedAt")),
        title=title,
        description=material["description"],
        require_candidate=material["require_candidate"],
        experience=str(raw.get("experience") or "").strip() or None,
        require_summary=clean_str_list(raw.get("requireSummary")),
        knowledge=knowledge,
        common_info=common_info,
        it_category_paths=it_paths,
        secondary_category_paths=secondary_paths,
        unresolved_categories=unresolved,
        content_hash="",
    )
    return draft.model_copy(
        update={"content_hash": job_content_hash(material, template_versions)}
    )


def _query_fingerprint(
    settings: MatchSettings,
    jobs_checksum: str,
    taxonomy_checksum: str,
    dense_model: dict[str, object],
    sparse_config: dict[str, object],
) -> str:
    """Everything that determines query-vector bytes (adoptable across builds)."""
    queries = settings.queries
    return sha256_bytes(
        stable_json_bytes(
            {
                "jobs_checksum": jobs_checksum,
                "taxonomy_checksum": taxonomy_checksum,
                "instruction": queries.instruction,
                "label_template": queries.label_template_version,
                "semantic_template": queries.semantic_template_version,
                "section_order": queries.section_order,
                "label_max_tokens": queries.label_max_tokens,
                "semantic_max_tokens": queries.semantic_max_tokens,
                "normalizer": NORMALIZER_VERSION,
                "dense_model": dense_model,
                "sparse_config": sparse_config,
                "shard_size": settings.shard_size,
            }
        )
    )


def _retrieval_fingerprint(
    query_fingerprint: str,
    esco_build_id: str,
    esco_collection: str,
    settings: MatchSettings,
) -> str:
    """Query vectors + ESCO snapshot + fusion params (adoptable across builds)."""
    retrieval = settings.retrieval
    return sha256_bytes(
        stable_json_bytes(
            {
                "query_fingerprint": query_fingerprint,
                "esco_build_id": esco_build_id,
                "esco_collection": esco_collection,
                "candidate_limit": retrieval.candidate_limit,
                "rrf_k": retrieval.rrf_k,
                "rrf_weights": retrieval.rrf_weights,
                "top_k": retrieval.top_k,
            }
        )
    )


def _match_config_fingerprint(
    settings: MatchSettings,
    jobs_checksum: str,
    taxonomy_checksum: str,
    esco_build_id: str,
    esco_collection: str,
    dense_model: dict[str, object],
    sparse_config: dict[str, object],
) -> tuple[str, str, str]:
    query_fp = _query_fingerprint(
        settings, jobs_checksum, taxonomy_checksum, dense_model, sparse_config
    )
    retrieval_fp = _retrieval_fingerprint(
        query_fp, esco_build_id, esco_collection, settings
    )
    rerank = settings.rerank
    full = sha256_bytes(
        stable_json_bytes(
            {
                "retrieval_fingerprint": retrieval_fp,
                "rerank_enabled": rerank.enabled,
                "rerank_model": rerank.model,
                "rerank_instruction": rerank.instruction,
                "rerank_top_n": rerank.top_n,
            }
        )
    )
    return full, query_fp, retrieval_fp


def ingest_jobs(
    settings: MatchSettings,
    jobs_path: Path,
    categories_path: Path,
    esco_build_dir: Path,
    esco_build_id: str,
    esco_collection: str,
    dense_model: dict[str, object],
    sparse_config: dict[str, object],
) -> Path:
    from esco_occupation_indexer.matching.settings import CHANNELS  # noqa: F401

    jobs_path = jobs_path.resolve()
    categories_path = categories_path.resolve()
    try:
        jobs_raw = json.loads(jobs_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SourceDataError(f"Cannot read jobs file: {jobs_path}") from error
    if not isinstance(jobs_raw, list):
        raise SourceDataError("Jobs file must contain a JSON list")

    taxonomy = load_taxonomy(categories_path)
    jobs_checksum = sha256_file(jobs_path)
    taxonomy_checksum = sha256_file(categories_path)
    template_versions = {
        "label": settings.queries.label_template_version,
        "semantic": settings.queries.semantic_template_version,
    }
    fingerprint, query_fp, retrieval_fp = _match_config_fingerprint(
        settings,
        jobs_checksum,
        taxonomy_checksum,
        esco_build_id,
        esco_collection,
        dense_model,
        sparse_config,
    )
    match_build_id = sha256_bytes(f"{jobs_checksum}:{fingerprint}".encode())[:12]
    match_dir = settings.match_work_dir / match_build_id
    manifest_file = match_dir / "manifest.json"

    if manifest_file.exists():
        manifest = load_match_manifest(match_dir)
        if (
            manifest.match_build_id == match_build_id
            and manifest.config_fingerprint == fingerprint
        ):
            from esco_occupation_indexer.matching.stages import (
                match_artifact_checksum_matches,
            )

            if manifest.stages["ingest"].status == "completed" and all(
                match_artifact_checksum_matches(match_dir, manifest, name)
                for name in ("canonical-jobs.jsonl.zst",)
            ):
                return match_dir
    else:
        match_dir.mkdir(parents=True, exist_ok=True)
        now = utc_now()
        manifest = MatchManifest(
            match_build_id=match_build_id,
            jobs_file=str(jobs_path),
            categories_file=str(categories_path),
            esco_build_dir=str(esco_build_dir.resolve()),
            esco_build_id=esco_build_id,
            esco_collection_snapshot=esco_collection,
            source_checksums={
                "jobs": jobs_checksum,
                "taxonomy": taxonomy_checksum,
            },
            config_fingerprint=fingerprint,
            query_fingerprint=query_fp,
            retrieval_fingerprint=retrieval_fp,
            match_config=settings.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            stages=new_match_stages(),
        )
        save_match_manifest(match_dir, manifest)
        write_match_resolved_settings(settings, match_dir / "resolved-config.json")

    start_match_stage(match_dir, "ingest")
    try:
        records: list[CanonicalJobPosting] = []
        skipped: Counter[str] = Counter()
        for raw in jobs_raw:
            if not isinstance(raw, dict):
                skipped["non_object"] += 1
                continue
            if raw.get("status") != "completed":
                skipped[str(raw.get("status"))] += 1
                continue
            records.append(canonicalize_job(raw, taxonomy, template_versions))
        records.sort(key=lambda record: record.job_id)
        write_jsonl_zst(match_dir / "canonical-jobs.jsonl.zst", records)
        atomic_write_json(
            match_dir / "ingest-stats.json",
            {
                "total_input": len(jobs_raw),
                "completed": len(records),
                "skipped_by_status": dict(sorted(skipped.items())),
            },
        )

        manifest = load_match_manifest(match_dir)
        manifest.counts.update(
            {
                "input_jobs": len(jobs_raw),
                "canonical_jobs": len(records),
            }
        )
        manifest.artifact_checksums.update(
            {
                "canonical-jobs.jsonl.zst": sha256_file(
                    match_dir / "canonical-jobs.jsonl.zst"
                ),
                "ingest-stats.json": sha256_file(match_dir / "ingest-stats.json"),
            }
        )
        save_match_manifest(match_dir, manifest)
        complete_match_stage(match_dir, "ingest")
        return match_dir
    except Exception as error:
        fail_match_stage(match_dir, "ingest", error)
        raise
