from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ObservedCategory(StrictModel):
    key: str
    name: str
    url: str | None = None
    group: str | None = None


class CategoryPathNode(StrictModel):
    external_id: str
    name: str


class ResolvedCategory(StrictModel):
    """One posting category with three layers of information.

    - observed: raw key/name as seen in the posting.
    - declared_path: level1Id → level2Id → level3Id with taxonomy names joined.
    - canonical_path: key walked up via parentExternalId in the taxonomy file.
    """

    observed: ObservedCategory
    declared_path: list[CategoryPathNode] = Field(default_factory=list)
    canonical_path: list[CategoryPathNode] = Field(default_factory=list)
    declared_root: str | None = None
    canonical_root: str | None = None
    terminal_label: str | None = None
    root_conflict: bool = False
    terminal_conflict: bool = False
    unresolved_declared_ids: list[str] = Field(default_factory=list)
    unresolved_key: bool = False
    missing_group: bool = False


class CanonicalJobPosting(StrictModel):
    schema_version: Literal["topcv_job_record/v1"] = "topcv_job_record/v1"
    job_id: str
    external_id: str
    platform: str
    source_url: str | None = None
    updated_at: datetime | None = None
    title: str
    description: str | None = None
    require_candidate: str | None = None
    experience: str | None = None
    require_summary: list[str] = Field(default_factory=list)
    knowledge: list[str] = Field(default_factory=list)
    common_info: dict[str, str] = Field(default_factory=dict)
    it_category_paths: list[ResolvedCategory] = Field(default_factory=list)
    secondary_category_paths: list[ResolvedCategory] = Field(default_factory=list)
    unresolved_categories: list[ObservedCategory] = Field(default_factory=list)
    content_hash: str


class ChannelResult(StrictModel):
    rank: int
    score: float


class MatchCandidate(StrictModel):
    rank: int
    esco_id: str
    esco_uri: str
    preferred_label: str
    isco_code: str
    path_text: str
    fused_score: float
    channels: dict[str, ChannelResult | None]
    rerank_score: float | None = None
    rerank_rank: int | None = None


class JobMatches(StrictModel):
    schema_version: Literal["topcv_esco_matches/v1"] = "topcv_esco_matches/v1"
    job_id: str
    job_content_hash: str
    esco_build_id: str
    esco_collection: str
    match_config_fingerprint: str
    query_template_versions: dict[str, str]
    candidates: list[MatchCandidate] = Field(default_factory=list)


class StageState(StrictModel):
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None


class MatchManifest(StrictModel):
    schema_version: Literal["topcv_match_build/v1"] = "topcv_match_build/v1"
    match_build_id: str
    jobs_file: str
    categories_file: str
    esco_build_dir: str
    esco_build_id: str
    esco_collection_snapshot: str
    source_checksums: dict[str, str]
    config_fingerprint: str
    query_fingerprint: str = ""
    retrieval_fingerprint: str = ""
    match_config: dict[str, object]
    created_at: datetime
    updated_at: datetime
    stages: dict[str, StageState]
    counts: dict[str, int] = Field(default_factory=dict)
    runtime: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    artifact_checksums: dict[str, str] = Field(default_factory=dict)
