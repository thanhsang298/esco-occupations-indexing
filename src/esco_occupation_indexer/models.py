from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OccupationHierarchyNode(StrictModel):
    uri: str
    code: str | None = None
    label: str
    node_type: Literal["isco_group", "occupation"]


class OccupationHierarchyData(StrictModel):
    level: int
    isco_major: str
    parent_uri: str
    parent_type: Literal["isco_group", "occupation"]
    terminal_isco_uri: str
    ancestor_uris: list[str]
    path: list[OccupationHierarchyNode]

    @property
    def path_text(self) -> str:
        return " > ".join(node.label for node in self.path)


class CanonicalOccupationRecord(StrictModel):
    schema_version: Literal["esco_occupation_record/v1"] = "esco_occupation_record/v1"
    esco_uri: str
    esco_id: str
    esco_version: str
    concept_type: str
    isco_group: str
    code: str
    status: Literal["released"] = "released"
    modified_at: datetime | None = None
    preferred_label: str
    alternative_labels: list[str] = Field(default_factory=list)
    hidden_labels: list[str] = Field(default_factory=list)
    description: str | None = None
    definition: str | None = None
    scope_note: str | None = None
    regulated_profession_note: str | None = None
    nace_code: str | None = None
    hierarchy: OccupationHierarchyData
    essential_skills: list[str] = Field(default_factory=list)
    optional_skills: list[str] = Field(default_factory=list)
    green_share: float | None = None
    collections: list[str] = Field(default_factory=list)
    source_language: Literal["en"] = "en"
    source_files: list[str]
    content_hash: str


class IscoGroupRecord(StrictModel):
    schema_version: Literal["esco_isco_group_record/v1"] = "esco_isco_group_record/v1"
    esco_uri: str
    code: str
    preferred_label: str
    alternative_labels: list[str] = Field(default_factory=list)
    description: str | None = None
    status: str | None = None


class LabelVariant(StrictModel):
    kind: Literal["preferred", "alternative"]
    language: Literal["en"] = "en"
    text: str
    normalized: str


class StageState(StrictModel):
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None


class BuildManifest(StrictModel):
    schema_version: Literal["esco_occupation_index_build/v1"] = (
        "esco_occupation_index_build/v1"
    )
    build_id: str
    esco_version: str
    source_language: str
    source_zip: str
    source_checksums: dict[str, str]
    config_fingerprint: str
    index_config: dict[str, object]
    collection_name: str
    created_at: datetime
    updated_at: datetime
    stages: dict[str, StageState]
    counts: dict[str, int] = Field(default_factory=dict)
    runtime: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    artifact_checksums: dict[str, str] = Field(default_factory=dict)
