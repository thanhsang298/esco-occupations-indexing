from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class QuerySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instruction: str = (
        "Given a job posting, retrieve the ESCO occupation that best represents "
        "its work and responsibilities."
    )
    label_template_version: str = "job-label-v1-title-it-terminal"
    semantic_template_version: str = "job-semantic-v1-title-context-resp-req"
    section_order: list[str] = Field(
        default_factory=lambda: ["title", "category", "responsibilities", "requirements"]
    )
    label_max_tokens: int = Field(default=128, gt=0)
    semantic_max_tokens: int = Field(default=4096, gt=0)


class RetrievalSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_limit: int = Field(default=50, gt=0)
    rrf_k: int = Field(default=60, gt=0)
    rrf_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "label_dense": 1.0,
            "semantic_dense": 1.0,
            "lexical_sparse": 1.0,
        }
    )
    top_k: int = Field(default=10, gt=0)
    qdrant_batch_jobs: int = Field(default=8, gt=0)
    query_timeout_seconds: int = Field(default=120, gt=0)


class RerankSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    base_url: str = "http://localhost:8989"
    model: str = "mixedbread-ai/mxbai-rerank-large-v2"
    instruction: str = (
        "Given a job posting, retrieve the ESCO occupation that best represents "
        "its work and responsibilities."
    )
    top_n: int = Field(default=20, gt=0)
    timeout_seconds: int = Field(default=120, gt=0)
    max_retries: int = Field(default=4, ge=0)
    retry_backoff_seconds: float = Field(default=1.0, ge=0)
    max_workers: int = Field(default=4, gt=0)


class MatchSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    match_work_dir: Path = Path("data/job-matching-work")
    shard_size: int = Field(default=256, gt=0)
    queries: QuerySettings = Field(default_factory=QuerySettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    rerank: RerankSettings = Field(default_factory=RerankSettings)

    def resolved(self, base_dir: Path | None = None) -> MatchSettings:
        data = self.model_dump(mode="json")
        work_dir = Path(data["match_work_dir"])
        if base_dir is not None and not work_dir.is_absolute():
            data["match_work_dir"] = str((base_dir / work_dir).resolve())
        return type(self).model_validate(data)


def load_match_settings(path: Path) -> MatchSettings:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return MatchSettings.model_validate(raw).resolved(path.parent.parent.resolve())


def write_match_resolved_settings(settings: MatchSettings, path: Path) -> None:
    path.write_text(
        json.dumps(settings.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_match_resolved_settings(match_dir: Path) -> MatchSettings:
    data = json.loads((match_dir / "resolved-config.json").read_text(encoding="utf-8"))
    return MatchSettings.model_validate(data)


CHANNELS: tuple[str, ...] = ("label_dense", "semantic_dense", "lexical_sparse")

RerankMode = Literal["off"]
