from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class TeiSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = "http://localhost:8080"
    api_key_env: str = "TEI_API_KEY"
    timeout_seconds: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=4, ge=0)
    retry_backoff_seconds: float = Field(default=1.0, ge=0)
    concurrent_requests: int = Field(default=16, gt=0)
    verify_model_identity: bool = True
    require_model_sha: bool = True

    @property
    def api_key(self) -> str | None:
        value = os.getenv(self.api_key_env)
        return value or None


class EmbeddingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: Literal["tei", "sentence_transformers"] = "tei"
    model_id: str = "microsoft/harrier-oss-v1-0.6b"
    model_revision: str = "f9b9dc8d367d443f2479d27aa5d8d2850c0774ee"
    dimension: int = Field(default=1024, gt=0)
    max_tokens: int = Field(default=512, gt=0)
    batch_size: int = Field(default=32, gt=0)
    shard_size: int = Field(default=256, gt=0)
    device: str = "auto"
    tei: TeiSettings = Field(default_factory=TeiSettings)


class SparseSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str = "Qdrant/bm25"
    language: str = "english"
    k: float = Field(default=1.2, gt=0)
    b: float = Field(default=0.75, ge=0, le=1)
    avg_len: float = Field(default=256.0, gt=0)
    token_max_length: int = Field(default=40, gt=0)
    disable_stemmer: bool = False


class QdrantSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = "http://localhost:6333"
    api_key_env: str = "QDRANT_API_KEY"
    collection_prefix: str = "esco_occupation_concepts"
    alias: str = "esco_occupation_concepts_current"
    upload_batch_size: int = Field(default=128, gt=0)
    timeout_seconds: int = Field(default=60, gt=0)

    @property
    def api_key(self) -> str | None:
        value = os.getenv(self.api_key_env)
        return value or None


class IndexingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    esco_version: Literal["1.2.1"] = "1.2.1"
    source_language: Literal["en"] = "en"
    work_dir: Path = Path("data/work")
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    sparse: SparseSettings = Field(default_factory=SparseSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)

    def resolved(self, base_dir: Path | None = None) -> IndexingSettings:
        data = self.model_dump(mode="json")
        work_dir = Path(data["work_dir"])
        if base_dir is not None and not work_dir.is_absolute():
            data["work_dir"] = str((base_dir / work_dir).resolve())
        if url := os.getenv("QDRANT_URL"):
            data["qdrant"]["url"] = url
        if url := os.getenv("TEI_URL"):
            data["embedding"]["tei"]["url"] = url
        return type(self).model_validate(data)


def load_settings(path: Path) -> IndexingSettings:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return IndexingSettings.model_validate(raw).resolved(path.parent.parent.resolve())


def write_resolved_settings(settings: IndexingSettings, path: Path) -> None:
    path.write_text(
        json.dumps(settings.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_resolved_settings(build_dir: Path) -> IndexingSettings:
    path = build_dir / "resolved-config.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    settings = IndexingSettings.model_validate(data)
    if url := os.getenv("QDRANT_URL"):
        settings.qdrant.url = url
    if url := os.getenv("TEI_URL"):
        settings.embedding.tei.url = url
    return settings
