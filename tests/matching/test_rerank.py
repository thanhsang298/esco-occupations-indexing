from pathlib import Path

import httpx
import pytest

from esco_occupation_indexer.errors import ValidationError
from esco_occupation_indexer.matching.canonicalize import ingest_jobs
from esco_occupation_indexer.matching.rerank_client import VllmRerankBackend
from esco_occupation_indexer.matching.settings import MatchSettings
from tests.matching.helpers import make_jobs_file, make_taxonomy_file


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_rerank_parses_and_sorts_by_score() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 2, "relevance_score": 0.1},
                    {"index": 0, "relevance_score": 0.9},
                    {"index": 1, "relevance_score": 0.5},
                ]
            },
        )

    backend = VllmRerankBackend("http://x:8989", "m", client=_client(handler), sleep=lambda s: None)
    assert backend.rerank("q", ["a", "b", "c"], top_n=3) == [(0, 0.9), (1, 0.5), (2, 0.1)]
    backend.close()


def test_rerank_rejects_bad_items() -> None:
    for results in (
        [{"index": 5, "relevance_score": 0.1}],
        [{"index": 0, "relevance_score": "high"}],
        [{"index": 0, "relevance_score": 0.1}, {"index": 0, "relevance_score": 0.2}],
        "not-a-list",
    ):
        def handler(request: httpx.Request, results=results) -> httpx.Response:
            return httpx.Response(200, json={"results": results})

        backend = VllmRerankBackend(
            "http://x:8989", "m", client=_client(handler), sleep=lambda s: None
        )
        with pytest.raises(ValidationError):
            backend.rerank("q", ["a", "b"], top_n=2)
        backend.close()


def test_rerank_retries_transient_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 1.0}]})

    backend = VllmRerankBackend(
        "http://x:8989", "m", client=_client(handler), sleep=lambda s: None
    )
    assert backend.rerank("q", ["a"], top_n=1) == [(0, 1.0)]
    assert calls["n"] == 3
    backend.close()


def test_rerank_empty_documents() -> None:
    backend = VllmRerankBackend("http://x:8989", "m", sleep=lambda s: None)
    assert backend.rerank("q", [], top_n=5) == []
    backend.close()


def test_rerank_sends_instruction_field() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content.decode()))
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.7}]})

    import json

    backend = VllmRerankBackend("http://x:8989", "m", client=_client(handler), sleep=lambda s: None)
    assert backend.rerank("q", ["a"], top_n=1, instruction="Do the thing.") == [(0, 0.7)]
    assert seen["instruction"] == "Do the thing."
    backend.close()


def test_rerank_config_change_invalidates_build(tmp_path: Path) -> None:
    jobs = make_jobs_file(tmp_path / "jobs.json")
    taxonomy = make_taxonomy_file(tmp_path / "cats.json")
    base = {
        "match_work_dir": str(tmp_path / "m"),
        "shard_size": 1,
    }
    first = ingest_jobs(
        MatchSettings.model_validate(base),
        jobs,
        taxonomy,
        tmp_path / "esco",
        "esco-x",
        "col-x",
        {"model_id": "m", "model_revision": "r", "dimension": 8},
        {"model_id": "Qdrant/bm25"},
    )
    altered = MatchSettings.model_validate(
        {**base, "rerank": {"enabled": True, "model": "other-model"}}
    )
    second = ingest_jobs(
        altered,
        jobs,
        taxonomy,
        tmp_path / "esco",
        "esco-x",
        "col-x",
        {"model_id": "m", "model_revision": "r", "dimension": 8},
        {"model_id": "Qdrant/bm25"},
    )
    assert first != second
