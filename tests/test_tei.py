from __future__ import annotations

import json

import httpx
import numpy as np
import pytest

from esco_occupation_indexer.errors import ValidationError
from esco_occupation_indexer.settings import IndexingSettings
from esco_occupation_indexer.tei import TeiDenseBackend

MODEL_ID = "microsoft/harrier-oss-v1-0.6b"
MODEL_SHA = "f9b9dc8d367d443f2479d27aa5d8d2850c0774ee"


def _settings(**tei_overrides: object) -> IndexingSettings:
    return IndexingSettings.model_validate(
        {
            "embedding": {
                "backend": "tei",
                "model_id": MODEL_ID,
                "model_revision": MODEL_SHA,
                "dimension": 3,
                "max_tokens": 4,
                "batch_size": 4,
                "tei": {
                    "url": "https://tei.example.test",
                    "max_retries": 0,
                    "retry_backoff_seconds": 0,
                    **tei_overrides,
                },
            }
        }
    )


def _info(model_id: str = MODEL_ID) -> dict[str, object]:
    return {
        "model_id": model_id,
        "model_sha": MODEL_SHA,
        "model_dtype": "float16",
        "model_type": {"embedding": {"pooling": "last_token"}},
        "max_input_length": 32768,
        "max_batch_tokens": 8,
        "max_client_batch_size": 2,
        "version": "1.9.3",
    }


def test_tei_backend_verifies_identity_tokenizes_truncates_and_embeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[tuple[str, dict[str, object] | None, str | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content) if request.content else None
        requests.append((request.url.path, payload, request.headers.get("Authorization")))
        if request.url.path == "/info":
            return httpx.Response(200, json=_info())
        if request.url.path == "/tokenize":
            assert payload is not None
            return httpx.Response(
                200,
                json=[
                    [
                        *[
                            {"id": index + 1, "text": word, "special": False}
                            for index, word in enumerate(text.split())
                        ],
                        {"id": 99, "text": "<eos>", "special": True},
                    ]
                    for text in payload["inputs"]
                ],
            )
        if request.url.path == "/embed":
            assert payload is not None
            token_rows = payload["inputs"]
            assert payload["normalize"] is True
            assert payload["truncate"] is False
            return httpx.Response(200, json=[[1.0, 0.0, 0.0] for _ in token_rows])
        raise AssertionError(f"Unexpected TEI path: {request.url.path}")

    monkeypatch.setenv("TEI_API_KEY", "test-secret")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    backend = TeiDenseBackend(_settings(), client=client)
    texts = ["one two", "one two three four five"]

    assert backend.count_truncated(texts, max_tokens=4) == 1
    vectors = backend.encode(texts, batch_size=8)

    assert vectors.dtype == np.float32
    assert vectors.shape == (2, 3)
    assert backend.max_batch_size == 2
    assert backend.last_effective_batch_size == 2
    assert backend.runtime_metadata["tei_model_sha"] == MODEL_SHA
    assert sum(path == "/tokenize" for path, _, _ in requests) == 1
    embed_payload = next(payload for path, payload, _ in requests if path == "/embed")
    assert embed_payload is not None
    assert embed_payload["inputs"][1] == [1, 2, 3, 99]
    assert all(auth == "Bearer test-secret" for _, _, auth in requests)


def test_tei_backend_rejects_wrong_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_info(model_id="other/model"))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(ValidationError, match="model ID mismatch"):
        TeiDenseBackend(_settings(), client=client)


def test_tei_backend_retries_overload_response() -> None:
    calls = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, text="busy", headers={"Retry-After": "0"})
        return httpx.Response(200, json=_info())

    client = httpx.Client(transport=httpx.MockTransport(handler))
    backend = TeiDenseBackend(
        _settings(max_retries=1),
        client=client,
        sleep=sleeps.append,
    )

    assert backend.info["model_id"] == MODEL_ID
    assert calls == 2
    assert sleeps == [0.0]


def test_tei_url_environment_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEI_URL", "https://gpu-vm.example.test:8080")
    settings = IndexingSettings().resolved()
    assert settings.embedding.backend == "tei"
    assert settings.embedding.tei.url == "https://gpu-vm.example.test:8080"
