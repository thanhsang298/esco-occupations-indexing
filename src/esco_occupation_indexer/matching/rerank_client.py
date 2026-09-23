from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol

import httpx

from esco_occupation_indexer.errors import ValidationError

_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class RerankBackend(Protocol):
    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int,
        instruction: str | None = None,
    ) -> list[tuple[int, float]]:
        """Return (document_index, relevance_score) sorted by score descending."""
        ...


class VllmRerankBackend:
    """Cross-encoder reranker served by vLLM (/v1/rerank)."""

    backend_name = "vllm-rerank"

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: float = 120.0,
        max_retries: int = 4,
        retry_backoff_seconds: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
        client: httpx.Client | None = None,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValidationError(f"Invalid rerank base URL: {base_url!r}")
        if not model.strip():
            raise ValidationError("Rerank model must not be empty")
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._max_retries = max_retries
        self._retry_backoff_seconds = retry_backoff_seconds
        self._sleep = sleep
        self._client = client or httpx.Client(timeout=httpx.Timeout(timeout_seconds))
        self._owns_client = client is None
        self.last_result_count = 0

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    @property
    def runtime_metadata(self) -> dict[str, str | int | float | bool | None]:
        return {"rerank_model": self._model, "rerank_base_url": self._base_url}

    def _post(self, payload: dict[str, object]) -> object:
        url = f"{self._base_url}/v1/rerank"
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.post(url, json=payload)
            except httpx.RequestError as error:
                last_error = error
                if attempt < self._max_retries:
                    self._sleep(min(self._retry_backoff_seconds * (2**attempt), 30.0))
                    continue
                raise ValidationError(f"Could not reach rerank endpoint {url}: {error}") from error
            if 200 <= response.status_code < 300:
                try:
                    return response.json()
                except ValueError as error:
                    raise ValidationError("Rerank endpoint returned invalid JSON") from error
            if response.status_code in _RETRYABLE_STATUS_CODES and attempt < self._max_retries:
                self._sleep(min(self._retry_backoff_seconds * (2**attempt), 30.0))
                continue
            raise ValidationError(
                f"Rerank endpoint returned HTTP {response.status_code}: "
                f"{response.text.strip()[:800]}"
            )
        assert last_error is not None
        raise ValidationError(f"Rerank endpoint unreachable: {last_error}")

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int,
        instruction: str | None = None,
    ) -> list[tuple[int, float]]:
        if not documents:
            return []
        payload: dict[str, object] = {
            "model": self._model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
        }
        if instruction:
            # Fills the <Instruct> slot of Qwen-style chat templates instead of
            # their web-search default. Verified: the server honors this field.
            payload["instruction"] = instruction
        data = self._post(payload)
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            raise ValidationError("Rerank response has no results list")
        results = data["results"]
        parsed: list[tuple[int, float]] = []
        for item in results:
            if not isinstance(item, dict):
                raise ValidationError("Rerank result is not an object")
            index = item.get("index")
            score = item.get("relevance_score")
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < len(documents)
                or isinstance(score, bool)
                or not isinstance(score, (int, float))
            ):
                raise ValidationError(f"Invalid rerank result item: {item!r}")
            parsed.append((index, float(score)))
        if len(parsed) != len(results) or len({index for index, _ in parsed}) != len(parsed):
            raise ValidationError("Rerank results contain duplicate indices")
        parsed.sort(key=lambda item: (-item[1], item[0]))
        self.last_result_count = len(parsed)
        return parsed
