from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np

from esco_occupation_indexer.errors import ValidationError
from esco_occupation_indexer.settings import IndexingSettings

_RETRYABLE_STATUS_CODES = {408, 424, 429, 500, 502, 503, 504}
_MAX_ERROR_DETAIL_LENGTH = 800


class _TeiHttpError(ValidationError):
    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        super().__init__(f"TEI returned HTTP {status_code}: {detail}")


@dataclass(frozen=True)
class _Token:
    token_id: int
    special: bool


class TeiDenseBackend:
    """Dense embedding backend for Hugging Face Text Embeddings Inference."""

    backend_name = "tei"
    device = "remote-tei"

    def __init__(
        self,
        settings: IndexingSettings,
        *,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        config = settings.embedding.tei
        try:
            endpoint = httpx.URL(config.url)
        except (TypeError, ValueError) as error:
            raise ValidationError(f"Invalid TEI URL: {config.url!r}") from error
        if endpoint.scheme not in {"http", "https"} or not endpoint.host:
            raise ValidationError(f"TEI URL must be an absolute HTTP(S) URL: {config.url!r}")

        self.dimension = settings.embedding.dimension
        self.max_tokens = settings.embedding.max_tokens
        self._base_url = str(endpoint).rstrip("/")
        self._max_retries = config.max_retries
        self._retry_backoff_seconds = config.retry_backoff_seconds
        self._concurrent_requests = config.concurrent_requests
        self._sleep = sleep
        self._headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if config.api_key:
            self._headers["Authorization"] = f"Bearer {config.api_key}"
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(config.timeout_seconds),
            follow_redirects=True,
        )
        self._owns_client = client is None
        self._token_cache: dict[str, tuple[_Token, ...]] = {}
        self.last_effective_batch_size = 1

        try:
            info = self._request_json("GET", "/info")
            if not isinstance(info, dict):
                raise ValidationError("TEI /info response must be a JSON object")
            self.info = info
            self._validate_info(settings)
        except Exception:
            if self._owns_client:
                self._client.close()
            raise

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    @property
    def runtime_metadata(self) -> dict[str, str | int | float | bool | None]:
        return {
            "tei_url": self._base_url,
            "tei_model_id": str(self.info.get("model_id")),
            "tei_model_sha": self.info.get("model_sha"),
            "tei_model_dtype": str(self.info.get("model_dtype")),
            "tei_server_version": str(self.info.get("version")),
            "tei_max_input_length": self.max_input_length,
            "tei_max_batch_tokens": self.max_batch_tokens,
            "tei_max_client_batch_size": self.max_batch_size,
            "tei_concurrent_requests": self._concurrent_requests,
        }

    def _validate_info(self, settings: IndexingSettings) -> None:
        embedding = settings.embedding
        tei = embedding.tei
        model_type = self.info.get("model_type")
        if not isinstance(model_type, dict) or "embedding" not in model_type:
            raise ValidationError("TEI endpoint is not serving an embedding model")

        served_model_id = self.info.get("model_id")
        served_model_sha = self.info.get("model_sha")
        if tei.verify_model_identity:
            if served_model_id != embedding.model_id:
                raise ValidationError(
                    f"TEI model ID mismatch: expected {embedding.model_id!r}, "
                    f"received {served_model_id!r}"
                )
            if tei.require_model_sha and not served_model_sha:
                raise ValidationError("TEI /info did not provide model_sha")
            if served_model_sha and served_model_sha != embedding.model_revision:
                raise ValidationError(
                    f"TEI model revision mismatch: expected {embedding.model_revision!r}, "
                    f"received {served_model_sha!r}"
                )

        self.max_input_length = self._positive_info_int("max_input_length")
        self.max_batch_tokens = self._positive_info_int("max_batch_tokens")
        self.max_batch_size = self._positive_info_int("max_client_batch_size")
        if self.max_input_length < self.max_tokens:
            raise ValidationError(
                "TEI max_input_length is lower than the configured embedding max_tokens: "
                f"{self.max_input_length} < {self.max_tokens}"
            )
        if self.max_batch_tokens < self.max_tokens:
            raise ValidationError(
                "TEI max_batch_tokens is lower than the configured embedding max_tokens: "
                f"{self.max_batch_tokens} < {self.max_tokens}"
            )

    def _positive_info_int(self, field: str) -> int:
        value = self.info.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValidationError(f"TEI /info contains invalid {field}: {value!r}")
        return value

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{self._base_url}{path}"
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.request(
                    method,
                    url,
                    json=payload,
                    headers=self._headers,
                )
            except httpx.RequestError as error:
                if attempt < self._max_retries:
                    self._sleep(self._retry_delay(attempt, None))
                    continue
                raise ValidationError(f"Could not reach TEI endpoint {url}: {error}") from error

            if 200 <= response.status_code < 300:
                try:
                    return response.json()
                except ValueError as error:
                    raise ValidationError(f"TEI returned invalid JSON from {path}") from error

            if response.status_code in _RETRYABLE_STATUS_CODES and attempt < self._max_retries:
                self._sleep(self._retry_delay(attempt, response.headers.get("Retry-After")))
                continue
            detail = response.text.strip()[:_MAX_ERROR_DETAIL_LENGTH] or "empty response body"
            raise _TeiHttpError(response.status_code, detail)

        raise AssertionError("unreachable")

    def _retry_delay(self, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return min(max(float(retry_after), 0.0), 30.0)
            except ValueError:
                pass
        return min(self._retry_backoff_seconds * (2**attempt), 30.0)

    def _tokenize_chunk(self, texts: list[str]) -> list[tuple[_Token, ...]]:
        try:
            response = self._request_json(
                "POST",
                "/tokenize",
                {"inputs": texts, "add_special_tokens": True},
            )
        except _TeiHttpError as error:
            if error.status_code == 413 and len(texts) > 1:
                middle = len(texts) // 2
                return [
                    *self._tokenize_chunk(texts[:middle]),
                    *self._tokenize_chunk(texts[middle:]),
                ]
            raise
        if not isinstance(response, list) or len(response) != len(texts):
            raise ValidationError("TEI /tokenize returned the wrong number of results")

        tokenized: list[tuple[_Token, ...]] = []
        for item in response:
            if not isinstance(item, list) or not item:
                raise ValidationError("TEI /tokenize returned an empty or invalid token list")
            tokens: list[_Token] = []
            for token in item:
                if not isinstance(token, dict):
                    raise ValidationError("TEI /tokenize returned an invalid token object")
                token_id = token.get("id")
                special = token.get("special")
                if (
                    isinstance(token_id, bool)
                    or not isinstance(token_id, int)
                    or token_id < 0
                    or not isinstance(special, bool)
                ):
                    raise ValidationError("TEI /tokenize returned invalid token metadata")
                tokens.append(_Token(token_id=token_id, special=special))
            tokenized.append(tuple(tokens))
        return tokenized

    def _tokens_for(self, texts: list[str]) -> list[tuple[_Token, ...]]:
        missing = list(dict.fromkeys(text for text in texts if text not in self._token_cache))
        chunks = [
            missing[start : start + self.max_batch_size]
            for start in range(0, len(missing), self.max_batch_size)
        ]
        for chunk, token_rows in zip(
            chunks,
            self._parallel_map(self._tokenize_chunk, chunks),
            strict=True,
        ):
            for text, tokens in zip(chunk, token_rows, strict=True):
                self._token_cache[text] = tokens
        return [self._token_cache[text] for text in texts]

    def token_id_lists(self, texts: list[str]) -> list[list[int]]:
        """Public token IDs (including special tokens) for measuring budgets."""
        return [[token.token_id for token in row] for row in self._tokens_for(texts)]

    def _parallel_map(self, operation: Callable[[Any], Any], items: list[Any]) -> list[Any]:
        if len(items) < 2 or self._concurrent_requests == 1:
            return [operation(item) for item in items]
        with ThreadPoolExecutor(
            max_workers=min(self._concurrent_requests, len(items))
        ) as executor:
            return list(executor.map(operation, items))

    @staticmethod
    def _truncate_token_ids(tokens: tuple[_Token, ...], max_tokens: int) -> list[int]:
        if len(tokens) <= max_tokens:
            return [token.token_id for token in tokens]

        trailing_special_count = 0
        for token in reversed(tokens):
            if not token.special:
                break
            trailing_special_count += 1
        trailing_special_count = min(trailing_special_count, max_tokens)
        content_count = max_tokens - trailing_special_count
        token_ids = [token.token_id for token in tokens[:content_count]]
        if trailing_special_count:
            token_ids.extend(token.token_id for token in tokens[-trailing_special_count:])
        return token_ids

    def count_truncated(self, texts: list[str], max_tokens: int) -> int:
        return sum(len(tokens) > max_tokens for tokens in self._tokens_for(texts))

    def _embedding_batches(
        self,
        token_rows: list[list[int]],
        batch_size: int,
    ) -> list[list[list[int]]]:
        request_limit = min(max(1, batch_size), self.max_batch_size)
        batches: list[list[list[int]]] = []
        current: list[list[int]] = []
        current_tokens = 0
        for row in token_rows:
            if len(row) > self.max_batch_tokens:
                raise ValidationError(
                    "One TEI input exceeds the server max_batch_tokens after truncation"
                )
            if current and (
                len(current) >= request_limit
                or current_tokens + len(row) > self.max_batch_tokens
            ):
                batches.append(current)
                current = []
                current_tokens = 0
            current.append(row)
            current_tokens += len(row)
        if current:
            batches.append(current)
        return batches

    def _embed_token_batch(self, token_rows: list[list[int]]) -> np.ndarray:
        try:
            response = self._request_json(
                "POST",
                "/embed",
                {
                    "inputs": token_rows,
                    "normalize": True,
                    "truncate": False,
                    "truncation_direction": "right",
                },
            )
        except _TeiHttpError as error:
            if error.status_code == 413 and len(token_rows) > 1:
                middle = len(token_rows) // 2
                return np.concatenate(
                    [
                        self._embed_token_batch(token_rows[:middle]),
                        self._embed_token_batch(token_rows[middle:]),
                    ],
                    axis=0,
                )
            raise
        try:
            vectors = np.asarray(response, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValidationError("TEI /embed returned non-numeric embeddings") from error
        expected_shape = (len(token_rows), self.dimension)
        if vectors.shape != expected_shape or not np.isfinite(vectors).all():
            raise ValidationError(
                f"TEI /embed returned shape {vectors.shape}; expected {expected_shape}"
            )
        return vectors

    def encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        tokens = self._tokens_for(texts)
        token_rows = [self._truncate_token_ids(row, self.max_tokens) for row in tokens]
        batches = self._embedding_batches(token_rows, batch_size)
        self.last_effective_batch_size = max(len(batch) for batch in batches)
        vectors = self._parallel_map(self._embed_token_batch, batches)
        return np.concatenate(vectors, axis=0)
