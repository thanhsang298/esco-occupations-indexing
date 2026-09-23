from __future__ import annotations

import numpy as np

from esco_occupation_indexer.embedding import FastEmbedBm25Backend, SparseEncoding
from esco_occupation_indexer.errors import ValidationError
from esco_occupation_indexer.settings import IndexingSettings


def build_query_dense_backend(
    esco_settings: IndexingSettings, max_tokens: int
):
    """TEI backend sharing ESCO model identity but allowing longer queries."""
    from esco_occupation_indexer.tei import TeiDenseBackend

    embedding = esco_settings.embedding.model_copy(update={"max_tokens": max_tokens})
    query_settings = esco_settings.model_copy(update={"embedding": embedding})
    backend = TeiDenseBackend(query_settings)
    if max_tokens > backend.max_input_length:
        backend.close()
        raise ValidationError(
            f"Query max_tokens {max_tokens} exceeds TEI max_input_length "
            f"{backend.max_input_length}"
        )
    if max_tokens > backend.max_batch_tokens:
        backend.close()
        raise ValidationError(
            f"Query max_tokens {max_tokens} exceeds TEI max_batch_tokens "
            f"{backend.max_batch_tokens}"
        )
    return backend


class QuerySparseBackend(FastEmbedBm25Backend):
    """BM25 query encoder: binary weights, deduped and sorted indices."""

    def query_encode(self, texts: list[str]) -> list[SparseEncoding]:
        encoded = self.model.query_embed(texts)
        results: list[SparseEncoding] = []
        for text, item in zip(texts, encoded, strict=True):
            indices = np.asarray(item.indices, dtype=np.uint32)
            values = np.asarray(item.values, dtype=np.float32)
            if indices.shape != values.shape or indices.ndim != 1:
                raise ValidationError(
                    f"Sparse query indices/values misaligned for {text[:60]!r}"
                )
            if not len(indices):
                raise ValidationError(f"Sparse query is empty for {text[:60]!r}")
            if len(set(indices.tolist())) != len(indices):
                raise ValidationError(f"Sparse query indices not unique for {text[:60]!r}")
            if not bool((values == 1.0).all()):
                raise ValidationError(
                    f"Sparse query weights must all be 1 for {text[:60]!r}"
                )
            order = np.argsort(indices, kind="stable")
            results.append(
                SparseEncoding(indices=indices[order], values=values[order])
            )
        if len(results) != len(texts):
            raise ValidationError("Sparse backend returned the wrong number of vectors")
        return results
