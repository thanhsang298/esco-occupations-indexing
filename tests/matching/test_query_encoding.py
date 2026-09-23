import numpy as np
import pytest

from esco_occupation_indexer.errors import ValidationError
from esco_occupation_indexer.matching.query_encoding import QuerySparseBackend
from esco_occupation_indexer.settings import SparseSettings


def _backend() -> QuerySparseBackend:
    return QuerySparseBackend(SparseSettings())


def test_query_encode_returns_binary_sorted_weights() -> None:
    backend = _backend()
    (encoding,) = backend.query_encode(["Backend Developer backend"])
    assert encoding.indices.ndim == 1
    assert len(encoding.indices) == len(encoding.values)
    assert bool((encoding.values == 1.0).all())
    assert (np.diff(encoding.indices.astype(np.int64)) > 0).all()


def test_query_encode_differs_from_document_embed() -> None:
    backend = _backend()
    (query,) = backend.query_encode(["developer developer developer"])
    (doc,) = backend.encode(["developer developer developer"])
    assert bool((query.values == 1.0).all())
    assert not bool((doc.values == 1.0).all())


def test_query_encode_rejects_empty() -> None:
    backend = _backend()
    with pytest.raises(ValidationError, match="empty"):
        backend.query_encode(["!!!"])
