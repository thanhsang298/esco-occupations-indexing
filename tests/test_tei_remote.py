from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from esco_occupation_indexer.settings import load_settings
from esco_occupation_indexer.tei import TeiDenseBackend


@pytest.mark.integration
def test_configured_tei_endpoint() -> None:
    if os.getenv("RUN_TEI_INTEGRATION") != "1":
        pytest.skip("set RUN_TEI_INTEGRATION=1 with TEI environment variables loaded")

    settings = load_settings(Path("config/indexing.yaml"))
    backend = TeiDenseBackend(settings)
    try:
        vectors = backend.encode(["ESCO indexing integration probe"], batch_size=1)
    finally:
        backend.close()

    assert vectors.shape == (1, settings.embedding.dimension)
    assert np.isfinite(vectors).all()
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3)
