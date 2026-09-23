import pytest

from esco_occupation_indexer.matching.retrieval import rrf_fuse

WEIGHTS = {"label_dense": 1.0, "semantic_dense": 1.0, "lexical_sparse": 1.0}


def test_rrf_ranks_start_at_one_and_fuse() -> None:
    fused = rrf_fuse(
        {
            "label_dense": [("a", 0.9), ("b", 0.8)],
            "semantic_dense": [("b", 0.7)],
            "lexical_sparse": [],
        },
        60,
        WEIGHTS,
        10,
    )
    assert [item[0] for item in fused] == ["b", "a"]
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)
    assert fused[0][2] == {
        "label_dense": (2, 0.8),
        "semantic_dense": (1, 0.7),
        "lexical_sparse": None,
    }


def test_rrf_tie_break_is_deterministic() -> None:
    channels = {
        "label_dense": [("b", 0.5), ("a", 0.5)],
        "semantic_dense": [("a", 0.5), ("b", 0.5)],
        "lexical_sparse": [("a", 1.0), ("b", 1.0)],
    }
    first = rrf_fuse(channels, 60, WEIGHTS, 10)
    second = rrf_fuse(channels, 60, WEIGHTS, 10)
    assert [item[0] for item in first] == [item[0] for item in second] == ["a", "b"]


def test_rrf_top_k_and_missing_channels() -> None:
    fused = rrf_fuse(
        {"label_dense": [(f"id-{i}", 1.0) for i in range(60)]},
        60,
        WEIGHTS,
        10,
    )
    assert len(fused) == 10
