from esco_occupation_indexer.normalize import (
    clean_optional,
    deduplicate_hidden,
    deduplicate_labels,
    normalize_label,
)


def test_normalization_preserves_technical_punctuation() -> None:
    assert normalize_label(" C++ ") == "c++"
    assert normalize_label("C#") == "c#"
    assert normalize_label(".NET") == ".net"
    assert normalize_label("Node.js") == "node.js"
    assert normalize_label("CI/CD") == "ci/cd"
    assert normalize_label("REST\u00a0API") == "rest api"


def test_normalization_unifies_unicode_punctuation() -> None:
    assert normalize_label("client–server’s API") == "client-server's api"
    assert normalize_label("O\u02bcBrien\u2015style") == "o'brien-style"


def test_label_dedup_is_local_and_deterministic() -> None:
    preferred, alternatives = deduplicate_labels(
        "Test APIs", ["test APIs", "API testing", "api Testing", "Endpoint testing"]
    )
    assert preferred == "Test APIs"
    assert alternatives == ["API testing", "Endpoint testing"]
    assert deduplicate_hidden(preferred, alternatives, ["API testing", "API checks"]) == [
        "API checks"
    ]


def test_optional_content_preserves_internal_official_whitespace() -> None:
    assert clean_optional("  First line\n  Second line  ") == "First line\n  Second line"
