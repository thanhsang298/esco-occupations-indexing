from __future__ import annotations

import re
import unicodedata

_DASH_TRANSLATION = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2043": "-",
        "\u2212": "-",
        "\u058a": "-",
        "\ufe58": "-",
        "\ufe63": "-",
        "\uff0d": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u02bc": "'",
        "\u2032": "'",
        "\u2035": "'",
        "\u201c": '"',
        "\u201d": '"',
    }
)


def normalize_label(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.translate(_DASH_TRANSLATION).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    # Descriptions, definitions, and scope notes are embedded and stored as
    # official text.  Only remove accidental edge whitespace; preserve all
    # internal punctuation and whitespace rather than applying label
    # normalization to semantic content.
    cleaned = value.strip()
    return cleaned or None


def split_labels(value: str | None) -> list[str]:
    if not value:
        return []
    return [label.strip() for label in re.split(r"[\r\n]+", value) if label.strip()]


def deduplicate_labels(preferred: str, alternatives: list[str]) -> tuple[str, list[str]]:
    preferred = preferred.strip()
    seen = {normalize_label(preferred)}
    unique: list[tuple[str, str]] = []
    for label in alternatives:
        text = label.strip()
        key = normalize_label(text)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append((key, text))
    unique.sort(key=lambda item: item[0])
    return preferred, [text for _, text in unique]


def deduplicate_hidden(
    preferred: str, alternatives: list[str], hidden: list[str]
) -> list[str]:
    seen = {normalize_label(preferred), *(normalize_label(item) for item in alternatives)}
    unique: list[tuple[str, str]] = []
    for label in hidden:
        text = label.strip()
        key = normalize_label(text)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append((key, text))
    unique.sort(key=lambda item: item[0])
    return [text for _, text in unique]
