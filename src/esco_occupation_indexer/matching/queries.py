from __future__ import annotations

from dataclasses import dataclass, field

from esco_occupation_indexer.matching.models import CanonicalJobPosting


@dataclass
class QuerySection:
    name: str
    header: str
    body: str
    # Lower cut_priority is shortened first when over budget.
    cut_priority: int = 99


@dataclass
class FittedQuery:
    text: str
    original_tokens: int
    actual_tokens: int
    truncated: bool
    shortened_sections: list[str] = field(default_factory=list)


def _terminal_labels(record: CanonicalJobPosting) -> list[str]:
    labels: list[str] = []
    for path in record.it_category_paths:
        label = path.terminal_label or path.observed.name
        if label and label not in labels:
            labels.append(label)
    return labels


def _it_paths_text(record: CanonicalJobPosting) -> list[str]:
    lines: list[str] = []
    for path in record.it_category_paths:
        names = [node.name for node in path.declared_path] or [path.observed.name]
        lines.append(" > ".join(names))
    return lines


def label_sections(record: CanonicalJobPosting) -> list[QuerySection]:
    return [
        QuerySection(
            name="title", header="Title: ", body=record.title, cut_priority=90
        ),
        QuerySection(
            name="category",
            header="IT categories: ",
            body="; ".join(_terminal_labels(record)),
            cut_priority=10,
        ),
    ]


def semantic_sections(record: CanonicalJobPosting) -> list[QuerySection]:
    return [
        QuerySection(
            name="title", header="Title: ", body=record.title, cut_priority=90
        ),
        QuerySection(
            name="category",
            header="IT category context: ",
            body="; ".join(_it_paths_text(record)),
            cut_priority=50,
        ),
        QuerySection(
            name="responsibilities",
            header="Job responsibilities: ",
            body=record.description or "",
            cut_priority=10,
        ),
        QuerySection(
            name="requirements",
            header="Candidate requirements: ",
            body=record.require_candidate or "",
            cut_priority=20,
        ),
    ]


def lexical_text(record: CanonicalJobPosting) -> str:
    parts = [record.title, *_terminal_labels(record), *record.knowledge]
    return "\n".join(part for part in parts if part)


def compose(sections: list[QuerySection]) -> str:
    return "\n".join(
        f"{section.header}{section.body}" for section in sections if section.body
    )


class TokenCounter:
    """Counts tokens via a TEI backend (server tokenizer is the source of truth)."""

    def __init__(self, backend) -> None:
        self._backend = backend

    def count(self, texts: list[str]) -> list[int]:
        return [len(ids) for ids in self._backend.token_id_lists(texts)]


def fit_to_budget(
    counter: TokenCounter,
    instruction: str,
    sections: list[QuerySection],
    budget: int,
    margin: int = 8,
) -> FittedQuery:
    """Deterministically shorten sections (lowest cut_priority first) to fit budget.

    The instruction prefix is never shortened. Bodies are shortened by character
    slices with a safety margin, then re-measured (server tokenizer is truth).
    """

    def full_text(current: list[QuerySection]) -> str:
        body = compose(current)
        return f"{instruction}\nQuery: {body}" if body else f"{instruction}\nQuery: "

    original = counter.count([full_text(sections)])[0]
    if original <= budget:
        return FittedQuery(
            text=full_text(sections),
            original_tokens=original,
            actual_tokens=original,
            truncated=False,
        )

    working = [
        QuerySection(
            name=section.name,
            header=section.header,
            body=section.body,
            cut_priority=section.cut_priority,
        )
        for section in sections
    ]
    shortened: list[str] = []
    for section in sorted(working, key=lambda item: item.cut_priority):
        if not section.body:
            continue
        for _ in range(3):
            total = counter.count([full_text(working)])[0]
            if total <= budget:
                break
            section_tokens = counter.count([section.header + section.body])[0]
            fixed_tokens = total - section_tokens
            full_chars = len(full_text(working)) or 1
            chars_per_token = full_chars / total
            allowance = max(0, budget - margin - fixed_tokens)
            new_chars = max(0, int(allowance * chars_per_token))
            if new_chars >= len(section.body):
                break
            section.body = section.body[:new_chars].rstrip()
            if section.name not in shortened:
                shortened.append(section.name)
        if counter.count([full_text(working)])[0] <= budget:
            break

    text = full_text(working)
    actual = counter.count([text])[0]
    if actual > budget:
        # Last resort: proportional hard cut of the whole text (deterministic).
        keep = max(1, int(len(text) * budget / actual) - margin)
        text = text[:keep].rstrip()
        actual = counter.count([text])[0]
    return FittedQuery(
        text=text,
        original_tokens=original,
        actual_tokens=actual,
        truncated=True,
        shortened_sections=sorted(shortened),
    )
