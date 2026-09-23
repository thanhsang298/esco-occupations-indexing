from esco_occupation_indexer.matching.models import CanonicalJobPosting
from esco_occupation_indexer.matching.queries import (
    TokenCounter,
    fit_to_budget,
    label_sections,
    lexical_text,
    semantic_sections,
)
from tests.matching.helpers import FakeQueryDenseBackend


def _job(**overrides) -> CanonicalJobPosting:
    base = {
        "job_id": "topcv:1",
        "external_id": "1",
        "platform": "topcv",
        "title": "Backend Developer",
        "description": "Develop REST APIs.",
        "require_candidate": "Java and SQL.",
        "knowledge": ["IT - Software"],
        "content_hash": "abc",
    }
    base.update(overrides)
    return CanonicalJobPosting(**base)


def test_query_snapshots() -> None:
    from esco_occupation_indexer.matching.queries import compose

    job = _job()
    assert compose(label_sections(job)) == "Title: Backend Developer"
    assert compose(semantic_sections(job)) == (
        "Title: Backend Developer\n"
        "Job responsibilities: Develop REST APIs.\n"
        "Candidate requirements: Java and SQL."
    )
    assert lexical_text(job) == "Backend Developer\nIT - Software"


def test_instruction_format() -> None:
    instruction = "Given a job posting, retrieve the thing."
    counter = TokenCounter(FakeQueryDenseBackend())
    fitted = fit_to_budget(counter, instruction, label_sections(_job()), 128)
    assert fitted.text.startswith(f"{instruction}\nQuery: Title: ")
    assert not fitted.truncated
    assert fitted.original_tokens == fitted.actual_tokens


def test_truncation_is_deterministic_and_reports() -> None:
    counter = TokenCounter(FakeQueryDenseBackend())
    job = _job(description="word " * 200, require_candidate="req " * 200)
    first = fit_to_budget(counter, "Instruct: x", semantic_sections(job), 32)
    second = fit_to_budget(counter, "Instruct: x", semantic_sections(job), 32)
    assert first.text == second.text
    assert first.truncated
    assert first.actual_tokens <= 32
    assert first.original_tokens > 32
    assert "responsibilities" in first.shortened_sections


def test_label_budget_128_with_vietnamese_title() -> None:
    counter = TokenCounter(FakeQueryDenseBackend())
    job = _job(title="Nhân Viên QA/QC Chuỗi Nhà Hàng Thu Nhập 12-16tr Tại Hà Nội")
    fitted = fit_to_budget(counter, "Instruct: x", label_sections(job), 128)
    assert not fitted.truncated
    assert "Nhân Viên QA/QC" in fitted.text
