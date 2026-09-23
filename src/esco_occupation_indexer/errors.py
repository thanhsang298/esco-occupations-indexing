class IndexingError(RuntimeError):
    """Base error for a failed indexing stage."""


class SourceDataError(IndexingError):
    """The ESCO source package is incomplete or malformed."""


class ValidationError(IndexingError):
    """A strict build invariant failed."""


class StageError(IndexingError):
    """A pipeline stage cannot run or did not complete."""
