"""Shared JSONL.ZST reader for maintenance scripts."""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import zstandard as zstd


def iter_jsonl_zst(path: Path) -> Iterator[dict[str, Any]]:
    with ExitStack() as stack:
        raw = stack.enter_context(path.open("rb"))
        reader = stack.enter_context(
            zstd.ZstdDecompressor().stream_reader(raw)
        )
        for line in io.TextIOWrapper(reader, encoding="utf-8"):
            if line.strip():
                yield json.loads(line)
