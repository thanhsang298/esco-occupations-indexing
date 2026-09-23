from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def make_taxonomy_file(path: Path) -> Path:
    nodes = [
        {"externalId": "257", "name": "IT", "parentExternalId": None},
        {"externalId": "265", "name": "Dev", "parentExternalId": "257"},
        {"externalId": "1021", "name": "Backend", "parentExternalId": "265"},
        {"externalId": "1", "name": "Sales", "parentExternalId": None},
        {"externalId": "92", "name": "Retail", "parentExternalId": "1"},
    ]
    return write_json(path, nodes)


def make_jobs_file(path: Path) -> Path:
    jobs = [
        {
            "externalId": "1001",
            "status": "completed",
            "platformId": "topcv",
            "sourceUrl": "https://topcv.vn/j1",
            "updatedAt": {"$date": "2026-08-13T18:01:00.460Z"},
            "_id": {"$oid": "6a7d48ffbd7445e8b753f3b2"},
            "title": "Backend Developer",
            "description": "Develop REST APIs and backend services.",
            "requireCandidate": "Java, SQL, 2 years experience.",
            "experience": "4",
            "requireSummary": ["2 years experience"],
            "knowledge": [{"id": "3", "title": "IT - Software"}],
            "commonInfo": {"level": "1"},
            "categories": [
                {
                    "key": "1021",
                    "name": "Backend",
                    "url": "https://topcv.vn/x",
                    "group": "r257~b265l1021",
                    "level1Id": "257",
                    "level2Id": "265",
                    "level3Id": "1021",
                }
            ],
        },
        {
            # Declared IT but canonical non-IT (key 92 lives under root 1).
            "externalId": "1002",
            "status": "completed",
            "platformId": "topcv",
            "sourceUrl": "https://topcv.vn/j2",
            "updatedAt": {"$date": "2026-08-13T18:01:00.460Z"},
            "_id": {"$oid": "6a7d48ffbd7445e8b753f3c3"},
            "title": "IT Sales",
            "description": "Sell software solutions.",
            "requireCandidate": "Communication skills.",
            "experience": "3",
            "requireSummary": [],
            "knowledge": [],
            "commonInfo": {},
            "categories": [
                {
                    "key": "92",
                    "name": "Retail",
                    "url": "",
                    "group": "",
                    "level1Id": "257",
                    "level2Id": "265",
                    "level3Id": "1021",
                }
            ],
        },
        {
            "externalId": "1003",
            "status": "retry",
            "platformId": "topcv",
            "title": "Skipped",
            "categories": [],
        },
    ]
    return write_json(path, jobs)


class FakeQueryDenseBackend:
    backend_name = "fake-query"
    device = "cpu"

    def __init__(self, dimension: int = 8) -> None:
        self.dimension = dimension
        self.encode_calls = 0
        self.last_effective_batch_size = 64
        self.max_batch_size = 64
        self.max_tokens = 10**6

    def encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        self.encode_calls += 1
        vectors = []
        for text in texts:
            raw = hashlib.sha256(text.casefold().encode()).digest()
            vector = np.asarray(
                [raw[index] + 1 for index in range(self.dimension)], dtype=np.float32
            )
            vector /= np.linalg.norm(vector)
            vectors.append(vector)
        return np.stack(vectors)

    def count_truncated(self, texts: list[str], max_tokens: int) -> int:
        return sum(len(text.split()) > max_tokens for text in texts)

    def token_id_lists(self, texts: list[str]) -> list[list[int]]:
        # Word-based fake tokenization: deterministic and budget-sensitive.
        return [
            [(index % 1000) + 1 for index, _ in enumerate(text.split())]
            for text in texts
        ]

    def close(self) -> None:
        pass
