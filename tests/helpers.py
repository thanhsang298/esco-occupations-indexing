from __future__ import annotations

import csv
import hashlib
import io
import re
import zipfile
from pathlib import Path

import numpy as np

from esco_occupation_indexer.embedding import SparseEncoding
from esco_occupation_indexer.settings import IndexingSettings

ROOT_URI = "http://data.europa.eu/esco/isco/C2"
UNIT_URI = "http://data.europa.eu/esco/isco/C251"
PARENT_OCC_URI = "http://data.europa.eu/esco/occupation/00000000-0000-4000-8000-000000000011"
CHILD_OCC_URI = "http://data.europa.eu/esco/occupation/00000000-0000-4000-8000-000000000012"
OBSOLETE_OCC_URI = "http://data.europa.eu/esco/occupation/00000000-0000-4000-8000-000000000013"

SKILL_1 = "http://data.europa.eu/esco/skill/11111111-1111-4111-8111-111111111111"
SKILL_2 = "http://data.europa.eu/esco/skill/22222222-2222-4222-8222-222222222222"
SKILL_3 = "http://data.europa.eu/esco/skill/33333333-3333-4333-8333-333333333333"


def _csv_bytes(headers: list[str], rows: list[dict[str, str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=headers, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def make_occupation_zip(path: Path) -> Path:
    occupation_headers = [
        "conceptType",
        "conceptUri",
        "iscoGroup",
        "preferredLabel",
        "altLabels",
        "hiddenLabels",
        "status",
        "modifiedDate",
        "regulatedProfessionNote",
        "scopeNote",
        "definition",
        "inScheme",
        "description",
        "code",
        "naceCode",
    ]
    occupations = [
        {
            "conceptType": "Occupation",
            "conceptUri": PARENT_OCC_URI,
            "iscoGroup": "251",
            "preferredLabel": "software developer",
            "altLabels": "software engineer",
            "hiddenLabels": "",
            "status": "released",
            "modifiedDate": "2025-12-10T00:00:00Z",
            "regulatedProfessionNote": "",
            "scopeNote": "",
            "definition": "",
            "inScheme": "occupations",
            "description": "Develops software applications.",
            "code": "2512.1",
            "naceCode": "",
        },
        {
            "conceptType": "Occupation",
            "conceptUri": CHILD_OCC_URI,
            "iscoGroup": "251",
            "preferredLabel": "backend developer",
            "altLabels": "backend engineer\nBackend Developer",
            "hiddenLabels": "be dev",
            "status": "released",
            "modifiedDate": "2025-12-10T00:00:00Z",
            "regulatedProfessionNote": "",
            "scopeNote": "Works on server-side logic.",
            "definition": "",
            "inScheme": "occupations",
            "description": "Develops REST APIs and backend services.",
            "code": "2512.1.1",
            "naceCode": "",
        },
        {
            "conceptType": "Occupation",
            "conceptUri": OBSOLETE_OCC_URI,
            "iscoGroup": "251",
            "preferredLabel": "obsolete operator",
            "altLabels": "",
            "hiddenLabels": "",
            "status": "obsolete",
            "modifiedDate": "2025-12-10T00:00:00Z",
            "regulatedProfessionNote": "",
            "scopeNote": "",
            "definition": "",
            "inScheme": "occupations",
            "description": "Old entry.",
            "code": "2512.9",
            "naceCode": "",
        },
    ]
    isco_groups = [
        {
            "conceptUri": ROOT_URI,
            "code": "2",
            "preferredLabel": "Professionals",
            "status": "released",
            "altLabels": "",
            "description": "Professionals.",
        },
        {
            "conceptUri": UNIT_URI,
            "code": "251",
            "preferredLabel": "Software and applications developers and analysts",
            "status": "released",
            "altLabels": "",
            "description": "Software developers.",
        },
    ]
    broader = [
        {"conceptUri": UNIT_URI, "broaderUri": ROOT_URI},
        {"conceptUri": PARENT_OCC_URI, "broaderUri": UNIT_URI},
        {"conceptUri": CHILD_OCC_URI, "broaderUri": PARENT_OCC_URI},
        {"conceptUri": OBSOLETE_OCC_URI, "broaderUri": UNIT_URI},
    ]
    relations = [
        {
            "occupationUri": CHILD_OCC_URI,
            "occupationLabel": "backend developer",
            "relationType": "essential",
            "skillType": "skill/competence",
            "skillUri": SKILL_1,
            "skillLabel": "develop software",
        },
        {
            "occupationUri": CHILD_OCC_URI,
            "occupationLabel": "backend developer",
            "relationType": "essential",
            "skillType": "knowledge",
            "skillUri": SKILL_2,
            "skillLabel": "databases",
        },
        {
            "occupationUri": CHILD_OCC_URI,
            "occupationLabel": "backend developer",
            "relationType": "optional",
            "skillType": "",
            "skillUri": SKILL_3,
            "skillLabel": "legacy skill",
        },
    ]

    files = {
        "occupations_en.csv": _csv_bytes(occupation_headers, occupations),
        "ISCOGroups_en.csv": _csv_bytes(
            ["conceptUri", "code", "preferredLabel", "status", "altLabels", "description"],
            isco_groups,
        ),
        "broaderRelationsOccPillar_en.csv": _csv_bytes(
            ["conceptUri", "broaderUri"], broader
        ),
        "occupationSkillRelations_en.csv": _csv_bytes(
            [
                "occupationUri",
                "occupationLabel",
                "relationType",
                "skillType",
                "skillUri",
                "skillLabel",
            ],
            relations,
        ),
        "researchOccupationsCollection_en.csv": _csv_bytes(
            ["conceptUri"], [{"conceptUri": CHILD_OCC_URI}]
        ),
        "greenShareOcc_en.csv": _csv_bytes(
            ["conceptUri", "greenShare"],
            [{"conceptUri": CHILD_OCC_URI, "greenShare": "0.5"}],
        ),
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename, data in files.items():
            archive.writestr(f"ESCO dataset/{filename}", data)
    return path


def make_settings(work_dir: Path, dimension: int = 8) -> IndexingSettings:
    return IndexingSettings.model_validate(
        {
            "work_dir": str(work_dir),
            "embedding": {
                "model_id": "test/deterministic",
                "model_revision": "test-revision",
                "dimension": dimension,
                "max_tokens": 16,
                "batch_size": 4,
                "shard_size": 1,
                "device": "cpu",
            },
        }
    )


class FakeDenseBackend:
    device = "cpu"

    def __init__(self, dimension: int = 8) -> None:
        self.dimension = dimension
        self.encode_calls = 0
        self.encoded_batches: list[tuple[str, ...]] = []

    def encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        self.encode_calls += 1
        self.encoded_batches.append(tuple(texts))
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


class FakeSparseBackend:
    def encode(self, texts: list[str]) -> list[SparseEncoding]:
        results = []
        for text in texts:
            tokens = sorted(set(re.findall(r"[a-z0-9]+", text.casefold())))
            token_indices = {
                int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big")
                for token in tokens
            }
            # Qdrant requires sparse indices to be ordered. Sorting here also makes
            # the synthetic backend a faithful stand-in for FastEmbed output.
            indices = np.asarray(sorted(token_indices), dtype=np.uint32)
            values = np.ones(indices.size, dtype=np.float32)
            results.append(SparseEncoding(indices=indices, values=values))
        return results
