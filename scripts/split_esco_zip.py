"""Split the official ESCO v1.2.1 EN CSV ZIP into pillar-specific ZIPs.

The skills pipeline only reads its 10 required members (matched by basename)
and the occupations pipeline only reads its own members, so both pipelines
could consume the full official ZIP.  We still split it into two derived,
gitignored ZIPs so that:

- each ``build-id`` is content-addressed from its own pillar checksum only
  (a skill data change does not invalidate occupation builds and vice versa);
- ``source-report.json`` stays small and pillar-scoped.

Usage:
    python scripts/split_esco_zip.py \
        --source ../esco-skills-indexing/data/escsv1.2.1-en-csv.zip \
        --out-dir data/raw
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

SKILLS_MEMBERS = [
    "skills_en.csv",
    "skillGroups_en.csv",
    "skillsHierarchy_en.csv",
    "broaderRelationsSkillPillar_en.csv",
    "digCompSkillsCollection_en.csv",
    "digitalSkillsCollection_en.csv",
    "greenSkillsCollection_en.csv",
    "languageSkillsCollection_en.csv",
    "researchSkillsCollection_en.csv",
    "transversalSkillsCollection_en.csv",
]

OCCUPATIONS_MEMBERS = [
    "occupations_en.csv",
    "ISCOGroups_en.csv",
    "broaderRelationsOccPillar_en.csv",
    "occupationSkillRelations_en.csv",
    "researchOccupationsCollection_en.csv",
    "greenShareOcc_en.csv",
]


def _find_member(archive: zipfile.ZipFile, basename: str) -> str:
    matches = [
        name
        for name in archive.namelist()
        if not name.endswith("/")
        and Path(name.replace("\\", "/")).name.casefold() == basename.casefold()
    ]
    if len(matches) != 1:
        raise SystemExit(f"Expected exactly one {basename!r}, found {len(matches)}")
    return matches[0]


def _write_derived(
    source: zipfile.ZipFile, out_path: Path, basenames: list[str]
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as dest:
        for basename in basenames:
            member = _find_member(source, basename)
            dest.writestr(basename, source.read(member))
    print(f"wrote {out_path} ({len(basenames)} files)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    with zipfile.ZipFile(args.source) as source:
        _write_derived(source, args.out_dir / "esco-skills-v1.2.1-en.zip", SKILLS_MEMBERS)
        _write_derived(
            source, args.out_dir / "esco-occupations-v1.2.1-en.zip", OCCUPATIONS_MEMBERS
        )


if __name__ == "__main__":
    main()
