# ruff: noqa: E501 -- long adjudication note literals
"""Add independently adjudicated primary_esco_ids to Gold v2.

Every multi-label row must be explicitly covered by either PRIMARY_OVERRIDES
or REVIEWED_KEEP_ALL. Single-label rows inherit their accepted label. An empty
override is valid when ESCO has no sufficiently relevant occupation.
"""

from __future__ import annotations

import json
from pathlib import Path

from _jsonl import iter_jsonl_zst

ROOT = Path(__file__).resolve().parents[1]
GOLD_DIR = ROOT / "data/gold"
ESCO = ROOT / "data/work/9ac16776cc2c/canonical-occupations.jsonl.zst"

# Rows where only a strict subset of acceptable occupations is primary.
PRIMARY_OVERRIDES: dict[str, list[str]] = {
    "topcv:1577614": ["ICT business analyst", "interpreter"],
    "topcv:1814704": ["ICT application developer", "ICT business analyst", "interpreter"],
    "topcv:1637170": ["ICT network engineer", "ICT system administrator"],
    "topcv:1948271": [
        "ICT account manager",
        "technical sales representative in electronic and telecommunications equipment",
    ],
    "topcv:1953614": ["ICT business analyst"],
    "topcv:1966660": ["embedded systems software developer", "embedded system designer"],
    "topcv:1989720": [
        "technical sales representative in electronic and telecommunications equipment"
    ],
    "topcv:2056186": ["ICT product manager"],
    "topcv:2082300": ["ICT test analyst"],
    "topcv:2082317": ["ICT system tester"],
    "topcv:2083817": ["ICT system tester"],
    "topcv:2144633": ["ICT product manager"],
    "topcv:2166550": ["ICT product manager"],
    "topcv:2168235": ["ICT system tester"],
    "topcv:2169853": ["web developer"],
    "topcv:2195592": ["digital games developer"],
    "topcv:2193692": ["ICT system administrator", "ICT network administrator"],
    "topcv:2208837": ["ICT project manager", "ICT business analyst", "interpreter"],
    "topcv:2227581": ["artificial intelligence engineer"],
    "topcv:2233218": ["ICT project manager"],
    "topcv:2238273": ["ICT test analyst"],
    "topcv:2240237": ["ICT project manager"],
    "topcv:2241572": ["artificial intelligence engineer", "computer vision engineer"],
    "topcv:2242784": ["ICT business analyst"],
    "topcv:2243796": ["ICT quality assurance manager", "ICT system tester"],
    "topcv:2248378": ["mobile application developer"],
    "topcv:2248674": ["cloud software developer"],
    "topcv:2249945": ["chief technology officer"],
    "topcv:2250430": ["ICT system tester"],
    "topcv:2256414": ["mobile application developer"],
    "topcv:2259362": ["ICT business analyst"],
    "topcv:2261733": ["digital games developer"],
    "topcv:2267680": ["ICT business analyst"],
    "topcv:2270289": ["digital artist", "user interface designer"],
    "topcv:2270608": ["ICT business analyst", "ICT system tester"],
    "topcv:2270789": ["ICT project manager"],
    "topcv:2271464": ["ICT system tester"],
    "topcv:2271554": ["cloud architect", "cloud engineer"],
    "topcv:2272037": ["quality engineer", "metal product quality control inspector"],
    "topcv:2272097": ["mobile application developer"],
    "topcv:2274037": ["ICT system tester"],
    "topcv:2274572": ["quality engineer", "food analyst"],
    "topcv:2274961": ["ICT business analyst"],
    "topcv:2275228": ["ICT system architect"],
    "topcv:2277072": ["digital media designer", "user interface designer"],
    "topcv:2277125": ["quality services manager"],
    "topcv:2281383": ["ICT business analyst"],
    "topcv:2281817": ["mobile application developer"],
    "topcv:2283451": ["digital transformation manager", "ICT project manager"],
    "topcv:2285116": ["ICT network engineer"],
    "topcv:2285281": ["ICT product manager"],
    "topcv:2286438": ["technical sales representative", "ICT business development manager"],
    "topcv:2287048": ["mobile application developer"],
    "topcv:2287208": ["ICT system tester"],
    "topcv:2287627": ["product quality inspector"],
    "topcv:2289773": ["digital games tester"],
    "topcv:2290696": ["ICT system tester"],
    "topcv:2290732": ["mobile application developer"],
}

# Explicitly reviewed hybrids/equivalent concepts for which every acceptable
# label is also primary. Listing them prevents accidental defaulting.
REVIEWED_KEEP_ALL = {
    "topcv:1163444",
    "topcv:1527429",
    "topcv:1644199",
    "topcv:1655990",
    "topcv:1739673",
    "topcv:1763894",
    "topcv:1794195",
    "topcv:1967495",
    "topcv:1972444",
    "topcv:1980989",
    "topcv:1997026",
    "topcv:2005380",
    "topcv:2128803",
    "topcv:2128820",
    "topcv:2133837",
    "topcv:2136005",
    "topcv:2165554",
    "topcv:2167040",
    "topcv:2173126",
    "topcv:2183117",
    "topcv:2189952",
    "topcv:2219085",
    "topcv:2225033",
    "topcv:2228899",
    "topcv:2235894",
    "topcv:2236849",
    "topcv:2237697",
    "topcv:2242440",
    "topcv:2242920",
    "topcv:2243277",
    "topcv:2246269",
    "topcv:2250650",
    "topcv:2250828",
    "topcv:2250996",
    "topcv:2252514",
    "topcv:2252851",
    "topcv:2252866",
    "topcv:2253449",
    "topcv:2253522",
    "topcv:2254418",
    "topcv:2254721",
    "topcv:2261078",
    "topcv:2261857",
    "topcv:2262390",
    "topcv:2263560",
    "topcv:2263682",
    "topcv:2264690",
    "topcv:2265306",
    "topcv:2266189",
    "topcv:2267163",
    "topcv:2267738",
    "topcv:2268942",
    "topcv:2269176",
    "topcv:2272254",
    "topcv:2275240",
    "topcv:2276387",
    "topcv:2276433",
    "topcv:2278632",
    "topcv:2280928",
    "topcv:2282098",
    "topcv:2282827",
    "topcv:2285229",
    "topcv:2286963",
    "topcv:2287923",
    "topcv:2287931",
    "topcv:2288871",
    "topcv:2290013",
    "topcv:2291251",
    "topcv:2292213",
    "topcv:635183",
    "topcv:930504",
}


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    label_to_ids: dict[str, list[str]] = {}
    for occupation in iter_jsonl_zst(ESCO):
        label_to_ids.setdefault(occupation["preferred_label"], []).append(occupation["esco_id"])
    required_labels = {label for labels in PRIMARY_OVERRIDES.values() for label in labels}
    ambiguous = {
        label: label_to_ids.get(label, [])
        for label in required_labels
        if len(label_to_ids.get(label, [])) != 1
    }
    if ambiguous:
        raise ValueError(f"Missing or ambiguous ESCO labels: {ambiguous}")
    label_to_id = {label: label_to_ids[label][0] for label in required_labels}

    aggregate_path = GOLD_DIR / "gold_v2.jsonl"
    aggregate = load(aggregate_path)
    multi_ids = {row["job_id"] for row in aggregate if len(row["accepted_esco_ids"]) > 1}
    covered = set(PRIMARY_OVERRIDES) | REVIEWED_KEEP_ALL
    if multi_ids != covered:
        raise ValueError(
            f"Multi-label adjudication coverage mismatch: missing={sorted(multi_ids - covered)}, "
            f"obsolete={sorted(covered - multi_ids)}"
        )

    def adjudicate(rows: list[dict]) -> list[dict]:
        for row in rows:
            job_id = row["job_id"]
            if job_id in PRIMARY_OVERRIDES:
                primary = [label_to_id[label] for label in PRIMARY_OVERRIDES[job_id]]
            else:
                primary = list(row["accepted_esco_ids"])
            if not set(primary).issubset(row["accepted_esco_ids"]):
                raise ValueError(f"Primary labels are not accepted for {job_id}")
            row["primary_esco_ids"] = primary
        return rows

    write(aggregate_path, adjudicate(aggregate))
    for part_path in sorted(GOLD_DIR.glob("gold_v2_part_*.jsonl")):
        write(part_path, adjudicate(load(part_path)))


if __name__ == "__main__":
    main()
