# ruff: noqa: E501 -- long adjudication note literals
"""Apply the reviewed semantic overrides to gold_v2 and its four part files.

The mapping uses ESCO preferred labels for readability, then resolves them to
UUIDs from the canonical local ESCO snapshot. Run the audit script afterwards.
"""

from __future__ import annotations

import json
from pathlib import Path

from _jsonl import iter_jsonl_zst

ROOT = Path(__file__).resolve().parents[1]
GOLD_DIR = ROOT / "data/gold"
ESCO = ROOT / "data/work/9ac16776cc2c/canonical-occupations.jsonl.zst"

# Only rows whose accepted set changes are listed. These decisions use the
# complete posting and ESCO descriptions, not candidate rank.
OVERRIDES: dict[str, tuple[list[str], str]] = {
    "topcv:1163444": (
        ["textile quality inspector", "clothing quality inspector"],
        "Prefer garment/textile-specific inspection occupations; generic product inspection/controller labels are redundant.",
    ),
    "topcv:1527429": (
        ["cyber incident responder", "digital forensics expert"],
        "Malware collection, reverse analysis and forensic incident work are core; ethical hacking is a different occupation.",
    ),
    "topcv:1577614": (
        ["ICT project manager", "ICT business analyst", "interpreter"],
        "BrSE combines software-requirement analysis, Japanese-Vietnamese communication and possible project coordination.",
    ),
    "topcv:1814704": (
        ["ICT application developer", "software developer", "ICT business analyst", "interpreter"],
        "SAP programming training and prior development experience make software development acceptable, while requirement bridging and Japanese-Vietnamese communication remain the core role.",
    ),
    "topcv:1948271": (
        [
            "ICT account manager",
            "technical sales representative in electronic and telecommunications equipment",
            "technical sales representative",
        ],
        "B2B sales of mini-PCs, displays and components is ICT/electronics technical sales; generic commercial sales is too broad.",
    ),
    "topcv:1980989": (
        ["quality engineer", "textile quality inspector", "clothing quality inspector"],
        "Buyer QA performs garment risk/quality review; remove generic product inspector/controller where textile/clothing-specific concepts exist.",
    ),
    "topcv:2005380": (
        ["quality engineer", "electronics production supervisor"],
        "The job plans and supervises IQC/PQC/OQC and corrective action for PCBA; operator and test-technician labels describe subordinates, not this role.",
    ),
    "topcv:2017288": (
        ["digital artist"],
        "The posting creates 2D game assets and UI art; it does not state animation duties.",
    ),
    "topcv:2066960": (
        ["ICT presales engineer"],
        "BOM preparation, solution advice, quotation and estimation are ICT presales; generic consulting/project labels are only adjacent.",
    ),
    "topcv:2110076": (
        ["ICT project manager"],
        "Despite the Product Manager title, the duties are software-project scope, plan, delivery and coordination rather than product lifecycle ownership.",
    ),
    "topcv:2128803": (
        ["chemical manufacturing quality technician", "pharmaceutical quality specialist"],
        "Laboratory QC, SOPs, chemical/microbiological testing and validation fit pharmaceutical/chemical quality; generic product inspection is too broad.",
    ),
    "topcv:2128820": (
        ["special effects artist", "animator"],
        "The role creates game VFX and animation; no specifically 3D-animation evidence is given.",
    ),
    "topcv:2185017": (
        ["database administrator"],
        "All stated duties concern database operation, backup, recovery, performance, DR and patching; system administration is overly broad.",
    ),
    "topcv:2195592": (
        ["digital games developer", "software developer"],
        "Playable ads are short interactive games built with Unity/Cocos; mobile application development is not stated.",
    ),
    "topcv:2193692": (
        ["ICT system administrator", "ICT network technician", "ICT network administrator"],
        "The role explicitly administers LAN/WAN, routing, VPN, firewall and IDS as well as servers; network administrator is a direct match.",
    ),
    "topcv:2208837": (
        ["ICT project manager", "project manager", "ICT business analyst", "interpreter"],
        "Lead BrSE combines project leadership, requirements analysis and Japanese-Vietnamese communication.",
    ),
    "topcv:2225759": (
        ["cybersecurity risk manager"],
        "The role develops information-security policy, risk and compliance controls; resilience and director-level compliance are not core occupations here.",
    ),
    "topcv:2235894": (
        ["product quality inspector", "product quality controller"],
        "IQC/PQC performs material and production inspection/control; quality engineering responsibilities are not established.",
    ),
    "topcv:2236849": (
        ["integration engineer", "ICT system administrator", "cloud engineer"],
        "Microsoft 365/Azure design, migration, deployment, operation and troubleshooting directly match cloud engineering in addition to integration/system work.",
    ),
    "topcv:2244270": (
        ["ICT operations manager"],
        "The job leads infrastructure, networks, cloud, security, support and local IT operations; CIO implies a broader enterprise strategy remit.",
    ),
    "topcv:2244376": (
        ["category manager"],
        "Product discovery, niche research, catalogue data and uploads fit category/assortment work; product-and-services management is too strategic.",
    ),
    "topcv:2248235": (
        ["digital transformation manager"],
        "The role leads AI-enabled organisational transformation; CIO is an adjacent executive occupation rather than the advertised role.",
    ),
    "topcv:2249385": (
        ["security alarm technician"],
        "The job builds, integrates, maintains and assures CCTV systems; hospitality security duties are absent.",
    ),
    "topcv:2252514": (
        ["user interface designer", "user experience analyst"],
        "The cross-platform role covers UI and UX; web designer is an unnecessary channel-specific label.",
    ),
    "topcv:2253522": (
        ["ICT account manager", "ICT presales engineer", "technical sales representative"],
        "The work is customer qualification, solution advice, quotation and closing ICT services; it is commercial presales, not independent ICT consulting.",
    ),
    "topcv:2254418": (
        ["chemistry technician", "scientific laboratory technician"],
        "Sample preparation and LCMS operation/data processing are laboratory technician duties, not quality engineering.",
    ),
    "topcv:2261078": (
        ["data quality specialist", "chief data officer"],
        "Company-wide data roadmap, governance, ownership and strategic use fit data governance/CDO work; DBA and pipeline engineering are not the core.",
    ),
    "topcv:2261695": (
        ["ICT quality assurance manager"],
        "The department head builds QC strategy, organisation, automation and quality governance; individual tester is not the target occupation.",
    ),
    "topcv:2261857": (
        ["graphic designer", "prepress technician", "prepress operator"],
        "The role both designs artwork and performs proof/file/film preparation and print-quality checks; prepress operator is a direct operational match.",
    ),
    "topcv:2265306": (
        ["industrial quality manager", "pharmaceutical quality specialist"],
        "The head of pharmaceutical QC owns department plans, staff and regulated quality; technician/engineer labels are subordinate roles.",
    ),
    "topcv:2272037": (
        ["quality engineer", "product quality inspector", "metal product quality control inspector"],
        "The posting performs incoming, in-process and final inspection of fabricated metal products; the metal-specific quality inspector is an exact match.",
    ),
    "topcv:2272254": (
        ["ICT business analyst", "software analyst", "interpreter"],
        "BrSE gathers, documents and reviews software requirements as the interface between Japanese clients and developers; both ICT business analyst and software analyst fit, alongside interpreting.",
    ),
    "topcv:2278632": (
        ["industrial quality manager", "pharmaceutical quality specialist"],
        "The laboratory QA manager owns QMS/SOP and regulated pharmaceutical quality; generic quality engineer is not the managerial occupation.",
    ),
    "topcv:2283711": (
        ["data analyst"],
        "Dashboards, business metrics, analysis and recommendations are data analysis; predictive-model research characteristic of data science is absent.",
    ),
    "topcv:2286146": (
        ["industrial quality manager"],
        "The department head establishes standards, procedures and organisation-wide manufacturing quality controls; quality engineer is a subordinate/general label.",
    ),
    "topcv:2290551": (
        ["artificial intelligence engineer"],
        "The role develops LLM agents and AI services; ESCO language engineer specifically concerns computational linguistics/translation and is not supported.",
    ),
}


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def update(rows: list[dict], label_to_id: dict[str, str]) -> list[dict]:
    for row in rows:
        decision = OVERRIDES.get(row["job_id"])
        if decision is None:
            continue
        labels, reason = decision
        row["accepted_esco_ids"] = [label_to_id[label] for label in labels]
        row["annotator"] = "muse-spark+codex-adjudication"
        row["notes"] = f"accepted-set adjudication: {reason}"
        row["labeled_at"] = "2026-09-23T00:00:00+07:00"
    return rows


def write(path: Path, rows: list[dict]) -> None:
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    path.write_text(text, encoding="utf-8")


def main() -> None:
    label_to_ids: dict[str, list[str]] = {}
    for row in iter_jsonl_zst(ESCO):
        label_to_ids.setdefault(row["preferred_label"], []).append(row["esco_id"])
    required = {label for labels, _ in OVERRIDES.values() for label in labels}
    ambiguous = {
        label: label_to_ids.get(label, [])
        for label in required
        if len(label_to_ids.get(label, [])) != 1
    }
    if ambiguous:
        raise ValueError(f"Missing or ambiguous ESCO preferred labels: {ambiguous}")
    label_to_id = {label: label_to_ids[label][0] for label in required}

    aggregate_path = GOLD_DIR / "gold_v2.jsonl"
    aggregate = update(load(aggregate_path), label_to_id)
    write(aggregate_path, aggregate)
    for part_path in sorted(GOLD_DIR.glob("gold_v2_part_*.jsonl")):
        write(part_path, update(load(part_path), label_to_id))


if __name__ == "__main__":
    main()
