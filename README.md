# ESCO Occupations Indexing

Reproducible batch pipeline that indexes released ESCO v1.2.1 English occupation
concepts into Qdrant. Forked from `esco-skills-indexing` for speed: `artifacts`,
`embedding`, `tei`, `normalize`, `utils`, `settings`, `cli`, and `pipeline` are
shared logic, while `models`, `source`, `hierarchy`, `ingest`, `documents`,
`validation`, and `qdrant_ops` are occupation-specific.

## Index schema

Each released ESCO occupation becomes one Qdrant point with:

- `label_dense`: preferred and alternative labels as a `MAX_SIM` multivector;
- `semantic_dense`: one controlled semantic document
  (`Occupation + Description + Definition + Scope note + full ISCO Context`);
- `lexical_sparse`: all preferred, alternative, and hidden labels using BM25 + IDF;
- payload metadata for exact normalized aliases, full ISCO hierarchy
  (`isco_major`, `terminal_isco_uri`, `path_text`), essential/optional skill URIs
  (format-validated only, resolved against the skill index at retrieval time),
  `green_share`, collections, and provenance.

Essential/optional skills are deliberately **not** embedded into the occupation
document in V1 to avoid skill content drowning occupation semantics. The existing
skill index plus the stored skill URIs provide the skill-support signal at
rerank time.

## Setup

Requirements: Python 3.12, `uv`, a reachable Qdrant + TEI, and the derived
occupation ZIP (see below).

```bash
uv python install 3.12
uv sync
cp ../../.env .env   # already done: QDRANT_URL=http://10.0.0.71:6333, TEI_URL=http://localhost:8787
```

`.env` (gitignored) is the source of truth for endpoints; `QDRANT_URL` and
`TEI_URL` override `config/occupations-indexing.yaml` at load time. The default
embedding is remote Harrier (`microsoft/harrier-oss-v1-0.6b`, dim 1024) via TEI.

## Source data

```bash
python scripts/split_esco_zip.py \
  --source ../esco-skills-indexing/data/escsv1.2.1-en-csv.zip \
  --out-dir data/raw
# -> data/raw/esco-occupations-v1.2.1-en.zip (6 files, gitignored)
```

See `docs/occupation-source-contract.md` for the file/column contract
(3043 rows / 3039 distinct occupations in v1.2.1, full ISCO path semantics).

## Run

```bash
uv run esco-occupation-index build \
  --config config/occupations-indexing.yaml \
  --source data/raw/esco-occupations-v1.2.1-en.zip
```

Add `--promote` to switch `esco_occupation_concepts_current` only after upload
verification. Stages can also be run independently (`ingest`, `validate`,
`embed`, `upload`, `verify`, `promote`).

Build layout under `data/work/<build-id>/`:

```text
manifest.json
resolved-config.json
source-report.json
canonical-occupations.jsonl.zst
isco-groups.jsonl.zst
validation-report.json
embedding-report.json
vectors/shard-*.npz
upload-report.json
verification-report.json
```

## Quality checks

```bash
uv run ruff check .
uv run pytest
RUN_QDRANT_INTEGRATION=1 uv run pytest -m integration
```

## Job matching (TopCV → ESCO)

Batch pipeline in `esco_occupation_indexer.matching`
(`ingest → validate → embed_queries → retrieve → verify`, no rerank):

```bash
uv run esco-occupation-index match-jobs \
  --config config/job-matching.yaml \
  --jobs data/iviec-job-crawler.job_details.topcv.it.json \
  --categories data/iviec-job-crawler.job_categories.topcv.level_id.json \
  --esco-build-dir data/work/<esco-build-id>
```

Add `--allow-collection` only when the shared alias points elsewhere: matching
then targets the verified snapshot collection directly and records
`alias_bypassed` in the manifest. Artifacts live in
`data/job-matching-work/<match-build-id>/` (`canonical-jobs.jsonl.zst`,
`query-vectors/`, `retrieval/`, `matches.jsonl.zst` sorted by `job_id`).
