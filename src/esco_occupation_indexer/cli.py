from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer

from esco_occupation_indexer.embedding import embed_build
from esco_occupation_indexer.errors import IndexingError
from esco_occupation_indexer.ingest import ingest_source
from esco_occupation_indexer.pipeline import run_build
from esco_occupation_indexer.qdrant_ops import promote_build, upload_build, verify_build
from esco_occupation_indexer.settings import load_settings
from esco_occupation_indexer.validation import validate_build

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


def _run[ResultT](operation: Callable[[], ResultT]) -> ResultT:
    try:
        return operation()
    except (IndexingError, OSError, ValueError) as error:
        typer.secho(f"ERROR: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error


@app.command()
def ingest(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    source: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
) -> None:
    """Parse an official ESCO ZIP and write canonical artifacts."""
    build_dir = _run(lambda: ingest_source(load_settings(config), source))
    typer.echo(build_dir)


@app.command()
def validate(
    build_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)],
) -> None:
    """Run strict canonical and hierarchy validation."""
    report = _run(lambda: validate_build(build_dir))
    typer.echo(f"Validated {report['record_count']} occupation concepts")


@app.command()
def embed(
    build_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)],
) -> None:
    """Generate resumable dense and BM25 vector shards."""
    report = _run(lambda: embed_build(build_dir))
    typer.echo(f"Embedded {report['record_count']} occupation concepts")


@app.command()
def upload(
    build_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)],
) -> None:
    """Create a staging collection and upsert all points."""
    report = _run(lambda: upload_build(build_dir))
    typer.echo(f"Uploaded collection {report['collection_name']}")


@app.command()
def verify(
    build_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)],
) -> None:
    """Verify the staging Qdrant collection against canonical artifacts."""
    report = _run(lambda: verify_build(build_dir))
    typer.echo(f"Verified {report['verified_point_count']} occupation concepts")


@app.command()
def promote(
    build_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)],
) -> None:
    """Atomically point the stable alias at a verified collection."""
    report = _run(lambda: promote_build(build_dir))
    typer.echo(f"Promoted {report['collection_name']} as {report['alias']}")


@app.command(name="build")
def build_command(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    source: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    promote: Annotated[bool, typer.Option("--promote/--no-promote")] = False,
) -> None:
    """Run the complete indexing pipeline."""
    build_dir = _run(lambda: run_build(load_settings(config), source, promote=promote))
    typer.echo(build_dir)


@app.command(name="match-jobs")
def match_jobs_command(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    jobs: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    categories: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    esco_build_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    allow_collection: Annotated[bool, typer.Option("--allow-collection")] = False,
) -> None:
    """Match TopCV jobs against the ESCO occupation index (no rerank)."""
    from esco_occupation_indexer.matching.pipeline import run_match
    from esco_occupation_indexer.matching.settings import load_match_settings

    match_dir = _run(
        lambda: run_match(
            load_match_settings(config),
            jobs,
            categories,
            esco_build_dir,
            allow_collection=allow_collection,
        )
    )
    typer.echo(match_dir)
