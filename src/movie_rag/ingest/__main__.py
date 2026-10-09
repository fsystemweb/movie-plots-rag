"""``python -m movie_rag.ingest`` (``make ingest``): build the Qdrant index from the dataset or the fixture."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from movie_rag.config import load_settings
from movie_rag.errors import MovieRagError
from movie_rag.ingest.download import say
from movie_rag.ingest.embed import Embedder, FastEmbedder
from movie_rag.ingest.pipeline import IngestReport, ingest_csv, resolve_source
from movie_rag.observability import configure_tracing

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILURE = 1


def format_report(report: IngestReport) -> str:
    """The user-facing summary printed at the end of a run."""
    return (
        f"ingested {report.source} into collection {report.collection!r}\n"
        f"  rows read:            {report.rows_read}\n"
        f"  rows dropped:         {report.rows_dropped}\n"
        f"  films:                {report.films}\n"
        f"  chunks total:         {report.chunks_total}\n"
        f"  chunks written:       {report.chunks_written}\n"
        f"  chunks skipped:       {report.chunks_skipped} (already in Qdrant)\n"
        f"  points in Qdrant:     {report.points_in_collection}\n"
        f"  elapsed:              {report.elapsed_s:.2f}s"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m movie_rag.ingest",
        description="Chunk, embed and upsert the movie plots into Qdrant. Without options it ingests the downloaded "
        "dataset if present, else the bundled fixture.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--fixture", action="store_true", help="ingest the bundled synthetic fixture")
    source.add_argument("--csv", type=Path, help="ingest this CSV (Kaggle schema)")
    parser.add_argument("--recreate", action="store_true", help="drop and rebuild the collection first")
    return parser


def main(
    argv: Sequence[str] | None = None, *, client: QdrantClient | None = None, embedder: Embedder | None = None
) -> int:
    """CLI entry point. ``client`` and ``embedder`` can be injected (tests); returns the process exit code."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one INFO line per Qdrant request drowns the report
    try:
        settings = load_settings()
    except Exception as exc:  # an unreadable config is a user error: report it, never a traceback
        say(f"cannot load the configuration: {type(exc).__name__}: fix config.yaml / .env and re-run")
        logger.debug("configuration error", exc_info=exc)
        return EXIT_FAILURE
    configure_tracing(settings)
    qdrant = client or QdrantClient(url=settings.qdrant.url, timeout=settings.qdrant.timeout_s)
    try:
        csv_path = resolve_source(settings, csv=args.csv, fixture=args.fixture)
        logger.info("ingesting %s into %s at %s", csv_path, settings.qdrant.collection, settings.qdrant.url)
        report = ingest_csv(
            settings,
            csv_path,
            client=qdrant,
            embedder=embedder or FastEmbedder(settings.embeddings),
            recreate=args.recreate,
        )
    except (ResponseHandlingException, UnexpectedResponse) as exc:
        say(f"cannot talk to Qdrant at {settings.qdrant.url}: {type(exc).__name__}. Is it running? Try `make up`.")
        logger.debug("qdrant error", exc_info=exc)
        return EXIT_FAILURE
    except MovieRagError as exc:
        say(f"ingest failed: {exc}")
        return EXIT_FAILURE
    say(format_report(report))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
