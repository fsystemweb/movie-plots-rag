"""``python -m movie_rag.retrieval`` (part of ``make demo``): a retrieval-only query, no LLM and no credentials."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from pydantic import ValidationError
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from movie_rag.config import RetrievalMode, load_settings
from movie_rag.errors import MovieRagError
from movie_rag.ingest.download import say
from movie_rag.ingest.embed import Embedder
from movie_rag.retrieval.search import MODES, MovieHit, Retriever, SearchFilters

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILURE = 1
ALL_MODES = "all"


def emit(text: str) -> None:
    """User-facing CLI output (results go to stdout so they can be piped)."""
    sys.stdout.write(text.rstrip("\n") + "\n")


def format_hits(mode: str, hits: Sequence[MovieHit]) -> str:
    """A readable ranked list for one mode."""
    lines = [f"== {mode}: {len(hits)} film(s) =="]
    if not hits:
        lines.append("  (no match)")
    for rank, hit in enumerate(hits, start=1):
        meta = ", ".join(v for v in (hit.genre, hit.origin, hit.director) if v)
        year = hit.release_year if hit.release_year is not None else "?"
        lines.append(f"{rank:>2}. {hit.title} ({year})  score={hit.score:.4f}  [{meta}]")
        lines.append(f"    {hit.movie_id}  chunk {hit.chunk_idx}  {hit.wiki_url or ''}".rstrip())
        lines.append(f"    {hit.snippet}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m movie_rag.retrieval",
        description="Retrieval-only search over the ingested films (no LLM, no credentials).",
    )
    parser.add_argument("query", nargs="?", help="the question; default: retrieval.demo_query from config.yaml")
    parser.add_argument(
        "--mode", choices=[*MODES, ALL_MODES], default=ALL_MODES, help="retrieval mode, or all three (default)"
    )
    parser.add_argument("--top-k", type=int, help="films to return (default: retrieval.top_k)")
    parser.add_argument("--year-from", type=int)
    parser.add_argument("--year-to", type=int)
    parser.add_argument("--genre")
    parser.add_argument("--origin")
    return parser


def main(
    argv: Sequence[str] | None = None, *, client: QdrantClient | None = None, embedder: Embedder | None = None
) -> int:
    """CLI entry point. ``client`` and ``embedder`` can be injected (tests); returns the process exit code."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        settings = load_settings()
    except Exception as exc:  # an unreadable config is a user error: report it, never a traceback
        say(f"cannot load the configuration: {type(exc).__name__}: fix config.yaml / .env and re-run")
        logger.debug("configuration error", exc_info=exc)
        return EXIT_FAILURE
    query = args.query or settings.retrieval.demo_query
    modes: tuple[RetrievalMode, ...] = MODES if args.mode == ALL_MODES else (args.mode,)
    retriever = Retriever(settings, client=client, embedder=embedder)
    try:
        filters = SearchFilters(year_from=args.year_from, year_to=args.year_to, genre=args.genre, origin=args.origin)
        emit(f"query: {query}")
        for mode in modes:
            emit(format_hits(mode, retriever.search(query, mode=mode, top_k=args.top_k, filters=filters)))
    except ValidationError as exc:
        say("invalid filters: " + "; ".join(str(e["msg"]) for e in exc.errors()))
        return EXIT_FAILURE
    except (ResponseHandlingException, UnexpectedResponse) as exc:
        say(
            f"cannot search Qdrant at {settings.qdrant.url}: {type(exc).__name__}. "
            "Is it running and ingested? Try `make up` and `make ingest`."
        )
        logger.debug("qdrant error", exc_info=exc)
        return EXIT_FAILURE
    except MovieRagError as exc:
        say(f"search failed: {exc}")
        return EXIT_FAILURE
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
