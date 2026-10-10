"""``python -m movie_rag.agent "question"`` (``make ask Q="..."``): ask the movie agent one question.

Without ``NEBIUS_API_KEY`` it prints the credentials hint and exits 2 before touching the network. The answer goes to
stdout (``--json`` for the whole :class:`Answer`), the run summary to stderr.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence

from movie_rag.agent.agent import MovieAgent
from movie_rag.agent.models import Answer
from movie_rag.config import load_settings
from movie_rag.errors import MissingCredentialError, MovieRagError
from movie_rag.ingest.download import say
from movie_rag.observability import configure_tracing
from movie_rag.retrieval.search import MODES

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2  # missing credentials or a bad command line


def emit(text: str) -> None:
    """User-facing CLI output (the answer goes to stdout so it can be piped)."""
    sys.stdout.write(text.rstrip("\n") + "\n")


def format_answer(answer: Answer) -> str:
    """The answer text followed by its sources (title, year, Wikipedia link), taken from the tool results."""
    lines = [answer.text]
    if answer.citations:
        lines += ["", "Sources:"]
        lines += [f"  - {c.label}" + (f" - {c.wiki_url}" if c.wiki_url else "") for c in answer.citations]
    return "\n".join(lines)


def format_summary(answer: Answer) -> str:
    """One line for stderr: tool calls, model, prompt version, git sha, latency."""
    calls = ", ".join(f"{c.name}({c.result_count})" for c in answer.tool_calls) or "none"
    blocked = f" (+{answer.blocked_tool_calls} refused)" if answer.blocked_tool_calls else ""
    meta = answer.metadata
    return (
        f"[tool calls: {calls}{blocked} | model {meta['chat_model']} | prompt {meta['prompt_version']} | "
        f"sha {meta['git_sha']} | {answer.latency_ms / 1000:.1f}s]"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m movie_rag.agent",
        description="Ask the movie agent a question about film plots (needs NEBIUS_API_KEY and `make serve`).",
    )
    parser.add_argument(
        "question", help="what to ask, e.g. 'a film about a hotel operator who overhears a murder plot'"
    )
    parser.add_argument("--mode", choices=list(MODES), help="pin the retrieval mode (default: the model decides)")
    parser.add_argument("--json", action="store_true", help="print the whole answer object as JSON")
    return parser


def main(argv: Sequence[str] | None = None, *, agent: MovieAgent | None = None) -> int:
    """CLI entry point; ``agent`` can be injected (tests). Returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    if not args.question.strip():
        parser.error('the question is empty: make ask Q="..."')  # exits 2
    try:
        settings = agent.settings if agent else load_settings()
    except Exception as exc:  # an unreadable config is a user error: report it, never a traceback
        say(f"cannot load the configuration: {type(exc).__name__}: fix config.yaml / .env and re-run")
        logger.debug("configuration error", exc_info=exc)
        return EXIT_FAILURE
    configure_tracing(settings)
    try:
        answer = asyncio.run((agent or MovieAgent(settings)).ask(args.question, mode=args.mode))
    except MissingCredentialError as exc:
        say(str(exc))
        return EXIT_USAGE
    except MovieRagError as exc:
        say(str(exc))
        return EXIT_FAILURE
    emit(answer.model_dump_json(indent=2) if args.json else format_answer(answer))
    say(format_summary(answer))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
