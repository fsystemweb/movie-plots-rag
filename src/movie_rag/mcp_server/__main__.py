"""``python -m movie_rag.mcp_server``: run the MCP server over HTTP (default, ``make serve``) or stdio.

``--healthcheck`` is what the Docker healthcheck runs: it GETs the server's own health route and exits 0 or 1.
"""

from __future__ import annotations

import argparse
import logging
import sys
import urllib.error
import urllib.request
from collections.abc import Sequence

from movie_rag.config import Settings, load_settings
from movie_rag.ingest.download import say
from movie_rag.mcp_server.server import build_server
from movie_rag.observability import configure_tracing

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILURE = 1
HTTP_OK = 200
LOOPBACK = "127.0.0.1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m movie_rag.mcp_server",
        description="Movie Plots RAG MCP server: search_movies, get_movie, find_similar, list_filters.",
    )
    parser.add_argument(
        "--transport",
        choices=["http", "stdio"],
        default="http",
        help="http (streamable HTTP at mcp.path, for Docker and the agent) or stdio (local MCP clients)",
    )
    parser.add_argument("--host", help="HTTP bind address (default: mcp.host)")
    parser.add_argument("--port", type=int, help="HTTP port (default: mcp.port)")
    parser.add_argument("--healthcheck", action="store_true", help="probe the running HTTP server and exit 0/1")
    return parser


def healthcheck(settings: Settings, port: int | None = None) -> bool:
    """True when the local HTTP server answers 200 on its health route."""
    cfg = settings.mcp
    url = f"http://{LOOPBACK}:{port or cfg.port}{cfg.health_path}"
    try:
        with urllib.request.urlopen(url, timeout=cfg.healthcheck_timeout_s) as response:
            return bool(response.status == HTTP_OK)
    except (urllib.error.URLError, OSError):
        return False


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns the process exit code (the server itself blocks until stopped)."""
    args = build_parser().parse_args(argv)
    # stdio uses stdout for the protocol: logs must go to stderr (the logging default).
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    try:
        settings = load_settings()
    except Exception as exc:  # an unreadable config is a user error: report it, never a traceback
        say(f"cannot load the configuration: {type(exc).__name__}: fix config.yaml / .env and re-run")
        logger.debug("configuration error", exc_info=exc)
        return EXIT_FAILURE
    if args.healthcheck:
        return EXIT_OK if healthcheck(settings, args.port) else EXIT_FAILURE
    configure_tracing(settings)
    server = build_server(settings)
    if args.transport == "stdio":
        server.run(transport="stdio", show_banner=False)
    else:
        server.run(
            transport="http",
            host=args.host or settings.mcp.host,
            port=args.port or settings.mcp.port,
            path=settings.mcp.path,
            show_banner=False,
        )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
