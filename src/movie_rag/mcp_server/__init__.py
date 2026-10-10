"""FastMCP server exposing the retrieval layer as four tools (no LLM calls inside this package)."""

from movie_rag.mcp_server.models import SearchResult, SimilarResult
from movie_rag.mcp_server.server import SERVER_NAME, TOOL_NAMES, build_server

__all__ = ["SERVER_NAME", "TOOL_NAMES", "SearchResult", "SimilarResult", "build_server"]
