"""The movie-discovery agent (LangChain over the MCP tools) and its CLI (``python -m movie_rag.agent``)."""

from movie_rag.agent.agent import MovieAgent, load_mcp_tools, make_chat_model
from movie_rag.agent.models import Answer, Citation, RetrievedFilm, ToolCallRecord, Usage
from movie_rag.agent.prompt import load_prompt

__all__ = [
    "Answer",
    "Citation",
    "MovieAgent",
    "RetrievedFilm",
    "ToolCallRecord",
    "Usage",
    "load_mcp_tools",
    "load_prompt",
    "make_chat_model",
]
