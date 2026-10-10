"""Stand-ins for the UI tests: an in-memory MCP client over the fixture index, a server that is down, a fake agent."""

from __future__ import annotations

from collections.abc import Callable
from types import TracebackType
from typing import Any

import httpx
from fastmcp import Client
from langchain_core.messages import AIMessage
from test_agent import ALPHA, ALPHA_TEXT, BETA, SEARCH, Tools

from fakes import ScriptedChatModel, tool_call_message
from index_fixture import Index
from movie_rag.agent import MovieAgent
from movie_rag.agent.models import Answer
from movie_rag.config import RetrievalMode, Settings
from movie_rag.mcp_server import build_server
from movie_rag.ui.service import MovieService


class DownClient:
    """What ``fastmcp.Client`` does when nothing listens at the URL: a RuntimeError chained to a ConnectError."""

    async def __aenter__(self) -> DownClient:
        try:
            raise httpx.ConnectError("All connection attempts failed")
        except httpx.ConnectError as exc:
            raise RuntimeError("Client failed to connect: All connection attempts failed") from exc

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        return None


class EmptyReplyClient:
    """A client whose tools answer without structured content."""

    async def __aenter__(self) -> EmptyReplyClient:
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        return None

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        return type("Result", (), {"structured_content": None})()


class FailingAgent:
    """Stands in for ``MovieAgent`` and raises ``error`` when asked."""

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.questions: list[str] = []

    async def ask(self, question: str, *, mode: RetrievalMode | None = None) -> Answer:
        self.questions.append(question)
        raise self.error


def mcp_client_factory(index: Index) -> Callable[[], Client[Any]]:
    """A client on the real MCP server (fixture index, fake embedder), connected in memory."""
    return lambda: Client(build_server(index.settings, index.retriever))


def scripted_agent(
    settings: Settings, replies: list[AIMessage] | None = None
) -> tuple[MovieAgent, ScriptedChatModel, Tools]:
    """An agent on a scripted model that searches once and then names the first film (no key, no network)."""
    tools = Tools([ALPHA, BETA])
    model = ScriptedChatModel(replies=replies or [tool_call_message(SEARCH, tokens=120), AIMessage(ALPHA_TEXT)])
    return MovieAgent(settings, model=model, tools=tools.as_tools()), model, tools


def make_service(index: Index, *, agent: Any = None, client_factory: Callable[[], Any] | None = None) -> MovieService:
    return MovieService(index.settings, agent=agent, client_factory=client_factory or mcp_client_factory(index))
