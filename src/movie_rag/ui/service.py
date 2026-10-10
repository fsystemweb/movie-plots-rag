"""Logic behind the Streamlit page: no ``streamlit`` import in here, so it is testable without a browser.

* **Retrieval-only** questions, the filter lists and the "Compare modes" tab call the MCP tools (``search_movies``,
  ``list_filters``) over the same HTTP endpoint the agent uses (``mcp.url``), through a plain MCP client. That keeps one
  retrieval path for the whole system, loads no embedding model in the Streamlit process, and makes no LLM call and
  needs no credential. The price is that ``make serve`` must be running, which the error message says.
* **Agent** questions go through :class:`movie_rag.agent.MovieAgent`; without ``NEBIUS_API_KEY`` it raises
  ``MissingCredentialError`` before anything else happens.
* Nothing here raises to the page: :meth:`MovieService.run_turn`, :meth:`~MovieService.compare` and
  :meth:`~MovieService.load_filters` return objects that carry a :class:`UiError` instead (a friendly message,
  never a traceback and never a secret).

The page is synchronous, so the public methods wrap one ``asyncio.run`` each; a single MCP session serves all the
calls of one user action (for example the three searches of the comparison).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastmcp import Client
from fastmcp.client.client import CallToolResult
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, Field

from movie_rag.agent import MovieAgent
from movie_rag.agent.agent import is_connection_failure
from movie_rag.agent.models import Answer, RetrievedFilm
from movie_rag.config import RetrievalMode, Settings, load_settings
from movie_rag.errors import AgentError, McpUnavailableError, MissingCredentialError, MovieRagError
from movie_rag.mcp_server.models import SearchResult
from movie_rag.observability import span
from movie_rag.retrieval import FilterOptions
from movie_rag.retrieval.search import MODES

logger = logging.getLogger(__name__)

SEARCH_TOOL = "search_movies"
FILTERS_TOOL = "list_filters"
RETRIEVAL_ONLY_HINT = "Switch on 'Retrieval only' in the sidebar to search without a key."

ErrorKind = Literal["credentials", "mcp_down", "tool", "agent", "unexpected"]
ClientFactory = Callable[[], "Client[Any]"]


class UiError(BaseModel):
    """A failure in words the user can act on."""

    kind: ErrorKind
    message: str


class SearchParams(BaseModel):
    """The sidebar's controls."""

    mode: RetrievalMode
    top_k: int = Field(ge=1)
    year_from: int | None = None
    year_to: int | None = None
    genre: str | None = None
    origin: str | None = None

    def tool_arguments(self, query: str) -> dict[str, Any]:
        """Arguments of ``search_movies``: unset filters are left out so the server's defaults apply."""
        arguments: dict[str, Any] = {"query": query, "mode": self.mode, "top_k": self.top_k}
        for name in ("year_from", "year_to", "genre", "origin"):
            value = getattr(self, name)
            if value is not None:
                arguments[name] = value
        return arguments

    def constraints_text(self) -> str:
        """The filters as a plain sentence for the agent, which picks the tool arguments itself ('' when none)."""
        parts: list[str] = []
        if self.year_from is not None and self.year_to is not None:
            parts.append(f"released between {self.year_from} and {self.year_to}")
        elif self.year_from is not None:
            parts.append(f"released in or after {self.year_from}")
        elif self.year_to is not None:
            parts.append(f"released in or before {self.year_to}")
        if self.genre:
            parts.append(f"genre '{self.genre}'")
        if self.origin:
            parts.append(f"origin '{self.origin}'")
        return f"Only consider films {', '.join(parts)}." if parts else ""


class RetrievalOutcome(BaseModel):
    """The films one retrieval returned, best first."""

    query: str
    mode: RetrievalMode
    films: list[RetrievedFilm]
    note: str | None = None
    latency_ms: float = 0.0


class Turn(BaseModel):
    """One chat exchange: the question and either an agent answer, a retrieval-only result, or an error."""

    question: str
    retrieval_only: bool
    answer: Answer | None = None
    outcome: RetrievalOutcome | None = None
    trace_url: str | None = None
    error: UiError | None = None


class FilterLoad(BaseModel):
    """What ``list_filters`` returned, or why it could not."""

    options: FilterOptions | None = None
    error: UiError | None = None


class Comparison(BaseModel):
    """The same query in every retrieval mode (retrieval only)."""

    query: str
    outcomes: dict[str, RetrievalOutcome] = Field(default_factory=dict)
    error: UiError | None = None

    def shared_movie_ids(self) -> list[str]:
        """Films returned by every mode, in the order of the first mode."""
        if not self.outcomes:
            return []
        lists = [[f.movie_id for f in o.films] for o in self.outcomes.values()]
        return [movie_id for movie_id in lists[0] if all(movie_id in other for other in lists[1:])]


def describe_error(error: BaseException, settings: Settings) -> UiError:
    """Turn an exception into a message for the page. Secrets are redacted; unknown errors are logged, not shown."""
    if isinstance(error, MissingCredentialError):
        return UiError(kind="credentials", message=f"{error} {RETRIEVAL_ONLY_HINT}")
    if isinstance(error, McpUnavailableError):
        return UiError(kind="mcp_down", message=str(error))
    if isinstance(error, ToolError):
        return UiError(kind="tool", message=settings.redact(str(error)))
    if isinstance(error, MovieRagError):
        return UiError(kind="agent", message=settings.redact(str(error)))
    if is_connection_failure(error):
        return UiError(kind="mcp_down", message=str(McpUnavailableError(settings.mcp.url)))
    logger.error("unexpected failure behind the page", exc_info=error)
    return UiError(
        kind="unexpected", message=f"Something went wrong ({type(error).__name__}). The details are in the server log."
    )


def to_films(result: SearchResult) -> list[RetrievedFilm]:
    """Search hits as the model the page renders (same one the agent reports)."""
    return [RetrievedFilm.model_validate({**hit.model_dump(), "tool": SEARCH_TOOL}) for hit in result.results]


def _structured(result: CallToolResult, tool: str) -> dict[str, Any]:
    if result.structured_content is None:
        raise AgentError(f"the {tool} tool returned no structured result")
    return dict(result.structured_content)


class MovieService:
    """What the page needs, with the agent and the MCP client injectable (tests pass fakes)."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        agent: MovieAgent | None = None,
        client_factory: ClientFactory | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self._agent = agent or MovieAgent(self.settings)
        self._client_factory = client_factory or self._default_client

    def _default_client(self) -> Client[Any]:
        return Client(self.settings.mcp.url, timeout=self.settings.ui.mcp_timeout_s)

    @asynccontextmanager
    async def _session(self) -> AsyncIterator[Client[Any]]:
        """One MCP session; a server that cannot be reached becomes :class:`McpUnavailableError`."""
        try:
            async with self._client_factory() as client:
                yield client
        except Exception as exc:
            if is_connection_failure(exc):
                raise McpUnavailableError(self.settings.mcp.url) from exc
            raise

    async def _search(self, client: Client[Any], query: str, params: SearchParams) -> RetrievalOutcome:
        started = time.perf_counter()
        result = await client.call_tool(SEARCH_TOOL, params.tool_arguments(query))
        found = SearchResult.model_validate(_structured(result, SEARCH_TOOL))
        return RetrievalOutcome(
            query=found.query,
            mode=found.mode,
            films=to_films(found),
            note=found.note,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    # --- the page's three entry points: they never raise ----------------------------------------------------

    def load_filters(self) -> FilterLoad:
        """Genres, origins and the year range for the sidebar."""

        async def fetch() -> FilterOptions:
            async with self._session() as client:
                result = await client.call_tool(FILTERS_TOOL, {})
                return FilterOptions.model_validate(_structured(result, FILTERS_TOOL))

        try:
            return FilterLoad(options=asyncio.run(fetch()))
        except Exception as exc:
            return FilterLoad(error=describe_error(exc, self.settings))

    def run_turn(self, question: str, params: SearchParams, *, retrieval_only: bool) -> Turn:
        """Answer ``question`` with the agent, or (``retrieval_only``) just list the matching films."""
        turn = Turn(question=question, retrieval_only=retrieval_only)
        try:
            if retrieval_only:
                turn.outcome = asyncio.run(self._retrieve(question, params))
            else:
                turn.answer, turn.trace_url = asyncio.run(self._ask(question, params))
        except Exception as exc:
            turn.error = describe_error(exc, self.settings)
        return turn

    def compare(self, query: str, params: SearchParams) -> Comparison:
        """``query`` in dense, sparse and hybrid mode over one MCP session (retrieval only: no LLM, no key)."""

        async def run() -> dict[str, RetrievalOutcome]:
            outcomes: dict[str, RetrievalOutcome] = {}
            async with self._session() as client:
                for mode in MODES:
                    outcomes[mode] = await self._search(client, query, params.model_copy(update={"mode": mode}))
            return outcomes

        try:
            return Comparison(query=query, outcomes=asyncio.run(run()))
        except Exception as exc:
            return Comparison(query=query, error=describe_error(exc, self.settings))

    # --- internals -----------------------------------------------------------------------------------------

    async def _retrieve(self, question: str, params: SearchParams) -> RetrievalOutcome:
        async with self._session() as client:
            return await self._search(client, question, params)

    async def _ask(self, question: str, params: SearchParams) -> tuple[Answer, str | None]:
        constraints = params.constraints_text()
        prompt = f"{question}\n\n{constraints}" if constraints else question
        with span("ui.ask", run_type="chain", settings=self.settings, inputs={"question": question}) as run:
            answer = await self._agent.ask(prompt, mode=params.mode)
            return answer, run.trace_url()


def build_service() -> MovieService:
    """The service the page uses: settings from ``config.yaml`` / the environment, real agent and MCP client."""
    return MovieService()
