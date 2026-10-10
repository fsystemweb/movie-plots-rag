"""The movie agent: a LangChain ``create_agent`` tool-calling loop over the MCP tools, with a hard tool-call cap.

* The chat model (Nebius, OpenAI-compatible) is created when a question is asked, never at import; without
  ``NEBIUS_API_KEY`` :class:`~movie_rag.errors.MissingCredentialError` is raised before anything else happens.
* At most ``llm.max_tool_calls`` tool calls run per question (``ToolCallLimitMiddleware``); further calls are refused
  with an error message the model sees, and counted in :attr:`Answer.blocked_tool_calls`.
* Citations come from the films the tools returned (:mod:`movie_rag.agent.citations`), never from the model's text.
* Every run is wrapped in a LangSmith span carrying ``observability.run_metadata`` (no-op without a key).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable, Iterator, Sequence
from typing import Any

import httpx
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError

from movie_rag.agent.citations import films_from_tool_message, merge_films, select_citations
from movie_rag.agent.models import Answer, RetrievedFilm, ToolCallRecord, Usage
from movie_rag.agent.prompt import load_prompt
from movie_rag.config import RetrievalMode, Settings, load_settings
from movie_rag.errors import AgentError, McpUnavailableError
from movie_rag.observability import run_metadata, scrub, span

logger = logging.getLogger(__name__)

SERVER_KEY = "movies"
SEARCH_TOOL = "search_movies"
ABSTENTION_TEXT = (
    "No film in the index matches that description well enough to recommend one. "
    "Try describing the plot differently, or relax a filter such as the year range."
)
BLOCKED_PREFIX = "Tool call limit exceeded"  # content of the ToolMessage that ToolCallLimitMiddleware injects
STEPS_PER_ROUND = 4  # graph steps one model/tools round may take (model, middleware, tools) with headroom
ERROR_TEXT_LIMIT = 300

ToolHandler = Callable[[MCPToolCallRequest], Awaitable[Any]]


def make_chat_model(settings: Settings) -> ChatOpenAI:
    """The Nebius chat model. Raises ``MissingCredentialError`` without ``NEBIUS_API_KEY`` (call time, not import)."""
    api_key = settings.require_nebius_api_key()
    return ChatOpenAI(
        model=settings.llm.chat_model,
        base_url=settings.llm.base_url,
        api_key=api_key,
        temperature=settings.llm.temperature,
    )


def _chain(error: BaseException) -> Iterator[BaseException]:
    """``error`` and everything it wraps: exception-group members (anyio) and ``__cause__`` links, depth first."""
    yield error
    for child in getattr(error, "exceptions", ()):
        yield from _chain(child)
    if error.__cause__ is not None:
        yield from _chain(error.__cause__)


def is_connection_failure(error: BaseException) -> bool:
    """True when ``error`` (or anything it wraps) means the MCP server could not be reached."""
    return any(isinstance(e, httpx.TransportError | httpx.HTTPStatusError | OSError) for e in _chain(error))


async def load_mcp_tools(settings: Settings, mode: RetrievalMode | None = None) -> list[BaseTool]:
    """The MCP server's tools as LangChain tools. ``mode`` forces the retrieval mode of every ``search_movies`` call.

    Raises :class:`McpUnavailableError` when the server cannot be reached.
    """

    async def force_mode(request: MCPToolCallRequest, handler: ToolHandler) -> Any:
        if mode is not None and request.name == SEARCH_TOOL:
            request = request.override(args={**request.args, "mode": mode})
        return await handler(request)

    client = MultiServerMCPClient(
        {
            SERVER_KEY: {
                "transport": "streamable_http",
                "url": settings.mcp.url,
                "timeout": settings.agent.mcp_timeout_s,
            }
        },
        tool_interceptors=[force_mode],
    )
    try:
        return list(await client.get_tools())
    except Exception as exc:
        if is_connection_failure(exc):
            raise McpUnavailableError(settings.mcp.url) from exc
        raise


def _tool_error_text(message: ToolMessage, settings: Settings) -> str:
    text = message.text if isinstance(message.text, str) else str(message.content)
    return str(scrub(text.strip()[:ERROR_TEXT_LIMIT], settings))


def _is_blocked(message: ToolMessage) -> bool:
    return message.status == "error" and message.text.startswith(BLOCKED_PREFIX)


def summarise_tool_calls(
    messages: Sequence[BaseMessage], settings: Settings
) -> tuple[list[ToolCallRecord], int, list[RetrievedFilm]]:
    """Executed tool calls, the number refused by the cap, and the films the tools returned (in order)."""
    results = {m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)}
    records: list[ToolCallRecord] = []
    films: list[RetrievedFilm] = []
    blocked = 0
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls:
            result = results.get(call["id"] or "")
            if result is None:  # the run ended before this call was answered
                continue
            if _is_blocked(result):
                blocked += 1
                continue
            found = films_from_tool_message(result)
            films.extend(found)
            failed = result.status == "error"
            records.append(
                ToolCallRecord(
                    name=call["name"],
                    args=scrub(dict(call["args"]), settings),
                    result_count=len(found),
                    error=_tool_error_text(result, settings) if failed else None,
                )
            )
    return records, blocked, films


def final_text(messages: Sequence[BaseMessage]) -> str:
    """The model's last message when it is a plain answer (no pending tool call), else the empty string."""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return "" if message.tool_calls else message.text.strip()
    return ""


def total_usage(messages: Sequence[BaseMessage]) -> Usage:
    usage = Usage()
    for message in messages:
        if isinstance(message, AIMessage) and message.usage_metadata:
            usage.input_tokens += message.usage_metadata.get("input_tokens", 0)
            usage.output_tokens += message.usage_metadata.get("output_tokens", 0)
            usage.total_tokens += message.usage_metadata.get("total_tokens", 0)
    return usage


class MovieAgent:
    """Answers one question at a time from the movie index. Construct it cheaply; the model and tools load lazily.

    ``model`` and ``tools`` can be injected (tests use a scripted chat model and in-process tools); by default the
    Nebius model and the tools of the MCP server at ``mcp.url`` are used.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        model: BaseChatModel | None = None,
        tools: Sequence[BaseTool] | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self._model = model
        self._tools = list(tools) if tools is not None else None

    async def ask(self, question: str, *, mode: RetrievalMode | None = None) -> Answer:
        """Answer ``question``. ``mode`` pins the retrieval mode; by default the model chooses it.

        Raises ``MissingCredentialError`` (no Nebius key), :class:`McpUnavailableError` (server down) or
        :class:`~movie_rag.errors.AgentError` (empty question, unknown prompt version).
        """
        cfg = self.settings
        question = str(
            scrub(question.strip(), cfg)
        )  # a key typed into a question reaches neither the model nor a trace
        if not question:
            raise AgentError("the question is empty")
        model = self._model or make_chat_model(cfg)  # credentials first: nothing else runs without them
        prompt = load_prompt(cfg)
        tools = self._tools if self._tools is not None else await load_mcp_tools(cfg, mode)
        metadata = run_metadata(cfg, retrieval_mode=mode or cfg.retrieval.default_mode)
        cap = cfg.llm.max_tool_calls
        agent = create_agent(
            model,
            tools,
            system_prompt=prompt,
            middleware=[ToolCallLimitMiddleware(run_limit=cap, exit_behavior="continue")],
            name="movie_agent",
        )
        run_config: Any = {
            "recursion_limit": STEPS_PER_ROUND * (cap + 2),
            "metadata": metadata,
            "run_name": "movie_agent",
        }
        started = time.perf_counter()
        with span("agent.ask", run_type="chain", settings=cfg, inputs={"question": question}, **metadata) as run:
            messages, stopped_early = await self._run(agent, question, run_config)
            answer = self._build_answer(question, messages, stopped_early, metadata)
            answer.latency_ms = round((time.perf_counter() - started) * 1000, 2)
            run.set(
                answer=answer.text,
                citations=[c.movie_id for c in answer.citations],
                abstained=answer.abstained,
                tool_calls=[(c.name, c.result_count) for c in answer.tool_calls],
                blocked_tool_calls=answer.blocked_tool_calls,
                usage=answer.usage.model_dump(),
            )
        return answer

    async def _run(self, agent: Any, question: str, run_config: Any) -> tuple[list[BaseMessage], bool]:
        """Drive the agent, keeping the latest state so a step-limit stop still yields the calls made so far."""
        messages: list[BaseMessage] = []
        try:
            async for state in agent.astream(
                {"messages": [{"role": "user", "content": question}]}, run_config, stream_mode="values"
            ):
                messages = list(state["messages"])
        except GraphRecursionError:
            logger.warning("agent stopped at the step limit without answering")
            return messages, True
        except Exception as exc:
            if is_connection_failure(exc):
                raise McpUnavailableError(self.settings.mcp.url) from exc
            raise
        return messages, False

    def _build_answer(
        self, question: str, messages: Sequence[BaseMessage], stopped_early: bool, metadata: dict[str, Any]
    ) -> Answer:
        cfg = self.settings
        records, blocked, films = summarise_tool_calls(messages, cfg)
        retrieved = merge_films(films)
        model_text = "" if stopped_early else final_text(messages)
        citations = select_citations(model_text, retrieved, cfg.agent.max_citations)
        return Answer(
            question=question,
            text=model_text if citations else ABSTENTION_TEXT,
            citations=citations,
            abstained=not citations,
            retrieved=retrieved,
            tool_calls=records,
            blocked_tool_calls=blocked,
            stopped_early=stopped_early,
            metadata={**metadata, "max_tool_calls": cfg.llm.max_tool_calls},
            usage=total_usage(messages),
        )
