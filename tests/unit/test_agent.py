"""The agent against a scripted chat model and in-process tools: no network, no key, no Qdrant, no MCP server."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool, ToolException
from langchain_openai import ChatOpenAI

from fakes import RunCapture, ScriptedChatModel, tool_call_message
from movie_rag.agent import MovieAgent, load_prompt, make_chat_model
from movie_rag.agent import agent as agent_module
from movie_rag.agent.agent import ABSTENTION_TEXT, is_connection_failure, load_mcp_tools
from movie_rag.config import Settings
from movie_rag.errors import AgentError, McpUnavailableError, MissingCredentialError

ALPHA = {
    "movie_id": "alpha-1999-1",
    "title": "Alpha",
    "release_year": 1999,
    "wiki_url": "https://en.wikipedia.org/wiki/Alpha",
    "score": 0.9,
    "snippet": "a telephone operator overhears a plot",
}
BETA = {**ALPHA, "movie_id": "beta-2004-2", "title": "Beta", "release_year": 2004, "wiki_url": "https://w/Beta"}
ALPHA_TEXT = "Alpha (1999) - https://en.wikipedia.org/wiki/Alpha [alpha-1999-1]: an operator overhears a plot."
SEARCH = ("search_movies", {"query": "operator overhears a murder plot"})


class Tools:
    """In-process stand-ins for the MCP tools that record every invocation."""

    def __init__(self, results: list[dict[str, Any]], *, fail: bool = False, artifact: bool = False) -> None:
        self.invocations: list[dict[str, Any]] = []
        self.results = results
        self.fail = fail
        self.artifact = artifact

    def search_movies(self, query: str, mode: str = "hybrid") -> Any:
        self.invocations.append({"query": query, "mode": mode})
        if self.fail:
            raise ToolException("The movie index (Qdrant) is unavailable; start it with `make up` and retry.")
        payload = {"query": query, "mode": mode, "count": len(self.results), "results": self.results}
        return (json.dumps(payload), {"structured_content": payload}) if self.artifact else json.dumps(payload)

    def list_filters(self) -> str:
        self.invocations.append({"tool": "list_filters"})
        return json.dumps({"genres": ["drama"], "origins": ["american"]})

    def as_tools(self) -> list[StructuredTool]:
        search = StructuredTool.from_function(
            self.search_movies,
            name="search_movies",
            description="Find films whose plot matches a description.",
            handle_tool_error=True,
            response_format="content_and_artifact" if self.artifact else "content",
        )
        filters = StructuredTool.from_function(
            self.list_filters, name="list_filters", description="Valid genres and origins."
        )
        return [search, filters]


def agent_for(
    settings: Settings, replies: list[AIMessage], tools: Tools | None = None
) -> tuple[MovieAgent, ScriptedChatModel, Tools]:
    tools = tools or Tools([ALPHA, BETA])
    model = ScriptedChatModel(replies=replies)
    return MovieAgent(settings, model=model, tools=tools.as_tools()), model, tools


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings()


# --- credentials: raised at call time, never at import ------------------------------------------------------


def test_importing_the_agent_needs_no_key_and_asking_without_one_fails_with_the_hint(
    settings: Settings,
) -> None:
    agent = MovieAgent(settings)  # constructing is free
    with pytest.raises(MissingCredentialError) as info:
        make_chat_model(settings)
    assert str(info.value) == "set NEBIUS_API_KEY — see docs/CREDENTIALS.md"
    assert agent.settings is settings


async def test_asking_without_a_key_fails_before_loading_tools_or_connecting(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the MCP server must not be contacted without credentials")

    monkeypatch.setattr(agent_module, "load_mcp_tools", forbidden)
    with pytest.raises(MissingCredentialError, match="NEBIUS_API_KEY"):
        await MovieAgent(settings).ask("which film?")


def test_the_chat_model_comes_from_the_configuration(make_settings: Callable[..., Settings]) -> None:
    cfg = make_settings(NEBIUS_API_KEY="fake-test-key", NEBIUS_BASE_URL="https://nebius.example/v1/")
    model = make_chat_model(cfg)
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == cfg.llm.chat_model
    assert model.openai_api_base == "https://nebius.example/v1/"
    assert model.temperature == cfg.llm.temperature == 0
    assert model.openai_api_key is not None and model.openai_api_key.get_secret_value() == "fake-test-key"


# --- the happy path ------------------------------------------------------------------------------------------


async def test_a_search_then_an_answer_yields_cited_films_from_the_tool_results(settings: Settings) -> None:
    agent, model, tools = agent_for(settings, [tool_call_message(SEARCH, tokens=100), AIMessage(ALPHA_TEXT)])
    answer = await agent.ask("  a film about an operator who overhears a murder plot  ")

    assert answer.question == "a film about an operator who overhears a murder plot"
    assert answer.text == ALPHA_TEXT and not answer.abstained and not answer.stopped_early
    assert [(c.movie_id, c.label, c.wiki_url) for c in answer.citations] == [
        ("alpha-1999-1", "Alpha (1999)", "https://en.wikipedia.org/wiki/Alpha")
    ]
    assert [f.movie_id for f in answer.retrieved] == ["alpha-1999-1", "beta-2004-2"]
    assert [(c.name, c.result_count, c.error) for c in answer.tool_calls] == [("search_movies", 2, None)]
    assert answer.tool_calls[0].args == SEARCH[1]
    assert answer.blocked_tool_calls == 0
    assert answer.usage.input_tokens == 100 and answer.usage.total_tokens == 101
    assert tools.invocations == [{"query": SEARCH[1]["query"], "mode": "hybrid"}]
    assert model.bound_tool_names == ["search_movies", "list_filters"]


async def test_the_model_sees_the_versioned_prompt_and_the_question(settings: Settings) -> None:
    agent, model, _ = agent_for(settings, [AIMessage("Nothing to search.")])
    await agent.ask("what is that film?")
    first_turn = model.prompts[0]
    assert first_turn[0].type == "system" and first_turn[0].content == load_prompt(settings)
    assert first_turn[-1].content == "what is that film?"


async def test_citations_also_work_with_the_mcp_artifact_shape(settings: Settings) -> None:
    agent, _, _ = agent_for(settings, [tool_call_message(SEARCH), AIMessage(ALPHA_TEXT)], Tools([ALPHA], artifact=True))
    answer = await agent.ask("operator")
    assert [c.movie_id for c in answer.citations] == ["alpha-1999-1"]


async def test_citations_are_capped_by_the_configuration(make_settings: Callable[..., Settings]) -> None:
    cfg = make_settings(AGENT__MAX_CITATIONS="1")
    text = "Alpha (1999) first, then Beta (2004)."
    agent, _, _ = agent_for(cfg, [tool_call_message(SEARCH), AIMessage(text)])
    answer = await agent.ask("films")
    assert [c.movie_id for c in answer.citations] == ["alpha-1999-1"]


# --- citations only from retrieved ids ---------------------------------------------------------------------


async def test_a_hallucinated_film_in_the_answer_never_becomes_a_citation(settings: Settings) -> None:
    text = (
        ALPHA_TEXT
        + " Ghost Town (2011) - https://en.wikipedia.org/wiki/Ghost_Town [ghost-town-2011-77]: also an operator."
    )
    agent, _, _ = agent_for(settings, [tool_call_message(SEARCH), AIMessage(text)])
    answer = await agent.ask("operator")
    assert [c.movie_id for c in answer.citations] == ["alpha-1999-1"]
    assert "ghost-town-2011-77" not in {f.movie_id for f in answer.retrieved}


async def test_citation_title_year_and_link_come_from_the_tool_result_not_the_model_text(settings: Settings) -> None:
    text = "Alpha (1999) https://evil.example/alpha [alpha-1999-1]: matches."
    agent, _, _ = agent_for(settings, [tool_call_message(SEARCH), AIMessage(text)])
    (citation,) = (await agent.ask("operator")).citations
    assert citation.wiki_url == "https://en.wikipedia.org/wiki/Alpha"


# --- abstention ---------------------------------------------------------------------------------------------


async def test_nothing_retrieved_is_an_abstention_with_no_citations(settings: Settings) -> None:
    agent, _, _ = agent_for(
        settings,
        [tool_call_message(SEARCH), AIMessage("I found no film that fits.")],
        Tools([]),
    )
    answer = await agent.ask("a film that does not exist")
    assert answer.abstained and answer.citations == [] and answer.retrieved == []
    assert answer.text == ABSTENTION_TEXT
    assert [(c.name, c.result_count) for c in answer.tool_calls] == [("search_movies", 0)]


async def test_an_answer_naming_only_unretrieved_films_is_an_abstention(settings: Settings) -> None:
    agent, _, _ = agent_for(
        settings,
        [tool_call_message(SEARCH), AIMessage("Casablanca (1942) [casablanca-1942-5] is the one.")],
        Tools([]),
    )
    answer = await agent.ask("old romance")
    assert answer.abstained and answer.text == ABSTENTION_TEXT and answer.citations == []


async def test_answering_without_any_tool_call_cites_nothing(settings: Settings) -> None:
    agent, _, tools = agent_for(settings, [AIMessage("Alpha (1999) [alpha-1999-1] from memory.")])
    answer = await agent.ask("from memory?")
    assert answer.abstained and answer.tool_calls == [] and tools.invocations == []


# --- the tool-call cap --------------------------------------------------------------------------------------


async def test_the_cap_is_four_tool_calls_even_when_the_model_keeps_asking(settings: Settings) -> None:
    assert settings.llm.max_tool_calls == 4
    agent, model, tools = agent_for(settings, [*[tool_call_message(SEARCH)] * 5, AIMessage(ALPHA_TEXT)])
    answer = await agent.ask("operator")

    assert len(tools.invocations) == 4  # the fifth call never ran
    assert len(answer.tool_calls) == 4 and answer.blocked_tool_calls == 1
    assert answer.metadata["max_tool_calls"] == 4
    assert [c.movie_id for c in answer.citations] == ["alpha-1999-1"]  # the model could still answer afterwards
    refusal = model.prompts[-1][-1]
    assert refusal.type == "tool" and "Tool call limit exceeded" in str(refusal.content)


async def test_calls_requested_in_one_batch_are_cut_at_the_cap(settings: Settings) -> None:
    agent, _, tools = agent_for(settings, [tool_call_message(*[SEARCH] * 6), AIMessage(ALPHA_TEXT)])
    answer = await agent.ask("operator")
    assert len(tools.invocations) == 4
    assert len(answer.tool_calls) == 4 and answer.blocked_tool_calls == 2


async def test_the_cap_comes_from_the_configuration(make_settings: Callable[..., Settings]) -> None:
    cfg = make_settings(LLM__MAX_TOOL_CALLS="2")
    agent, _, tools = agent_for(cfg, [*[tool_call_message(SEARCH)] * 3, AIMessage(ALPHA_TEXT)])
    answer = await agent.ask("operator")
    assert len(tools.invocations) == 2 and answer.blocked_tool_calls == 1
    assert "at most 2 tool calls" in load_prompt(cfg)


async def test_a_model_that_never_stops_calling_tools_is_stopped_with_an_abstention(settings: Settings) -> None:
    agent, _, tools = agent_for(settings, [tool_call_message(SEARCH)])  # repeats forever
    answer = await agent.ask("operator")
    assert len(tools.invocations) == 4
    assert answer.stopped_early and answer.abstained and answer.text == ABSTENTION_TEXT
    assert len(answer.tool_calls) == 4 and answer.blocked_tool_calls >= 1
    assert [f.movie_id for f in answer.retrieved] == ["alpha-1999-1", "beta-2004-2"]


# --- tool errors, forced mode ----------------------------------------------------------------------------


async def test_a_failing_tool_is_recorded_and_the_model_can_still_answer(settings: Settings) -> None:
    agent, model, _ = agent_for(
        settings, [tool_call_message(SEARCH), AIMessage("The index is down.")], Tools([], fail=True)
    )
    answer = await agent.ask("operator")
    (call,) = answer.tool_calls
    assert call.name == "search_movies" and call.result_count == 0
    assert call.error is not None and "make up" in call.error
    assert answer.abstained
    assert "unavailable" in str(model.prompts[-1][-1].content)


async def test_tool_arguments_in_the_record_are_scrubbed_of_secrets(make_settings: Callable[..., Settings]) -> None:
    cfg = make_settings(NEBIUS_API_KEY="fake-test-key-123")
    leaky = ("search_movies", {"query": "plot fake-test-key-123 sk-abcdefghijklmnop"})
    agent, _, _ = agent_for(cfg, [tool_call_message(leaky), AIMessage("none")])
    answer = await agent.ask("operator")
    assert "fake-test-key-123" not in answer.model_dump_json() and "sk-abcdefghijklmnop" not in answer.model_dump_json()


# --- metadata and tracing ---------------------------------------------------------------------------------


async def test_every_answer_carries_the_run_metadata(settings: Settings) -> None:
    agent, _, _ = agent_for(settings, [tool_call_message(SEARCH), AIMessage(ALPHA_TEXT)])
    answer = await agent.ask("operator", mode="dense")
    meta = answer.metadata
    assert {"git_sha", "prompt_version", "chat_model", "embedding_model", "retrieval_mode", "config_hash"} <= set(meta)
    assert meta["prompt_version"] == "system_v1" == settings.agent.prompt_version
    assert meta["chat_model"] == settings.llm.chat_model
    assert meta["embedding_model"] == settings.embeddings.dense_model
    assert meta["config_hash"] == settings.config_hash()
    assert meta["retrieval_mode"] == "dense"
    assert answer.latency_ms > 0


async def test_the_default_retrieval_mode_in_the_metadata_is_the_configured_one(settings: Settings) -> None:
    agent, _, _ = agent_for(settings, [AIMessage("nothing")])
    assert (await agent.ask("q")).metadata["retrieval_mode"] == settings.retrieval.default_mode


async def test_no_secret_reaches_the_answer_object_or_the_trace(make_settings: Callable[..., Settings]) -> None:
    cfg = make_settings(NEBIUS_API_KEY="fake-test-key-123", LANGSMITH_API_KEY="fake-ls-key-456")
    agent, _, _ = agent_for(cfg, [tool_call_message(SEARCH), AIMessage(ALPHA_TEXT)])
    with RunCapture() as capture:
        answer = await agent.ask("operator, and my key is fake-test-key-123")
    for secret in ("fake-test-key-123", "fake-ls-key-456"):
        assert secret not in answer.model_dump_json()
        assert secret not in capture.payloads


async def test_the_run_is_traced_with_the_metadata_when_tracing_is_on(settings: Settings) -> None:
    agent, _, _ = agent_for(settings, [tool_call_message(SEARCH), AIMessage(ALPHA_TEXT)])
    with RunCapture() as capture:
        answer = await agent.ask("operator")
    run = capture.runs["agent.ask"]
    metadata = run["extra"]["metadata"]
    assert metadata["prompt_version"] == "system_v1" and metadata["chat_model"] == settings.llm.chat_model
    assert metadata["config_hash"] == settings.config_hash()
    assert run["outputs"]["citations"] == ["alpha-1999-1"] and run["outputs"]["tool_calls"] == [["search_movies", 2]]
    assert answer.citations


# --- prompt -----------------------------------------------------------------------------------------------


def test_the_prompt_states_the_answer_rules_with_the_configured_limits(settings: Settings) -> None:
    prompt = load_prompt(settings)
    assert "{" not in prompt  # every placeholder was filled
    assert f"at most {settings.llm.max_tool_calls} tool calls" in prompt
    assert f"at most {settings.agent.max_citations} films" in prompt
    for rule in ("only facts that appear in tool results", "Title (Year)", "Wikipedia link", "say so plainly"):
        assert rule.lower() in " ".join(prompt.lower().split())


@pytest.mark.parametrize("version", ["system_v99", "../config", "System_V1", ""])
def test_an_unknown_or_unsafe_prompt_version_is_an_agent_error(settings: Settings, version: str) -> None:
    settings.agent.prompt_version = version
    with pytest.raises(AgentError, match="unknown prompt version"):
        load_prompt(settings)


async def test_an_empty_question_is_rejected(settings: Settings) -> None:
    agent, _, _ = agent_for(settings, [AIMessage("x")])
    with pytest.raises(AgentError, match="empty"):
        await agent.ask("   ")


# --- the MCP server is down ----------------------------------------------------------------------------------


def connect_error() -> BaseExceptionGroup[Exception]:
    return ExceptionGroup("unhandled errors in a TaskGroup", [httpx.ConnectError("All connection attempts failed")])


def test_connection_failures_are_recognised_inside_exception_groups_and_causes() -> None:
    assert is_connection_failure(connect_error())
    assert is_connection_failure(httpx.ReadTimeout("slow"))
    assert is_connection_failure(ConnectionRefusedError())
    wrapped = RuntimeError("wrapper")
    wrapped.__cause__ = httpx.ConnectError("refused")
    assert is_connection_failure(wrapped)
    assert not is_connection_failure(ValueError("a bug"))
    assert not is_connection_failure(ExceptionGroup("g", [ValueError("a bug")]))


async def test_loading_tools_from_a_down_server_names_the_url_and_the_fix(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def down(self: Any, **kwargs: Any) -> None:
        raise connect_error()

    monkeypatch.setattr(agent_module.MultiServerMCPClient, "get_tools", down)
    with pytest.raises(McpUnavailableError) as info:
        await load_mcp_tools(settings)
    assert settings.mcp.url in str(info.value) and "make serve" in str(info.value)


async def test_other_errors_while_loading_tools_are_not_disguised(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def broken(self: Any, **kwargs: Any) -> None:
        raise ValueError("schema bug")

    monkeypatch.setattr(agent_module.MultiServerMCPClient, "get_tools", broken)
    with pytest.raises(ValueError, match="schema bug"):
        await load_mcp_tools(settings)


async def test_asking_with_a_down_server_raises_the_actionable_error(
    make_settings: Callable[..., Settings], monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = make_settings(MCP_URL="http://127.0.0.1:9/mcp")

    async def down(self: Any, **kwargs: Any) -> None:
        raise connect_error()

    monkeypatch.setattr(agent_module.MultiServerMCPClient, "get_tools", down)
    agent = MovieAgent(cfg, model=ScriptedChatModel(replies=[AIMessage("x")]))
    with pytest.raises(McpUnavailableError, match=r"http://127\.0\.0\.1:9/mcp.*make serve"):
        await agent.ask("operator")


async def test_a_server_that_dies_mid_run_is_the_same_error(settings: Settings) -> None:
    def dies(query: str) -> str:
        raise httpx.ConnectError("connection lost")

    tool = StructuredTool.from_function(dies, name="search_movies", description="search")
    agent = MovieAgent(settings, model=ScriptedChatModel(replies=[tool_call_message(SEARCH)]), tools=[tool])
    with pytest.raises(McpUnavailableError):
        await agent.ask("operator")


async def test_unexpected_errors_in_a_run_propagate(settings: Settings) -> None:
    def bug(query: str) -> str:
        raise ZeroDivisionError("bug")

    tool = StructuredTool.from_function(bug, name="search_movies", description="search")
    agent = MovieAgent(settings, model=ScriptedChatModel(replies=[tool_call_message(SEARCH)]), tools=[tool])
    with pytest.raises(ZeroDivisionError):
        await agent.ask("operator")


def test_helpers_cope_with_runs_that_ended_early(settings: Settings) -> None:
    from langchain_core.messages import HumanMessage

    unanswered = tool_call_message(SEARCH)
    records, blocked, films = agent_module.summarise_tool_calls([HumanMessage("q"), unanswered], settings)
    assert (records, blocked, films) == ([], 0, [])
    assert agent_module.final_text([HumanMessage("q")]) == ""
    assert agent_module.final_text([HumanMessage("q"), unanswered]) == ""
    assert agent_module.total_usage([HumanMessage("q"), AIMessage("no usage")]).total_tokens == 0
