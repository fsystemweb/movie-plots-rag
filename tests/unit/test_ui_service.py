"""The page's logic (no streamlit): retrieval-only through MCP, the agent path, comparison, friendly errors."""

from __future__ import annotations

import logging
from collections.abc import Callable

import httpx
import pytest
from fastmcp.exceptions import ToolError

from index_fixture import Index, premise
from movie_rag.agent import agent as agent_module
from movie_rag.agent.models import RetrievedFilm
from movie_rag.config import Settings
from movie_rag.errors import AgentError, McpUnavailableError, MissingCredentialError, MovieRagError
from movie_rag.observability import Span
from movie_rag.ui.service import (
    Comparison,
    MovieService,
    RetrievalOutcome,
    SearchParams,
    describe_error,
    to_films,
)
from ui_support import DownClient, EmptyReplyClient, FailingAgent, make_service, scripted_agent

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")


@pytest.fixture(scope="module")
def index() -> Index:
    return Index()


@pytest.fixture
def no_llm(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Fail the test if anything builds a chat model; the list records the attempts."""
    attempts: list[str] = []

    def forbidden(settings: Settings) -> None:
        attempts.append("make_chat_model")
        raise AssertionError("the chat model must not be created in retrieval-only mode")

    monkeypatch.setattr(agent_module, "make_chat_model", forbidden)
    return attempts


def params(index: Index, **overrides: object) -> SearchParams:
    base: dict[str, object] = {"mode": "dense", "top_k": index.settings.retrieval.top_k}
    return SearchParams.model_validate({**base, **overrides})


# --- SearchParams ----------------------------------------------------------------------------------------------


def test_tool_arguments_leave_out_unset_filters(index: Index) -> None:
    assert params(index, top_k=3).tool_arguments("a plot") == {"query": "a plot", "mode": "dense", "top_k": 3}
    full = params(index, year_from=1990, year_to=1999, genre="drama", origin="american").tool_arguments("q")
    assert full == {
        "query": "q",
        "mode": "dense",
        "top_k": index.settings.retrieval.top_k,
        "year_from": 1990,
        "year_to": 1999,
        "genre": "drama",
        "origin": "american",
    }


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, ""),
        ({"year_from": 1990, "year_to": 1999}, "Only consider films released between 1990 and 1999."),
        ({"year_from": 1990}, "Only consider films released in or after 1990."),
        ({"year_to": 1999}, "Only consider films released in or before 1999."),
        (
            {"year_to": 1999, "genre": "drama", "origin": "british"},
            "Only consider films released in or before 1999, genre 'drama', origin 'british'.",
        ),
    ],
)
def test_constraints_text_describes_the_filters_for_the_agent(
    index: Index, overrides: dict[str, object], expected: str
) -> None:
    assert params(index, **overrides).constraints_text() == expected


# --- filters ----------------------------------------------------------------------------------------------------


def test_load_filters_returns_the_server_s_lists(index: Index) -> None:
    loaded = make_service(index).load_filters()
    assert loaded.error is None and loaded.options is not None
    assert loaded.options.genres and loaded.options.origins
    assert loaded.options.year_min is not None and loaded.options.year_max is not None
    assert loaded.options.year_min < loaded.options.year_max


def test_load_filters_with_the_server_down_is_an_error_object_not_an_exception(index: Index) -> None:
    loaded = make_service(index, client_factory=DownClient).load_filters()
    assert loaded.options is None and loaded.error is not None
    assert loaded.error.kind == "mcp_down"


# --- retrieval only --------------------------------------------------------------------------------------------


def test_retrieval_only_lists_films_through_the_mcp_tool_without_any_llm(index: Index, no_llm: list[str]) -> None:
    service = make_service(index)
    turn = service.run_turn(premise(index.original), params(index, top_k=5), retrieval_only=True)

    assert turn.error is None and turn.answer is None and turn.outcome is not None
    outcome = turn.outcome
    assert outcome.mode == "dense" and 1 <= len(outcome.films) <= 5 and outcome.latency_ms > 0
    top = outcome.films[0]
    assert top.movie_id == index.original.movie_id and top.title == index.original.title
    assert top.score is not None and top.snippet and top.tool == "search_movies"
    assert no_llm == []


def test_retrieval_only_applies_the_sidebar_filters(index: Index) -> None:
    service = make_service(index)
    unfiltered = service.run_turn("a plot about a secret", params(index, top_k=20), retrieval_only=True).outcome
    assert unfiltered is not None
    genre = next(f.genre for f in unfiltered.films if f.genre)
    filtered = service.run_turn(
        "a plot about a secret", params(index, top_k=20, genre=genre), retrieval_only=True
    ).outcome
    assert filtered is not None and filtered.films
    assert {f.genre for f in filtered.films} == {genre}


def test_retrieval_only_with_no_match_carries_the_server_note(index: Index) -> None:
    turn = make_service(index).run_turn("anything", params(index, year_from=3000), retrieval_only=True)
    assert turn.error is None and turn.outcome is not None
    assert turn.outcome.films == [] and turn.outcome.note and "list_filters" in turn.outcome.note


def test_retrieval_only_with_the_server_down_names_the_url_and_the_fix(index: Index) -> None:
    turn = make_service(index, client_factory=DownClient).run_turn("q", params(index), retrieval_only=True)
    assert turn.outcome is None and turn.error is not None and turn.error.kind == "mcp_down"
    assert index.settings.mcp.url in turn.error.message and "make serve" in turn.error.message


def test_a_tool_error_from_the_server_is_shown_as_is(index: Index) -> None:
    too_many = params(index, top_k=index.settings.mcp.max_top_k + 1)
    turn = make_service(index).run_turn("a plot", too_many, retrieval_only=True)
    assert turn.error is not None and turn.error.kind == "tool"
    assert "top_k" in turn.error.message


def test_a_reply_without_structured_content_is_an_agent_error(index: Index) -> None:
    turn = make_service(index, client_factory=EmptyReplyClient).run_turn("q", params(index), retrieval_only=True)
    assert turn.error is not None and turn.error.kind == "agent" and "search_movies" in turn.error.message


# --- agent path --------------------------------------------------------------------------------------------------


def test_the_agent_path_returns_the_answer_with_citations_and_passes_constraints(index: Index) -> None:
    agent, model, tools = scripted_agent(index.settings)
    service = make_service(index, agent=agent)
    turn = service.run_turn("an operator overhears a plot", params(index, year_from=1990), retrieval_only=False)

    assert turn.error is None and turn.outcome is None and turn.answer is not None
    assert [c.movie_id for c in turn.answer.citations] == ["alpha-1999-1"]
    assert turn.answer.usage.total_tokens > 0 and turn.answer.latency_ms > 0
    assert turn.question == "an operator overhears a plot"  # the page shows what the user typed
    sent = str(model.prompts[0][-1].content)
    assert sent.startswith("an operator overhears a plot") and "released in or after 1990" in sent
    assert tools.invocations[0]["mode"] == "hybrid"  # the fake tool's default; the pin is applied by the MCP loader
    assert turn.trace_url is None  # tracing is off


def test_the_trace_link_comes_from_the_ask_span(index: Index, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Span, "trace_url", lambda self: "https://smith.example/r/123")
    agent, _, _ = scripted_agent(index.settings)
    turn = make_service(index, agent=agent).run_turn("operator", params(index), retrieval_only=False)
    assert turn.trace_url == "https://smith.example/r/123"


def test_without_a_key_the_agent_path_shows_the_credentials_hint_and_the_way_out(index: Index) -> None:
    # The real agent with no key and no injected model: it must fail before touching the model or the network.
    service = make_service(index, client_factory=DownClient)
    turn = service.run_turn("a film", params(index), retrieval_only=False)
    assert turn.answer is None and turn.error is not None and turn.error.kind == "credentials"
    assert "set NEBIUS_API_KEY — see docs/CREDENTIALS.md" in turn.error.message
    assert "Retrieval only" in turn.error.message


def test_an_unreachable_mcp_server_in_agent_mode_names_the_url(index: Index) -> None:
    agent = FailingAgent(McpUnavailableError(index.settings.mcp.url))
    turn = make_service(index, agent=agent).run_turn("q", params(index), retrieval_only=False)
    assert turn.error is not None and turn.error.kind == "mcp_down"
    assert index.settings.mcp.url in turn.error.message and "make serve" in turn.error.message


def test_an_unexpected_failure_is_logged_but_its_text_is_not_shown(
    index: Index, caplog: pytest.LogCaptureFixture
) -> None:
    agent = FailingAgent(ValueError("boom with detail"))
    with caplog.at_level(logging.ERROR):
        turn = make_service(index, agent=agent).run_turn("q", params(index), retrieval_only=False)
    assert turn.error is not None and turn.error.kind == "unexpected"
    assert "ValueError" in turn.error.message and "boom" not in turn.error.message
    assert "unexpected failure" in caplog.text


def test_the_empty_question_error_of_the_agent_is_a_plain_message(index: Index) -> None:
    turn = make_service(index, agent=FailingAgent(AgentError("the question is empty"))).run_turn(
        "q", params(index), retrieval_only=False
    )
    assert turn.error is not None and turn.error.kind == "agent" and turn.error.message == "the question is empty"


# --- compare modes ----------------------------------------------------------------------------------------------


def test_compare_runs_the_three_modes_over_one_session(index: Index) -> None:
    sessions: list[int] = []
    factory = make_service(index)._client_factory

    def counting() -> object:
        sessions.append(1)
        return factory()

    service = MovieService(index.settings, client_factory=counting)  # type: ignore[arg-type]
    result = service.compare(premise(index.original), params(index, top_k=5))

    assert result.error is None and list(result.outcomes) == ["dense", "sparse", "hybrid"]
    assert all(o.mode == mode and o.films for mode, o in result.outcomes.items())
    assert sessions == [1]
    # Hybrid is not meaningful on the in-memory engine (see docs/BACKLOG.md PR-04), so only dense and sparse are
    # compared by content; the shared list must be exactly the intersection of the three rankings.
    for mode in ("dense", "sparse"):
        assert result.outcomes[mode].films[0].movie_id == index.original.movie_id
    ranked = [{f.movie_id for f in o.films} for o in result.outcomes.values()]
    assert set(result.shared_movie_ids()) == set.intersection(*ranked)


def test_compare_with_the_server_down_is_an_error(index: Index) -> None:
    result = make_service(index, client_factory=DownClient).compare("q", params(index))
    assert result.outcomes == {} and result.error is not None and result.error.kind == "mcp_down"
    assert result.shared_movie_ids() == []


def test_shared_movie_ids_keeps_the_order_of_the_first_mode() -> None:
    films = [RetrievedFilm(movie_id=i, title=i.upper()) for i in ("a", "b", "c")]
    comparison = Comparison(
        query="q",
        outcomes={
            "dense": RetrievalOutcome(query="q", mode="dense", films=films),
            "sparse": RetrievalOutcome(query="q", mode="sparse", films=[films[2], films[0]]),
        },
    )
    assert comparison.shared_movie_ids() == ["a", "c"]


# --- error descriptions -------------------------------------------------------------------------------------------


def test_describe_error_covers_every_kind(make_settings: Callable[..., Settings]) -> None:
    cfg = make_settings(NEBIUS_API_KEY="fake-test-key-123")
    assert describe_error(MissingCredentialError("NEBIUS_API_KEY"), cfg).kind == "credentials"
    assert describe_error(McpUnavailableError("http://x/mcp"), cfg).kind == "mcp_down"
    assert describe_error(ToolError("bad top_k"), cfg).message == "bad top_k"
    assert describe_error(MovieRagError("index missing"), cfg).kind == "agent"
    raw = describe_error(httpx.ConnectError("refused"), cfg)  # a connection failure nobody wrapped
    assert raw.kind == "mcp_down" and cfg.mcp.url in raw.message and "make serve" in raw.message


def test_describe_error_never_shows_a_configured_secret(make_settings: Callable[..., Settings]) -> None:
    cfg = make_settings(NEBIUS_API_KEY="fake-test-key-123")
    for error in (ToolError("bad fake-test-key-123"), MovieRagError("worse fake-test-key-123")):
        assert "fake-test-key-123" not in describe_error(error, cfg).message


# --- models and the default client -------------------------------------------------------------------------------


def test_to_films_maps_hits_to_the_shared_film_model(index: Index) -> None:
    from movie_rag.mcp_server.models import SearchResult

    hits = index.retriever().search(premise(index.original), mode="dense", top_k=2)
    films = to_films(SearchResult(query="q", mode="dense", count=len(hits), results=hits))
    assert [f.movie_id for f in films] == [h.movie_id for h in hits]
    assert films[0].score == hits[0].score and films[0].tool == "search_movies"


def test_the_default_client_uses_the_configured_url_and_timeout(index: Index) -> None:
    client = MovieService(index.settings)._default_client()
    assert client.transport.url == index.settings.mcp.url  # type: ignore[attr-defined]
    assert client._session_kwargs["read_timeout_seconds"].total_seconds() == index.settings.ui.mcp_timeout_s


def test_the_ui_config_is_complete(index: Index) -> None:
    ui = index.settings.ui
    assert len(ui.example_questions) == 4 and ui.title and ui.mcp_timeout_s > 0
