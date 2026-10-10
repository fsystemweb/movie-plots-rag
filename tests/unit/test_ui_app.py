"""The Streamlit page through ``streamlit.testing.v1.AppTest``: real rendering code, fake agent, in-memory MCP."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import streamlit as st
from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest
from test_agent import ALPHA_TEXT, SEARCH

from fakes import tool_call_message
from index_fixture import Index, premise
from movie_rag.agent import agent as agent_module
from movie_rag.errors import McpUnavailableError, MovieRagError
from movie_rag.ui import service as service_module
from movie_rag.ui.service import MovieService
from ui_support import DownClient, FailingAgent, make_service, scripted_agent

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")

APP = Path(__file__).resolve().parents[2] / "src" / "movie_rag" / "ui" / "app.py"
HINT = "set NEBIUS_API_KEY — see docs/CREDENTIALS.md"


@pytest.fixture(scope="module")
def index() -> Index:
    return Index()


@pytest.fixture(autouse=True)
def _fresh_resources() -> Iterator[None]:
    """``st.cache_resource`` is process-wide: forget the service between tests."""
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


@pytest.fixture
def no_llm(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    attempts: list[str] = []

    def forbidden(settings: Any) -> None:
        attempts.append("make_chat_model")
        raise AssertionError("the chat model must not be created in retrieval-only mode")

    monkeypatch.setattr(agent_module, "make_chat_model", forbidden)
    return attempts


def launch(monkeypatch: pytest.MonkeyPatch, service: MovieService, *, mode: str | None = "dense") -> AppTest:
    """The page, wired to ``service`` (the factory the page calls is replaced), already run once.

    The mode defaults to dense: hybrid is only meaningful on a real Qdrant server (see docs/BACKLOG.md, PR-04).
    """
    monkeypatch.setattr(service_module, "build_service", lambda: service)
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    if mode is not None:
        at.sidebar.selectbox(key="mode").set_value(mode)
        at.run()
    return at


def texts(elements: Any) -> list[str]:
    return [e.value for e in elements]


def ask(at: AppTest, question: str) -> AppTest:
    at.chat_input[0].set_value(question)
    return at.run()


# --- first load ---------------------------------------------------------------------------------------------------


def test_the_page_loads_with_the_configured_title_examples_and_sidebar(
    monkeypatch: pytest.MonkeyPatch, index: Index
) -> None:
    at = launch(monkeypatch, make_service(index), mode=None)
    cfg = index.settings

    assert not at.exception
    assert at.title[0].value == cfg.ui.title
    assert [b.label for b in at.button if b.key and b.key.startswith("example:")] == cfg.ui.example_questions
    assert len(cfg.ui.example_questions) == 4
    assert [t.label for t in at.tabs] == ["Chat", "Compare modes"]
    assert at.sidebar.toggle[0].label.startswith("Retrieval only")
    assert at.sidebar.selectbox(key="mode").options == ["dense", "sparse", "hybrid"]
    assert at.sidebar.selectbox(key="mode").value == cfg.retrieval.default_mode
    assert at.sidebar.slider(key="top_k").value == cfg.retrieval.top_k
    assert at.sidebar.slider(key="top_k").max == cfg.mcp.max_top_k


def test_filters_in_the_sidebar_come_from_list_filters(monkeypatch: pytest.MonkeyPatch, index: Index) -> None:
    at = launch(monkeypatch, make_service(index), mode=None)
    options = make_service(index).load_filters().options
    assert options is not None
    assert at.sidebar.selectbox(key="genre").options == ["(any)", *options.genres]
    assert at.sidebar.selectbox(key="origin").options == ["(any)", *options.origins]
    years = at.sidebar.slider(key="years")
    assert (years.min, years.max, years.value) == (
        options.year_min,
        options.year_max,
        (options.year_min, options.year_max),
    )


def test_without_a_key_the_page_starts_in_retrieval_only_mode(monkeypatch: pytest.MonkeyPatch, index: Index) -> None:
    assert index.settings.nebius_api_key is None
    assert launch(monkeypatch, make_service(index), mode=None).sidebar.toggle(key="retrieval_only").value is True


# --- retrieval only -----------------------------------------------------------------------------------------------


def test_retrieval_only_happy_path_lists_films_and_never_calls_an_llm(
    monkeypatch: pytest.MonkeyPatch, index: Index, no_llm: list[str]
) -> None:
    agent, model, _ = scripted_agent(index.settings)
    at = launch(monkeypatch, make_service(index, agent=agent))
    at = ask(at, premise(index.original))

    assert not at.exception and not at.error and not at.info
    assert [m.name for m in at.chat_message] == ["user", "assistant"]
    assert any("No LLM was called" in c for c in texts(at.caption))
    films = next(e for e in at.expander if e.label.startswith("Retrieved films"))
    assert films.proto.expanded is True
    rendered = " ".join(texts(films.markdown))
    assert index.original.title in rendered and "score" in rendered
    assert texts(films.caption)  # the snippets
    assert model.prompts == [] and no_llm == []


def test_an_example_button_runs_that_question(monkeypatch: pytest.MonkeyPatch, index: Index) -> None:
    at = launch(monkeypatch, make_service(index))
    example = index.settings.ui.example_questions[0]
    at.button(key=f"example:{example}").click()
    at.run()
    assert not at.exception
    assert example in texts(at.chat_message[0].markdown)
    assert any("Retrieved films" in e.label for e in at.expander)


def test_the_sidebar_mode_top_k_and_genre_reach_the_search(monkeypatch: pytest.MonkeyPatch, index: Index) -> None:
    at = launch(monkeypatch, make_service(index))
    genre = make_service(index).load_filters().options.genres[0]  # type: ignore[union-attr]
    at.sidebar.selectbox(key="mode").set_value("sparse")
    at.sidebar.slider(key="top_k").set_value(2)
    at.sidebar.selectbox(key="genre").set_value(genre)
    at = ask(at.run(), "a secret plan")

    assert not at.exception
    assert any("sparse mode" in c for c in texts(at.caption))
    films = next(e for e in at.expander if e.label.startswith("Retrieved films"))
    assert len(texts(films.markdown)) <= 2  # at most top_k films
    assert all(genre in line for line in texts(films.markdown))


def test_a_question_without_matches_shows_the_servers_hint(monkeypatch: pytest.MonkeyPatch, index: Index) -> None:
    present = {(r.genre, r.origin) for r in index.records}
    options = make_service(index).load_filters().options
    assert options is not None
    genre, origin = next((g, o) for g in options.genres for o in options.origins if (g, o) not in present)
    at = launch(monkeypatch, make_service(index))
    at.sidebar.selectbox(key="genre").set_value(genre)
    at.sidebar.selectbox(key="origin").set_value(origin)
    at = ask(at.run(), "anything at all")

    assert not at.exception and not at.error
    (info,) = at.info
    assert "No film matched" in info.value
    assert any(e.label == "Retrieved films (0)" for e in at.expander)


# --- agent --------------------------------------------------------------------------------------------------------


def test_the_agent_path_shows_answer_sources_steps_and_metrics(monkeypatch: pytest.MonkeyPatch, index: Index) -> None:
    agent, model, tools = scripted_agent(index.settings)
    at = launch(monkeypatch, make_service(index, agent=agent))
    at.sidebar.toggle(key="retrieval_only").set_value(False)
    at = ask(at.run(), "an operator overhears a murder plot")

    assert not at.exception and not at.error and not at.info
    assistant = at.chat_message[1]
    assert ALPHA_TEXT.split(" - ")[0] in " ".join(texts(assistant.markdown))
    sources = [m for m in texts(assistant.markdown) if m.startswith("- [Alpha (1999)]")]
    assert sources == ["- [Alpha (1999)](https://en.wikipedia.org/wiki/Alpha)"]

    films = next(e for e in at.expander if e.label == "Retrieved films (2)")
    assert "Alpha (1999)" in " ".join(texts(films.markdown)) and "Beta (2004)" in " ".join(texts(films.markdown))
    steps = next(e for e in at.expander if e.label.startswith("Agent steps (1 tool call"))
    assert "search_movies" in texts(steps.markdown)[0] and "2 film(s)" in texts(steps.markdown)[0]

    metrics = texts(assistant.caption)[-1]
    assert "121 tokens" in metrics and index.settings.llm.chat_model in metrics and "tracing off" in metrics
    assert tools.invocations and "operator overhears" in str(model.prompts[0][-1].content)


def test_the_agent_path_links_the_trace_when_there_is_one(monkeypatch: pytest.MonkeyPatch, index: Index) -> None:
    from movie_rag.observability import Span

    monkeypatch.setattr(Span, "trace_url", lambda self: "https://smith.example/r/9")
    agent, _, _ = scripted_agent(index.settings)
    at = launch(monkeypatch, make_service(index, agent=agent))
    at.sidebar.toggle(key="retrieval_only").set_value(False)
    at = ask(at.run(), "operator")
    assert "[LangSmith trace](https://smith.example/r/9)" in texts(at.chat_message[1].caption)[-1]


def test_an_abstention_and_refused_calls_are_visible(monkeypatch: pytest.MonkeyPatch, index: Index) -> None:
    agent, _, _ = scripted_agent(index.settings, [*[tool_call_message(SEARCH)] * 5, AIMessage("No luck, sorry.")])
    at = launch(monkeypatch, make_service(index, agent=agent))
    at.sidebar.toggle(key="retrieval_only").set_value(False)
    at = ask(at.run(), "operator")
    assert not at.exception
    assistant_text = " ".join(texts(at.chat_message[1].markdown))
    assert "No film in the index matches" in assistant_text
    assert "1 further call(s) were refused" in assistant_text


def test_the_model_answering_without_tools_is_explained(monkeypatch: pytest.MonkeyPatch, index: Index) -> None:
    agent, _, _ = scripted_agent(index.settings, [AIMessage("From memory.")])
    at = launch(monkeypatch, make_service(index, agent=agent))
    at.sidebar.toggle(key="retrieval_only").set_value(False)
    at = ask(at.run(), "operator")
    assert "The model answered without calling a tool." in " ".join(texts(at.chat_message[1].markdown))


# --- friendly errors ----------------------------------------------------------------------------------------------


def test_agent_mode_without_a_key_shows_the_hint_not_a_traceback(monkeypatch: pytest.MonkeyPatch, index: Index) -> None:
    at = launch(monkeypatch, make_service(index))  # the real MovieAgent, no key, no injected model
    at.sidebar.toggle(key="retrieval_only").set_value(False)
    at = ask(at.run(), "a film about a lighthouse")

    assert not at.exception and not at.error
    (info,) = at.info
    assert HINT in info.value and "Retrieval only" in info.value
    assert [m.name for m in at.chat_message] == ["user", "assistant"]


def test_the_mcp_server_being_down_names_the_url_and_make_serve(monkeypatch: pytest.MonkeyPatch, index: Index) -> None:
    at = launch(monkeypatch, make_service(index, client_factory=DownClient))
    assert not at.exception
    (warning,) = at.sidebar.warning  # the filters could not be loaded
    assert "Filters unavailable" in warning.value and "make serve" in warning.value

    at = ask(at, "a film")
    assert not at.exception
    (error,) = at.error
    assert index.settings.mcp.url in error.value and "make serve" in error.value


def test_an_agent_side_mcp_failure_is_the_same_friendly_error(monkeypatch: pytest.MonkeyPatch, index: Index) -> None:
    agent = FailingAgent(McpUnavailableError(index.settings.mcp.url))
    at = launch(monkeypatch, make_service(index, agent=agent))
    at.sidebar.toggle(key="retrieval_only").set_value(False)
    at = ask(at.run(), "a film")
    (error,) = at.error
    assert index.settings.mcp.url in error.value and "make serve" in error.value
    assert agent.questions == ["a film"]


def test_an_unexpected_failure_shows_a_generic_message(monkeypatch: pytest.MonkeyPatch, index: Index) -> None:
    at = launch(monkeypatch, make_service(index, agent=FailingAgent(RuntimeError("secret internals"))))
    at.sidebar.toggle(key="retrieval_only").set_value(False)
    at = ask(at.run(), "a film")
    (error,) = at.error
    assert "RuntimeError" in error.value and "secret internals" not in error.value


def test_no_secret_is_rendered(monkeypatch: pytest.MonkeyPatch, index: Index) -> None:
    monkeypatch.setenv("NEBIUS_API_KEY", "fake-test-key-123")
    settings = service_module.load_settings(env_file=None)
    agent = FailingAgent(MovieRagError("the provider rejected fake-test-key-123"))
    service = MovieService(settings, agent=agent, client_factory=make_service(index)._client_factory)
    at = launch(monkeypatch, service)
    assert at.sidebar.toggle(key="retrieval_only").value is False  # a key is configured: agent mode is the default
    at = ask(at, "a film")

    (error,) = at.error
    assert error.value == "the provider rejected ***"
    everything = [e.value for group in (at.markdown, at.caption, at.info, at.warning, at.text) for e in group]
    assert not any("fake-test-key-123" in str(v) for v in everything)


# --- compare modes ------------------------------------------------------------------------------------------------


def test_compare_modes_shows_the_three_rankings(
    monkeypatch: pytest.MonkeyPatch, index: Index, no_llm: list[str]
) -> None:
    at = launch(monkeypatch, make_service(index))
    assert at.text_input(key="compare_query").value == index.settings.retrieval.demo_query
    at.text_input(key="compare_query").set_value(premise(index.original))
    at.button(key="compare_button").click()
    at.run()

    assert not at.exception and not at.error
    assert [s.value for s in at.subheader] == ["dense", "sparse", "hybrid"]
    assert any("appear in all three modes" in c for c in texts(at.caption))
    assert any(index.original.title in m for m in texts(at.markdown))
    assert no_llm == []


def test_compare_modes_with_the_server_down_is_a_friendly_error(monkeypatch: pytest.MonkeyPatch, index: Index) -> None:
    at = launch(monkeypatch, make_service(index, client_factory=DownClient))
    at.button(key="compare_button").click()
    at.run()
    assert not at.exception
    (error,) = at.error
    assert "make serve" in error.value
