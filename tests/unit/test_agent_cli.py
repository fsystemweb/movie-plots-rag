"""``python -m movie_rag.agent`` / ``make ask``: output, exit codes and the credentials hint."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from test_agent import ALPHA_TEXT, SEARCH, Tools

from fakes import ScriptedChatModel, tool_call_message
from movie_rag.agent import MovieAgent
from movie_rag.agent import __main__ as cli
from movie_rag.config import Settings, load_settings
from movie_rag.errors import McpUnavailableError

ROOT = Path(__file__).resolve().parents[2]
HINT = "set NEBIUS_API_KEY — see docs/CREDENTIALS.md"


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings()


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI loads its own settings: keep a developer's ``.env`` (and its key) out of the tests."""
    monkeypatch.setattr(cli, "load_settings", lambda: load_settings(env_file=None))


def scripted_agent(settings: Settings, replies: list[AIMessage], tools: Tools | None = None) -> MovieAgent:
    return MovieAgent(settings, model=ScriptedChatModel(replies=replies), tools=(tools or Tools([])).as_tools())


def test_without_a_key_the_cli_prints_the_hint_and_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["a film about a hotel"]) == cli.EXIT_USAGE == 2
    captured = capsys.readouterr()
    assert captured.err.strip() == HINT
    assert captured.out == ""


def test_an_empty_question_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        cli.main(["   "])
    assert info.value.code == 2
    assert "question is empty" in capsys.readouterr().err


def test_a_missing_question_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        cli.main([])
    assert info.value.code == 2


def test_an_answer_is_printed_with_its_sources_and_a_run_summary(
    settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    agent = scripted_agent(
        settings,
        [tool_call_message(SEARCH), AIMessage(ALPHA_TEXT)],
        Tools([{"movie_id": "alpha-1999-1", "title": "Alpha", "release_year": 1999, "wiki_url": "https://w/Alpha"}]),
    )
    assert cli.main(["an operator overhears a plot"], agent=agent) == cli.EXIT_OK
    captured = capsys.readouterr()
    assert captured.out.startswith(ALPHA_TEXT)
    assert "Sources:\n  - Alpha (1999) - https://w/Alpha" in captured.out
    assert "tool calls: search_movies(1)" in captured.err
    assert f"model {settings.llm.chat_model}" in captured.err and "prompt system_v1" in captured.err


def test_an_abstention_prints_the_message_and_no_sources(
    settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    agent = scripted_agent(settings, [tool_call_message(SEARCH), AIMessage("nothing fits")])
    assert cli.main(["unfindable"], agent=agent) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "No film in the index matches" in out and "Sources:" not in out


def test_json_output_is_the_whole_answer_object(settings: Settings, capsys: pytest.CaptureFixture[str]) -> None:
    agent = scripted_agent(settings, [tool_call_message(SEARCH), AIMessage("nothing fits")])
    assert cli.main(["--json", "unfindable"], agent=agent) == cli.EXIT_OK
    data = json.loads(capsys.readouterr().out)
    assert data["abstained"] is True and data["metadata"]["prompt_version"] == "system_v1"
    assert data["tool_calls"][0]["name"] == "search_movies"


def test_refused_calls_are_shown_in_the_summary(settings: Settings, capsys: pytest.CaptureFixture[str]) -> None:
    agent = scripted_agent(settings, [tool_call_message(*[SEARCH] * 6), AIMessage("done")])
    assert cli.main(["many searches"], agent=agent) == cli.EXIT_OK
    assert "(+2 refused)" in capsys.readouterr().err


def test_the_mode_option_is_passed_to_the_agent(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    async def ask(self: MovieAgent, question: str, *, mode: Any = None) -> Any:
        seen.update(question=question, mode=mode)
        raise McpUnavailableError("http://x/mcp")

    monkeypatch.setattr(MovieAgent, "ask", ask)
    assert cli.main(["--mode", "sparse", "q"], agent=scripted_agent(settings, [AIMessage("x")])) == cli.EXIT_FAILURE
    assert seen == {"question": "q", "mode": "sparse"}


def test_an_unreachable_mcp_server_is_a_plain_message_and_exit_1(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def down(self: MovieAgent, question: str, *, mode: Any = None) -> Any:
        raise McpUnavailableError(self.settings.mcp.url)

    monkeypatch.setattr(MovieAgent, "ask", down)
    assert cli.main(["q"], agent=scripted_agent(settings, [AIMessage("x")])) == cli.EXIT_FAILURE
    err = capsys.readouterr().err
    assert settings.mcp.url in err and "make serve" in err and "Traceback" not in err


def test_an_unreadable_configuration_is_reported_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken() -> Settings:
        raise ValueError("bad yaml")

    monkeypatch.setattr(cli, "load_settings", broken)
    assert cli.main(["q"]) == cli.EXIT_FAILURE
    err = capsys.readouterr().err
    assert "cannot load the configuration" in err and "Traceback" not in err


def test_make_ask_without_a_key_prints_the_hint_and_exits_2() -> None:
    env = {**os.environ, "NEBIUS_API_KEY": "", "LANGSMITH_TRACING": "false"}  # blank = unset, even if .env has a key
    done = subprocess.run(
        ["make", "ask", "Q=a film about a hotel operator"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 2
    assert HINT in done.stderr
    assert "Traceback" not in done.stderr


def test_make_ask_passes_the_question_and_options_through(monkeypatch: pytest.MonkeyPatch) -> None:
    dry = subprocess.run(
        ["make", "-n", "ask", "Q=what film?", "MODE=dense", "JSON=1"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert 'python -m movie_rag.agent --mode dense --json "what film?"' in dry
