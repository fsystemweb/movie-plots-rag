"""MCP server CLI and transports: the HTTP app at /mcp (in-process ASGI) and a stdio subprocess smoke test."""

from __future__ import annotations

import runpy
import sys
import urllib.error
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from fastmcp import Client, FastMCP
from fastmcp.client.transports import StdioTransport, StreamableHttpTransport

from movie_rag.config import Settings
from movie_rag.mcp_server import TOOL_NAMES, build_server
from movie_rag.mcp_server import __main__ as cli

MakeSettings = Callable[..., Settings]
ROOT = Path(__file__).resolve().parents[2]


class Recorder:
    """Records how ``FastMCP.run`` (which blocks until the server stops) was called."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    rec = Recorder()

    def fake_run(_server: FastMCP, **kwargs: Any) -> None:
        rec.calls.append(kwargs)

    monkeypatch.setattr(FastMCP, "run", fake_run)
    return rec


# --- CLI -------------------------------------------------------------------------------------------------------


def test_http_is_the_default_transport_and_host_port_path_come_from_the_configuration(recorder: Recorder) -> None:
    assert cli.main([]) == cli.EXIT_OK
    (call,) = recorder.calls
    assert call["transport"] == "http"
    assert (call["host"], call["port"], call["path"]) == ("127.0.0.1", 8000, "/mcp")


def test_environment_and_flags_override_host_and_port(recorder: Recorder, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP__HOST", "0.0.0.0")
    monkeypatch.setenv("MCP__PORT", "9001")
    assert cli.main([]) == cli.EXIT_OK
    assert (recorder.calls[0]["host"], recorder.calls[0]["port"]) == ("0.0.0.0", 9001)
    assert cli.main(["--host", "10.0.0.5", "--port", "7000"]) == cli.EXIT_OK
    assert (recorder.calls[1]["host"], recorder.calls[1]["port"]) == ("10.0.0.5", 7000)


def test_stdio_transport_takes_no_network_arguments(recorder: Recorder) -> None:
    assert cli.main(["--transport", "stdio"]) == cli.EXIT_OK
    assert recorder.calls == [{"transport": "stdio", "show_banner": False}]


def test_unknown_transport_is_rejected_by_the_parser() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--transport", "carrier-pigeon"])


def test_starting_the_server_configures_tracing(recorder: Recorder, monkeypatch: pytest.MonkeyPatch) -> None:
    configure = MagicMock()
    monkeypatch.setattr(cli, "configure_tracing", configure)
    cli.main([])
    configure.assert_called_once()


def test_an_unreadable_configuration_is_reported_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str], recorder: Recorder
) -> None:
    bad = tmp_path / "config.yaml"
    bad.write_text("mcp: [not, a, mapping]\n")
    monkeypatch.setenv("MOVIE_RAG_CONFIG", str(bad))
    assert cli.main([]) == cli.EXIT_FAILURE
    assert "cannot load the configuration" in capsys.readouterr().err
    assert recorder.calls == []


class FakeResponse:
    def __init__(self, status: int) -> None:
        self.status = status

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def test_healthcheck_probes_the_configured_health_route(
    make_settings: MakeSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    urlopen = MagicMock(return_value=FakeResponse(200))
    monkeypatch.setattr(cli.urllib.request, "urlopen", urlopen)
    settings = make_settings()
    assert cli.healthcheck(settings) is True
    assert urlopen.call_args.args == ("http://127.0.0.1:8000/health",)
    assert urlopen.call_args.kwargs["timeout"] == settings.mcp.healthcheck_timeout_s
    assert cli.healthcheck(settings, port=9001) is True
    assert urlopen.call_args.args == ("http://127.0.0.1:9001/health",)


@pytest.mark.parametrize("failure", [urllib.error.URLError("refused"), ConnectionResetError(), TimeoutError()])
def test_healthcheck_is_false_when_the_server_does_not_answer(
    make_settings: MakeSettings, monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    monkeypatch.setattr(cli.urllib.request, "urlopen", MagicMock(side_effect=failure))
    assert cli.healthcheck(make_settings()) is False


def test_healthcheck_is_false_on_a_non_200_answer(make_settings: MakeSettings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.urllib.request, "urlopen", MagicMock(return_value=FakeResponse(503)))
    assert cli.healthcheck(make_settings()) is False


def test_the_healthcheck_flag_maps_to_the_exit_code(recorder: Recorder, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "healthcheck", MagicMock(side_effect=[True, False]))
    assert cli.main(["--healthcheck"]) == cli.EXIT_OK
    assert cli.main(["--healthcheck"]) == cli.EXIT_FAILURE
    assert recorder.calls == []  # a probe never starts a server


@pytest.mark.filterwarnings("ignore:'movie_rag.mcp_server.__main__' found in sys.modules")
def test_module_entry_point_exits_through_main(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["movie_rag.mcp_server", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("movie_rag.mcp_server", run_name="__main__", alter_sys=True)
    assert exit_info.value.code == 0


# --- transports ------------------------------------------------------------------------------------------------


async def test_the_http_app_serves_the_tools_at_the_configured_path_and_a_health_route(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings()
    app = build_server(settings).http_app(path=settings.mcp.path)

    def asgi_client(**kwargs: Any) -> httpx.AsyncClient:
        kwargs.pop("follow_redirects", None)
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), **kwargs)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
            health = await http.get(settings.mcp.health_path)
            assert health.status_code == 200 and health.json() == {"status": "ok", "server": "movie-rag"}
        transport = StreamableHttpTransport(f"http://test{settings.mcp.path}", httpx_client_factory=asgi_client)
        async with Client(transport) as client:
            assert sorted(t.name for t in await client.list_tools()) == sorted(TOOL_NAMES)


async def test_stdio_smoke_lists_the_four_tools() -> None:
    """Spawn ``python -m movie_rag.mcp_server --transport stdio`` and talk MCP over its pipes."""
    env = {"LANGSMITH_TRACING": "false", "PATH": "", "HOME": str(ROOT)}
    transport = StdioTransport(
        command=sys.executable, args=["-m", "movie_rag.mcp_server", "--transport", "stdio"], env=env, cwd=str(ROOT)
    )
    async with Client(transport, timeout=60) as client:
        tools = await client.list_tools()
    assert sorted(t.name for t in tools) == sorted(TOOL_NAMES)
