"""The page against a real streamable-HTTP MCP server on loopback, and a headless ``streamlit run`` smoke test.

The MCP server runs in a background thread (own event loop) on a free port, backed by the fixture in an in-memory
Qdrant with the fake embedder. No credentials, no external network.
"""

from __future__ import annotations

import asyncio
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import suppress
from pathlib import Path

import httpx
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from index_fixture import Index, premise
from movie_rag.config import Settings
from movie_rag.mcp_server import build_server
from movie_rag.ui.service import MovieService, SearchParams

pytestmark = [
    pytest.mark.integration,
    pytest.mark.filterwarnings("ignore:Payload indexes have no effect"),
]

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "src" / "movie_rag" / "ui" / "app.py"
STARTUP_S = 60


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for(url: str, ok: Callable[[httpx.Response], bool], *, seconds: float = STARTUP_S) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        with suppress(httpx.TransportError):
            if ok(httpx.get(url, timeout=2)):
                return True
        time.sleep(0.2)
    return False


@pytest.fixture(scope="module")
def index() -> Index:
    return Index()


@pytest.fixture
def mcp_settings(index: Index, make_settings: Callable[..., Settings]) -> Iterator[Settings]:
    port = free_port()
    settings = make_settings(MCP_URL=f"http://127.0.0.1:{port}/mcp")
    server = build_server(index.settings, index.retriever)
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    served = asyncio.run_coroutine_threadsafe(
        server.run_async(transport="http", host="127.0.0.1", port=port, path="/mcp", show_banner=False), loop
    )
    try:
        assert wait_for(f"http://127.0.0.1:{port}{settings.mcp.health_path}", lambda r: r.status_code == 200)
        yield settings
    finally:
        served.cancel()
        with suppress(Exception):
            served.result(timeout=10)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=10)


def test_the_service_reaches_the_server_over_http_without_an_llm(index: Index, mcp_settings: Settings) -> None:
    service = MovieService(mcp_settings)
    params = SearchParams(mode="dense", top_k=3)

    loaded = service.load_filters()
    assert loaded.error is None and loaded.options is not None and loaded.options.genres

    turn = service.run_turn(premise(index.original), params, retrieval_only=True)
    assert turn.error is None and turn.outcome is not None
    assert turn.outcome.films[0].movie_id == index.original.movie_id

    comparison = service.compare(premise(index.original), params)
    assert comparison.error is None and list(comparison.outcomes) == ["dense", "sparse", "hybrid"]


def test_a_closed_port_is_the_friendly_error_with_the_url(make_settings: Callable[..., Settings]) -> None:
    url = f"http://127.0.0.1:{free_port()}/mcp"
    turn = MovieService(make_settings(MCP_URL=url)).run_turn(
        "a film", SearchParams(mode="dense", top_k=3), retrieval_only=True
    )
    assert turn.error is not None and turn.error.kind == "mcp_down"
    assert url in turn.error.message and "make serve" in turn.error.message


def test_the_unpatched_page_works_against_the_real_server(
    monkeypatch: pytest.MonkeyPatch, index: Index, mcp_settings: Settings
) -> None:
    """No injected service: ``build_service()`` reads ``MCP_URL`` like ``make ui`` does."""
    monkeypatch.setenv("MCP_URL", mcp_settings.mcp.url)
    st.cache_resource.clear()
    try:
        at = AppTest.from_file(str(APP), default_timeout=60).run()
        at.sidebar.toggle(key="retrieval_only").set_value(True)
        at.sidebar.selectbox(key="mode").set_value("dense")
        at.run()
        at.chat_input[0].set_value(premise(index.original))
        at.run()
    finally:
        st.cache_resource.clear()
    assert not at.exception and not at.error
    films = next(e for e in at.expander if e.label.startswith("Retrieved films"))
    assert index.original.title in " ".join(m.value for m in films.markdown)


def test_streamlit_starts_headless_and_reports_healthy() -> None:
    port = free_port()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(APP),
            "--server.headless",
            "true",
            "--server.port",
            str(port),
            "--server.address",
            "127.0.0.1",
            "--browser.gatherUsageStats",
            "false",
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert wait_for(
            f"http://127.0.0.1:{port}/_stcore/health", lambda r: r.status_code == 200 and r.text.strip() == "ok"
        )
    finally:
        process.terminate()
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=15)
        if process.poll() is None:
            process.kill()
