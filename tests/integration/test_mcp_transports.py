"""The real HTTP transport: ``python -m movie_rag.mcp_server`` bound to a loopback port, spoken to over streamable HTTP.

Only loopback sockets are used. The embedding models are not loaded: ``list_filters`` needs Qdrant only, and Qdrant is
pointed at a closed loopback port so the "index unavailable" error path is the one exercised end to end.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from movie_rag.mcp_server import TOOL_NAMES

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
STARTUP_TIMEOUT_S = 60


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def server_url() -> Iterator[str]:
    port, dead_qdrant = free_port(), free_port()
    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith(("LANGSMITH", "QDRANT"))},
        "LANGSMITH_TRACING": "false",
        "QDRANT_URL": f"http://127.0.0.1:{dead_qdrant}",
    }
    process = subprocess.Popen(
        [sys.executable, "-m", "movie_rag.mcp_server", "--port", str(port)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while True:
            assert process.poll() is None, "the server exited during startup"
            try:
                if httpx.get(f"{base}/health", timeout=2).status_code == 200:
                    break
            except httpx.TransportError:
                pass
            assert time.monotonic() < deadline, "the server did not start in time"
            time.sleep(0.2)
        yield base
    finally:
        process.terminate()
        process.wait(timeout=15)


async def test_the_http_server_lists_the_tools_at_slash_mcp(server_url: str) -> None:
    async with Client(f"{server_url}/mcp") as client:
        assert sorted(t.name for t in await client.list_tools()) == sorted(TOOL_NAMES)


async def test_an_unreachable_qdrant_is_a_tool_error_over_http(server_url: str) -> None:
    async with Client(f"{server_url}/mcp") as client:
        with pytest.raises(ToolError, match="unavailable"):
            await client.call_tool("list_filters", {})


def test_the_healthcheck_command_succeeds_against_the_running_server(server_url: str) -> None:
    port = server_url.rsplit(":", 1)[1]
    done = subprocess.run(
        [sys.executable, "-m", "movie_rag.mcp_server", "--healthcheck", "--port", port], cwd=ROOT, check=False
    )
    assert done.returncode == 0
    closed = subprocess.run(
        [sys.executable, "-m", "movie_rag.mcp_server", "--healthcheck", "--port", str(free_port())],
        cwd=ROOT,
        check=False,
    )
    assert closed.returncode == 1
