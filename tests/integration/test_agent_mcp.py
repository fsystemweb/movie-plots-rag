"""The agent over the real streamable-HTTP transport and ``langchain-mcp-adapters`` (loopback only).

The MCP server runs in this process on a free port, backed by the fixture in an in-memory Qdrant with the fake
embedder; the chat model is scripted. This is the one place where the adapter's real tool results (and their structured
artifacts) feed the citation logic.
"""

from __future__ import annotations

import asyncio
import os
import socket
from collections.abc import AsyncIterator, Callable
from contextlib import suppress

import httpx
import pytest
from langchain_core.messages import AIMessage

from fakes import ScriptedChatModel, tool_call_message
from index_fixture import Index, premise
from movie_rag.agent import MovieAgent, load_mcp_tools
from movie_rag.config import Settings, load_settings
from movie_rag.errors import McpUnavailableError
from movie_rag.mcp_server import TOOL_NAMES, build_server

LIVE_KEY = os.environ.get("NEBIUS_API_KEY")  # captured before the per-test environment scrub

pytestmark = [
    pytest.mark.integration,
    pytest.mark.filterwarnings("ignore:Payload indexes have no effect"),
]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def index() -> Index:
    return Index()


@pytest.fixture
async def mcp_settings(index: Index, make_settings: Callable[..., Settings]) -> AsyncIterator[Settings]:
    port = free_port()
    settings = make_settings(MCP_URL=f"http://127.0.0.1:{port}/mcp")
    server = build_server(index.settings, index.retriever)
    task = asyncio.create_task(
        server.run_async(transport="http", host="127.0.0.1", port=port, path="/mcp", show_banner=False)
    )
    try:
        async with httpx.AsyncClient() as http:
            for _ in range(100):
                with suppress(httpx.TransportError):
                    if (await http.get(f"http://127.0.0.1:{port}{settings.mcp.health_path}")).status_code == 200:
                        break
                await asyncio.sleep(0.1)
            else:
                pytest.fail("the MCP server did not start")
        yield settings
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await task


async def test_the_adapter_loads_the_four_server_tools(mcp_settings: Settings) -> None:
    tools = await load_mcp_tools(mcp_settings)
    assert sorted(t.name for t in tools) == sorted(TOOL_NAMES)


async def test_the_agent_answers_from_the_real_tools_with_citations_from_the_tool_results(
    index: Index, mcp_settings: Settings
) -> None:
    film = index.original
    question = premise(film)
    wrong_year = film.release_year + 1
    reply = (
        f"{film.title} ({film.release_year}) [{film.movie_id}]: matches the premise. "
        f"Imaginary Film ({wrong_year}) [imaginary-{wrong_year}-0] is invented."
    )
    model = ScriptedChatModel(
        replies=[
            tool_call_message(("search_movies", {"query": question, "mode": "dense", "top_k": 3})),
            tool_call_message(("get_movie", {"movie_id": film.movie_id})),
            AIMessage(reply),
        ]
    )
    answer = await MovieAgent(mcp_settings, model=model).ask(question)

    assert [c.name for c in answer.tool_calls] == ["search_movies", "get_movie"]
    assert answer.tool_calls[0].result_count >= 1 and answer.tool_calls[1].result_count == 1
    assert [c.movie_id for c in answer.citations] == [film.movie_id]
    assert answer.citations[0].wiki_url == film.wiki_url
    assert film.movie_id in {f.movie_id for f in answer.retrieved}
    assert not answer.abstained


async def test_the_mode_option_overrides_the_mode_the_model_asked_for(index: Index, mcp_settings: Settings) -> None:
    question = premise(index.original)
    model = ScriptedChatModel(
        replies=[tool_call_message(("search_movies", {"query": question, "mode": "sparse"})), AIMessage("done")]
    )
    answer = await MovieAgent(mcp_settings, model=model).ask(question, mode="dense")
    tool_message = next(m for m in model.prompts[-1] if m.type == "tool")
    assert '"mode":"dense"' in str(tool_message.content).replace(" ", "")
    assert answer.metadata["retrieval_mode"] == "dense"


async def test_a_server_error_comes_back_as_a_tool_error_the_model_can_read(mcp_settings: Settings) -> None:
    model = ScriptedChatModel(
        replies=[tool_call_message(("get_movie", {"movie_id": "no-such-film-0"})), AIMessage("no such film")]
    )
    answer = await MovieAgent(mcp_settings, model=model).ask("a film that is not there")
    assert answer.tool_calls[0].error is not None and "no film" in answer.tool_calls[0].error.lower()
    assert answer.abstained


async def test_a_server_that_is_not_running_is_an_actionable_error(
    make_settings: Callable[..., Settings],
) -> None:
    cfg = make_settings(MCP_URL=f"http://127.0.0.1:{free_port()}/mcp")
    with pytest.raises(McpUnavailableError, match="make serve"):
        await load_mcp_tools(cfg)
    agent = MovieAgent(cfg, model=ScriptedChatModel(replies=[AIMessage("x")]))
    with pytest.raises(McpUnavailableError, match=cfg.mcp.url):
        await agent.ask("anything")


@pytest.mark.live
async def test_live_nebius_model_answers_from_the_real_tools(
    index: Index, mcp_settings: Settings, make_settings: Callable[..., Settings]
) -> None:
    """Real Nebius chat model, real MCP transport, fixture index. Needs NEBIUS_API_KEY (environment or ``.env``)."""
    stored = load_settings().nebius_api_key
    key = LIVE_KEY or (stored.get_secret_value() if stored else None)
    if not key:
        pytest.skip("set NEBIUS_API_KEY to run the live agent test")
    live = make_settings(NEBIUS_API_KEY=key, MCP_URL=mcp_settings.mcp.url)
    answer = await MovieAgent(live).ask(premise(index.original))

    assert answer.text and len(answer.tool_calls) <= live.llm.max_tool_calls
    assert answer.retrieved, "the model never searched"
    assert {c.movie_id for c in answer.citations} <= {f.movie_id for f in answer.retrieved}
    assert answer.metadata["chat_model"] == live.llm.chat_model
