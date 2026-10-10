"""Contract tests for the MCP tools, through an in-memory FastMCP client (no sockets, no models, no Qdrant service).

The retriever runs on ``QdrantClient(":memory:")`` with the fixture ingested through the deterministic fake embedder.
qdrant-client 1.15.1's in-memory engine ignores prefetch filters in ``query_points_groups`` (see
tests/unit/test_retrieval.py), so these tests exercise dense and sparse ranking and use hybrid only for the shape of
the answer; hybrid correctness is covered by tests/integration/test_retrieval_qdrant.py.
"""

from __future__ import annotations

import ast
import json
import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from fakes import FakeEmbedder, RunCapture
from movie_rag import mcp_server
from movie_rag.config import Settings, load_settings
from movie_rag.ingest.clean import MovieRecord, clean_csv
from movie_rag.ingest.index import ensure_collection
from movie_rag.ingest.pipeline import ingest_csv, write_records
from movie_rag.mcp_server import TOOL_NAMES, build_server
from movie_rag.retrieval import Retriever

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "movies_sample.csv"
SNAPSHOT = ROOT / "tests" / "fixtures" / "mcp_tools_schema.json"
PACKAGE = ROOT / "src" / "movie_rag" / "mcp_server"
LLM_PACKAGES = {
    "langchain",
    "langchain_core",
    "langchain_openai",
    "langchain_mcp_adapters",
    "langgraph",
    "openai",
    "anthropic",
    "ragas",
    "litellm",
    "transformers",
}


class Index:
    """The fixture in an in-memory collection plus one remake, so that a title can be ambiguous."""

    def __init__(self) -> None:
        self.settings = load_settings(env_file=None)
        self.client = QdrantClient(":memory:")
        ingest_csv(self.settings, FIXTURE, client=self.client, embedder=FakeEmbedder(), recreate=True)
        records, _ = clean_csv(FIXTURE, self.settings.ingest.min_plot_words)
        self.records = records
        self.original = records[0]
        self.remake = self.original.model_copy(
            update={"movie_id": f"{self.original.movie_id}-remake", "release_year": self.original.release_year + 30}
        )
        ensure_collection(self.client, self.settings)
        write_records(self.client, FakeEmbedder(), self.settings, [self.remake])

    def retriever(self) -> Retriever:
        return Retriever(self.settings, client=self.client, embedder=FakeEmbedder())


@pytest.fixture(scope="module")
def index() -> Index:
    return Index()


@pytest.fixture
async def client(index: Index) -> AsyncIterator[Client[Any]]:
    async with Client(build_server(index.settings, index.retriever)) as c:
        yield c


def premise(record: MovieRecord) -> str:
    return ". ".join(record.plot.split(". ")[:2])


def unique_title_record(index: Index) -> MovieRecord:
    return next(r for r in index.records if r.title != index.original.title and r.genre and r.origin)


async def call(client: Client[Any], tool: str, **arguments: Any) -> dict[str, Any]:
    result = await client.call_tool(tool, arguments)
    assert result.structured_content is not None
    return dict(result.structured_content)


async def error_of(client: Client[Any], tool: str, **arguments: Any) -> str:
    with pytest.raises(ToolError) as info:
        await client.call_tool(tool, arguments)
    return str(info.value)


# --- the tool list ---------------------------------------------------------------------------------------------


async def test_exactly_the_four_documented_tools_are_exposed(client: Client[Any]) -> None:
    tools = await client.list_tools()
    assert (
        sorted(t.name for t in tools)
        == sorted(TOOL_NAMES)
        == ["find_similar", "get_movie", "list_filters", "search_movies"]
    )
    for tool in tools:
        assert tool.description and len(tool.description) > 80, tool.name  # LLM-oriented docstring
        assert tool.annotations is not None and tool.annotations.readOnlyHint is True


async def test_every_argument_is_described_for_the_model(client: Client[Any]) -> None:
    for tool in await client.list_tools():
        for name, prop in tool.inputSchema.get("properties", {}).items():
            assert prop.get("description"), f"{tool.name}.{name} has no description"


async def test_tool_schemas_match_the_committed_snapshot(client: Client[Any]) -> None:
    """The tool contract (names, descriptions, input and output schemas) is committed; changing it is deliberate.

    Regenerate with ``UPDATE_MCP_SNAPSHOT=1 uv run pytest tests/unit/test_mcp_server.py -k snapshot``, review the diff.
    """
    tools = sorted(await client.list_tools(), key=lambda t: t.name)
    current = [
        {"name": t.name, "description": t.description, "inputSchema": t.inputSchema, "outputSchema": t.outputSchema}
        for t in tools
    ]
    if os.environ.get("UPDATE_MCP_SNAPSHOT"):
        SNAPSHOT.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
    assert json.loads(SNAPSHOT.read_text()) == json.loads(json.dumps(current))


def test_the_server_package_imports_no_llm_library() -> None:
    """The MCP server must stay a pure retrieval service: no LangChain, OpenAI or other model clients."""
    files = sorted(PACKAGE.rglob("*.py"))
    assert len(files) >= 4
    offenders: list[str] = []
    for path in files:
        for node in ast.walk(ast.parse(path.read_text())):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            offenders += [f"{path.name}: {n}" for n in names if n.split(".")[0] in LLM_PACKAGES]
    assert offenders == []


def test_the_server_is_importable_without_llm_modules_loaded() -> None:
    import subprocess
    import sys

    code = (
        "import sys, movie_rag.mcp_server;"
        f"bad = sorted(m for m in sys.modules if m.split('.')[0] in {sorted(LLM_PACKAGES)!r});"
        "print(bad); sys.exit(1 if bad else 0)"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stdout + done.stderr


# --- search_movies ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["dense", "sparse"])
async def test_search_finds_a_film_by_its_premise(index: Index, client: Client[Any], mode: str) -> None:
    target = index.records[3]
    out = await call(client, "search_movies", query=premise(target), mode=mode, top_k=3)
    assert out["mode"] == mode and out["count"] == len(out["results"]) <= 3
    top = out["results"][0]
    assert top["movie_id"] == target.movie_id
    assert (top["title"], top["release_year"], top["wiki_url"]) == (target.title, target.release_year, target.wiki_url)
    assert top["snippet"] and len(top["snippet"]) <= index.settings.retrieval.snippet_max_chars
    assert out["note"] is None


async def test_search_defaults_come_from_the_configuration(index: Index, client: Client[Any]) -> None:
    out = await call(client, "search_movies", query="a heist at a casino")
    assert out["mode"] == index.settings.retrieval.default_mode == "hybrid"
    assert out["count"] == index.settings.retrieval.top_k
    films = [h["movie_id"] for h in out["results"]]
    assert len(films) == len(set(films))


async def test_search_applies_year_genre_and_origin_filters(index: Index, client: Client[Any]) -> None:
    target = unique_title_record(index)
    assert target.genre is not None and target.origin is not None
    out = await call(
        client,
        "search_movies",
        query=premise(target),
        mode="dense",
        top_k=20,
        year_from=target.release_year,
        year_to=target.release_year,
        genre=target.genre.upper(),
        origin=target.origin,
    )
    assert out["results"] and out["results"][0]["movie_id"] == target.movie_id
    assert {h["release_year"] for h in out["results"]} == {target.release_year}
    assert {h["genre"] for h in out["results"]} == {target.genre}
    assert {h["origin"] for h in out["results"]} == {target.origin}


async def test_search_with_no_match_returns_an_empty_list_and_a_hint(client: Client[Any]) -> None:
    out = await call(client, "search_movies", query="anything at all", mode="dense", genre="no-such-genre")
    assert out["count"] == 0 and out["results"] == []
    assert "list_filters" in out["note"]


async def test_search_trims_the_query(client: Client[Any]) -> None:
    out = await call(client, "search_movies", query="   a heist at a casino   ", mode="dense", top_k=1)
    assert out["query"] == "a heist at a casino"


@pytest.mark.parametrize(
    ("arguments", "fragment"),
    [
        ({"query": "x"}, "at least"),
        ({"query": "heist", "top_k": 0}, "top_k"),
        ({"query": "heist", "top_k": 10_000}, "top_k"),
        ({"query": "heist", "mode": "fuzzy"}, "mode"),
        ({"query": "heist", "year_from": 2000, "year_to": 1990}, "year_from"),
        ({"query": "heist", "year_from": "soon"}, "year_from"),
        ({"query": "   "}, "empty"),
        ({"query": "x" * 5000}, "query"),
        ({}, "query"),
    ],
)
async def test_search_rejects_bad_input_with_a_tool_error(
    client: Client[Any], arguments: dict[str, Any], fragment: str
) -> None:
    message = await error_of(client, "search_movies", **arguments)
    assert fragment in message
    assert "Traceback" not in message


# --- get_movie -------------------------------------------------------------------------------------------------


async def test_get_movie_by_id_returns_metadata_and_the_full_plot(index: Index, client: Client[Any]) -> None:
    record = index.records[5]
    out = await call(client, "get_movie", movie_id=record.movie_id)
    assert out["movie_id"] == record.movie_id and out["plot"] == record.plot
    assert (out["title"], out["release_year"], out["director"], out["genre"], out["origin"]) == (
        record.title,
        record.release_year,
        record.director,
        record.genre,
        record.origin,
    )
    assert out["wiki_url"] == record.wiki_url and out["n_chunks"] >= 1


async def test_get_movie_by_exact_title(index: Index, client: Client[Any]) -> None:
    record = unique_title_record(index)
    out = await call(client, "get_movie", title=f"  {record.title} ")
    assert out["movie_id"] == record.movie_id and out["plot"] == record.plot


async def test_get_movie_by_title_and_year_picks_the_right_remake(index: Index, client: Client[Any]) -> None:
    out = await call(client, "get_movie", title=index.original.title, year=index.remake.release_year)
    assert out["movie_id"] == index.remake.movie_id


async def test_an_ambiguous_title_lists_the_candidates(index: Index, client: Client[Any]) -> None:
    message = await error_of(client, "get_movie", title=index.original.title)
    assert index.original.movie_id in message and index.remake.movie_id in message
    assert "movie_id" in message and "year" in message
    assert message.index(index.original.movie_id) < message.index(index.remake.movie_id)  # oldest first


@pytest.mark.parametrize(
    ("arguments", "fragment"),
    [
        ({}, "exactly one of movie_id or title"),
        ({"movie_id": "a", "title": "b"}, "exactly one of movie_id or title"),
        ({"movie_id": "  "}, "exactly one of movie_id or title"),
        ({"movie_id": "a", "year": 1990}, "year can only be combined with title"),
        ({"movie_id": "no-such-film-1900-0"}, "no film with movie_id 'no-such-film-1900-0'"),
        ({"title": "No Such Film"}, "no film titled 'No Such Film'"),
        ({"title": "No Such Film", "year": 1999}, "from 1999"),
    ],
)
async def test_get_movie_reports_bad_input_and_unknown_films(
    client: Client[Any], arguments: dict[str, Any], fragment: str
) -> None:
    assert fragment in await error_of(client, "get_movie", **arguments)


async def test_an_unknown_title_suggests_search(client: Client[Any]) -> None:
    assert "search_movies" in await error_of(client, "get_movie", title="No Such Film")


# --- find_similar ----------------------------------------------------------------------------------------------


async def test_find_similar_excludes_the_input_and_ranks_by_similarity(index: Index, client: Client[Any]) -> None:
    record = index.records[7]
    out = await call(client, "find_similar", movie_id=record.movie_id, top_k=5)
    ids = [h["movie_id"] for h in out["results"]]
    assert out["movie_id"] == record.movie_id and out["count"] == len(ids) == 5
    assert record.movie_id not in ids and len(set(ids)) == 5
    scores = [h["score"] for h in out["results"]]
    assert scores == sorted(scores, reverse=True)


async def test_find_similar_finds_the_remake_first(index: Index, client: Client[Any]) -> None:
    out = await call(client, "find_similar", movie_id=index.original.movie_id, top_k=3)
    assert out["results"][0]["movie_id"] == index.remake.movie_id  # identical plot, different film


async def test_find_similar_defaults_top_k_from_the_configuration(index: Index, client: Client[Any]) -> None:
    out = await call(client, "find_similar", movie_id=index.records[7].movie_id)
    assert out["count"] == index.settings.retrieval.top_k


@pytest.mark.parametrize(
    ("arguments", "fragment"),
    [
        ({"movie_id": "no-such-film-1900-0"}, "no film with movie_id"),
        ({"movie_id": ""}, "movie_id"),
        ({"movie_id": "x", "top_k": 0}, "top_k"),
        ({"movie_id": "x", "top_k": 1000}, "top_k"),
        ({}, "movie_id"),
    ],
)
async def test_find_similar_rejects_bad_input_and_unknown_films(
    client: Client[Any], arguments: dict[str, Any], fragment: str
) -> None:
    assert fragment in await error_of(client, "find_similar", **arguments)


# --- list_filters ----------------------------------------------------------------------------------------------


async def test_list_filters_returns_values_usable_as_filters(index: Index, client: Client[Any]) -> None:
    out = await call(client, "list_filters")
    genres = {r.genre for r in index.records if r.genre}
    origins = {r.origin for r in index.records if r.origin}
    years = [r.release_year for r in index.records]
    assert set(out["genres"]) == genres and set(out["origins"]) == origins
    assert (out["year_min"], out["year_max"]) == (min(years), max([*years, index.remake.release_year]))
    assert out["truncated"] is False
    # every advertised value works as a filter
    hit = await call(client, "search_movies", query="a film", mode="dense", genre=out["genres"][0])
    assert hit["count"] > 0


async def test_list_filters_is_capped_by_the_configuration(index: Index) -> None:
    settings = index.settings.model_copy(deep=True)
    settings.mcp.max_filter_values = 2
    async with Client(build_server(settings, index.retriever)) as c:
        out = await call(c, "list_filters")
    assert len(out["genres"]) == len(out["origins"]) == 2 and out["truncated"] is True


# --- failures of the index -------------------------------------------------------------------------------------


def broken_server(settings: Settings, error: Exception) -> FastMCP:
    qdrant = MagicMock(spec=QdrantClient)
    for method in ("query_points_groups", "retrieve", "scroll", "facet"):
        getattr(qdrant, method).side_effect = error
    return build_server(settings, lambda: Retriever(settings, client=qdrant, embedder=FakeEmbedder()))


CALLS: list[tuple[str, dict[str, Any]]] = [
    ("search_movies", {"query": "a heist"}),
    ("get_movie", {"movie_id": "x-1999-1"}),
    ("get_movie", {"title": "X"}),
    ("find_similar", {"movie_id": "x-1999-1"}),
    ("list_filters", {}),
]


@pytest.mark.parametrize(("tool", "arguments"), CALLS)
@pytest.mark.parametrize(
    "error",
    [ResponseHandlingException(ConnectionError("refused")), ConnectionError("down"), TimeoutError("slow")],
)
async def test_an_unreachable_qdrant_becomes_a_clear_tool_error(
    index: Index, tool: str, arguments: dict[str, Any], error: Exception
) -> None:
    async with Client(broken_server(index.settings, error)) as c:
        message = await error_of(c, tool, **arguments)
    assert "unavailable" in message and "make up" in message
    assert "refused" not in message and "Traceback" not in message


@pytest.mark.parametrize(("tool", "arguments"), CALLS)
async def test_a_missing_collection_tells_the_caller_to_ingest(
    index: Index, tool: str, arguments: dict[str, Any]
) -> None:
    async with Client(broken_server(index.settings, UnexpectedResponse(404, "Not Found", b"", None))) as c:
        message = await error_of(c, tool, **arguments)
    assert "make ingest" in message and index.settings.qdrant.collection in message


async def test_any_other_qdrant_error_is_reported_without_details(index: Index) -> None:
    error = UnexpectedResponse(500, "Internal Server Error", b"secret server detail", None)
    async with Client(broken_server(index.settings, error)) as c:
        message = await error_of(c, "search_movies", query="a heist")
    assert "returned an error" in message and "secret server detail" not in message


async def test_an_unexpected_bug_is_masked(index: Index) -> None:
    async with Client(broken_server(index.settings, RuntimeError("db password is hunter2"))) as c:
        message = await error_of(c, "search_movies", query="a heist")
    assert "hunter2" not in message


# --- laziness and wiring ---------------------------------------------------------------------------------------


async def test_the_retriever_is_created_once_and_only_when_a_tool_runs(index: Index) -> None:
    created: list[Retriever] = []

    def factory() -> Retriever:
        created.append(index.retriever())
        return created[-1]

    async with Client(build_server(index.settings, factory)) as c:
        await c.list_tools()
        assert created == []
        await call(c, "search_movies", query="a heist", mode="dense")
        await call(c, "list_filters")
    assert len(created) == 1


def test_the_default_server_builds_a_real_retriever_lazily(
    make_settings: Callable[..., Settings], monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_settings()
    server = build_server(settings)
    assert server.name == mcp_server.SERVER_NAME == "movie-rag"
    assert isinstance(server, FastMCP)


# --- tracing ---------------------------------------------------------------------------------------------------


@asynccontextmanager
async def traced(index: Index, settings: Settings | None = None) -> AsyncIterator[tuple[RunCapture, Client[Any]]]:
    """A client whose server task starts *inside* the capture, so the tracing context reaches the tool code."""
    with RunCapture() as capture:
        async with Client(build_server(settings or index.settings, index.retriever)) as c:
            yield capture, c


async def test_search_opens_a_tool_span_with_a_nested_retriever_span(index: Index) -> None:
    async with traced(index) as (capture, client):
        await call(client, "search_movies", query="a heist", mode="dense", top_k=3, year_from=1950, genre="Drama")
    tool, retriever = capture.runs["mcp.search_movies"], capture.runs["retriever.search"]
    assert tool["run_type"] == "tool" and retriever["run_type"] == "retriever"
    assert retriever["parent_run_id"] == tool["id"]  # the span context crosses the worker thread
    meta = tool["extra"]["metadata"]
    assert (meta["tool"], meta["retrieval_mode"], meta["top_k"], meta["year_from"]) == (
        "search_movies",
        "dense",
        3,
        1950,
    )
    assert (meta["year_to"], meta["genre"], meta["origin"]) == (None, "Drama", None)
    assert meta["config_hash"] == index.settings.config_hash() and meta["git_sha"]
    assert tool["inputs"] == {"query": "a heist"}
    assert tool["outputs"]["result_count"] == 3 and tool["outputs"]["latency_ms"] >= 0
    assert retriever["outputs"]["result_count"] == 3 and len(retriever["outputs"]["movie_ids"]) == 3
    assert retriever["extra"]["metadata"]["filters"] == {"year_from": 1950, "genre": "drama"}


@pytest.mark.parametrize(
    ("tool", "arguments", "span_name"),
    [
        ("get_movie", {"title": "No Such Film"}, "mcp.get_movie"),
        ("find_similar", {"movie_id": "no-such-film-1900-0"}, "mcp.find_similar"),
        ("list_filters", {}, "mcp.list_filters"),
        ("search_movies", {"query": "   "}, "mcp.search_movies"),
    ],
)
async def test_every_tool_call_is_traced_including_failures(
    index: Index, tool: str, arguments: dict[str, Any], span_name: str
) -> None:
    async with traced(index) as (capture, client):
        with suppress(ToolError):
            await client.call_tool(tool, arguments)
    run = capture.runs[span_name]
    assert run["run_type"] == "tool" and run["extra"]["metadata"]["tool"] == tool
    assert run["outputs"]["latency_ms"] >= 0


async def test_get_movie_find_similar_and_list_filters_spans_record_their_results(index: Index) -> None:
    record = index.records[2]
    async with traced(index) as (capture, client):
        await call(client, "get_movie", movie_id=record.movie_id)
        await call(client, "find_similar", movie_id=record.movie_id, top_k=2)
        await call(client, "list_filters")
    assert capture.runs["mcp.get_movie"]["outputs"]["movie_id"] == record.movie_id
    assert capture.runs["mcp.find_similar"]["outputs"]["result_count"] == 2
    assert capture.runs["retriever.find_similar"]["parent_run_id"] == capture.runs["mcp.find_similar"]["id"]
    assert capture.runs["mcp.list_filters"]["outputs"]["genres"] > 0


async def test_traces_never_contain_a_key_typed_into_a_query(index: Index) -> None:
    typed = "sk-" + "abcdefghijklmnop1234"
    async with traced(index) as (capture, client):
        await call(client, "search_movies", query=f"my key is {typed}", mode="dense", genre=typed)
    assert typed not in capture.payloads
    assert capture.runs["mcp.search_movies"]["inputs"] == {"query": "my key is ***"}
