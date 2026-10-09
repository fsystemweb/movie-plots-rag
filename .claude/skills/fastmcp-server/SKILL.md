---
name: fastmcp-server
description: Use when building, testing or reviewing the FastMCP server in src/movie_rag/mcp_server (tools, transports, contract tests).
---
# FastMCP server

**Version first:** `uv pip show fastmcp`. The installed version's docs win; record version + doc URL in the PR.

## Docs
- Tools: https://gofastmcp.com/servers/tools
- Running the server: https://gofastmcp.com/deployment/running-server
- Testing: https://gofastmcp.com/patterns/testing
- Client: https://gofastmcp.com/clients/client

## Working pattern
```python
from typing import Annotated
from pydantic import BaseModel, Field
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

def build_server(retriever_factory: Callable[[], Retriever]) -> FastMCP:
    mcp = FastMCP("movie-rag")

    @mcp.tool
    def search_movies(
        query: Annotated[str, Field(min_length=2, description="Natural-language description of the plot or film")],
        mode: Annotated[Literal["dense", "sparse", "hybrid"], Field(description="...")] = "hybrid",
        top_k: Annotated[int, Field(ge=1, le=20, description="...")] = 8,
        ...
    ) -> SearchResult:
        """LLM-oriented docstring: when to use the tool, what it returns, how to cite."""
        ...
        raise ToolError("year_from must be <= year_to")
    return mcp

# transports
mcp.run(transport="http", host=..., port=..., path="/mcp")   # older releases: "streamable-http"
mcp.run()                                                      # stdio
```
- Retriever injected via factory → tests pass a fake retriever.
- No LLM imports (`langchain`, `openai`, ...) anywhere in `mcp_server/`.
- Each tool wrapped in a LangSmith span (mode, filters, latency, result count) — see `langsmith-tracing`.

## Testing
```python
from fastmcp import Client
async with Client(build_server(lambda: FakeRetriever())) as c:
    tools = await c.list_tools()
    res = await c.call_tool("search_movies", {"query": "..."})
    with pytest.raises(ToolError): await c.call_tool("search_movies", {"query": "x", "year_from": 2000, "year_to": 1990})
```
- Schema snapshot test: dump `list_tools()` input schemas to a JSON fixture and compare.
- stdio smoke test: spawn the server via the stdio client and list tools.
- Import-guard test: walk `mcp_server/` AST and assert no LLM packages imported.

## Review checklist
- [ ] Exactly four tools: search_movies, get_movie, find_similar, list_filters
- [ ] Docstrings are LLM-readable (when to use, args, output, citation hint)
- [ ] Every ToolError path has a test
- [ ] No LLM imports in `mcp_server/` (test enforces it)
- [ ] Span on every tool call
- [ ] Both HTTP (`/mcp`) and stdio transports start (tested or smoke-checked)
- [ ] Host/port/path from config
