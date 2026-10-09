---
name: langchain-agent
description: Use when building, testing or reviewing the LangChain agent in src/movie_rag/agent (model setup, MCP tools, tool-call cap, citations, CLI).
---
# LangChain agent over MCP

**Version first:** `uv pip show langchain langchain-openai langchain-mcp-adapters langgraph`. The installed version's
docs win; record versions + doc URLs in the PR.

## Docs
- Agents: https://docs.langchain.com/oss/python/langchain/agents
- MCP: https://docs.langchain.com/oss/python/langchain/mcp
- Middleware (tool call limit): https://docs.langchain.com/oss/python/langchain/middleware
- langchain-mcp-adapters: https://github.com/langchain-ai/langchain-mcp-adapters
- Nebius Token Factory: https://docs.tokenfactory.nebius.com/ (OpenAI-compatible API)

## Working pattern
```python
def make_chat_model(cfg: Settings) -> ChatOpenAI:
    if not cfg.nebius_api_key:
        raise MissingCredentialError("set NEBIUS_API_KEY — see docs/CREDENTIALS.md")
    return ChatOpenAI(model=cfg.chat_model, base_url=cfg.nebius_base_url,
                      api_key=cfg.nebius_api_key, temperature=0)

client = MultiServerMCPClient({"movies": {"transport": "streamable_http", "url": cfg.mcp_url}})
tools = await client.get_tools()
agent = create_agent(model, tools, system_prompt=load_prompt(cfg.prompt_version))
# cap: ToolCallLimitMiddleware(run_limit=cfg.max_tool_calls) if available, else counter + recursion_limit
```
- `Answer(text, citations, tool_calls, usage)` typed (Pydantic). Citations are built **only** from movie_ids present
  in tool results; anything the model mentions that was not retrieved is dropped.
- Prompt lives in `src/movie_rag/agent/prompts/system_v1.md`; version recorded in trace metadata.
- The model is created lazily at call time — importing the module never needs a key.

## Testing
- `GenericFakeChatModel` (or a `BaseChatModel` subclass with `bind_tools`) scripted to emit tool calls → assert cap
  (5th call refused/stopped), abstention when results are empty, citation filtering, metadata attached.
- Fake MCP tools (plain LangChain tools) in unit tests; no network.
- One `@pytest.mark.live` test hitting Nebius, skipped by default.
- CLI: without `NEBIUS_API_KEY`, `make ask` prints the hint and exits 2.

## Review checklist
- [ ] No hard-coded URL or model id (all from config)
- [ ] Citations only from retrieved ids (test)
- [ ] 4-call cap tested; abstention tested
- [ ] Prompt version + run metadata in trace
- [ ] MissingCredentialError at call time, not import
