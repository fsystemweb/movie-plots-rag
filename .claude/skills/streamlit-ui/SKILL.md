---
name: streamlit-ui
description: Use when building, testing or reviewing the Streamlit test page in src/movie_rag/ui (app.py rendering, service.py logic, AppTest tests).
---
# Streamlit test page

**Version first:** `uv pip show streamlit`. The installed version's docs win; record version + doc URL in the PR.

## Docs
- API reference: https://docs.streamlit.io/develop/api-reference
- App testing: https://docs.streamlit.io/develop/concepts/app-testing , https://docs.streamlit.io/develop/api-reference/app-testing
- Caching: https://docs.streamlit.io/develop/concepts/architecture/caching

## Working pattern
- `ui/service.py`: pure logic (retrieve, ask agent, list filters, compare modes) returning Pydantic models; no `st`.
- `ui/app.py`: rendering only. Sidebar: mode, top_k, year range, genre/origin from `list_filters`, retrieval-only
  toggle. Chat input + 4 example questions. Answer with citations; "Retrieved films" expander (score, snippet, year,
  genre); "Agent steps" expander; latency/tokens/trace link; "Compare modes" tab (retrieval only).
- `st.cache_resource` for clients/services.
- Errors → friendly `st.error`/`st.info`; `MissingCredentialError` shows how to fix (link to docs/CREDENTIALS.md);
  MCP down shows how to start it (`make serve`).
- Never display env values or keys.

## Testing
```python
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("src/movie_rag/ui/app.py", default_timeout=30)
# inject a fake service (monkeypatch the factory module attribute, or session_state hook)
at.run(); at.chat_input[0].set_value("...").run()
assert not at.exception
```
- Cases: retrieval-only happy path (assert fake LLM never called), agent path with fake service,
  missing-credential message, MCP-down error.
- Headless start smoke: `streamlit run ... --server.headless true` reaches health endpoint (optional, integration).

## Review checklist
- [ ] Starts headless
- [ ] Retrieval-only works with no keys and makes no LLM call
- [ ] Friendly errors for missing creds / MCP down
- [ ] No secrets displayed
- [ ] Logic in service.py, rendering in app.py
