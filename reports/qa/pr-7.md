# QA report: PR #7, PR-07 Streamlit test page

- Branch `pr/07-streamlit-ui`, head `2daee34b4f6c6409d3fa247643688ceab5af0492`
- In scope: `17e26d4` (feat) and `2daee34` (PR description footer). `441e8a1` is the orchestrator's tracking commit and is not reviewed.
- Skill applied: `streamlit-ui` (review checklist below). Installed streamlit is 1.65.0 (`uv pip show streamlit`), and the lockfile resolves it to 1.65.0 (`uv.lock:2116-2117`).

## Summary

The PR adds `src/movie_rag/ui/service.py` (logic, no `streamlit` import), `src/movie_rag/ui/app.py` (rendering only), the `make ui` target, a `ui:` config section, `Span.trace_url()`, 18 AppTest tests, 28 service tests, 4 integration tests and a real screenshot. Every acceptance criterion in the plan row and every §1 element has evidence I produced myself. `make check` is green with 784 passed tests and 99.62% coverage. The integration suite passes against Docker Qdrant, and I ran the unpatched page in hybrid mode against a real `make serve`. I found no blockers and no majors. There are five minors and two nits, and most of them are already in `docs/BACKLOG.md`.

## Evidence I produced

| What | Command | Result |
|---|---|---|
| Full gate | `make check` (log `logs/qa-pr7-check.log`) | ruff: "All checks passed!" · format: "92 files already formatted" · mypy: "Success: no issues found in 29 source files" · pytest: **784 passed, 1 deselected** · "Required test coverage of 80% reached. Total coverage: **99.62%**" · `ui/service.py` 100%, `ui/app.py` 96%, `observability.py` 100% |
| Integration on Docker Qdrant | `QDRANT_URL=http://localhost:6333 uv run pytest -m integration -q` | **32 passed, 1 skipped** (the skip is the existing PR-06 live test `test_agent_mcp.py:141`, "set NEBIUS_API_KEY"), 0 failed |
| Lockfile | `uv lock --check` | "Resolved 161 packages", consistent |
| CI | `gh pr checks 7` | quality pass, tests pass, secret-scan pass, smoke-eval skipping (it is a stub until PR-09) |
| Live run | Docker Qdrant `movie_plots` (304 points, `dense` + `bm25`) + `make serve` + `make ui HEADLESS=1 PORT=8599` (`/_stcore/health` returned `ok`) + unpatched `AppTest` on `app.py` with no `NEBIUS_API_KEY` and a counting wrapper around `make_chat_model` | toggle defaulted to retrieval-only, mode `hybrid`, 18 genres, years (1915, 2017). The question returned "Retrieval only: 8 film(s), hybrid mode, 0.42s. No LLM was called." with the top film "The Overheard Midnight Call (1968)" and **0** chat-model builds. Compare tab subheaders were `['dense','sparse','hybrid']` with no error. Agent mode without a key showed `st.info` "set NEBIUS_API_KEY — see docs/CREDENTIALS.md Switch on 'Retrieval only'…" with no exception and no `st.error`. Afterwards I stopped the server and the UI myself; both ports return 000 |
| Screenshot | Viewed `docs/img/ui.png` | This is the real page: "Movie Plots RAG" title, sidebar (Retrieval only on, mode hybrid, top_k 8, year range 1915–2017, Genre/Origin "(any)"), Chat / Compare modes tabs, 4 example buttons, the first example asked, the caption "Retrieval only: 8 film(s), hybrid mode, 0.19s. No LLM was called.", and the expanded "Retrieved films (8)" with linked title (year), score, genre, origin, director and snippet. The fixture titles match my live run |

## Acceptance criteria

| Criterion | Evidence | Status |
|---|---|---|
| AppTest: retrieval-only happy path, no LLM call | `tests/unit/test_ui_app.py::test_retrieval_only_happy_path_lists_films_and_never_calls_an_llm` passes. It is proven twice: `make_chat_model` is patched to raise and record attempts (`no_llm == []`), and the injected scripted model's `prompts == []`. The service-level test uses the real default `MovieAgent` together with the same `no_llm` guard (`test_ui_service.py::test_retrieval_only_lists_films_through_the_mcp_tool_without_any_llm`). My live run counted 0 chat-model builds | ✅ |
| AppTest: agent path with fake service | `test_the_agent_path_shows_answer_sources_steps_and_metrics` (real `MovieAgent` on a scripted model with in-process tools: answer, Sources link, "Retrieved films (2)", "Agent steps (1 tool call…)", "121 tokens", model name, "tracing off"), plus the trace-link, abstention/refused and no-tool tests | ✅ |
| AppTest: missing-credential message | `test_agent_mode_without_a_key_shows_the_hint_not_a_traceback` (real agent, no key: exactly one `st.info` containing the hint and "Retrieval only", no exception, no `st.error`). Reproduced live | ✅ |
| AppTest: MCP-down error | `test_the_mcp_server_being_down_names_the_url_and_make_serve` (sidebar warning and chat error both name `mcp.url` and `make serve`), `test_compare_modes_with_the_server_down_is_a_friendly_error`. Real closed port: `tests/integration/test_ui_mcp.py::test_a_closed_port_is_the_friendly_error_with_the_url` passed in my integration run | ✅ |
| Screenshot `docs/img/ui.png` | Viewed, shows the real page (see above). A headless browser was available, so no CREDENTIALS.md follow-up is needed | ✅ |
| §1 sidebar: mode, top_k, year range, genre/origin from `list_filters`, retrieval-only toggle | `app.py:41-81`. Tests: `test_the_page_loads_with_the_configured_title_examples_and_sidebar`, `test_filters_in_the_sidebar_come_from_list_filters`, `test_the_sidebar_mode_top_k_and_genre_reach_the_search`. In agent mode, top_k and the filters are only a soft hint (minor m1) | ✅ |
| §1 chat input + 4 example questions | `app.py:181-193`, `config.yaml` `ui.example_questions` (4), `test_an_example_button_runs_that_question` | ✅ |
| §1 answer with citations | `app.py:115-121`, Sources assertion in the agent-path test | ✅ |
| §1 "Retrieved films" expander (score, snippet, year, genre) | `app.py:94-101`, asserted in the AppTest tests, visible in the screenshot | ✅ |
| §1 "Agent steps" expander | `app.py:124-134`, asserted ("Agent steps (1 tool call", "search_movies", "2 film(s)", refused calls, no-tool note) | ✅ |
| §1 latency / tokens / trace link | `app.py:135-140`. "121 tokens" and "tracing off" are asserted, and `[LangSmith trace](…)` is asserted when `Span.trace_url` returns a URL | ✅ |
| §1 "Compare modes" tab (retrieval only) | `app.py:195-202`, `service.py:232-245`, `test_compare_modes_shows_the_three_rankings` (with the `no_llm` guard), one session (`test_compare_runs_the_three_modes_over_one_session`). Live in hybrid | ✅ |
| §1 friendly errors | `service.describe_error` (`service.py:136-151`). Unexpected errors show only the type (`test_an_unexpected_failure_shows_a_generic_message` asserts "secret internals" is absent). Secrets are redacted (`test_no_secret_is_rendered` scans markdown/caption/info/warning/text for the fake key) | ✅ |
| §1 logic in `ui/service.py`, rendering in `ui/app.py`, AppTest | `service.py` has no `streamlit` import. `app.py` only calls `service.*` and renders | ✅ |
| Starts headless (skill checklist) | `test_streamlit_starts_headless_and_reports_healthy` passed. `make ui HEADLESS=1` gave a live health `ok` | ✅ |

## Judgement calls (requested explicitly)

- **Retrieval-only goes through MCP rather than the retriever.** Accepted. This mode needs `make serve` but no credential, and the credential-free requirement in CLAUDE.md/§2 is about keys, not about local processes. It keeps one retrieval path (the same validation and errors the agent sees) and keeps embedding models out of the Streamlit process. `list_filters` is an MCP tool per §1 anyway. The dependency is documented in the README, in the `make ui` comment and in the friendly MCP-down message. `make demo` (up + ingest + CLI query) does not involve the UI, so nothing regresses.
- **In agent mode the filters go in as question text and top_k is not passed.** This meets §1 only partially. The sidebar intent is satisfied for retrieval-only and Compare. In agent mode the page shows a caption that discloses it (`app.py:54`), the mode is pinned exactly (`MovieAgent.ask(mode=…)`, existing interceptor), and the item is in the backlog. The agent design in §1 lets the model choose tool arguments, so I rate this **minor** (m1), not major. The exact fix is small and already sketched.
- **Does the retrieval-only test prove no LLM call?** Yes. The fixture swaps the module global that `MovieAgent.ask` looks up at call time (`agent.py:194`) for one that raises, and the injected scripted model records zero prompts. The service-level twin uses the real default agent. My live run confirmed 0 builds.
- **Error rendering and leaks.** All service entry points catch `Exception` and return a `UiError`. `MovieRagError` and `ToolError` text is passed through `settings.redact`, the server uses `mask_error_details=True`, unexpected errors show the type only and are logged, and `McpUnavailableError` contains only `mcp.url`. Failures *outside* the service (for example `get_service()` → `load_settings()` on a bad `config.yaml`) still fall through to Streamlit's default exception display (m4).
- **Markdown escaping (only `$`).** There is no HTML or script injection, because Streamlit renders markdown without `unsafe_allow_html` and sanitises link URLs. The remaining risks are rendering corruption from `*`, `_`, `[`, `]`, `#` and `:color[]` directives in titles and plots, and model-answer markdown such as `![](…)` images that the browser fetches. The second one is a prompt-injection or exfiltration vector through indexed plot text. That is low risk for a local test page, but real (m2).
- **AppTest uses dense instead of hybrid.** Acceptable given the documented in-memory hybrid limitation (PR-04 backlog). The AppTest Compare test and the HTTP integration test do execute hybrid. No automated test runs the *page* against real Qdrant hybrid (m5). I verified that path by hand above.
- **`Span.trace_url`.** It is correct and safe: `None` without a run or when tracing is off, and exceptions are logged and swallowed. All four branches are tested (`test_observability.py`). Moving the `Span` construction inside `ls.trace` does not change behaviour. One caveat: `RunTree.get_url()` may call the LangSmith API synchronously inside `async _ask` (nit, backlogged).
- **Streamlit dependency and lockfile.** `streamlit>=1.65.0` is in `[project].dependencies` (`pyproject.toml:19`), `uv.lock` is committed and current (`uv lock --check` passes), and CI's `uv sync --frozen` passed.

## Integrity checks

| Check | Result |
|---|---|
| `pragma: no cover`, coverage `omit`/`fail_under` changes | None. I grepped the `+` lines of `git diff 441e8a1 2daee34 -- src tests Makefile pyproject.toml`. The gate is still `--cov-fail-under=$(COV_FAIL_UNDER)` on the command line (`Makefile:10`) and was not touched |
| skip/xfail outside live tests | None added. The only skip in my run is the existing live test |
| Protected files (`.github/workflows`, `.claude`, `CLAUDE.md`, `KICKOFF.md`, `docs/TOKEN_USAGE.md`, `docs/token_usage.jsonl`, `.env*`) | Untouched by `17e26d4`/`2daee34` (`git diff 441e8a1 2daee34 --name-only` grep returned nothing). The token files appear in `main...branch` only through the orchestrator commit `441e8a1` |
| Secrets, `.env` content, raw data | None. The only secret-shaped strings are fake test keys (`fake-test-key-123`, `fake-ls-key-456`). The PNG is a screenshot of synthetic fixture data. gitleaks/secret-scan passed in CI |
| Hard-coded models, URLs or retrieval parameters in `src/` | None. No `http(s)://` literals were added in `src/`. Title, MCP timeout and example questions are in `config.yaml` `ui:` + `UiConfig` (`config.py:97-100`). URL is `mcp.url`. Defaults come from `retrieval.default_mode`, `retrieval.top_k`, `mcp.max_top_k` and `retrieval.demo_query` |
| `print`/env reads in `src/` | None added |
| Tests assert something meaningful | Yes. They check concrete rendered values, the no-LLM guards, secret absence and error kinds. One test is weak but valid (`test_the_span_handle_carries_the_run…` only asserts `None` with tracing off) |
| Credentials needed | No. All tests ran with an empty `NEBIUS_API_KEY`/`LANGSMITH_API_KEY` |
| Scope | Within the plan row: the UI package, `make ui`, the `ui:` config, the small `observability` addition needed for §1's trace link, tests, README section, PR doc and backlog entries (per the pr-workflow skill) |

## streamlit-ui review checklist

- [x] Starts headless: integration test plus live `make ui HEADLESS=1`.
- [x] Retrieval-only works with no keys and makes no LLM call: tests plus live run (0 builds).
- [x] Friendly errors for missing credentials and MCP down: tests plus live missing-key run and closed-port integration test.
- [x] No secrets displayed: `test_no_secret_is_rendered`, redaction in `describe_error` (with the m4 caveat for errors outside the service).
- [x] Logic in service.py, rendering in app.py.

## Findings

### Blockers
None.

### Majors
None.

### Minors
- **m1. Agent mode ignores sidebar `top_k`, and the filters are only a prompt hint.** `src/movie_rag/ui/service.py:253-258`, `src/movie_rag/ui/app.py:54`. The model may ignore "Only consider films …", and the top_k slider has no effect in agent mode. The caption at `app.py:54` says "the filters below are passed on", which overstates a hint. **Fix:** extend the `force_mode` interceptor in `load_mcp_tools` (`agent/agent.py:82-84`) into a `force_args` that overrides `top_k`/`year_from`/`year_to`/`genre`/`origin` on `search_movies`, pass them through `MovieAgent.ask`, and reword the caption until then. (Already in BACKLOG [PR-07].)
- **m2. Only `$` is escaped when data and model text are rendered as markdown.** `src/movie_rag/ui/app.py:29-31`, used at `:98`, `:101` and `:116`. Titles and plots containing `*`, `_`, `[`/`]`, `#` or `:red[...]` render incorrectly (for example a `]` in a title breaks the link at `:98`). The model answer at `:116` can contain `![](http://…)` images that the browser loads automatically, which is a prompt-injection or exfiltration vector through indexed plot text. **Fix:** escape markdown metacharacters (`\ ` `` ` `` `* _ [ ] ( ) # + - ! < > : $`) in `md()` for data fields (titles, snippets, details), and for `answer.text` neutralise image syntax (`![` → `!\[`) or render the answer with `st.text`/`st.write` of plain text. (Partly in BACKLOG [PR-07].)
- **m3. Failures inside an open session are reported as "server unreachable".** `src/movie_rag/ui/service.py:183-191` wraps the whole `yield` body. With `is_connection_failure` (`agent/agent.py:71-73`, which counts any `httpx.TransportError`/`HTTPStatusError`), a `ReadTimeout` after `ui.mcp_timeout_s`, or an HTTP 5xx during `call_tool`, becomes "unreachable: start it with `make serve`" although the server is running. **Fix:** map connection failures only around `client.__aenter__` (or check `httpx.ConnectError`/`ConnectTimeout` specifically), and add a `timeout` `ErrorKind` for `httpx.TimeoutException`.
- **m4. Errors outside the service bypass the friendly path.** `src/movie_rag/ui/app.py:23-26,177`. `get_service()` → `load_settings()` (bad `config.yaml`/env) or a rendering bug raises into Streamlit, which shows the full traceback in the browser by default. Secrets are `SecretStr`, so key values should not appear, but this still contradicts "friendly errors, never a traceback". **Fix:** wrap `get_service()` in `main()` with `try/except MovieRagError|ValidationError` → `st.error(...)` + `st.stop()`, and/or add `.streamlit/config.toml` with `[client] showErrorDetails = "none"`.
- **m5. No automated test runs the page in hybrid mode against real Qdrant.** `tests/integration/test_ui_mcp.py:57` always builds `Index()` in memory, even when `QDRANT_URL` is set. The AppTest suite pins dense mode (`test_ui_app.py:58-66`). I verified the real hybrid path by hand only. **Fix:** like `tests/integration/test_ingest_qdrant.py:29-35`, back the loopback MCP server with `QdrantClient(url=QDRANT_URL)` when it is set, and run `test_the_unpatched_page_works_against_the_real_server` with `mode="hybrid"`.

### Nits
- `src/movie_rag/ui/app.py:188`: `use_container_width` is deprecated in streamlit 1.65 (docstring at `streamlit/elements/widgets/button.py:256-258`). Use `width="stretch"`.
- `src/movie_rag/observability.py:157`: `RunTree.get_url()` may block on a LangSmith API call inside `async _ask` (`service.py:258`). This only happens with tracing on, and it is already backlogged.

## Coverage

Line+branch on `src/movie_rag`: **99.62%** (gate 80%). 784 passed, 1 deselected (live). `ui/service.py` 100%, `ui/app.py` 96% (missing: `64->74, 79, 100->95, 132, 151->exit`), `observability.py` 100%.

VERDICT: PASS
