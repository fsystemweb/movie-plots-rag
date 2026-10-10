# QA report: PR #5, PR-05 FastMCP server

- Branch `pr/05-mcp-server`, head `a0a0f7f65ac322b8a0fda8796cd2347076c9d1c5`. Reviewed on 2026-10-10 by qa-validator.
- In scope: builder commits `031406c` (server, tests), `f374765` (compose, Dockerfile, `make serve`) and `a0a0f7f` (PR description).
- Out of scope: `5b81f86`. The orchestrator wrote it (tracking docs, token log, `reports/qa/pr-4.md`).
- Skills applied: `fastmcp-server`, `langsmith-tracing` and `pr-workflow` (for the backlog convention).
- Installed versions: fastmcp 3.4.8, langsmith 0.14.7, uv 0.12.24.

## Summary

The PR adds `src/movie_rag/mcp_server/` with a FastMCP server called `movie-rag` that exposes exactly four read-only tools.

- The retriever comes from a factory and is created lazily.
- Arguments are validated by pydantic, and the limits come from `config.yaml` (`mcp:`).
- Callers get a `ToolError` with an actionable message for:
  - bad input
  - an unknown film
  - an ambiguous title
  - a missing collection
  - an unreachable Qdrant
- Errors we did not anticipate are masked.
- Every tool call runs inside a LangSmith span (`mcp.<tool>`), and a `retriever.*` span is nested under it. This delivers the tracing that PR-04 deferred.
- The server runs over HTTP at `/mcp` and over stdio.
- The compose service has a healthcheck, and `make serve` supports `DOCKER=1` and `STDIO=1`.
- `make check` is green with 100% line+branch coverage.
- I ran the compose service from this branch and called all four tools over HTTP. Every result was correct.
- I found no blockers and no majors, and five minors.

## Acceptance criteria

| Criterion | Evidence (produced by QA) | Status |
|---|---|---|
| Four tools per §1 (`search_movies`, `get_movie`, `find_similar`, `list_filters`) | Over HTTP against the running container: `tools: ['find_similar', 'get_movie', 'list_filters', 'search_movies']`. Test: `tests/unit/test_mcp_server.py::test_exactly_the_four_documented_tools_are_exposed`. | ✅ |
| `search_movies` args and returns (score, snippet ≤400, wiki URL) | Over HTTP: `search hybrid: hybrid 3 [('The Forgetting Hour', 1947, 400, 'https://en.wikipedia.org/...'), ...]`. Snippet lengths were 400, 399 and 395. Filters, defaults and the empty-result hint are tested in `test_mcp_server.py`. | ✅ |
| `get_movie` by movie_id or exact title, full plot | Over HTTP: `get_movie id: the-forgetting-hour-1947-282 ... 586` (plot chars) and `get_movie title: the-forgetting-hour-1947-282`. | ✅ |
| `find_similar` excludes the input | Over HTTP: `find_similar: 3 [...] False` (the input is not in the results). Also covered by unit and integration tests. | ✅ |
| `list_filters` returns genres, origins and the year range | Over HTTP: `list_filters: {"genres": ["drama", ...], "origins": ["american", ...` | ✅ |
| Bad input returns a `ToolError` with a clear message | Over HTTP: `invalid arguments: year_from (2000) must not be after year_to (1990)`, `no film titled 'the forgetting hour'. Titles must match exactly; ...`, `no film with movie_id 'nope'. Use search_movies ...`, `pass exactly one of movie_id or title`. A `top_k=0` call returns the pydantic multi-line message (see m4). | ✅ |
| In-memory contract tests for all tools and error paths | `test_mcp_server.py` uses `fastmcp.Client(build_server(...))` with no sockets. It has parametrised error tests for every `ToolError` raise in `server.py`. It also tests an unreachable Qdrant (3 exception types × 5 call shapes), a 404 collection, a 500 response and a masked `RuntimeError`. All pass in `make check`. | ✅ |
| Schema snapshot | `tests/fixtures/mcp_tools_schema.json` (name, description, input and output schemas) is compared in `test_tool_schemas_match_the_committed_snapshot`. It passes. | ✅ |
| stdio smoke test lists the tools | `tests/unit/test_mcp_main.py::test_stdio_smoke_lists_the_four_tools` spawns `python -m movie_rag.mcp_server --transport stdio` and passes. | ✅ |
| No LLM imports in `mcp_server/` (test) | `test_the_server_package_imports_no_llm_library` walks the AST. `test_the_server_is_importable_without_llm_modules_loaded` checks `sys.modules` in a fresh interpreter. Both pass. | ✅ |
| LangSmith span per tool call (mode, filters, latency, result count) | `server.py:127-141, 173-182, 226-236, 249`. Tests `test_search_opens_a_tool_span_with_a_nested_retriever_span` and `test_every_tool_call_is_traced_including_failures` check the metadata fields, `latency_ms`, `result_count` and the parent/child link. | ✅ |
| HTTP (`/mcp`) and stdio both start | HTTP: `make serve DOCKER=1` exits 0, the container is `Up 20 seconds (healthy)`, `curl /health` returns `{"status":"ok","server":"movie-rag"}`, and `tests/integration/test_mcp_transports.py` (real process on loopback) passes. stdio: the smoke test above. | ✅ |
| `mcp-server` compose service with healthcheck | `docker-compose.yml`: `depends_on qdrant: service_healthy`, healthcheck `python -m movie_rag.mcp_server --healthcheck`. Live: the compose output shows `Container movie-rag-mcp Healthy`. | ✅ |
| `make serve` | Ran `make serve DOCKER=1` (exit 0). Test: `test_make_serve_runs_the_mcp_server_over_http_stdio_or_compose`. | ✅ |
| Host, port and path from config | `config.yaml` `mcp:` block and `config.py:79-88` (`McpConfig`). Test: `test_http_is_the_default_transport_and_host_port_path_come_from_the_configuration`. | ✅ |
| PR-04 deferred retrieval tracing delivered | `search.py` opens `retriever.search` and `retriever.find_similar` spans. `test_retrieval.py::test_search_and_find_similar_open_retriever_spans` passes, and the spans nest under the MCP span. This closes the PR-04 QA condition (`reports/qa/pr-4.md:65-69`) and BACKLOG item `[PR-04]` line 27. | ✅ |
| PR-04 QA m4 (exact-title lookup) | `Retriever.find_by_title`. Unit tests, plus the integration test `test_exact_title_lookup_finds_the_film_and_its_whole_plot` against real Qdrant. | ✅ |

## Commands run and results

- **`make check` (exit 0):**
  - ruff: `All checks passed!`
  - format: `69 files already formatted`
  - mypy: `Success: no issues found in 20 source files`
  - pytest: `661 passed in 185.97s`
  - coverage: `TOTAL 1352 0 258 0 100%` and `Required test coverage of 80% reached. Total coverage: 100.00%`
  - Every `mcp_server/*` file, `observability.py` and `retrieval/search.py` is at 100%.
- **`QDRANT_URL=http://localhost:6333 uv run pytest -m integration tests/integration -v`:** `23 passed`. This includes 3 new MCP transport tests and 3 new retrieval tests against Qdrant v1.15.4.
- **`uv lock --check`:** `Resolved 129 packages`, so the lockfile is current.
- **Compose run:**
  - `make serve DOCKER=1` built the image from this branch. The `ghcr.io/astral-sh/uv:0.12.24` pin resolves.
  - The Qdrant container was left running and was not recreated (I confirmed this with `--dry-run` first).
  - I called the server over HTTP with `fastmcp.Client("http://localhost:8000/mcp")`. The output is quoted in the table above.
  - Teardown: `docker compose stop mcp-server && docker compose rm -f mcp-server`. Only the container I started was removed. Qdrant stayed up, and the image and volume already existed before my run.
- **GitHub CI (`gh pr checks 5`):** quality pass, secret-scan pass, tests pass. smoke-eval was skipped, which is expected because eval-smoke belongs to PR-09.

## Integrity checks

| Check | Result |
|---|---|
| `pragma: no cover`, skip/xfail, `pytest.skip(` added | None. I grepped the added lines of `5b81f86..a0a0f7f`. |
| Coverage `omit` / `fail_under` / `--cov-fail-under` changed | None. The `pyproject.toml` diff only adds the `fastmcp` dependency and rewords a marker description. |
| Protected files (`.github/workflows/`, `.claude/`, `CLAUDE.md`, `KICKOFF.md`, token files, `.env*`) | The builder commits touch none of them. The token files appear in the diff only through the orchestrator commit `5b81f86`. |
| Secrets, `.env` content, raw data committed | None. `.dockerignore` excludes `.env`, `.env.*`, `data` and `.claude`. The image copies only `pyproject.toml`, `uv.lock`, `README.md`, `config.yaml` and `src`. |
| Secret-shaped strings in `src/` | None. No `sk-` or `lsv2_` strings were added. The only existing match is the detection regex in `observability.py:36`. |
| Hard-coded models, URLs or retrieval params in `src/` | None of substance. Limits, host, port, path, health path and timeout are all in `config.yaml`. The only exception is the loopback host of the healthcheck probe, `__main__.py:25` (see m3). |
| Tests need a credential | No. Tracing tests use a LangSmith client with a mocked session (`tests/fakes.py::RunCapture`), so nothing leaves the process. |
| Meaningless tests | One weak test (m5). The rest assert real behaviour. |
| Scope | The diff stays within the plan row. Every retriever addition (`find_by_title`, `find_similar`, `list_filters`) and the `span` helper exists to serve the tools. It also fixes PR-04 QA m2, a test-only change listed in BACKLOG. That fix is small and acceptable. |

## The builder's declared decisions

1. **Optional `year` argument on `get_movie`: accepted.**
   - §1 specifies "movie_id or exact title", but titles are not unique because of remakes.
   - The optional `year` resolves the ambiguity in one call. It is rejected unless `title` is also given, and it is documented in the docstring and the schema snapshot.
   - It extends the contract without changing it, so this is not scope creep.
2. **Exact, case-sensitive title lookup on an unindexed field: accepted.**
   - §1 says "exact title", and the agent copies titles from tool results.
   - Over HTTP, a lowercased title returns a clear error that points to `search_movies`.
   - A payload scan of about 35k films (filtered to `chunk_idx == 0`) is acceptable for now. The title index is tracked in BACKLOG.
3. **`list_filters` cap of 200 with a `truncated` flag: accepted.**
   - The cap comes from config (`mcp.max_filter_values`), and the reason (thousands of combined Kaggle genres) is sound.
   - The flag has an off-by-one false positive (m1).
4. **Pydantic's verbose validation errors in `ToolError`: accepted as minor (m4).**
   - The message still names the field and the constraint (`top_k ... greater than or equal to 1`), so it is clear.
   - It is verbose and includes a pydantic docs URL. It is declared in BACKLOG for PR-06.
5. **FastEmbed models download on the first query inside the container: accepted.**
   - The healthcheck does not depend on the models, and the models go into a named volume.
   - The cost is first-query latency and needing network at that moment. This is tracked in BACKLOG.
6. **Fake `sk-` keys built by string concatenation in tests: not a concern.**
   - The guardrail `SECRET_RE` (`.claude/hooks/guardrails.py:125`) applies only to `src/`.
   - `tests/` already contained the literal `"sk-abcdefghijklmnop"` on `main` (`tests/unit/test_observability.py:110`).
   - The values are obviously fake (`abcdefghijklmnop1234`). Concatenation only avoids false positives from scanners such as gitleaks, and CI secret-scan passes.
   - Nothing in `src/` uses the trick.
7. **Span content: no secrets, and a no-op without a key. Verified.**
   - `span()` scrubs inputs, metadata and outputs (`observability.py:634-689` in the diff). It removes configured secrets and `sk-`/`lsv2_` shapes.
   - `test_traces_never_contain_a_key_typed_into_a_query` checks the whole captured upload payload.
   - Empty-string keys (compose passes `LANGSMITH_API_KEY: ${LANGSMITH_API_KEY:-}`) are normalised to `None`. I checked: `secret_values() == []`, `redact('abc') == 'abc'` and `tracing_requested == False`.
   - Without a key, `configure_tracing` forces `LANGSMITH_TRACING=false`, and the code path is the same (`test_span_runs_the_block_identically_with_tracing_off`).
8. **PR-04 deferred tracing: delivered.** See the criteria table.

## Findings

### Blocker
None.

### Major
None.

### Minor

- **m1: `src/movie_rag/retrieval/search.py:317`, `truncated` is a false positive when the number of values exactly equals the limit.**
  - The current code is `truncated=limit in (len(genres), len(origins))`.
  - Reproduced on the fixture, which has 17 genres: `list_filters(limit=17)` returns `genres=17 truncated=True` even though nothing was cut.
  - **Fix:** request `limit + 1` facet values and set `truncated = len(raw) > limit` before slicing to `limit`. Add a test where the limit equals the count.
- **m2: `src/movie_rag/mcp_server/server.py:197,207` together with `search.py:241,255`, the ambiguous-title message misreports the count when more than `title_lookup_limit` films share a title.**
  - The scroll is capped at `limit`, so the message reads "10 films are titled X" even when there are more.
  - `find_by_title` also hard-codes a default `limit: int = 10` that duplicates the `config.yaml` value.
  - **Fix:** scroll `limit + 1` and say "more than N films are titled X" when it overflows. Remove the default, or derive it from settings.
- **m3: `src/movie_rag/mcp_server/__main__.py:25,48`, the healthcheck always probes `127.0.0.1`.**
  - If `mcp.host` or `--host` binds a specific non-loopback interface (for example `10.0.0.5`), `--healthcheck` reports unhealthy even though the server is up.
  - It works for both shipped configs: `127.0.0.1` by default and `0.0.0.0` in Docker.
  - **Fix:** probe `cfg.host` unless it is a wildcard (`0.0.0.0` or `::`), in which case use loopback. Alternatively, add `mcp.healthcheck_host` to `config.yaml`.
- **m4: argument validation done by FastMCP itself returns pydantic's multi-line message.**
  - Example: `'1 validation error for call[search_movies]\ntop_k\n  Input should be greater than or equal to 1 ... For further information visit https://errors.pydantic.dev/...'`.
  - Our own validation is one line. This is declared in BACKLOG.
  - **Fix (PR-06 or earlier):** compact these through FastMCP middleware or an error handler into `invalid arguments: top_k must be >= 1`.
- **m5: `tests/unit/test_mcp_server.py:440`, `test_the_default_server_builds_a_real_retriever_lazily` does not test what its name says.**
  - It only asserts the server name and type.
  - It requests `monkeypatch` but never uses it.
  - **Fix:** patch `movie_rag.mcp_server.server.Retriever` with a `MagicMock` and assert it is not called after `build_server(settings)` and `list_tools()`. Then call a tool and assert it was called once with `settings`.

Nits (no action required):
- `server.py:196` uses `assert title is not None` in production code to narrow the type for mypy. It is harmless, but an explicit `if` would also survive `python -O`.
- The retriever spans use `run_type="retriever"` but do not return LangSmith Document-shaped outputs. This is fine for §1, but the LangSmith UI will not render the results as documents.

## Coverage
Line+branch coverage on `src/movie_rag` is **100.00%** (gate 80%). 661 tests passed (`make check`), plus 23 integration tests against Docker Qdrant.

VERDICT: PASS
