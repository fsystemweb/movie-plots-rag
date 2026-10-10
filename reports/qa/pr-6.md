# QA report: PR #6, PR-06 Agent + CLI

- Branch: `pr/06-agent-cli` @ `28522201990b2102ddbd90ab526c2eab6b3e6e95` (same as the GitHub PR head)
- In scope: `24b0251` (feat) and `2852220` (docs). `dae1752` is orchestrator-authored (tracking docs, token log, `reports/qa/pr-5.md`) and was not reviewed.
- Skills applied: `langchain-agent`, `langsmith-tracing`, `pr-workflow` (the PR description checklist).
- Installed versions (`uv pip show`): langchain 1.4.4, langchain-core 1.6.9, langchain-openai 1.7.0, langchain-mcp-adapters 0.3.2, langgraph 1.2.14, langsmith 0.14.7. These match the PR description.

## Summary

The PR adds `src/movie_rag/agent/`, which runs a `create_agent` loop over the MCP tools. The pieces are:
- `ToolCallLimitMiddleware(run_limit=llm.max_tool_calls, exit_behavior="continue")`, which caps tool calls.
- A recursion limit, which stops a model that keeps asking for tools after the cap.
- Citations that only select among films the tools returned (`Title (Year)` or `movie_id`).
- A typed `Answer` with run metadata.
- A CLI (`make ask`) that prints the credentials hint and exits 2 when there is no key.

I verified every acceptance criterion myself. I also probed the cap, the step limit, the trace tree, secret scrubbing and citation edge cases. I found no blockers and no majors. I found six minors, all in citation and abstention heuristics or brittleness, and none of them breaks an acceptance criterion.

## Acceptance criteria

| Criterion | Evidence (produced by QA) | Result |
|---|---|---|
| Fake-model test: 4-call cap | `tests/unit/test_agent.py:214-248` counts real tool executions (`tools.invocations`), not message counts. `make check`: all pass. My probe (scripted model: 4 rounds, then 0..7 extra tool rounds, then an answer) gave executions = 4 every time. With 0-4 refused retries the model still answers (`stopped_early=False`). With ≥5 it stops at the step limit and abstains. A batch of 3 calls × 4 rounds gave 4 executed and 8 refused. No run ever executed a 5th tool. Middleware source (`.venv/.../langchain/agents/middleware/tool_call_limit.py`, `_separate_tool_calls` / `after_model`) blocks per call, both within a batch and across rounds. | ✅ |
| Fake-model test: abstention | `test_agent.py:183-208`: nothing retrieved, only unretrieved films named, and no tool call all give `abstained=True`, `ABSTENTION_TEXT`, no citations. `test_agent.py:242` covers the step-limit path. | ✅ |
| Fake-model test: citations only from retrieved ids | `test_agent.py:162-177` and `tests/unit/test_agent_citations.py` (15 tests). The integration test `tests/integration/test_agent_mcp.py:75` runs over the real adapter. Title, year and URL come from the tool result even when the model writes `evil.example`. My probes (below) confirm that a hallucinated film never becomes a citation. | ✅ |
| Fake-model test: metadata attached | `test_agent.py:277-314` checks that the `Answer.metadata` and the captured LangSmith run carry git_sha, prompt_version, chat_model, embedding_model, retrieval_mode and config_hash. My trace probe found all 7 runs (`agent.ask` → `movie_agent` → `model` / `ScriptedChatModel` / `ToolCallLimitMiddleware.after_model` / `tools` / `search_movies`) correctly nested, each with `prompt_version` and `retrieval_mode` in its metadata. | ✅ |
| Without `NEBIUS_API_KEY`, `make ask` prints the hint and exits 2 | `NEBIUS_API_KEY= LANGSMITH_TRACING=false make ask Q="a film about a hotel operator who overhears a murder plot"` gave `exit=2`, empty stdout, and stderr `set NEBIUS_API_KEY — see docs/CREDENTIALS.md` with no traceback. `python -m movie_rag.agent "x"` also gave `direct exit=2`. Unit and subprocess tests: `tests/unit/test_agent_cli.py:191,286`. | ✅ |
| One `live` test ready for later | `uv run pytest -m live --collect-only -q` collected exactly one test, `tests/integration/test_agent_mcp.py::test_live_nebius_model_answers_from_the_real_tools`. `make check` reports "1 deselected". Under `-m integration` it reports `SKIPPED ... set NEBIUS_API_KEY`. The `pytest.skip(` at line 141 is inside the live test, which is allowed. | ✅ |
| §1 Agent: `create_agent`, MCP via adapters, `ChatOpenAI(base_url, temperature=0)` | `agent.py:51-59` and `agent.py:76-102`. Model, URL and temperature all come from config (`test_agent.py:106`). | ✅ |
| §1 answer rules in versioned `prompts/system_v1.md` | Rules 1-6 cover: only tool facts, ≤ `{max_films}` films with one sentence each, `Title (Year)` plus Wikipedia link, say so when nothing fits. The limits are templated from config (`prompt.py:15-25`, test `test_agent.py:320`). | ✅ |
| §1 citations built from tool results, never free text | `citations.py:82-103`: `Citation` fields always come from `RetrievedFilm`, and the text only selects among them. | ✅ |
| §1 Observability: metadata on every run, no secrets in traces | `RunCapture` probe with `sk-abcdefghijkl123` in the question: `leak: False`. `test_agent.py:266,295` passes. The question is scrubbed before it reaches the model or a span (`agent.py:189-191`). Tool-call args are scrubbed (`agent.py:138`). | ✅ |
| Credentials policy: error at call time, not import | Grep: `nebius_api_key` is only read through `require_nebius_api_key()` in `make_chat_model` (`agent.py:53`), which is called inside `ask` before tools load (`test_agent.py:95` forbids `load_mcp_tools`). Importing `movie_rag.agent` needs no key (the whole suite runs with the env scrubbed). | ✅ |

## Own verification runs

- `make check` (exit 0):
  - ruff: "All checks passed!"; format: "83 files already formatted"
  - mypy: "Success: no issues found in 26 source files"
  - pytest: **728 passed, 1 deselected**; coverage **99.90 %** (1670 stmts / 1 miss, 322 branches / 1 partial). The only miss is `agent/__main__.py:96` (`sys.exit(main())`).
- Integration: `QDRANT_URL=http://localhost:6333 uv run pytest -m integration -q` (Qdrant healthz 200) gave **28 passed, 1 skipped** (the live test), 703 deselected.
- GitHub checks on the head: quality SUCCESS, tests SUCCESS, secret-scan SUCCESS, smoke-eval SKIPPED (PR-09 not built yet).

## Integrity checks

| Check | Result |
|---|---|
| `pragma: no cover` added | none (grep over the `src` and `tests` diff) |
| Coverage `omit` / `fail_under` / `--cov-fail-under` changes | none. `Makefile:9 COV_FAIL_UNDER := 80` is unchanged, and the gate is still on the command line. |
| skip/xfail outside live tests | none. The only `pytest.skip(` is inside the `@pytest.mark.live` test (`test_agent_mcp.py:141`). |
| Protected files (`.github/workflows/`, `.claude/`, `CLAUDE.md`, `KICKOFF.md`, `docs/TOKEN_USAGE.md`, `docs/token_usage.jsonl`, `.env*`) | none touched in `dae1752..HEAD`. The token-log changes on the PR come only from the orchestrator commit `dae1752`. |
| Secrets / .env content / raw data committed | none. `sk-…` strings appear only as test fixtures in `tests/`. `.env` was never read or printed by QA. |
| Hard-coded model names, URLs, retrieval params in `src/` | none. Model, base_url, temperature, cap, mcp.url, prompt version, max_citations and mcp_timeout_s all come from `config.yaml`. The only URL in `src/` is the example in `system_v1.md` rule 3, which is prompt text. `STEPS_PER_ROUND=4` and `ERROR_TEXT_LIMIT=300` are internal implementation constants, not retrieval tunables. |
| Tests assert something meaningful | Yes. The cap is asserted on real tool executions, citations on ids and URLs from tool results, and the trace on captured HTTP payloads. |
| Needs a credential to pass | No. The suite and the integration run pass with `NEBIUS_API_KEY` blank. |
| Scope vs plan row | In scope: agent package, CLI, `make ask`, config additions for the agent, `MissingCredentialError` hint use, `AgentError` / `McpUnavailableError`, README and PR doc, backlog bullets. |
| `tests/fakes.py` / `tests/index_fixture.py` refactor is behaviour-neutral | `fakes.py`: additions only, no removed lines. `Index` and `premise` were moved verbatim from `test_mcp_server.py` to `tests/index_fixture.py`. `test_mcp_server.py` has 33 tests before and after, and all pass. |
| `uv.lock` current | CI `uv sync --frozen` succeeded (quality and tests jobs green). |

## Probes of the points I was asked to judge

**Cap.** The cap is enforced on actual tool executions. In 10 scripted scenarios (single calls, a batch of 6, batches of 3 × 4 rounds, models that never stop) the tool ran at most 4 times. Because of `exit_behavior="continue"`, the model receives `"Tool call limit exceeded. Do not make additional tool calls."` and can still answer. The step-limit path works as follows:
- `recursion_limit = 4 × (cap + 2) = 24` leaves room for 4 tool rounds plus up to 4 refused retries before cutting off.
- `_run` keeps the last streamed state, so the calls made so far and the retrieved films survive (`stopped_early=True`).
- The text is dropped and the run is an abstention.

This path is sound.

**Citation edge cases** (`select_citations`, my probes):

| Case | Result | Verdict |
|---|---|---|
| Same title, different years, text names `Alpha (2029)` | `['alpha-2029-2']` | correct |
| Id prefix of another id: `[alpha-1999-12]` with `alpha-1999-1` also retrieved | only `alpha-1999-12` | correct |
| Remake id suffix: `[alpha-1999-1-remake]` | only the remake | correct |
| Id followed by `.`, or inside a URL path | matched (same film) | fine |
| Hallucinated film with fake `[id]` and wiki link | dropped | correct |
| Title only, or lowercase label | no citation, so a false abstention | design limit, backlogged |
| **Title substring with the same year**: text `The Alpha (1999)` when both `Alpha (1999)` and `The Alpha (1999)` were retrieved | **both cited** | minor m1 |
| Negative mention: `Unlike Alpha (1999), Beta (2004) fits` | both cited | minor m6 |
| Film without a year (label = bare title, e.g. `It`): `It is a weak match.` | cited | needs a payload without `release_year`; `MovieRecord.release_year` is `int`, so this is unreachable in practice |

A hallucinated film never becomes a citation. The false positives above are all films that **were** retrieved, so the plan criterion "citations only from retrieved ids" holds.

**Fixed abstention vs §1 "if nothing fits, say so".** The fixed message tells the user that nothing in the index matches and suggests rephrasing or relaxing a filter, so it satisfies §1. It does throw away the model's own explanation, which prompt rule 5 invites (minor m3).

**`BLOCKED_PREFIX` detection.** This depends on the wording of a library string. It is guarded by a canary test: `test_the_cap_is_four...` asserts `blocked_tool_calls == 1`, `len(tool_calls) == 4`, and that the refusal text is present. A rewording would fail CI instead of silently miscounting. The cap itself does not depend on this detection, only the reporting does (minor m4).

**Credentials and secrets.** The key is read only at call time, and the error is raised before the MCP server is contacted. The question is scrubbed before it reaches the model, the span or `Answer.question`. Tool args are scrubbed. Run metadata goes through `run_metadata`, which rejects credential-like keys and values. `ChatOpenAI` holds the key as a `SecretStr`.

## Findings

### Blocker
None.

### Major
None.

### Minor

- **m1, `src/movie_rag/agent/citations.py:95`.** The label match is a plain substring test (`film.label in text`). When `Alpha (1999)` and `The Alpha (1999)` are both retrieved, an answer naming only `The Alpha (1999)` cites both.
  - Fix: match the label with a boundary regex, e.g. `re.search(rf"(?<![\w'’]){re.escape(film.label)}", text)`, or match longer labels first and mask spans already matched.
  - Add a test for this case.
- **m2, `src/movie_rag/agent/agent.py:253`.** When at least one retrieved film is cited, the model's text is shown verbatim, including invented films and links the model wrote. The PR's own transcript shows `Ghost Call (2011) [ghost-call-2011-9]` in the printed answer. Citations stay correct, but the user still reads the hallucination.
  - Fix: detect `Title (Year)` / `[id]` mentions that match no retrieved film and either strip those sentences or expose them as `Answer.unverified_mentions`, with a CLI and UI warning.
  - At minimum, add a backlog item for PR-07/PR-09 (faithfulness will measure it).
- **m3, `src/movie_rag/agent/agent.py:253` vs `prompts/system_v1.md` rule 5.** The prompt invites a short rephrase or filter suggestion when nothing fits, but the code always replaces an uncited answer with `ABSTENTION_TEXT`, so that suggestion is lost.
  - Fix: either keep the model's text when it names no unretrieved `Title (Year)` or id (abstained stays `True`), or remove the suggestion sentence from a future prompt version.
- **m4, `src/movie_rag/agent/agent.py:44,110`.** Blocked calls are recognised by the middleware's English message prefix. The canary test catches a rewording, but this couples the code to a library string.
  - Fix: derive `blocked = requested calls - executed calls`. Executions can be counted by an interceptor or wrapper around the tools, or by matching ToolMessages produced by the tools node rather than by `after_model`.
- **m5, `src/movie_rag/agent/agent.py:197`.** When `mode` is not pinned, `retrieval_mode` in the metadata is the configured default (`hybrid`), even if the model chose `dense` or `sparse` in its `search_movies` args. The trace can misstate the mode actually used.
  - Fix: record `retrieval_mode="auto"` (or the set of modes found in the tool-call args) when `mode is None`.
- **m6, `src/movie_rag/agent/citations.py:82-103`.** Any mention counts as a citation, including negative ones ("Unlike Alpha (1999)…"). This is inherent to text-based selection.
  - Fix: document it in the backlog and check it in PR-09 eval.
  - The prompt's `[movie_id]` convention could become the primary signal, with label matching used only as a fallback.

Note (already backlogged by the builder, not counted): `docs/CREDENTIALS.md`, the file named in the hint, does not exist until PR-11.

## Coverage
Line+branch on `src/movie_rag`: **99.90 %** (gate 80 %). Agent modules are at 97-100 %. I found no gaming: no pragmas, no omits, the gate is unchanged, and the single uncovered line is the `__main__` guard.

VERDICT: PASS
