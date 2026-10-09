# Movie Plots RAG — Autonomous Build Kickoff

> **For the human (2 minutes):**
> ```bash
> mkdir movie-rag && cd movie-rag && git init -b main
> mkdir -p .claude/agents .claude/skills      # must exist before Claude starts, so new agents/skills hot-load
> cp /path/to/KICKOFF.md .                     # this file
> # Optional: gh repo create movie-rag --private --source . --push   (enables real PRs + GitHub CI)
> claude --permission-mode bypassPermissions "Read KICKOFF.md and execute it end to end. Do not stop to ask questions."
> ```
> Run it inside a disposable container or VM: permission prompts are off, the hooks created in Phase 0 are the safety layer.
> **Resume after any interruption:** `claude --permission-mode bypassPermissions "Continue the build per KICKOFF.md from docs/PR_TRACKER.md."`
> **Pause:** `touch .claude/STOP` (the orchestrator stops after the current PR).
> **No credentials are needed to build.** You add them later with `docs/CREDENTIALS.md`.

---

## 0. Instructions to Claude

You are the **orchestrator** of an autonomous build. Execute this file top to bottom without asking questions.
When something is ambiguous, choose the option most consistent with this file, record it under "Decisions" in the
PR description, and continue.

1. **Phase 0** (you, directly, no subagents): create the bootstrap files in §4 exactly as specified, make the hook
   tests pass, commit to `main` as `chore: bootstrap autonomous build`. After this commit you never write product
   code yourself again.
2. **Phases 1–11**: for each PR in §6, in order, run the per-PR loop in §5 using the `builder` and `qa-validator`
   subagents. One PR at a time, never two in parallel.
3. **Handoff** (§9): when PR-11 is merged, print the final summary and stop.

Before each PR: if `.claude/STOP` exists, stop. Re-read `docs/PR_TRACKER.md` — it is your memory across context
compaction and restarts. A PR with status `merged` is never redone; `in-progress` is resumed from its branch.

If after Phase 0 the `builder` / `qa-validator` subagents are not available (the `.claude/agents` directory did not
exist at startup), print exactly: `RESTART NEEDED: run the resume command in KICKOFF.md` and stop.

---

## 1. Product spec

A movie-discovery RAG agent that answers fuzzy plot questions ("a film where a detective loses his memory") with
cited results. It must demonstrate production-minded AI engineering:

- **Hybrid retrieval**: dense + sparse (BM25) vectors in Qdrant, fused with RRF in one Query API call, metadata filters.
- **Retrieval as a service**: a FastMCP server any MCP client can call; the agent and the evaluator both use it.
- **Measured quality**: an evaluation comparing dense-only, sparse-only and hybrid (Hit@k, MRR, RAGAS metrics).
- **Observability**: every ingestion, retrieval and generation step traceable in LangSmith.
- **Test UI**: a Streamlit page to try the system.
- **Stakeholder doc**: a plain-language overview for non-technical readers.

**Fixed stack:** Python 3.12 · uv · ruff · mypy (strict on `src/`) · pytest + pytest-cov + pytest-asyncio · Qdrant
(Docker; `QdrantClient(":memory:")` in tests) · FastEmbed (dense `BAAI/bge-small-en-v1.5`, 384 dims; sparse
`Qdrant/bm25`) · FastMCP · LangChain + langchain-openai + langchain-mcp-adapters · Nebius Token Factory
(OpenAI-compatible, base URL `https://api.tokenfactory.nebius.com/v1/`) · RAGAS · LangSmith · Streamlit.

**Dataset:** Kaggle "Wikipedia Movie Plots" (`jrobischon/wikipedia-movie-plots`), ~35k films, columns
`Release Year, Title, Origin/Ethnicity, Director, Cast, Genre, Wiki Page, Plot`. Plot text is CC BY-SA: keep
attribution and Wiki links in every answer. English only, under 2 GB.

**Preprocessing:** drop plots under 50 words; normalise genre/origin to lowercase, `"unknown"` → null; stable
`movie_id` = slug(title) + year + row index; fixed random seed for any sampling.

**Ingestion (`make ingest`):** sentence-aware chunks of ~250 tokens with 40 overlap (plots under 250 tokens stay
whole); prepend `Title (Year) | Genre | Director` to every chunk; dense + BM25 vectors per chunk (named vectors
`dense`, `bm25` with `Modifier.IDF`); point id = uuid5(`movie_id:chunk_idx`); batches of 256; resumable (skip existing
ids); payload indexes on `release_year` (int), `origin`, `genre`, `movie_id` (keyword); logs rows read/dropped,
chunks written, elapsed; one LangSmith run per ingest when tracing is on.

**Retrieval:** modes `dense | sparse | hybrid` (default hybrid) through ONE code path. Hybrid = two prefetches (top 50
each) fused with RRF; filters (`year_from`, `year_to`, `genre`, `origin`) applied **inside** each prefetch; group by
`movie_id` so each film appears once with its best chunk; default `top_k` 8. All parameters in `config.yaml`.
Stretch: cross-encoder reranker as a 4th mode.

**MCP server (FastMCP), four tools, no LLM calls inside:**

| Tool | Args | Returns |
|---|---|---|
| `search_movies` | query, mode, top_k, year_from, year_to, genre, origin | ranked films: movie_id, title, year, director, genre, origin, score, snippet (≤400 chars), wiki URL |
| `get_movie` | movie_id or exact title | full metadata + full plot |
| `find_similar` | movie_id, top_k | closest films by plot vector, excluding the input |
| `list_filters` | — | valid genres, origins, year range |

Pydantic-validated inputs; bad input → `ToolError` with a clear message; HTTP transport (path `/mcp`) for Docker
and the agent, stdio for local MCP clients; LLM-oriented docstrings; a LangSmith span per tool call (mode, filters,
latency, result count).

**Agent:** LangChain tool-calling agent (`create_agent` in LangChain ≥1.0) using the MCP tools via
`langchain-mcp-adapters`, model `ChatOpenAI(base_url=<nebius>, temperature=0)`. Max 4 tool calls per question.
Answer rules (versioned prompt `prompts/system_v1.md`): only facts from tool results; cite every film as
`Title (Year)` + Wikipedia link; at most 5 films, one sentence each on why it matches; if nothing fits, say so.
Citations are built from tool results, never from free text. CLI: `make ask Q="..."`.

**Evaluation:** 40 questions, 10 per type — fuzzy plot, exact entity, filtered, unanswerable. Generated with a fixed
seed by paraphrasing sampled plots (n-gram overlap guard: no 4-word overlap with the plot), reviewed by the QA agent.
Metrics: Hit@k, MRR, abstention rate (own code, deterministic); RAGAS faithfulness, response relevancy, context
precision, context recall with a judge model different from the generator; latency p50/p95 and tokens per question.
One LangSmith experiment per mode. Report in `reports/EVAL_RESULTS.md`.

**Observability:** env `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`. Metadata on every run: git sha,
prompt version, chat model, embedding model, retrieval mode, config hash. Never put secrets in traces.

**Streamlit page (`make ui`):** sidebar (mode, top_k, year range, genre/origin from `list_filters`, retrieval-only
toggle), chat input with 4 example questions, answer with citations, "Retrieved films" expander (score, snippet,
year, genre), "Agent steps" expander (tool calls), latency/tokens/trace link, "Compare modes" tab (retrieval only),
friendly errors. Logic in `ui/service.py`, rendering in `ui/app.py`, tested with `streamlit.testing.v1.AppTest`.

**Non-goals:** production UI, fine-tuning, multi-user auth, caching layers, deployment, non-English data.

---

## 2. Credentials are deferred — build everything to run without them

No API keys or accounts exist yet. The human adds them later. **Every PR must be completable, testable and mergeable
with no credentials.** Rules:

| Integration | Env vars (names only, in `.env.example`) | Behaviour without credentials |
|---|---|---|
| Nebius (chat + judge) | `NEBIUS_API_KEY`, `NEBIUS_BASE_URL`, model ids in `config.yaml` | Agent and RAGAS raise a clear `MissingCredentialError("set NEBIUS_API_KEY — see docs/CREDENTIALS.md")` at call time, never at import. Tests use fake chat models. Live tests marked `@pytest.mark.live`, skipped by default |
| LangSmith | `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` | Tracing is a no-op; code paths identical; dataset upload and experiments skip with a log line |
| Kaggle | `KAGGLE_USERNAME`, `KAGGLE_KEY` | `make download` explains what to set. Development uses the committed fixture (below). You may also look for a public no-auth mirror of the same dataset (e.g. on Hugging Face); use it only if its license and columns match, and document it in `docs/DATASET.md` |
| GitHub | `gh auth status` | Local mode (§7): PR descriptions in `docs/prs/`, CI run locally with `make ci` |
| Embeddings | none (FastEmbed runs locally) | Works offline after the first model download |

**Fixture data** (`tests/fixtures/movies_sample.csv`): ~300 rows in the exact Kaggle schema. If no real data is
reachable, write realistic synthetic records (invented titles, clearly marked synthetic in
`tests/fixtures/README.md`), covering several genres, origins and decades, plus a few short plots that cleaning must
drop and a few long plots that must be chunked. Everything — ingestion, retrieval, MCP, UI retrieval-only mode,
deterministic metrics and the smoke eval's retrieval half — must run end to end on this fixture with `make demo`.

**Model defaults in `config.yaml`** (validated later against `GET /v1/models` by `make doctor`): generator
`Qwen/Qwen3-30B-A3B-Instruct-2507`, judge `openai/gpt-oss-120b`.

**`make doctor`**: checks Docker/Qdrant reachability, each credential (present? valid with one cheap call?), model ids
against `/v1/models`, and prints a table with what works and what to do next. Exit 0 even when credentials are
missing (it is a report, not a gate).

**`docs/CREDENTIALS.md`** (written in PR-11): step-by-step for the human — where to create each key, where to put it
(`.env` locally, GitHub Actions secrets for CI), `make doctor` to verify, then the exact command sequence to produce
real results: `make download && make ingest && make eval MODE=dense && make eval MODE=sparse && make eval MODE=hybrid
&& make report`, which refreshes `reports/EVAL_RESULTS.md`, the README results table and the numbers in
`docs/SOLUTION_OVERVIEW.md`.

---

## 3. Repository layout (target)

```text
movie-rag/
  KICKOFF.md  CLAUDE.md  README.md  Makefile  pyproject.toml  uv.lock  config.yaml
  docker-compose.yml   .env.example  .gitignore  .pre-commit-config.yaml
  .claude/   agents/  skills/  hooks/  settings.json  state/ (gitignored)
  .github/   workflows/ci.yml  pull_request_template.md
  docs/      BUILD_PLAN.md  PR_TRACKER.md  TOKEN_USAGE.md  token_usage.jsonl  DATASET.md  EVAL_SET.md
             CREDENTIALS.md  SOLUTION_OVERVIEW.md  DEMO.md  BACKLOG.md  BLOCKERS.md  adr/  prs/  img/
  src/movie_rag/  config.py  observability.py  errors.py
             ingest/  retrieval/  mcp_server/  agent/ (prompts/)  ui/  eval/ (data/)  cli.py
  tests/     unit/  integration/  hooks/  fixtures/
  reports/   qa/  EVAL_RESULTS.md
  data/      (gitignored)
```

---

## 4. Phase 0 — bootstrap (you write these directly)

Write the files in this order. `.claude/settings.json` goes **last**, so the guardrails do not block Phase 0 itself.

### 4.1 `CLAUDE.md`
Project rules every agent loads. Include: the three roles; the non-negotiables from §4.4; the stack from §1; the
credentials policy from §2 (one paragraph); the Makefile targets from §4.7; coding conventions (src layout, a
`tests/unit/test_<module>.py` per module, no network in unit tests, `@pytest.mark.live` for real API calls,
`@pytest.mark.integration` for the Qdrant service, all tunables in `config.yaml` + pydantic-settings, type hints,
Pydantic at boundaries, `logging` not `print`, conventional commits); "load the matching skill before touching a
layer; when a skill disagrees with the installed library version, the installed version's docs win — record version
and doc URL in the PR"; "Bash calls time out at 10 minutes: run long jobs in the background and poll the log".

### 4.2 Subagents — `.claude/agents/*.md`

`builder.md`
```markdown
---
name: builder
description: Implements exactly one PR from docs/BUILD_PLAN.md — code, tests, docs — then opens the PR (GitHub or local mode). Use for all product code changes.
tools: Read, Write, Edit, Grep, Glob, Bash, Skill, WebFetch, WebSearch, TodoWrite
model: sonnet
effort: high
maxTurns: 150
---
You are the builder. Implement one PR end to end to a standard a senior reviewer would merge.
Before coding: read CLAUDE.md, the PR's row in docs/BUILD_PLAN.md, and load every relevant skill (always `pr-workflow`).
If a library API is uncertain, check the installed version (`uv pip show <pkg>`) and its docs with WebFetch; note version + URL in the PR.
Write tests with the code: unit tests for every public function, contract tests for MCP tools, AppTest for UI. Assert behaviour.
No network in unit tests; no credentials needed (see CLAUDE.md credentials policy).
Run `make check` until green; coverage ≥ 80% line+branch on src/movie_rag.
Finish per the `pr-workflow` skill and return: PR number or local PR id, 5-line summary, coverage %, open risks.
Never: merge, approve, write reports/qa/, edit .github/workflows/, .claude/, CLAUDE.md, KICKOFF.md, docs/TOKEN_USAGE.md;
add `# pragma: no cover`, coverage omit, skip/xfail outside live tests; read .env; hard-code models, URLs or thresholds.
A stop hook re-runs lint, types and tests when you finish; if red you will be sent back.
```

`qa-validator.md`
```markdown
---
name: qa-validator
description: Independent reviewer for one PR. Verifies acceptance criteria with its own evidence, test quality, coverage integrity and guardrails, then writes a PASS/FAIL verdict. Read-only on product code.
tools: Read, Grep, Glob, Bash, Write, Skill
model: opus
effort: high
maxTurns: 80
---
You did not write this code and you do not fix it. Default to FAIL when evidence is missing.
1. Read the diff (`gh pr diff N` or `git diff main...<branch>`) and the PR's row in docs/BUILD_PLAN.md.
2. Every acceptance criterion needs evidence you produced yourself (command + output, or file:line).
3. Run `make check` yourself; record coverage % and test count.
4. Automatic FAIL if the diff: adds `pragma: no cover`, coverage omit/fail_under changes, skip/xfail outside live tests;
   touches .github/workflows/, .claude/, CLAUDE.md, KICKOFF.md; commits secrets, .env content or raw data;
   hard-codes model names, URLs or retrieval parameters in src/; has tests that assert nothing meaningful;
   needs a credential to pass; goes beyond the plan row's scope.
5. Load the skill for each touched layer and apply its review checklist.
Write reports/qa/pr-<N>.md: summary, criteria table (criterion | evidence | ✅/❌), integrity checks, findings by severity
(blocker/major/minor) with file:line and a concrete fix, coverage %, and as the very last line exactly `VERDICT: PASS` or `VERDICT: FAIL`.
PASS requires zero blockers and zero majors. In GitHub mode also run `gh pr comment N --body-file reports/qa/pr-<N>.md`.
You may only write under reports/qa/. You never commit, push or edit the PR.
```

You (the main session) are the orchestrator; no agent file is needed for you, but the guardrail hook treats a tool
call without `agent_type` as the orchestrator role.

### 4.3 Skills — `.claude/skills/<name>/SKILL.md`
Each has frontmatter (`name`, one-line `description` of when to use it), doc URLs, the working pattern, how to test,
and a **review checklist** the QA agent applies. Every skill says: check the installed version first; that version's
docs win.

- **`qdrant-hybrid`** — docs: qdrant.tech hybrid-queries, Query API, grouping, FastEmbed sparse. Collection with
  `vectors_config={"dense": VectorParams(size, COSINE)}`, `sparse_vectors_config={"bm25": SparseVectorParams(modifier=Modifier.IDF)}`;
  payload indexes; `TextEmbedding.embed` for documents vs `query_embed` for queries (same for `SparseTextEmbedding("Qdrant/bm25")`);
  uuid5 ids + skip-existing resumable upserts; hybrid = `query_points_groups(prefetch=[Prefetch(query=dense, using="dense", limit, filter),
  Prefetch(query=SparseVector, using="bm25", limit, filter)], query=FusionQuery(fusion=Fusion.RRF), group_by="movie_id", group_size=1)`;
  single modes query their vector directly with the same filter/grouping code. Checklist: query_embed for queries, IDF modifier,
  filters inside prefetch, one shared function for all modes, idempotency test, params from config.
- **`fastmcp-server`** — docs: gofastmcp.com tools, running-server, testing, client. `mcp = FastMCP("movie-rag")`, `@mcp.tool`
  with `Annotated[..., Field(description=...)]` args and Pydantic return models; `ToolError` for bad input; `mcp.run(transport="http")`
  (older releases: `"streamable-http"`) and stdio; retriever injected via a factory so tests pass a fake; contract tests with
  `async with Client(server) as c: await c.call_tool(...)`; schema snapshot test. Checklist: four tools, LLM-readable docstrings,
  ToolError paths tested, no LLM imports in mcp_server/, spans on every tool, both transports start.
- **`langchain-agent`** — docs: docs.langchain.com agents + MCP pages, langchain-mcp-adapters README, Nebius Token Factory API intro.
  `ChatOpenAI(model=cfg.chat_model, base_url=cfg.nebius_base_url, api_key=..., temperature=0)` created lazily (MissingCredentialError
  without key); `MultiServerMCPClient({"movies": {"transport": "streamable_http", "url": cfg.mcp_url}})`; `create_agent(model, tools,
  system_prompt)`; tool-call cap via the installed version's limit middleware or a counter + `recursion_limit`; typed `Answer(text,
  citations, tool_calls, usage)`; tests with `GenericFakeChatModel` scripted to emit tool calls. Checklist: no hard-coded URL/model,
  citations only from retrieved ids, cap and abstention tested, prompt version in trace metadata.
- **`langsmith-tracing`** — docs: docs.smith.langchain.com observability + evaluation. `@traceable(run_type="retriever")` with
  document-shaped outputs; `observability.run_metadata()`; `Client().create_dataset/create_examples`, `client.evaluate(...,
  experiment_prefix=mode)`; everything no-ops without a key; `LANGSMITH_TRACING=false` in conftest. Checklist: works with tracing off,
  metadata fields present, no secrets in payloads.
- **`ragas-eval`** — docs: docs.ragas.io (metrics list, custom LLM). Pin the version first; 0.2/0.3-era API: `evaluate`,
  `EvaluationDataset`, `Faithfulness`, `ResponseRelevancy`, `LLMContextPrecisionWithReference`, `LLMContextRecall`,
  `LangchainLLMWrapper`; newer releases use `ragas.metrics.collections` + `llm_factory` — use what the pinned version documents.
  Judge ≠ generator. Deterministic metrics first (no LLM). Report format per §1. Checklist: version + judge recorded,
  overlap guard tested, deterministic metrics tested on hand-computed cases, report numbers traceable to JSON.
- **`streamlit-ui`** — docs: docs.streamlit.io API + app testing. Layout per §1; `st.cache_resource` for clients; AppTest tests
  with a faked `ui.service`; retrieval-only mode makes no LLM call; missing-credential state shows how to fix it. Checklist: starts
  headless, retrieval-only works with no keys, friendly errors, no secrets displayed.
- **`pr-workflow`** — branch `pr/<NN>-<slug>` from `main`; conventional commits; `make check` green before opening; open the PR per
  §7 with `.github/pull_request_template.md` fully filled (summary, plan row, acceptance checklist with evidence, library versions +
  doc URLs, decisions, coverage %, follow-ups moved to `docs/BACKLOG.md`); fixes go as new commits on the same branch (no
  force-push, no new PR); reply to QA findings point by point.

### 4.4 Guardrail hooks — `.claude/hooks/`
Python 3 scripts, stdlib only, reading the hook JSON from stdin; exit 2 + stderr message = block; exit 0 = no
objection. Role = `agent_type` field (`builder`, `qa-validator`), absent = `orchestrator`. Repo paths resolved
against `CLAUDE_PROJECT_DIR`.

**`guardrails.py`** (PreToolUse on `Bash|Write|Edit|MultiEdit|NotebookEdit|Read`) must block:

| Rule | Applies to |
|---|---|
| `git push` to `main`/`master`; any force-push; `--no-verify` or `git commit -n`; `git config`; `gh repo delete/edit`, `gh secret`, `gh pr review --approve` | everyone |
| Read/print `.env` or `.env.*` except `.env.example` (Read tool, `cat/grep/head/tail/less/source .env`); `env`/`printenv`/`export -p` dumps | everyone |
| `curl/wget ... \| sh`; `rm -rf` of `/`, `~`, `$HOME`, `..`; `pip install` (use uv) | everyone |
| `--cov-fail-under` below 80 | everyone |
| Writing (tool or shell redirect/tee) `.github/workflows/**`, `.claude/**` except `.claude/state/**`, `CLAUDE.md`, `KICKOFF.md`, `docs/TOKEN_USAGE.md`, `docs/token_usage.jsonl`, `.env*` except `.env.example`; any path outside the repo | everyone |
| New content in `src/` or `tests/` containing `pragma: no cover`, `pytest.mark.skip/xfail` or `pytest.skip(` (unless the text also mentions `live`); `pyproject.toml` edits setting coverage `omit` or `fail_under`; secret-shaped strings (`sk-…`, `lsv2_…`) in `src/` | everyone |
| Writing anything except `docs/PR_TRACKER.md`, `docs/BLOCKERS.md`, `docs/BACKLOG.md`, `.claude/state/**`; `sed -i`/`perl -pi` on `src/` or `tests/` | orchestrator |
| Writing anything except `reports/qa/**`; `git commit/push`, `gh pr create/edit/close` | qa-validator |
| Writing `reports/qa/**` | builder |
| Merge (`gh pr merge N` or local `git merge` into `main`): only the orchestrator; only when `reports/qa/pr-N.md` exists and its last non-empty line is `VERDICT: PASS`; GitHub mode also requires `gh pr checks N` exit 0 and refuses `--admin`; local mode requires `.claude/state/ci-<N>.ok` written by `make ci` within the last hour for the branch's current commit | merge |

**`stop_gates.py`** (SubagentStop, matcher `builder|qa-validator`): for `builder`, run `make lint typecheck test`
(timeout 9 min) and exit 2 with the last 3,000 chars of output if red; for `qa-validator`, exit 2 unless a
`reports/qa/pr-*.md` modified in the last 3 hours ends with a `VERDICT:` line. After 3 blocks for the same
`agent_id` (counter in `.claude/state/`), let it stop so it can report failure.

**`track_tokens.py`** (Stop and SubagentStop, no matcher; never fails the run — catch everything, exit 0):
- Hook mode: find the transcript (`agent_transcript_path` if present; else `<dir of transcript_path>/<session_id>/subagents/agent-<agent_id>.jsonl`;
  else glob `~/.claude/projects/**/agent-<agent_id>.jsonl`; main session = `transcript_path`). Sum `message.usage`
  `input_tokens`, `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`, **deduplicating by
  `message.id`** (one API message spans several lines). Append a JSON row to `docs/token_usage.jsonl`: ts, pr
  (from `.claude/state/current_pr`, else branch `pr/NN-…` → `PR-NN`), session, agent, agent_key, the four counts.
- `--render`: rebuild `docs/TOKEN_USAGE.md` keeping the last row per (session, agent_key): a table by PR × agent
  (input, output, cache write, cache read, estimated cost USD), and run totals. Estimated cost uses a per-model price
  table in `.claude/hooks/prices.json` (fill it with current public Claude prices; label the column "estimate").

**Hook tests** — `tests/hooks/test_guardrails.py` and `test_track_tokens.py` (run by `make test`): one parametrised
case per rule above (blocked and allowed variants, e.g. `git push -u origin pr/03-x` allowed, `git push origin main`
blocked, `cat .env.example` allowed), merge gate with PASS/FAIL/missing verdict, and the dedup logic with a fake
transcript. They must pass before you commit Phase 0.

### 4.5 `.claude/settings.json` (write last)
```json
{
  "env": { "CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH": "1", "BASH_DEFAULT_TIMEOUT_MS": "600000" },
  "permissions": {
    "deny": ["Read(./.env)", "Read(./.env.*)", "Edit(./.github/workflows/**)", "Edit(./CLAUDE.md)", "Edit(./KICKOFF.md)",
             "Bash(git push --force*)", "Bash(git push -f*)", "Bash(git push origin main*)", "Bash(gh pr merge * --admin*)", "Bash(gh repo delete*)"],
    "allow": ["Bash(uv *)", "Bash(make *)", "Bash(git *)", "Bash(gh pr *)", "Bash(gh run *)", "Bash(docker compose *)", "WebFetch", "WebSearch"]
  },
  "hooks": {
    "PreToolUse": [{ "matcher": "Bash|Write|Edit|MultiEdit|NotebookEdit|Read",
      "hooks": [{ "type": "command", "command": "python3", "args": ["${CLAUDE_PROJECT_DIR}/.claude/hooks/guardrails.py"], "timeout": 90 }] }],
    "SubagentStop": [
      { "matcher": "builder|qa-validator",
        "hooks": [{ "type": "command", "command": "python3", "args": ["${CLAUDE_PROJECT_DIR}/.claude/hooks/stop_gates.py"], "timeout": 600 }] },
      { "hooks": [{ "type": "command", "command": "python3", "args": ["${CLAUDE_PROJECT_DIR}/.claude/hooks/track_tokens.py"], "timeout": 60 }] }],
    "Stop": [{ "hooks": [{ "type": "command", "command": "python3", "args": ["${CLAUDE_PROJECT_DIR}/.claude/hooks/track_tokens.py"], "timeout": 60 }] }]
  }
}
```

### 4.6 CI — `.github/workflows/ci.yml` (human-owned after Phase 0)
On `pull_request` and `workflow_dispatch`, `concurrency` cancel-in-progress, `LANGSMITH_TRACING=false`. Jobs:
1. **quality** — `astral-sh/setup-uv`, `uv sync --frozen --all-extras --dev`, `ruff check .`, `ruff format --check .`, `mypy src`.
2. **tests** — Qdrant service container (`qdrant/qdrant`, port 6333, `QDRANT_URL`), FastEmbed model cache,
   `pytest -m "not live" --cov=src/movie_rag --cov-branch --cov-report=term-missing --cov-report=xml --cov-fail-under=80`,
   coverage markdown in `$GITHUB_STEP_SUMMARY`, upload `coverage.xml` artifact.
3. **secret-scan** — `gitleaks/gitleaks-action@v2` with full history.
4. **smoke-eval** — needs tests; runs on label `eval` or dispatch; retrieval half (Hit@k/MRR on the fixture) always runs;
   the LLM half runs only `if: env.NEBIUS_API_KEY != ''` and otherwise writes "skipped: no NEBIUS_API_KEY" to the summary.
   Secrets referenced: `NEBIUS_API_KEY`, `LANGSMITH_API_KEY` (absent for now — must not fail the job).

Also write `.github/pull_request_template.md` with the sections listed in the `pr-workflow` skill.

### 4.7 Makefile contract
Targets (later PRs implement them; until then they print `not implemented yet (PR-NN)` and exit 0):
`setup lint format typecheck test cov check ci up down download ingest serve ask ui eval eval-smoke report doctor demo`.
- `check` = lint + `ruff format --check` + typecheck + test with the 80% coverage gate.
- `ci` = `check` + gitleaks (if installed) + the retrieval half of `eval-smoke`; on success writes
  `.claude/state/ci-<branch-or-PR>.ok` containing the commit sha (used by the local-mode merge gate).
- `demo` = `up` + `ingest` on the fixture + a sample retrieval-only query in all three modes.

### 4.8 Tracking files
`docs/BUILD_PLAN.md` = §6 copied verbatim. `docs/PR_TRACKER.md`:

```markdown
| ID | Title | Branch | Status | PR | CI | QA | Attempts | Merged (UTC) | Notes |
|---|---|---|---|---|---|---|---:|---|---|
| PR-01 | Scaffold, tooling, CI | pr/01-scaffold | todo | | | | 0 | | |
```
(one row per PR in §6; statuses `todo · in-progress · merged · blocked · skipped`).
`docs/TOKEN_USAGE.md` = rendered by `track_tokens.py --render` (initially "No runs recorded yet").
Also `.env.example` (names from §2, empty values), `.gitignore` (`.env`, `.env.*`, `!.env.example`, `data/`, `logs/`,
`.claude/state/`, `.claude/STOP`, caches, `coverage.xml`), empty `docs/BACKLOG.md` and `docs/BLOCKERS.md`.

Commit Phase 0, then push if a GitHub remote exists.

---

## 5. Per-PR loop (orchestrator)

For the next PR whose status is not `merged`/`skipped`:

1. Write the PR id to `.claude/state/current_pr`. `git checkout main && git pull --ff-only` (if remote). Create or
   check out `pr/<NN>-<slug>`. Set the tracker row `in-progress`, attempts +1.
2. **Delegate to `builder`** with: PR id, the full plan row (scope + acceptance criteria), branch name, GitHub or local
   mode, and any findings from a previous round. Ask for the PR number/id and a short summary back.
3. **CI**: GitHub mode → `gh pr checks N --watch --interval 30`; local mode → run `make ci` yourself. Red → send the
   failing excerpt to the builder (resume the same builder with SendMessage when possible). Max 3 CI rounds.
4. **Delegate to `qa-validator`** with PR number/id, plan row, mode. It writes `reports/qa/pr-<N>.md`.
5. FAIL → findings to builder → back to 3. Max 3 QA rounds.
6. PASS → merge: GitHub `gh pr merge N --squash --delete-branch`; local `git checkout main && git merge --squash
   pr/<NN>-<slug> && git commit` (message = PR title + summary). The hook re-checks the gates.
7. Update the tracker row (`merged`, link/id, CI ✅, QA ✅, UTC time), run `python3 .claude/hooks/track_tokens.py --render`,
   commit both as `chore(tracking): <PR-id> [skip ci]` (on `main` in local mode; in GitHub mode include them in the next
   PR branch's first commit). Delete `.claude/state/current_pr`.
8. Stuck after 3 rounds → row `blocked` with a one-line reason, details in `docs/BLOCKERS.md`, **skip to the next PR only
   if it does not depend on the blocked one**; otherwise stop and print the blocker.

Keep your own context small: ask subagents for summaries, not logs. Never lower a gate, skip a check, or widen scope.

---

## 6. Build plan (copy to `docs/BUILD_PLAN.md`)

Done = every acceptance criterion has QA-verified evidence, `make check` green, coverage ≥ 80%, **no credential
needed**. Cut order if behind: PR-12 → `find_similar` → Compare-modes tab.

**PR-01 — Scaffold, tooling, CI.** uv project, `pyproject.toml` (ruff, mypy strict, pytest markers `live` and
`integration`, coverage `branch = true`, `source = ["src/movie_rag"]`), Makefile per §4.7, pre-commit (ruff, ruff-format,
check-yaml, end-of-file, gitleaks), `docker-compose.yml` with Qdrant (pinned tag, healthcheck, volume), `config.yaml`
skeleton, `src/movie_rag/__init__.py`, smoke test, README skeleton (quickstart, make targets, "Built autonomously" with
links to the trackers). ✅ `make setup && make check` green · hook tests still green · `docker compose up -d qdrant`
healthy (or documented if Docker is unavailable in this environment) · CI workflow green in GitHub mode.

**PR-02 — Settings, errors, observability, data.** `config.py` (pydantic-settings, env + yaml, config hash),
`errors.py` (`MissingCredentialError`), `observability.py` (tracing switch, `run_metadata()`), `ingest/download.py`
(Kaggle; clear message without credentials; optional no-auth mirror per §2), `ingest/clean.py`, the fixture per §2,
`docs/DATASET.md`, `make doctor` (first version). ✅ cleaning rules tested on hand-built rows · id stable across runs ·
missing credentials produce the documented message, never a stack trace · secrets never logged (test).

**PR-03 — Chunking, embeddings, ingestion.** Per §1. ✅ chunker tests (short = 1 chunk, sentence boundaries, header on
every chunk) · re-ingest leaves point count unchanged · `make ingest` on the fixture logs counts that match Qdrant.

**PR-04 — Hybrid retrieval.** Per §1. ✅ one code path for three modes · filters correct inside prefetch (tests) ·
one row per film · sample query output for all modes pasted in the PR.

**PR-05 — FastMCP server.** Per §1, plus `mcp-server` service in docker-compose with healthcheck and `make serve`.
✅ in-memory contract tests for all tools and error paths · schema snapshot · stdio smoke test lists the tools ·
no LLM imports in `mcp_server/` (test).

**PR-06 — Agent + CLI.** Per §1. ✅ fake-model tests: 4-call cap, abstention, citations only from retrieved ids,
metadata attached · without `NEBIUS_API_KEY`, `make ask` prints the credentials hint and exits 2 · one `live` test
ready for later.

**PR-07 — Streamlit test page.** Per §1. ✅ AppTest: retrieval-only happy path (no LLM call), agent path with fake
service, missing-credential message, MCP-down error · screenshot `docs/img/ui.png` (headless browser if available,
otherwise documented as a follow-up in CREDENTIALS.md).

**PR-08 — Evaluation set.** Generator script (fixed seed, paraphrase prompt, n-gram overlap guard) that runs once a
key exists; **now**: write the 40 questions directly from fixture films (and, for unanswerable ones, plausible but
absent films), each with type, gold `movie_id`/filters, in `eval/data/questions_v1.jsonl`; `docs/EVAL_SET.md`;
LangSmith upload command that skips without a key. ✅ overlap guard tested and passing for all fuzzy questions · QA
reviewed all 40 and listed rejections (regenerated).

**PR-09 — Evaluation runner.** Hit@k, MRR, abstention (own code); RAGAS wiring with judge ≠ generator (runs with key,
clear skip without); `make eval MODE=…` writing `reports/eval_<mode>.json`; `make report` building
`reports/EVAL_RESULTS.md`; `make eval-smoke`. ✅ deterministic metrics tested on hand-computed cases · `make eval` for
all three modes runs **now** on the fixture for retrieval metrics, and `EVAL_RESULTS.md` shows those numbers with LLM
metrics marked "pending credentials" · report records ragas version and judge model.

**PR-10 — ADRs and README.** `docs/adr/` 001 Qdrant, 002 RRF + parameters, 003 chunking, 004 local embeddings; README:
architecture (Mermaid), quickstart without keys (`make demo`), results section generated by `make report` (fixture
retrieval numbers now, full numbers after credentials), honest note on what the fixture can and cannot show.
✅ a fresh clone reaches `make demo` using only the README · every number in README traces to a report file.

**PR-11 — Stakeholder overview, credentials guide, demo.** `docs/SOLUTION_OVERVIEW.md` per §8; `docs/CREDENTIALS.md` per
§2; `docs/DEMO.md` (2-minute script with questions to ask in the UI); final acceptance checklist (§9) ticked with
evidence. ✅ overview ≤ 1,500 words, no unexplained jargon, numbers match reports or are marked pending ·
CREDENTIALS.md commands verified up to the point where a key is required.

**PR-12 — Stretch: reranker.** Fourth mode `hybrid_rerank` (cross-encoder over top 30), in eval, UI and MCP.
✅ eval shows retrieval gain and latency cost on the fixture.

---

## 7. GitHub mode vs local mode

At the start of each PR run `gh auth status && git remote get-url origin`.
- **GitHub mode** (both succeed): push the branch, `gh pr create --base main --body-file …`, CI on GitHub, merge with
  `gh pr merge`. Add label `eval` from PR-09 on.
- **Local mode** (otherwise): the PR description goes to `docs/prs/PR-NN.md` (same template, committed on the branch);
  CI = `make ci`; merge = squash merge into local `main`. When the human later adds a remote, `git push -u origin main`
  publishes everything; the PR descriptions remain in `docs/prs/`.

---

## 8. Stakeholder overview — `docs/SOLUTION_OVERVIEW.md`

For readers with no AI background: ≤ 1,500 words, sentences under 25 words, a short glossary, every number copied
from `reports/EVAL_RESULTS.md` (or marked "pending credentials").
1. The problem — people remember films by story, not title; keyword search fails them.
2. What it does — one worked example: question, answer, cited films, Wikipedia links.
3. How it works in five steps — Mermaid flow: plot library → split and indexed two ways (meaning and exact words) →
   the assistant searches → answers only from what it found → cites each film.
4. How we know it works — 40 test questions; how often the right film is in the top results; how often answers stay
   true to sources; how often it says "I don't know" when it should. One small table.
5. Cost and speed — seconds per answer, cost per 1,000 questions (pending credentials if not measured).
6. Limits — English, Wikipedia plots only, no ratings or opinions, weak on very short plots.
7. How it was built — two days, AI agents under automated checks, every change reviewed by a separate AI reviewer and
   tested before acceptance; PR and token trackers as the audit trail (with totals from `docs/TOKEN_USAGE.md`).
8. What's next — connect credentials and run the full evaluation; two or three options with rough effort.
9. Glossary — RAG, embedding, vector database, hybrid search, MCP, evaluation.

---

## 9. Handoff (after PR-11, or PR-12 if built)

Final acceptance checklist (tick each with evidence in PR-11):
- [ ] Fresh clone → `make setup && make demo` works with no credentials
- [ ] `make check` green on `main`; every merged PR has `VERDICT: PASS` in `reports/qa/`
- [ ] Streamlit page answers in retrieval-only mode and shows the credentials hint for agent mode
- [ ] `reports/EVAL_RESULTS.md` has fixture retrieval metrics for all three modes; LLM metrics marked pending
- [ ] `docs/PR_TRACKER.md` and `docs/TOKEN_USAGE.md` complete for every PR
- [ ] `docs/CREDENTIALS.md`, `docs/SOLUTION_OVERVIEW.md`, `docs/DEMO.md` present
- [ ] No secrets, `.env` or raw data committed (gitleaks clean)

Then print a final summary: PRs merged/blocked/skipped, total tokens and estimated cost from `docs/TOKEN_USAGE.md`,
coverage on `main`, and the three commands the human runs next:
`cp .env.example .env  # fill keys` → `make doctor` → the full-results sequence from `docs/CREDENTIALS.md`.
