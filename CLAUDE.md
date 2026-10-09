# Movie Plots RAG — project rules

A movie-discovery RAG agent that answers fuzzy plot questions with cited results (full spec: `KICKOFF.md` §1,
build plan: `docs/BUILD_PLAN.md`, progress: `docs/PR_TRACKER.md`).

## Roles

| Role | Who | May write | May not |
|---|---|---|---|
| **orchestrator** | main session (tool calls without `agent_type`) | `docs/PR_TRACKER.md`, `docs/BLOCKERS.md`, `docs/BACKLOG.md`, `.claude/state/**`; runs CI and merges | write product code, docs or tests |
| **builder** | `.claude/agents/builder.md` | product code, tests, docs for exactly one PR | merge, approve, write `reports/qa/**`, touch protected files |
| **qa-validator** | `.claude/agents/qa-validator.md` | `reports/qa/**` only | commit, push, create/edit/close PRs, change product code |

One PR at a time. Every merge requires `reports/qa/pr-<N>.md` ending in `VERDICT: PASS` plus green CI.

## Non-negotiables (enforced by `.claude/hooks/guardrails.py`)

- No `git push` to `main`/`master`, no force-push, no `--no-verify` / `git commit -n`, no `git config`,
  no `gh repo delete/edit`, `gh secret`, `gh pr review --approve`.
- Never read or print `.env` / `.env.*` (only `.env.example`); no `env` / `printenv` / `export -p` dumps.
- No `curl|sh` / `wget|sh`, no `rm -rf` of `/`, `~`, `$HOME`, `..`; no `pip install` — use `uv`.
- Coverage gate is 80% line+branch; never lower `--cov-fail-under`, never add coverage `omit`/`fail_under` changes.
- Protected files (nobody writes them): `.github/workflows/**`, `.claude/**` (except `.claude/state/**`),
  `CLAUDE.md`, `KICKOFF.md`, `docs/TOKEN_USAGE.md`, `docs/token_usage.jsonl`, `.env*` (except `.env.example`),
  anything outside the repo.
- No `pragma: no cover`, no `pytest.mark.skip`/`xfail`/`pytest.skip(` in `src/` or `tests/` except for `live` tests.
- No secret-shaped strings (`sk-…`, `lsv2_…`) in `src/`.
- Only the orchestrator merges, and only through the QA + CI gate.

## Stack

Python 3.12 · uv · ruff · mypy (strict on `src/`) · pytest + pytest-cov + pytest-asyncio · Qdrant (Docker;
`QdrantClient(":memory:")` in tests) · FastEmbed (dense `BAAI/bge-small-en-v1.5`, 384 dims; sparse `Qdrant/bm25`) ·
FastMCP · LangChain + langchain-openai + langchain-mcp-adapters · Nebius Token Factory (OpenAI-compatible) · RAGAS ·
LangSmith · Streamlit.

## Credentials policy

No credentials exist yet and **every PR must be buildable, testable and mergeable without them**. Nebius-backed code
(agent, RAGAS) raises `MissingCredentialError("set NEBIUS_API_KEY — see docs/CREDENTIALS.md")` at call time, never at
import; tests use fake chat models and real-API tests are `@pytest.mark.live` (skipped by default). LangSmith tracing
is a no-op without a key, with identical code paths. Kaggle download explains what to set; development uses
`tests/fixtures/movies_sample.csv`. GitHub absent → local mode (`docs/prs/`, `make ci`). FastEmbed runs locally.
Env var names live in `.env.example`; never read `.env`.

## Makefile targets

`setup lint format typecheck test cov check ci up down download ingest serve ask ui eval eval-smoke report doctor demo`

- `check` = lint + `ruff format --check` + typecheck + test with the 80% coverage gate.
- `ci` = `check` + gitleaks (if installed) + retrieval half of `eval-smoke`; on success writes
  `.claude/state/ci-<branch-or-PR>.ok` with the commit sha (local-mode merge gate).
- `demo` = `up` + `ingest` on the fixture + a sample retrieval-only query in all three modes.
- Targets not built yet print `not implemented yet (PR-NN)` and exit 0.
- `eval-smoke` runs the retrieval half (Hit@k/MRR on the fixture, no credentials); `eval-smoke LLM=1` adds the
  RAGAS half. CI calls exactly these two forms.
- `test` must keep running `tests/hooks` (they are part of the suite) and pass `--cov-fail-under=80` on
  `src/movie_rag` with `--cov-branch`; the gate lives on the command line, never in `pyproject.toml`.

## Contracts the hooks and CI rely on

- Qdrant image tag is `qdrant/qdrant:v1.15.4` (pinned in `.github/workflows/ci.yml`); `docker-compose.yml` uses the same tag.
- CI installs with `uv sync --frozen --all-extras --dev`, so `uv.lock` must be committed and current.
- QA reports: `reports/qa/pr-<NN>.md` (two-digit, e.g. `pr-03.md`; GitHub mode: the GitHub PR number).
  Local CI marker: `.claude/state/ci-<NN>.ok` containing the branch head sha.
- `logs/` (gitignored) is writable by every role for background-job output.

## Coding conventions

- src layout: `src/movie_rag/...`; one `tests/unit/test_<module>.py` per module.
- No network in unit tests. `@pytest.mark.live` for real API calls, `@pytest.mark.integration` for the Qdrant service.
- All tunables (models, URLs, thresholds, retrieval params) in `config.yaml` + pydantic-settings; never hard-coded.
- Type hints everywhere; Pydantic models at boundaries (MCP I/O, config, eval records).
- `logging`, not `print` (CLI user-facing output excepted via a dedicated function).
- Conventional commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`, `refactor:`).

## Skills and library versions

Load the matching skill in `.claude/skills/` before touching a layer (`qdrant-hybrid`, `fastmcp-server`,
`langchain-agent`, `langsmith-tracing`, `ragas-eval`, `streamlit-ui`, always `pr-workflow`). When a skill disagrees
with the installed library version, the installed version's docs win — record the version and doc URL in the PR.

## Operational notes

- Bash calls time out at 10 minutes: run long jobs in the background (`nohup … > logs/x.log 2>&1 &`) and poll the log.
- `.claude/STOP` present → the orchestrator stops after the current PR.
