# Movie Plots RAG

A movie-discovery RAG agent that answers fuzzy plot questions ("a heist movie where the crew is betrayed by the
getaway driver") with cited results. Hybrid retrieval (dense + BM25 with RRF) over Qdrant, served through an MCP
server, consumed by a LangChain agent, with a Streamlit test page and a RAGAS evaluation harness.

> Status: scaffold. Most `make` targets print `not implemented yet (PR-NN)` until the PR that owns them lands.
> See [`docs/PR_TRACKER.md`](docs/PR_TRACKER.md).

## Quickstart

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), Docker (for Qdrant).

```bash
make setup                  # uv sync --all-extras --dev, installs pre-commit hooks
make check                  # lint + format check + mypy (strict) + tests with the 80% coverage gate
docker compose up -d qdrant # Qdrant v1.15.4 on :6333, with a healthcheck and a named volume
docker compose ps           # STATUS shows "healthy" once /readyz answers
docker compose down         # stop it (data stays in the qdrant_storage volume)
```

No credentials are needed to build, test or run the retrieval path. Tunables live in [`config.yaml`](config.yaml);
environment variable names are in `.env.example`.

## Make targets

| Target | What it does | Status |
|---|---|---|
| `setup` | install dependencies and pre-commit hooks | ready |
| `lint` / `format` / `typecheck` | ruff check / ruff format / mypy strict on `src` | ready |
| `test` | full suite (including `tests/hooks`) with `--cov-branch --cov-fail-under=80` | ready |
| `cov` | test run plus an HTML coverage report | ready |
| `check` | lint + format check + typecheck + test | ready |
| `ci` | `check` + gitleaks (if installed) + retrieval smoke eval (stub until PR-09); writes `.claude/state/ci-<NN>.ok` | ready (smoke eval: PR-09) |
| `up` / `down` | start / stop Qdrant via docker compose | ready |
| `download`, `doctor` | dataset download, environment report | PR-02 |
| `ingest` | chunk, embed and upsert into Qdrant | PR-03 |
| `demo` | `up` + `ingest` on the fixture + sample queries | PR-04 |
| `serve` | MCP server | PR-05 |
| `ask` | CLI agent | PR-06 |
| `ui` | Streamlit page | PR-07 |
| `eval`, `eval-smoke`, `report` | evaluation harness (`eval-smoke LLM=1` adds RAGAS) | PR-09 |

## Built autonomously

This repository is built by an autonomous orchestrator, builder and QA-validator loop, one reviewed PR at a time.
The trail is in the repo:

- Plan: [`docs/BUILD_PLAN.md`](docs/BUILD_PLAN.md) and progress: [`docs/PR_TRACKER.md`](docs/PR_TRACKER.md)
- Token usage and estimated cost: [`docs/TOKEN_USAGE.md`](docs/TOKEN_USAGE.md)
- Blockers and backlog: [`docs/BLOCKERS.md`](docs/BLOCKERS.md), [`docs/BACKLOG.md`](docs/BACKLOG.md)
- QA reports: [`reports/qa/`](reports/qa/) and PR write-ups: [`docs/prs/`](docs/prs/)
- Project rules: [`CLAUDE.md`](CLAUDE.md)
