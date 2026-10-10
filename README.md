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
environment variable names are in `.env.example`. `make doctor` shows what is configured and what to do next.
Data: development and CI use the synthetic [`tests/fixtures/movies_sample.csv`](tests/fixtures/README.md); the real
dataset and its licence are described in [`docs/DATASET.md`](docs/DATASET.md).

## Ingestion

```bash
make up                  # Qdrant on :6333
make ingest              # downloaded dataset if data/raw has it, otherwise the synthetic fixture (with a warning)
make ingest FIXTURE=1    # force the fixture;  CSV=path/to.csv ingests another file;  RECREATE=1 rebuilds the collection
```

* **Chunks.** Plots are cut at sentence boundaries into passages of about `ingest.chunk_tokens` (250) tokens with
  `ingest.chunk_overlap` (40) tokens of shared trailing sentences; a plot at or under the budget stays whole. A *token*
  is a WordPiece token of the dense model (`BAAI/bge-small-en-v1.5`), without `[CLS]`/`[SEP]`. The budget covers the
  plot text; every chunk additionally gets the header `Title (Year) | Genre | Director` prepended before embedding.
* **Points.** One point per chunk with named vectors `dense` (384, cosine) and `bm25` (sparse, `Modifier.IDF`), id
  `uuid5(movie_id:chunk_idx)`, written in batches of `ingest.batch_size` (256). Payload: `movie_id`, `title`,
  `release_year`, `director`, `cast`, `genre`, `origin`, `wiki_page`, `chunk_idx`, `n_chunks`, `text`; chunk 0 also
  carries `full_plot` (what `get_movie` reads, by id). Payload indexes: `release_year` (integer), `origin`, `genre`,
  `movie_id` (keyword).
* **Re-runs.** Ids that already exist are skipped before embedding, so a re-run (or a resume after a crash) is cheap and
  leaves the point count unchanged. Changed chunking parameters or data need `RECREATE=1`.
* **Output.** Rows read/dropped, films, chunks total/written/skipped, points in Qdrant (read back) and elapsed time,
  logged and printed; one LangSmith run per ingest when `LANGSMITH_TRACING=true` and a key is set.
* **Models** are downloaded on first use into FastEmbed's cache (`FASTEMBED_CACHE_PATH`, default
  `<tmp>/fastembed_cache`).

## Retrieval

```bash
make demo                                              # Qdrant + fixture + the sample query in all three modes
uv run python -m movie_rag.retrieval "a detective loses his memory" --mode hybrid --top-k 5 \
    --year-from 1950 --year-to 1999 --genre thriller   # --origin too; omit --mode to compare all three
```

* **One code path.** `dense`, `sparse` and `hybrid` all go through `Retriever.search` and one `query_points_groups`
  call; the mode only picks the query: one named vector (`dense` or `bm25`), or two prefetches (the same two vectors,
  `retrieval.prefetch_limit` chunks each) fused with RRF.
* **Filters** (`year_from`, `year_to`, `genre`, `origin`) are built once and attached to **each prefetch** in hybrid mode
  (and as `query_filter` in the single-vector modes). `genre` and `origin` match exactly, case-insensitively.
* **One row per film.** Results are grouped by `movie_id`; each film shows its best chunk (`chunk_idx`), the score of
  that chunk, a snippet of at most `retrieval.snippet_max_chars` characters and its Wikipedia link. Scores are only
  comparable within a mode (cosine, BM25 or RRF). `Retriever.get_movie` reads the whole plot from chunk 0 by id.
* **Parameters** (`config.yaml`, `retrieval:`): `default_mode`, `top_k`, `prefetch_limit`, `snippet_max_chars`,
  `demo_query`. The RRF constant is not configurable on Qdrant 1.15 (needs qdrant-client and server 1.16).

## MCP server

```bash
make serve                  # streamable HTTP on http://127.0.0.1:8000/mcp (needs `make up` + `make ingest` to answer)
make serve DOCKER=1         # the same as the `mcp-server` compose service on :8000, healthy once /health answers
make serve STDIO=1          # stdio, for local MCP clients (Claude Desktop, MCP Inspector)
```

Four read-only tools, no LLM calls inside, built with FastMCP on top of `movie_rag.retrieval`:

| Tool | Arguments | Returns |
|---|---|---|
| `search_movies` | `query`, `mode` (hybrid), `top_k`, `year_from`, `year_to`, `genre`, `origin` | ranked films: id, title, year, director, genre, origin, score, snippet, wiki URL |
| `get_movie` | `movie_id` or exact `title` (+ `year` for remakes) | full metadata and plot |
| `find_similar` | `movie_id`, `top_k` | closest films by plot vector, never the input film |
| `list_filters` | none | valid genres and origins (most frequent first) and the year range |

Bad input, unknown films and an unreachable or missing index come back as `ToolError` with a message that says what to
do; internals are never exposed. Every call is a LangSmith span (`mcp.<tool>` with mode, filters, latency and result
count, and a nested `retriever.*` span); without `LANGSMITH_API_KEY` tracing is a no-op on the same code path.
Host, port, path and limits live under `mcp:` in `config.yaml` (`MCP__HOST`, `MCP__PORT`, ... override them). The
tool schemas are snapshot-tested (`tests/fixtures/mcp_tools_schema.json`; regenerate with
`UPDATE_MCP_SNAPSHOT=1 uv run pytest tests/unit/test_mcp_server.py -k snapshot`).

## Agent

```bash
make ask Q="a film where a hotel telephone operator overhears a murder being planned"
```

`src/movie_rag/agent/` is a LangChain (`create_agent`) tool-calling agent. Its tools are the four MCP tools, loaded
through `langchain-mcp-adapters` from `mcp.url` (so `make serve` must be running); the chat model is
`ChatOpenAI(base_url=llm.base_url, model=llm.chat_model, temperature=0)` on Nebius Token Factory and is created when a
question is asked, so importing the package or running the tests never needs a key. Without `NEBIUS_API_KEY`,
`make ask` prints `set NEBIUS_API_KEY — see docs/CREDENTIALS.md` and exits 2.

- At most `llm.max_tool_calls` (4) tool calls per question, enforced with `ToolCallLimitMiddleware`; further calls are
  refused (the model sees an error) and counted in `Answer.blocked_tool_calls`.
- The versioned prompt `agent/prompts/system_v1.md` (`agent.prompt_version`) holds the answer rules: only facts from
  tool results, at most `agent.max_citations` films with one sentence each, cite as `Title (Year)` plus Wikipedia link,
  say so when nothing fits.
- Citations are built from the films the tools returned, never from the model's text: the text only selects which
  retrieved films are mentioned (by exact `Title (Year)` or `movie_id`). An invented film matches nothing and is dropped;
  an answer that cites no retrieved film is replaced by a standard abstention message (`Answer.abstained`).
- `Answer` (Pydantic) carries the text, citations, every retrieved film, the tool calls, token usage, latency and the
  run metadata (git sha, prompt version, chat model, embedding model, retrieval mode, config hash), which is also on
  the LangSmith spans. Secrets never reach the model or a trace.

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
| `download` | fetch the Kaggle CSV into `data/raw/`; without credentials prints what to set and exits 2 (`FORCE=1` re-downloads) | ready |
| `doctor` | report on Docker, Qdrant, dataset, credentials and model ids with next steps; always exits 0 | ready |
| `ingest` | chunk, embed and upsert into Qdrant; downloaded dataset if present, else the fixture (`FIXTURE=1` forces it, `CSV=path`, `RECREATE=1`) | ready |
| `demo` | `up` + `ingest` on the fixture + the sample query in dense, sparse and hybrid mode (`Q="..."` to ask your own); no credentials | ready |
| `serve` | MCP server over HTTP at `mcp.url` (`STDIO=1` for stdio, `DOCKER=1` for the compose service with healthcheck) | ready |
| `ask` | CLI agent: `make ask Q="..."` (`MODE=dense\|sparse\|hybrid`, `JSON=1`); needs `make serve` and `NEBIUS_API_KEY` (without it: prints the hint, exits 2) | ready |
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
