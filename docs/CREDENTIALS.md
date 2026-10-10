# Credentials guide

Everything in this repository builds, tests and runs **without any key** (`make demo`, the retrieval-only test page,
the retrieval metrics). Keys unlock three things: the agent's answers and RAGAS scoring (Nebius), the real 35,000-film
dataset (Kaggle) and traces (LangSmith). This page is the step-by-step for the human who adds them, and ends with the
command sequence that produces the real results.

Every "set NEBIUS_API_KEY — see docs/CREDENTIALS.md" hint in the code points here.

| Key | Variables | Unlocks | Without it |
|---|---|---|---|
| Nebius Token Factory | `NEBIUS_API_KEY` (optional: `NEBIUS_BASE_URL`) | `make ask`, the page's agent mode, RAGAS, the LLM half of `make eval` | `make ask` prints the hint and exits 2; the page falls back to "Retrieval only"; reports say "pending credentials" |
| Kaggle | `KAGGLE_USERNAME`, `KAGGLE_KEY` | `make download` (the real dataset) | `make download` prints what to set and exits 2; `make ingest` uses the synthetic fixture |
| LangSmith | `LANGSMITH_API_KEY`, `LANGSMITH_TRACING`, `LANGSMITH_PROJECT` | traces, the evaluation dataset upload, experiments | tracing is a no-op with identical code paths |

Embeddings run locally (FastEmbed) and Qdrant runs in Docker, so neither needs a key.

## 1. Create the keys

### Nebius Token Factory (`NEBIUS_API_KEY`)

1. Sign up at <https://tokenfactory.nebius.com> (Google or GitHub login).
2. Open <https://tokenfactory.nebius.com/project/api-keys>, click **Create API key**, give it a name, click **Create**.
3. Copy the key now: it is shown once.

The endpoint is OpenAI-compatible: `llm.base_url` in `config.yaml` is `https://api.tokenfactory.nebius.com/v1/`
(override with `NEBIUS_BASE_URL`). The default models are `Qwen/Qwen3-30B-A3B-Instruct-2507` (answers, `llm.chat_model`)
and `openai/gpt-oss-120b` (RAGAS judge, `llm.judge_model`); `make doctor` checks both against `GET /v1/models`, and
the evaluator refuses to run when judge and generator are the same model. Docs:
<https://docs.tokenfactory.nebius.com/api-reference/introduction>.

### LangSmith (`LANGSMITH_API_KEY`, optional)

1. Sign up at <https://smith.langchain.com>.
2. In the app open **Settings** and then **API Keys**, create a key and copy it.
3. Set `LANGSMITH_TRACING=true` to send traces and, optionally, `LANGSMITH_PROJECT` (default `movie-plots-rag`,
   `observability.project`).
4. An account in the EU region also needs `LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com` (no trailing slash)
   for the SDK and `OBSERVABILITY__LANGSMITH_API_URL=https://eu.api.smith.langchain.com` for `make doctor`.

Docs: <https://docs.langchain.com/langsmith/trace-with-langchain>. A key with `LANGSMITH_TRACING` unset or `false`
sends nothing.

### Kaggle (`KAGGLE_USERNAME`, `KAGGLE_KEY`)

1. Sign in at <https://www.kaggle.com>, open <https://www.kaggle.com/settings> and find the **API** section.
2. Create a **legacy** API key: it downloads `kaggle.json` with `{"username": "...", "key": "..."}`. Copy the two
   values into the two variables.
3. Open the dataset page once while signed in, <https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots>,
   and accept its terms (Kaggle refuses downloads otherwise). Licence notes: [`DATASET.md`](DATASET.md).

Only the `KAGGLE_USERNAME` / `KAGGLE_KEY` pair is read. `~/.kaggle/kaggle.json` and Kaggle's newer single-token
(`KGAT_...`) form are not supported yet (backlog).

## 2. Put them where they are read

**Locally**: copy the template and fill in the values. `.env` is gitignored and must never be committed.

```bash
cp .env.example .env     # then edit .env: NEBIUS_API_KEY=..., KAGGLE_USERNAME=..., KAGGLE_KEY=..., LANGSMITH_*=...
```

A blank value (`KEY=`) means "not set". The settings loader reads `.env` from the repository root wherever you run
from; a variable exported in your shell wins over `.env`. Other overrides: `QDRANT_URL`, `MCP_URL`; any
`config.yaml` key can be overridden with a double underscore (`LLM__CHAT_MODEL=...`).

**GitHub Actions (CI)**: repository **Settings > Secrets and variables > Actions > New repository secret**, or with the
GitHub CLI (it prompts for the value, nothing lands in your shell history):

```bash
gh secret set NEBIUS_API_KEY
gh secret set LANGSMITH_API_KEY
```

CI uses them only in the `eval` job (pull requests labelled `eval`, or manual runs): with `NEBIUS_API_KEY` it adds
`make eval-smoke LLM=1`, without it that step is skipped with a note. Nothing else in CI needs a key.

Optional, to stop Docker Hub rate limits on the Qdrant service container (PR #3 needed many reruns): add secrets
`DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` (a Docker Hub access token), then add `credentials:` to the `qdrant`
services in `.github/workflows/ci.yml` (protected file: a human edit; see `docs/BACKLOG.md`, `[Orchestrator]`).

## 3. Check them: `make doctor`

```bash
make doctor
```

It prints a table (Docker, Qdrant, dataset, Nebius key and both model ids, LangSmith, Kaggle) with the next step for
each row, and always exits 0. Each key is checked with one cheap call (`GET /models`, `GET /api/v1/sessions`, a dataset
lookup), so an OK row means the key is valid, not just present. Without keys it looks like this (captured on this branch):

```text
CHECK           STATUS   DETAIL                                                         NEXT STEP
--------------  -------  -------------------------------------------------------------  --------------------------------------------
Docker          OK       daemon 29.0.1                                                  -
Qdrant          OK       ready at http://localhost:6333                                 -
Dataset         WARN     data/raw/wiki_movie_plots_deduped.csv not downloaded; ...      after setting Kaggle keys: `make download`
Nebius API key  MISSING  NEBIUS_API_KEY is not set (agent and RAGAS need it)            set NEBIUS_API_KEY — see docs/CREDENTIALS.md
Model: chat     SKIPPED  Qwen/Qwen3-30B-A3B-Instruct-2507 (no key to list models)      -
Model: judge    SKIPPED  openai/gpt-oss-120b (no key to list models)                   -
LangSmith       MISSING  LANGSMITH_API_KEY is not set; tracing is a no-op (switch off) optional: set LANGSMITH_API_KEY and ...
Kaggle          MISSING  KAGGLE_USERNAME and KAGGLE_KEY not set                         set them — see docs/CREDENTIALS.md (...)
```

(columns shortened here; the real output is wider). Once the keys are valid the Nebius row reads `valid; N models
listed` and both model rows read `OK`. If a model row says "not listed by Nebius", pick an id from `/v1/models` and set
`llm.chat_model` / `llm.judge_model` in `config.yaml`.

What each command does without its key (all verified without credentials):

| Command | Without the key |
|---|---|
| `make download` | prints `set KAGGLE_USERNAME and KAGGLE_KEY — see docs/CREDENTIALS.md`, what to do, exits 2 |
| `make ask Q="..."` | prints `set NEBIUS_API_KEY — see docs/CREDENTIALS.md`, exits 2 |
| `make eval-smoke LLM=1` | runs the retrieval half, prints `LLM half skipped: pending credentials`, exits 0 |
| `make eval` | writes the retrieval metrics, records the LLM half as "pending credentials", exits 0 |

## 4. Produce the real results

Make sure the tools are installed with the `eval` extra (RAGAS is optional): `make setup` already does
(`uv sync --all-extras --dev`); otherwise run `uv sync --extra eval`. Then, with the three keys set and Docker running:

```bash
make up                  # Qdrant (the sequence below assumes it is running)
make download && make ingest && make eval MODE=dense && make eval MODE=sparse && make eval MODE=hybrid && make report
```

| Step | What it does |
|---|---|
| `make download` | fetches `wiki_movie_plots_deduped.csv` into `data/raw/` (gitignored; `FORCE=1` downloads again) |
| `make ingest` | chunks, embeds (locally, on CPU: untimed for 35k films) and upserts the downloaded dataset into the main collection (`qdrant.collection`); `RECREATE=1` drops the fixture films a previous `make demo` put there, so the real index holds real films only |
| `make eval MODE=...` | per mode: Hit@k, MRR and latency, then the agent and RAGAS half (needs `NEBIUS_API_KEY`); writes `reports/eval_<mode>.json`; `LLM=0` skips the LLM half. It indexes and searches its own collection `eval.collection` (`movie_plots_eval`), not the main one |
| `make report` | renders `reports/EVAL_RESULTS.md`, the Results block of `README.md` and the two result blocks of `docs/SOLUTION_OVERVIEW.md` from those JSON files |

Then commit `reports/eval_*.json`, `reports/EVAL_RESULTS.md`, `README.md` and `docs/SOLUTION_OVERVIEW.md` **together**:
`tests/unit/test_readme_results.py` and `tests/unit/test_solution_overview.py` fail when a generated block and the JSON
disagree. Hand-written prose is never touched by `make report`; re-read it after a live run:

* `README.md`: "What the fixture can and cannot show".
* `docs/SOLUTION_OVERVIEW.md`: section 2's illustrative answer, "How to read this" (section 4) and "Cost per 1,000
  questions: pending credentials" (section 5). Compute the cost from the tokens-per-question row of
  `reports/EVAL_RESULTS.md` and Nebius's price list for the two models, then replace the sentence with the figure.
  The build totals in section 7 are "as of change 10": regenerate them from `docs/TOKEN_USAGE.md`.

**What the sequence measures, and what it leaves alone.** The 40 questions and their gold film ids belong to the
synthetic fixture, so the evaluation runs on the fixture, in a collection of its own (`eval.collection`, default
`movie_plots_eval`, override with `EVAL__COLLECTION`). The first `make eval` fills that collection with the 304 fixture
passages; later runs reuse it. The main collection that `make ingest` fills, and that `make ask`, the page and
`make serve` read, is never written by an evaluation (`make eval-smoke` and `make ci` included). The generated captions
say what was indexed (collection and passage count). The committed overview and README numbers are therefore
**fixture numbers**, whatever the real index holds. Earlier versions of the evaluator added the fixture films to the main
collection; that behaviour is gone. If your main collection already holds them, `make ingest RECREATE=1` rebuilds it
from the downloaded dataset. The evaluator also refuses (clear error, nothing written) to add the fixture to a
non-empty collection that holds other films.

For real-data numbers a real-data question set is needed (backlog): run `python -m movie_rag.eval.generate` (needs
`NEBIUS_API_KEY`), review the output and point `eval.questions_path` at it (`docs/EVAL_SET.md`).

Optional extras once the keys exist:

```bash
make eval-smoke LLM=1                      # 24 agent calls (eval.smoke_llm_per_type x 4 types x 3 modes) + RAGAS sample
uv run pytest -m live                      # the live agent test, never run so far
uv run python -m movie_rag.eval.upload     # creates the LangSmith dataset (bump eval.dataset_name when questions change)
make serve & make ask Q="a film where a hotel telephone operator overhears a murder being planned"
```

## 5. Read this before the first live run

These are the open backlog items (`docs/BACKLOG.md`) most likely to matter when a real model answers for the first time.

1. **Invented titles can stay in the answer text** (PR-06 QA m2, priority). The citation list is built only from
   retrieved films, but a film the model made up, or a link it wrote, remains in the displayed text whenever at least
   one real film is cited. Treat answer text as unverified until this is fixed.
2. **Citation rate is unverified.** A model that paraphrases a title without the year yields no citation and the
   standard "I don't know" text (PR-06). Check the citation hit rate and the abstention rows after the first
   run, and tune `agent/prompts/system_v1.md` as a new version (never edit in place).
3. **RAGAS has only run against a fake judge** (PR-09): instructor's tool-calling mode against
   `openai/gpt-oss-120b` on Nebius, and the cost of 30 samples x 4 metrics x 3 modes, are untested. RAGAS 0.4.3 needs a
   compatibility shim and forces older versions of `openai`, `rich`, `jiter` and `fsspec` into the shared lock.
4. **Hybrid fusion uses a fixed constant** (effective RRF k = 2): qdrant-client 1.15.1 cannot send a parametric RRF.
   Upgrading to 1.16 or later and re-running `make eval` is open (PR-04 QA m1). Hybrid rankings below the top are not
   exactly repeatable between identical calls; the metrics are tie-aware.
5. **Agent mode in the page** has only run against a scripted model (PR-07): check the trace link and latency, and
   that the sidebar filters (sent to the model as a sentence) are respected.
6. **Cost control.** The LLM half of `make eval` runs whenever `NEBIUS_API_KEY` is set (`LLM=0` turns it off);
   `make eval-smoke LLM=1` is the small version. Lower `eval.smoke_llm_per_type` if your rate limit is tight.
7. **Refreshing the reports changes committed files** (git sha and timestamp are inside each JSON): regenerate only
   when the code or the question set changes.

## 6. Never

Never commit `.env`, a key, or the raw dataset. CI runs gitleaks, and `src/` is scanned for secret-shaped strings by a
test. Settings keep keys as `SecretStr`, so logs, traces and error messages show `***`.
