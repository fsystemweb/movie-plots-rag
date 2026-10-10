# Evaluation runner

`python -m movie_rag.eval` (through `make eval`, `make eval-smoke`, `make report`) measures the three retrieval modes on
the 40 questions of [`EVAL_SET.md`](EVAL_SET.md) and, when a key exists, the agent and RAGAS. Code:
`src/movie_rag/eval/` (`metrics.py`, `retrieval_eval.py`, `agent_eval.py`, `ragas_judge.py`, `langsmith_experiment.py`,
`runner.py`, `report.py`, `__main__.py`). Tunables are under `eval:` in `config.yaml`.

## Commands

| Command | What it does |
|---|---|
| `make eval [MODE=dense\|sparse\|hybrid]` | Evaluates one mode or all three and writes `reports/eval_<mode>.json`. The agent + RAGAS half runs when `NEBIUS_API_KEY` is set (`LLM=0` skips it) and is recorded as `pending_credentials` otherwise; the run still succeeds. |
| `make eval-smoke` | The retrieval half for all modes, printed, nothing written. Fails if a mode's MRR is below `eval.smoke_min_mrr`. This is the form CI calls. |
| `make eval-smoke LLM=1` | Adds a small sample (`eval.smoke_llm_per_type` questions per type) through the agent and RAGAS; prints a skip notice and exits 0 without a key. The second form CI calls. |
| `make report` | Renders `reports/EVAL_RESULTS.md` from the `eval_<mode>.json` files (and only from them). |

Everything needs the Qdrant **service** (`make up`, or `QDRANT_URL`). The evaluator ingests the fixture itself when the
gold films are not in the collection (ingestion is idempotent), so a fresh Qdrant works. It refuses the in-process
`:memory:` engine: qdrant-client 1.15.1's local engine ignores the prefetch queries and filters of grouped hybrid
queries, so hybrid numbers from it would be wrong (docs/BACKLOG.md, PR-04 and PR-08 QA m7).

## What is measured

The evaluator calls the project's MCP server in process (`fastmcp.Client(server)`): the same `search_movies` tool, with
the question's filters, that the agent and the UI use. No `make serve` is needed.

| Metric | Definition | Needs a key |
|---|---|---|
| Hit@k (k in `eval.k_values`) | the gold film is among the top k films | no |
| MRR | mean of 1/rank of the gold film (0 when not retrieved) | no |
| Latency p50 / p95 | wall time of one in-process `search_movies` call (after one untimed warm-up call) | no |
| Correct abstention rate | share of unanswerable questions on which the agent cites no retrieved film | yes |
| False abstention rate | share of answerable questions on which it abstains | yes |
| Citation hit rate | share of answerable questions whose gold film is cited | yes |
| Tokens per question, agent latency | p50 / p95 over the agent's runs | yes |
| RAGAS faithfulness, response relevancy, context precision, context recall | judged by `llm.judge_model` over the answerable questions | yes |

Hit@k and MRR are averaged over the 30 questions that have a gold film; the 10 unanswerable ones are what abstention
is about. **Why abstention needs the LLM:** abstaining is the agent's behaviour (an answer that cites no film). A
retrieval-only proxy would need a score threshold, a new tunable that is not in the spec and is not comparable across
cosine, BM25 and RRF scores, so it is not computed; the report shows `pending credentials` instead.

### Ties and repeatability

Reciprocal rank fusion gives equal scores to a film that is first in one list and second in the other, so a plain rank
of a tied gold film is arbitrary (observed before this was handled: hybrid Hit@1 moving between 0.867 and 0.900 on
identical data). The metrics are therefore tie-aware: the best-placed gold film is taken to be at any position of its
tie group with equal probability, and Hit@k and the reciprocal rank are the expectations (`metrics.rank_distribution`).
Without ties the rank is exact.

`Retriever.search` and `find_similar` now list films with equal scores by `movie_id` (`retrieval.search.by_score_then_id`),
so exactly tied output scores no longer depend on the server's order. **That does not make hybrid results exactly
repeatable.** Repeating one hybrid query 15 times showed two different result sets (the films below the top, and their
fused scores, changed): the server breaks ties *inside* each prefetch list (BM25 scores tie on near-identical plots)
arbitrarily before fusing, and which tied films make the `top_k` cut is also its choice. A deterministic hybrid ranking
would need client-side fusion with a stable tie-break. The tie-aware metrics are what keep the reported numbers stable:
they were identical in every re-run we made. The per-question `rank` in the JSON is the position in the order
returned, for inspection only.

### RAGAS

ragas is the optional `eval` extra (`uv sync --extra eval`; `make setup` and CI install all extras). Without it the
retrieval metrics still run, and asking for the LLM half stops before any model call with the command to install it.

ragas 0.4.3 (`ragas.metrics.collections`: `Faithfulness`, `AnswerRelevancy` = response relevancy,
`ContextPrecisionWithReference`, `ContextRecall`), judge via `llm_factory(llm.judge_model, client=AsyncOpenAI(base_url=llm.base_url))`,
embeddings for relevancy from the local FastEmbed dense model. The judge must differ from the generator
(`llm.chat_model`); the run is refused otherwise. Reference for precision/recall is the gold film's title, year and full
plot. A metric that fails or returns NaN for one sample is recorded as failed for it (`n_scored/n_failed` in the JSON)
and does not discard the rest. The ragas version and judge model are written into every report.

## Reading the results

The fixture is a 291-film synthetic corpus whose plots are built around rare invented nouns, so BM25 has an easy job.
The numbers favour BM25 and **cannot show the advantage hybrid retrieval is expected to have on real plots**. The gold
ids belong to the fixture; a meaningful comparison needs the full dataset and a question set generated for it
(`python -m movie_rag.eval.generate`). With 30 answerable questions one question is 3.3 points.

## LangSmith

With `LANGSMITH_API_KEY`, `make eval` also records one experiment per mode on the dataset uploaded by
`python -m movie_rag.eval.upload` (created if missing): the target returns the ranking already computed and the
evaluators recompute Hit@k and the reciprocal rank, so the experiment matches the JSON. Without a key it logs
`skipped: no LANGSMITH_API_KEY`; a LangSmith outage marks the experiment `failed` in the report and never fails the run.
`make eval-smoke` records no experiments.
