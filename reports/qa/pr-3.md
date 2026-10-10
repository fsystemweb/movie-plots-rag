# QA report: PR #3 (PR-03, chunking, embeddings, ingestion)

Branch `pr/03-ingestion`, HEAD `fe6eed0`. Reviewed `origin/main...origin/pr/03-ingestion`. The first commit, `5f81454` (orchestrator tracking: BACKLOG/PR_TRACKER/TOKEN_USAGE/`reports/qa/pr-2.md`), is not builder work and is not counted against this PR. Builder commits: `7fc9caf`, `4335a6d`, `fdbed65`, `82dfd43`, `fe6eed0`.

## Summary

The PR adds a sentence-aware chunker (`ingest/chunk.py`), FastEmbed dense and BM25 embedders (`ingest/embed.py`), the Qdrant schema, ids and payload (`ingest/index.py`), a resumable batched pipeline with a single LangSmith run (`ingest/pipeline.py`), and the `make ingest` CLI (`ingest/__main__.py`). I re-ran every acceptance criterion myself:
- `make check`: 496 tests passed, coverage 100.00% (line and branch).
- The 7 integration tests pass against a real Qdrant v1.15.4.
- I ran `make ingest FIXTURE=1 RECREATE=1` twice and then a plain re-ingest. Each run reported 304 points, and the Qdrant REST count was 304 each time.

There are no blockers and no majors. I found four minors, listed below.

## Acceptance criteria

| Criterion | Evidence (produced by QA) | Result |
|---|---|---|
| Chunker: a short plot is 1 chunk | `make check` ran `tests/unit/test_chunk.py` (44 passed). The tests include `test_short_plot_is_a_single_whole_chunk`, `test_plot_exactly_at_the_budget_stays_whole_and_one_over_is_split` and `test_short_plot_record_has_one_chunk_with_the_header`. I also ran a real-tokenizer probe over the fixture: 291 films, 278 single-chunk. Every film at or under 250 tokens has exactly one chunk; `test_real_chunks_stay_within_the_token_budget` asserts this and passed against the live Qdrant run. | ✅ |
| Chunker: cuts at sentence boundaries | `test_long_plot_is_cut_only_at_sentence_boundaries_and_respects_the_budget`, `test_chunks_overlap_by_whole_sentences_within_the_overlap_budget`, `test_chunks_cover_every_sentence_in_order`, and 20 seeded random plots. Real-tokenizer probe on the 13 multi-chunk fixture films: every chunk ends on a sentence terminator (for example `'...behind the altar tomb.'`). Chunk sizes are 225–250 tokens (max 250, none over budget). The overlap between chunks is 0–40 tokens (`[25, 27, 27, 27, 18, 35, 30, 15, 0, 39, 20, 16, 40]`). | ✅ |
| Chunker: header on every chunk | `test_every_chunk_carries_the_header_and_its_position`; `test_every_embedded_text_starts_with_the_film_header` checks this at the pipeline level on all 304 embedded texts. Real tokens added by the header: at most 22, so the largest embedded text is 267 tokens, well under bge's 512. | ✅ |
| Re-ingest leaves the point count unchanged | Live run. `make ingest FIXTURE=1 RECREATE=1` twice gave `points_in_collection=304`, and REST `points/count {"exact":true}` returned `{"count":304}` after each run. A plain `make ingest FIXTURE=1` then logged `chunks_written=0 chunks_skipped=304 points_in_collection=304 elapsed_s=0.44`, and REST still returned 304. The integration test `test_reingest_leaves_the_point_count_unchanged` passed with `QDRANT_URL=http://localhost:6333`. | ✅ |
| `make ingest` on the fixture logs counts that match Qdrant | Live log: `ingest done: rows_read=300 rows_dropped=9 films=291 chunks_total=304 chunks_written=304 chunks_skipped=0 points_in_collection=304 elapsed_s=31.95`. An independent REST read gives `points_count=304`. The report is also printed (`rows read: 300 … points in Qdrant: 304`). | ✅ |
| §1: named vectors `dense` and `bm25` with `Modifier.IDF` | REST `GET /collections/movie_plots` returned `{'dense': {'size': 384, 'distance': 'Cosine'}}` and `{'bm25': {'modifier': 'idf'}}`. Code: `src/movie_rag/ingest/index.py:97`. | ✅ |
| §1: payload indexes on release_year (int) and origin/genre/movie_id (keyword) | REST `payload_schema`: `{'movie_id': 'keyword', 'origin': 'keyword', 'release_year': 'integer', 'genre': 'keyword'}`. I also confirmed with `--cov=tests/integration` that the guarded assertion block at `tests/integration/test_ingest_qdrant.py:91` actually ran against the live service (the branch report shows only `91->exit` missing, so the true branch was taken). | ✅ |
| §1: ids are uuid5(`movie_id:chunk_idx`) | `index.py:516-518`. `test_point_id_is_pinned_so_that_the_namespace_never_changes` pins the value. | ✅ |
| §1: batches of 256, resumable by skipping existing ids | `pipeline.py:96-98` (retrieve by id before embedding). `test_default_batch_size_is_256`, `test_batches_never_exceed_the_configured_size`, `test_an_interrupted_ingest_keeps_finished_batches_and_the_rerun_finishes_the_rest`, `test_ingest_resumes_after_deleted_points_writing_only_the_missing_ones`. The live log shows `256 chunks seen` then `304 chunks seen`. | ✅ |
| §1: one LangSmith run per ingest, required metadata, no secrets | `pipeline.py:137` (`@traceable(name="ingest", process_inputs=_trace_inputs)`), metadata at `:150`. `test_one_langsmith_run_per_ingest_with_metadata_and_counts` asserts `create_run.call_count == 1` and the metadata keys. `test_traces_never_contain_configured_secrets` passes and includes a positive control (the run is recorded, so the absence check is meaningful). | ✅ |
| `query_embed` for queries, `embed` for documents (dense and BM25) | `embed.py:87,90` use `.embed`; `embed.py:93,97` use `.query_embed`. `test_documents_use_embed_and_queries_use_query_embed` asserts the call order on both fakes. | ✅ |
| Parameters come from config | `chunk_tokens`, `chunk_overlap`, `batch_size`, model names, `dense_dim`, collection and timeout are all read from `Settings`. A grep of `src/movie_rag/ingest/` for model names, sizes and URLs found literals only in docstrings and in the uuid namespace seed (see integrity checks). | ✅ |

## Integrity checks

| Check | Result |
|---|---|
| `pragma: no cover`, coverage omit or `fail_under` changes | None. The gate stays on the Makefile command line at 80. |
| skip/xfail outside live tests | None. The integration module uses an in-memory fallback instead of skipping (assessed below). |
| Touches `.github/workflows/`, `.claude/`, `CLAUDE.md`, `KICKOFF.md` | No (`git diff --name-only 5f81454..fe6eed0`). |
| Secrets, `.env` content, raw data | None. The only "secrets" in the diff are the fake test strings `fake-*-secret-*`. No `data/` files are committed. |
| Hard-coded model names, URLs or retrieval parameters in `src/` | None that count. `index.py:29` contains `"https://github.com/fsystemweb/movie-plots-rag/points"`, but it is only the seed for the uuid5 namespace, not an endpoint. It is deliberately fixed (changing it would change every point id) and is pinned by a test. Accepted. |
| Tests that assert nothing meaningful | None found. `test_ingest_runs_identically_with_tracing_off_and_no_key` is light, but it asserts that writes happened. |
| Needs a credential to pass | No. `make check` passed with no keys. Model download needs network on a cold cache, but no credential; CI caches `FASTEMBED_CACHE_PATH`. |
| Scope | Within the PR-03 row. `embed_dense_query`/`embed_sparse_query` reach slightly into PR-04, but they are small and needed to prove that `query_embed` is used (and they are exercised by the integration test). `qdrant.timeout_s` is a new config key, not a hard-code. |
| `tests/unit/test_package.py` change | Legitimate. `ingest` is no longer a stub, so the stub parametrisation now checks `demo` (still a PR-04 stub, `Makefile`), and a new `make -n ingest` test asserts that the flags are forwarded. No gate is weakened. |
| Conventional commits, branch name | `pr/03-ingestion`; commits are `chore(deps)`, `feat(ingest)` ×2 and `docs` ×2. |
| PR template filled, versions and doc URLs, backlog | All 8 template sections are present in `docs/prs/PR-03.md`, and the PR body matches it. Versions match `uv pip show`: qdrant-client 1.15.1, fastembed 0.9.0, langsmith 0.14.7, tokenizers 0.23.3. Four follow-ups were appended to `docs/BACKLOG.md`. |
| Remote CI | `quality` and `secret-scan` succeed. `tests` failed at "Initialize containers": the GitHub runner timed out on Docker Hub auth for the service image, so no repo code ran. This is not attributed to the builder; the orchestrator's merge gate depends on the rerun. |

### Skill checklists

**pr-workflow**
- Branch and commits: ✅
- Template with evidence: ✅
- Versions and URLs: ✅
- Backlog: ✅
- No force-push: ✅ (linear history on top of `5f81454`)

**qdrant-hybrid** (the ingestion-relevant items)
- `query_embed` for queries and `embed` for documents: ✅
- IDF on `bm25`: ✅
- uuid5 ids plus a re-ingest count test: ✅
- Four payload indexes: ✅
- Config-driven parameters: ✅
- Hybrid filter-in-prefetch and the shared search function belong to PR-04: N/A

**langsmith-tracing**
- Works with tracing off and no key: ✅ (all tests run with `LANGSMITH_TRACING=false`)
- Metadata fields present: ✅ (`retrieval_mode="n/a"`, `git_sha`, `prompt_version`, `chat_model`, `embedding_model`, `config_hash`)
- No secrets: ✅ (`process_inputs` limits the inputs to the csv path, collection and recreate flag)
- Upload/experiments: N/A

### Points the orchestrator asked me to scrutinise

**1. Integration-test fallback to `:memory:`.** I judge this honest, not a weakening.
- In CI (`QDRANT_URL` is set in the `tests` job) and in my local run, every assertion runs against the real service, including the payload-index schema. I proved that branch executed (see the criteria table).
- Without `QDRANT_URL`, the same tests still exercise the real FastEmbed models, the pipeline and the counts. Only the payload-index assertion is skipped, and it cannot work in local mode anyway because Qdrant local mode keeps no indexes.
- The unit test `test_ensure_collection_creates_the_four_payload_indexes` independently asserts the index calls and their types.
- The trade-off is documented in the module docstring and in the backlog.
- Visibility could be better (minor 3).

**2. Token definition and chunk sizes.** A token is a WordPiece token of the dense model, excluding `[CLS]`/`[SEP]`, which is the right unit. On the 13 long fixture plots (292–345 real tokens):
- Chunk sizes are 225–250 tokens for first chunks and 53–152 for tails.
- No chunk exceeds 250.
- Overlap is at most 40 everywhere.
- One boundary (Glass Harvest) has 0 overlap, because the last sentence alone exceeds 40 tokens (minor 2).

Counts saturate at 512 because of FastEmbed truncation. This is harmless, since the only comparison is against 250.

**3. Sentence boundaries.** These are correct on the fixture. Probes found false negatives on real-world patterns (minor 1).

**4. Idempotency.** Verified live (304 → 304 → 304). The re-run skips embedding entirely (0.44 s).

**5. `full_plot` on chunk 0.** A sound design:
- `get_movie` needs a single `retrieve` by `point_id(movie_id, 0)`, with no scroll and no filter.
- The payload is not multiplied by the chunk count.
- It is verified live by `test_chunk_zero_holds_the_full_plot_for_get_movie` and offline by `test_film_ids_are_recoverable_from_movie_id_and_index`.
- Chunk ≥1 is asserted to lack `full_plot`.

## Findings

### Blocker
None.

### Major
None.

### Minor

**1. The sentence splitter misses common sentence ends.** `src/movie_rag/ingest/chunk.py:36-40`
- `"no"` is in `_ABBREVIATIONS`, so "She refuses to say no. The next day he leaves." stays one sentence.
- `_INITIALISM` matches any single letter, so "They choose Plan B. The heist begins." is not split.
- A trailing initialism such as "He moves to the U.S. There he meets Ann." is not split either.

All three were reproduced with `split_sentences`. The effect is only coarser sentence units, never a budget violation, because over-long sentences are split on words.

Fix:
- Treat `No.` as an abbreviation only when the next token starts with a digit.
- Treat a single capital letter as an initial only when the next word is also capitalised and is not a common sentence starter, or drop single-letter initials outside `X. Y.` runs.
- Add these three cases to `test_chunk.py`.

Partly backlogged already.

**2. Overlap can be 0 tokens.** `src/movie_rag/ingest/chunk.py:127-131` builds the overlap only from whole trailing sentences, so when the last sentence is over 40 tokens the next chunk shares nothing (observed on 1 of 13 fixture boundaries). The docstring documents this, but KICKOFF says "40 overlap".

Fix: either fall back to the last ≤40 tokens of words from the final sentence, or record the "whole sentences, possibly none" rule in ADR-003 (PR-10) so the deviation is an explicit decision.

**3. The `:memory:` fallback is silent.** `tests/integration/test_ingest_qdrant.py:35,91`: running `pytest -m integration` locally without `QDRANT_URL` passes while silently skipping the payload-index assertions, even though the marker text says "needs the Qdrant service".

Fix: report the backend that was used, for example a `pytest_report_header` hook or a module-level `print`/`warnings.warn` saying "integration: using in-memory Qdrant, payload-index assertions not run". Alternatively, parametrise the test id with the backend so the weaker run is visible in the output.

**4. Token counting depends on FastEmbed internals, with no upper version bound.** `src/movie_rag/ingest/embed.py:83` calls `cast(Any, self.dense_model.model).tokenize(...)`, which is internal FastEmbed API, while `pyproject.toml:8` pins `fastembed>=0.7`.

Fix: bound the pin to the verified minor (`fastembed>=0.9,<0.10`), as was done for qdrant-client. The integration test `test_real_tokenizer_counts_wordpieces_without_special_tokens` already catches a break.

## Coverage

`make check` (run by QA, `QDRANT_URL` unset):
- ruff and format check: clean
- `mypy src`: Success (13 source files)
- Tests: **496 passed** in 52.91 s
- Coverage: **100.00%** line+branch on `src/movie_rag` (gate 80%). The new modules `chunk`, `embed`, `index`, `pipeline` and `__main__` are each at 100%.

Integration tests against real Qdrant v1.15.4 (`QDRANT_URL=http://localhost:6333 uv run pytest -m integration`): **7 passed** in 36.78 s. The throw-away collection was dropped afterwards. I ran `make down` at the end.

VERDICT: PASS
