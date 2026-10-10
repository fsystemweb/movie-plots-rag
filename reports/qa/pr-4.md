# QA report — PR #4 (PR-04 Hybrid retrieval)

- Branch `pr/04-retrieval`, head `0adbbecc0699d0e5612ab79b297fbf85db13b16a` (same sha as the green CI run 38038160981).
- Scope reviewed: builder commits `45665d8`, `6d10884`, `0adbbec`. `3d883b7` was written by the orchestrator (tracking docs, token log, `reports/qa/pr-3.md`), so it is out of scope.
- Plan row: "PR-04 — Hybrid retrieval. Per §1. ✅ one code path for three modes · filters correct inside prefetch (tests) · one row per film · sample query output for all modes pasted in the PR."
- Skills applied: `qdrant-hybrid` (review checklist below) and `pr-workflow` (it allows the builder to add follow-ups to `docs/BACKLOG.md`).
- Installed versions: qdrant-client 1.15.1, fastembed 0.9.0; local and CI server `qdrant/qdrant:v1.15.4`.

## Summary

The PR adds `src/movie_rag/retrieval/` with these pieces:
- `Retriever.search(query, mode, top_k, filters)`: one `build_request` and one `query_points_groups` call for dense, sparse and hybrid.
- `SearchFilters`: a validated Pydantic model, turned into one Qdrant filter that is attached to each hybrid prefetch.
- Grouping by `movie_id` (`group_size=1`).
- `MovieHit`, which carries the citation fields.
- `get_movie`, which reads the full plot from chunk 0.
- A CLI and `make demo`.

All four acceptance criteria are met, and I checked each one myself. `make check` passes with 555 tests and 100.00% line+branch coverage. Against the real Qdrant v1.15.4, all 10 integration cases pass, including hybrid. A mutation check shows the hybrid integration test fails if the filter is removed from the prefetches. I verified the in-memory engine limitation in the library source and by running it. The RRF-k claim is only half right: the **client** cannot send a parametric RRF, but the **server 1.15.4 accepts it**. The decision to remove `rrf_k` still holds, but its written reason is wrong. That is a minor finding. There are no blockers and no majors.

## Acceptance criteria

| Criterion | Evidence (produced by QA) | Result |
|---|---|---|
| One code path for three modes | `src/movie_rag/retrieval/search.py:198-210`: `search` calls `build_request(...)` and then makes the single `self.client.query_points_groups(**request)` call. In `build_request` (`search.py:121-155`) the mode only changes the query part: hybrid sets `prefetch` plus `FusionQuery(RRF)`; dense and sparse set `query`, `using` and `query_filter`. The `group_by`, `group_size`, `limit` and `with_payload` values are shared. Tests `test_hybrid_request_has_two_prefetches_...` and `test_dense_and_sparse_requests_query_one_named_vector_with_the_filter` pass. | ✅ |
| Filters correct inside prefetch (tests) | **Request shape:** `search.py:143-144` passes `filter=query_filter` on both `Prefetch`es. Unit tests assert both prefetch filters and that there is no top-level `query_filter` (`tests/unit/test_retrieval.py` `test_hybrid_request_has_two_prefetches_...`, `test_hybrid_sends_the_filter_in_both_prefetches_to_the_client`). **Behaviour on the real server:** `QDRANT_URL=http://localhost:6333 uv run pytest tests/integration/test_retrieval_qdrant.py -v` gives `10 passed`, including `test_the_filter_is_applied_before_the_candidate_cut_not_after_it[hybrid]` and `test_results_are_one_row_per_film_and_respect_filters[hybrid]`. **Mutation check** (monkeypatched `build_request`, crowded collection with `prefetch_limit=2`): the original code returns `['weak-0','weak-1','weak-2']`. With the filter dropped from the prefetches it returns `['strong-0','strong-1','strong-3']`, so the integration test catches it. With the filter moved to top level only, the server pushes it down and the result is correct, but the unit shape test catches the change (`"query_filter" not in req`). **CI** run 38038160981 (job `tests`, `QDRANT_URL` set) shows `tests/integration/test_retrieval_qdrant.py ..........` (10 cases, hybrid included) and `558 passed`. | ✅ |
| One row per film | `query_points_groups(group_by="movie_id", group_size=1)` (`search.py:131-135`), and `search.py:211` builds one hit per group. Tests: `test_results_have_one_row_per_film_best_first_with_citation_fields` (20 distinct films), `test_a_multi_chunk_film_is_returned_once_through_its_best_chunk` (the film appears once, via its last chunk), and integration `test_results_are_one_row_per_film_and_respect_filters[hybrid]` (15 distinct films on the real server, which passed in my run). | ✅ |
| Sample query output for all modes pasted in the PR | `gh pr view 4 --json body` contains `== dense: 3 film(s) ==`, `== sparse: 3 film(s) ==` and `== hybrid: 3 film(s) ==`, plus a filtered run in all three modes. I re-ran `QDRANT_URL=http://localhost:6333 uv run python -m movie_rag.retrieval --top-k 3` and got the same titles and scores (dense 0.7687/0.6804/0.6301, sparse 50.7431/16.1906/12.9317, hybrid 1.0000/0.5833/0.3788). | ✅ |
| §1: default hybrid, top_k 8, prefetch 50, params in config.yaml | `config.yaml` has `retrieval: default_mode: hybrid, top_k: 8, prefetch_limit: 50, snippet_max_chars: 400, demo_query`. `search.py:192-193` and `:137` read them. `test_mode_and_top_k_default_to_the_configuration` passes. A grep of `src/movie_rag/retrieval/` for model names, URLs or numeric limits finds nothing hard-coded (the only hit is a docstring). | ✅ |
| §1: query embedders used for queries | `search.py:205-206` uses `embed_dense_query` and `embed_sparse_query`. `test_only_the_query_embeddings_a_mode_needs_are_computed` asserts the document path is never used. | ✅ |
| Citation fields (title, year, Wikipedia link, snippet ≤ 400) | `MovieHit` (`search.py:64-77`) has `movie_id, title, release_year, director, genre, origin, wiki_url, score, chunk_idx, snippet`. The test compares every field against the source record and asserts `len(snippet) <= snippet_max_chars`. | ✅ |
| Error handling | Live CLI runs: a missing collection prints "cannot search Qdrant at http://localhost:6333: UnexpectedResponse... Try `make up` and `make ingest`." and exits 1. An unreachable server (127.0.0.1:1) prints the same hint and exits 1. `--top-k 0` prints "search failed: top_k must be at least 1, got 0" and exits 1. Inverted years are rejected by the validator (tested). None of these print a traceback. | ✅ |
| `make check` green | `make check` exit 0: ruff OK, format OK, mypy OK, **555 passed**, **Total coverage: 100.00%** (1107 stmts, 222 branches, 0 missed). Log: `logs/qa-pr4-check.log`. | ✅ |
| CI green | `gh pr checks 4`: quality pass, secret-scan pass, tests pass. `smoke-eval` was skipping, as expected before PR-09. | ✅ |

## Integrity checks

| Check | Result |
|---|---|
| `pragma: no cover`, `skip`, `xfail`, `pytest.skip(` added | None. The only grep hits are the English word "omit" in README and PR text. The hybrid integration cases are left out by conditional parametrisation (`MODES` at `tests/integration/test_retrieval_qdrant.py:37`), not by a skip. This is documented, follows the PR-03 precedent, and the cases do run in CI. |
| Coverage omit / `fail_under` changes | None; `pyproject.toml` is untouched. |
| Protected files (`.github/workflows/`, `.claude/`, `CLAUDE.md`, `KICKOFF.md`, token files) | The builder commits do not touch any of them. Builder files are Makefile, README, config.yaml, docs/BACKLOG.md, docs/prs/PR-04.md, src/*, tests/*. The token-file and tracking changes are only in the orchestrator commit `3d883b7`. |
| Secrets / `.env` / raw data | None. No `sk-` or `lsv2_` strings, no data files added. |
| Hard-coded models, URLs or retrieval params in `src/` | None. The grep only matches docstring text. The `"movie_id"`, `"release_year"`, `"genre"` and `"origin"` strings are payload schema keys, not tunables. |
| Needs a credential | No. No LLM is involved, and FastEmbed runs locally. |
| Tests assert real behaviour | Yes. The mutation check above confirms the hybrid filter test catches a real regression. |
| Scope | Matches the plan row, with one small addition (`get_movie`, see m4). |
| `print` in src | None. CLI output goes through `emit` and `say`. |

## Builder decisions — verdicts

1. **`retrieval.rrf_k` removed: I accept the decision but reject part of its reasoning.**
   - **Client: confirmed.** qdrant-client 1.15.1 has only `Fusion.RRF` and `Fusion.DBSF` (`.venv/.../qdrant_client/http/models/models.py:858-867`). It has no `RrfQuery` or `Rrf` model. Passing a raw `{"rrf":{"k":60}}` to `client.query_points` raises `ValueError: Unsupported query type: <class 'dict'>`.
   - **Server: refuted.** Qdrant server **v1.15.4 accepts parametric RRF** over REST: `POST /collections/movie_plots/points/query` with `"query":{"rrf":{"k":K}}` returns top score 0.1 for k=10, 0.001 for k=1000, and 1/60≈0.016667 for k=60. Plain `{"fusion":"rrf"}` returns 0.5, so the fixed constant in use today is **k=2**.
   - **Why I still accept it:** dropping a setting the client cannot send is right. A config value that is silently ignored would mislead.
   - **What is wrong:** the docs say the server and the pinned image are the obstacle, which is false (see m1).
2. **In-memory engine drops prefetch filters in `query_points_groups`: confirmed. I accept the testing strategy.**
   - **Source:** `local/local_collection.py:1016-1024` rewrites every prefetch through `set_prefetch_limit_recursively` (`:2847-2855`). That function returns `types.Prefetch(limit=limit, prefetch=list())`, so the query, `using` and `filter` are lost.
   - **Running it:** the same 6-point collection with a year filter gives `:memory:` groups `['f0'...'f5']` (filter ignored), but `:memory:` `query_points` gives `['f4','f5']`. The server gives `['f4','f5']` for both calls.
   - **Coverage is adequate.** Request-shape unit tests catch a filter that is moved or removed from the prefetches. The real-server integration test catches a filter that is not applied before the candidate cut (mutation check). CI runs those cases (`QDRANT_URL` is set in `ci.yml` and the log shows 10 integration cases).
3. **Sparse can return fewer than top_k: accepted.**
   - BM25 gives no score to chunks with no term overlap. Padding the list would add irrelevant films.
   - Checked live: the query `"xyzzyplugh" --mode sparse` prints `== sparse: 0 film(s) == (no match)`. The PR body's filtered run shows `sparse: 1 film(s)`.
   - The CLI shows this clearly, and there is a backlog note for PR-09.
4. **Retrieval tracing spans deferred to PR-05: accepted.**
   - The PR-04 plan row has no observability criterion.
   - §1 assigns "a LangSmith span per tool call (mode, filters, latency, result count)" to the MCP server, which is PR-05.
   - A backlog entry exists (`docs/BACKLOG.md`, last `[PR-04]` line).
   - Condition: PR-05 QA must check that it is delivered, because the §1 goal "every … retrieval … step traceable" depends on it.

## qdrant-hybrid review checklist

- [x] `query_embed` for queries (`search.py:205-206` through `embed_*_query`, tested) and `embed` for documents (PR-03, unchanged).
- [x] `bm25` has `Modifier.IDF` (PR-03, unchanged).
- [x] Filters inside each prefetch in hybrid (`search.py:143-144`, tested by shape and behaviour).
- [x] One shared function serves all three modes (`build_request` and `Retriever.search`).
- [x] uuid5 ids and idempotent re-ingest (PR-03). `get_movie` reuses `point_id(movie_id, 0)`.
- [x] Payload indexes (PR-03, unchanged).
- [x] Limits, collection name and model names come from `config.yaml`.

## Findings

### Blockers
None.

### Majors
None.

### Minors

- **m1: The written reason for removing `rrf_k` is factually wrong about the server.**
  - Where: `src/movie_rag/retrieval/search.py:16-18`, `README.md:68`, `docs/BACKLOG.md:23` and `docs/prs/PR-04.md:94` (with `:91`).
  - What they say: "Qdrant 1.15.x exposes only `Fusion.RRF`", and re-adding the setting "needs qdrant-client and server 1.16 (the CI/compose image tag is protected/pinned)".
  - What is true: server v1.15.4 accepts and honours `{"rrf":{"k":K}}` (evidence above). Only qdrant-client 1.15.1 lacks the model. The current effective constant is k=2, not the usual 60, and the PR does not say so.
  - **Fix:**
    - Reword the four places to say the limit is in the client: no `RrfQuery` in qdrant-client 1.15.1, while server 1.15.4 accepts `rrf.k`.
    - State that the effective RRF constant is 2.
    - Change the backlog item to "bump qdrant-client to ≥1.16 (no protected-file change needed), then re-add `retrieval.rrf_k`". This is worth doing before PR-09 compares modes.

- **m2: A unit test opens a real TCP connection.**
  - Where: `tests/unit/test_retrieval_cli.py:133-139` (`test_without_an_injected_client_main_connects_to_the_configured_url`) connects to `127.0.0.1:1`.
  - It is loopback and is refused immediately, but it still uses the network stack. CLAUDE.md says "No network in unit tests".
  - **Fix:** monkeypatch `movie_rag.retrieval.search.QdrantClient` with a `MagicMock` whose `query_points_groups` raises `ResponseHandlingException`. Then assert both the constructor's `url=` argument and the error message.

- **m3: Some unit tests run hybrid on an engine known to ignore prefetches.**
  - Where: `tests/unit/test_retrieval.py:355` (`test_mode_and_top_k_default_to_the_configuration`) and `:365` (`test_top_k_is_honoured`), both using the default hybrid mode, and `tests/unit/test_retrieval_cli.py:72` (all modes).
  - All three run hybrid on `QdrantClient(":memory:")`. Their count-only assertions would pass even if hybrid were wrong. They do not claim to check correctness, but a reader could take them as hybrid coverage.
  - **Fix:** use a `MagicMock` client for the hybrid default-mode tests and assert `call_args`, or add a comment pointing to the integration test, as done at `test_retrieval.py:191-194`.

- **m4: `get_movie` is a small addition beyond the strict plan row.**
  - Where: `search.py:215-233`, with tests.
  - It is a 19-line, tested, read-only retrieval primitive that §1 implies (full plot on chunk 0). It adds no new interface (no MCP tool, no CLI flag). I accept it as in scope for the retrieval layer, so this is not a scope-creep FAIL.
  - Watch-out for PR-05: §1 also wants lookup "by exact title", which this does not support.
  - **Fix:** none required now. PR-05 should add the title lookup.

## Coverage

`make check`: **100.00%** line+branch on `src/movie_rag` (gate 80%), **555 passed** locally without `QDRANT_URL`. CI with `QDRANT_URL` (run 38038160981): 100.00%, **558 passed**, the 3 extra being the hybrid integration cases. Integration suite against local Qdrant v1.15.4: **10 passed**.

VERDICT: PASS
