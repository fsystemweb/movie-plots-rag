# QA report: PR #9, PR-09 Evaluation runner (round 2)

- **Branch and head:** `pr/09-eval-runner`, head `cd7aac1`. Local HEAD, `origin/pr/09-eval-runner` and the GitHub PR head are all `cd7aac113b6e`.
- **Commits reviewed:** round-2 commits `6ffa441`, `b5c024d`, `cd7aac1` (diff `232cccf..cd7aac1`), plus integrity checks over the whole branch `b8784d2..cd7aac1`. `b8784d2` is orchestrator-authored and out of scope.
- **Skills applied:** `ragas-eval`, `langsmith-tracing` (unchanged from round 1), and `qdrant-hybrid`, because `retrieval/search.py` changed in this round.

## Summary

Every round-1 finding is fixed or accepted with a justification, and I checked each one myself.

- **B1 (`make test` red under make): fixed.** The builder stop-gate command `make lint typecheck test` now passes when run plainly.
- **M1 (suite time): fixed.**
  - Uncontended run: **288 s** wall, down from 376 s.
  - Contended run (another full suite running at the same time, plus my Qdrant probes): 439 s, down from 493 s.
  - Both are well under the 540 s gate timeout.
- **m3 (ragas as a main dependency): fixed.** ragas is now an optional extra. Without it:
  - the retrieval half runs;
  - the LLM half stops before any model call, with an install hint.
- **m1 (tie order): fixed as far as the client can go.** I confirmed the builder's explanation against the Docker Qdrant: per-film fused scores change between identical hybrid calls. The remaining nondeterminism is therefore upstream in Qdrant's prefetch. Keeping the tie-aware metrics and documenting it is the right call.
- **Reproducibility:** a fresh `make eval` reproduces the committed numbers exactly.
- **CI:** green on the new head.

## Round 1 findings → status

| # | Finding | Status | Evidence (mine) |
|---|---|---|---|
| B1 | `make test` red under a parent make (`test_package.py::_dry_run`) | ✅ fixed | `tests/unit/test_package.py:43-47` now runs `make --no-print-directory -n`. Plain `make lint typecheck test` → `1000 passed, 1 deselected`, `EXIT=0` (twice: once contended, once not) |
| M1 | Suite time near the 540 s stop-gate timeout | ✅ fixed | `PYTEST_ADDOPTS=--durations=25 make lint typecheck test`, uncontended: **WALL=288 s** (pytest 284.05 s). Contended run (a second full `make lint typecheck test` started 20 s earlier, plus 240 hybrid queries and two smoke runs of mine): **WALL=439 s**. `make check`: 278 s. The slowest test is now 43.5 s (FastEmbed setup in the integration eval), and the in-memory CLI tests take 3-11 s each, down from 73 s total |
| m1 | Nondeterministic tie order in the retrieval layer | ✅ fixed in the client; the residual upstream variation is documented | See "Tie-break verification" below |
| m2 | Literal "30 questions / 3.3 points" in `report.py` | ✅ fixed | `src/movie_rag/eval/report.py:164,181` derives the sentence from `overall.n`. Tested for n=6 and n=30 (`test_eval_report.py::test_the_size_of_one_question_is_derived_from_the_data`) |
| m3 | ragas as a main runtime dependency | ✅ fixed | See "Dependency split" below |
| m4 | A tie group cut at the retrieval depth | accepted (documented, no effect on the committed numbers) | Unchanged. It remains a documented limitation |
| m5 | LangSmith target keyed by question text | ✅ fixed | `langsmith_experiment.py:41-49` keys by `question_id`, and `upload.py:60-64` puts `question_id` in the example inputs. `test_two_questions_with_the_same_text_do_not_collide` and `test_eval_upload.py:79` cover it |
| nit | No-op `assert sys.modules` | ✅ removed | `test_eval_ragas.py:68-71` |

## Tie-break verification (Docker Qdrant v1.15.4 at http://localhost:6333, collection `movie_plots`)

- **Code:** `src/movie_rag/retrieval/search.py:132-138` adds `by_score_then_id`, applied in `search` (`:240`) and `find_similar` (`:299`). Unit tests use three server orders plus `find_similar` (`tests/unit/test_retrieval.py:313-343`).
- **30 identical hybrid calls each, `top_k=8`:**
  - fuzzy-01: **1 distinct order** (round 1: 12/18 split). The gold film is tied at 0.625 and is now always 2nd, placed by `movie_id`.
  - exact-09 and exact-10: 1 distinct order each.
  - exact-08: the top 7 are identical in every call. Only the 8th slot (two films tied at 0.2) alternates 17/13. This is "which tied films make the cut".
- **6 identical calls each over all 40 questions:**
  - All 240 returned lists are sorted by `(-score, movie_id)`.
  - 8/40 questions still vary (round 1: 14/40 orderings differed).
  - In 2 of them **the fused score of the same film changes**: fuzzy-10 `echo-chamber-9-…` 0.333 vs 0.5, and `the-bottled-conf…` 0.355 vs 0.520. That can only happen when the prefetch lists themselves differ. It confirms the builder's explanation: Qdrant breaks ties inside a prefetch before RRF. A client-side sort cannot fix that.
- **Judgement: acceptable.**
  - The client now does everything it can, and user-facing order is deterministic for any given result set.
  - The tie-aware metrics stay correct, and they do not depend on the id order.
  - The docs no longer over-claim. `EVAL_RESULTS.md` says "Hybrid results are not exactly repeatable…". `docs/EVALUATION.md` has a "Ties and repeatability" section. `docs/BACKLOG.md` has an entry for client-side fusion.
  - The retrieval change is small (a stable sort applied after the hits are returned), is tested, and was made in direct response to QA m1, so I do not count it as scope creep.
- **qdrant-hybrid checklist:**
  - Filters are still inside both prefetches: `test_hybrid_sends_the_filter_in_both_prefetches_to_the_client` is unchanged and green.
  - One shared search function serves all three modes.
  - No new tunables. The other items are untouched by this round.

## Dependency split (m3)

- **Lock and manifest:**
  - `uv lock --check` → `Resolved 186 packages`, rc=0.
  - `pyproject.toml:22-27` declares ragas only under the `eval` extra.
  - `uv.lock` has `provides-extras = ["eval"]`.
  - CI (`ci.yml:30,55,113`) and `make setup` (`Makefile:27`) install `--all-extras`.
- **Base export:** `uv export --frozen --no-dev` contains no ragas, instructor, datasets, langchain-community or scipy. With `--extra eval` all five appear.
- **Downgrades:** openai 3.3.0, rich 14.3.4, jiter 0.14.0 and fsspec 2026.7.0 are present either way, because the lock is a single resolution. This is now correctly called "downgrades" and listed in `docs/prs/PR-09.md` and `docs/BACKLOG.md`.
- **Behaviour without the extra** (separate venv from `uv sync --frozen --dev`, `find_spec('ragas')` → None):
  - The eval CLI imports fine.
  - `python -m movie_rag.eval smoke` → the committed numbers, rc=0.
  - `smoke --llm` without a key → `LLM half skipped: pending credentials …`, rc=0.
  - `NEBIUS_API_KEY=test-key-not-real … smoke --llm` → `evaluation failed: the RAGAS metrics need the optional `eval` extra, which is not installed: run `uv sync --extra eval` (or `make setup`, which installs all extras).`, rc=1, no traceback.
- **Fail-fast order:** `runner.py:121` now builds the scorers before the agent runs. `test_a_missing_eval_extra_stops_the_llm_half_before_the_agent_is_asked` asserts `model.prompts == []`. `require_ragas` re-raises other missing modules unchanged (tested with `scipy`).

## Acceptance criteria (re-verified on cd7aac1)

| Criterion | Evidence | Status |
|---|---|---|
| Hit@k, MRR, abstention in own code, tested on hand-computed cases | `src/movie_rag/eval/metrics.py` unchanged since round 1 (100 % covered). `tests/unit/test_eval_metrics.py` is green | ✅ |
| `make eval` writes `reports/eval_<mode>.json`; committed numbers reproduce | `EVAL__REPORTS_DIR=logs/qa9r2/out QDRANT_URL=http://localhost:6333 make eval` → three files, rc=0. Against the committed JSON, `retrieval.overall` and `by_type` are **identical** for dense, sparse and hybrid. Per-question lists: dense 0/40 and sparse 0/40 differ, hybrid 4/40 (tail ties, see above). All committed lists are sorted by `(-score, movie_id)`. The config-hash difference comes only from my `EVAL__REPORTS_DIR` override (the default `config_hash()` = `7fc61e900353`, as committed) | ✅ |
| `make report` builds `reports/EVAL_RESULTS.md` | Rendered from my fresh JSON and diffed against the committed page: only timestamps, git sha, config hash (override) and latency columns differ. Every Hit@k and MRR cell is the same | ✅ |
| `make eval-smoke` and `LLM=1`; CI calls exactly these | CI run 38057027050 (head `cd7aac1`, `pull_request`): smoke-eval printed dense .792, sparse .983, hybrid .942 MRR on a fresh Qdrant. The LLM step is skipped with "no NEBIUS_API_KEY". `test_package.py` dry-run tests are green under make | ✅ |
| LLM metrics "pending credentials"; ragas version and judge recorded | `EVAL_RESULTS.md` run-metadata table: ragas 0.4.3, judge `openai/gpt-oss-120b`, generator `Qwen/Qwen3-30B-A3B-Instruct-2507`. The LLM rows say pending credentials | ✅ |
| Judge != generator, clear skip without a key | Unchanged code, tests green. Re-observed in the smoke runs above | ✅ |
| One LangSmith experiment per mode, clean skip without a key | `test_eval_langsmith.py` and `test_eval_runner.py::test_a_langsmith_experiment_is_recorded_per_mode_when_a_key_exists` are green. Keyed by `question_id` now | ✅ |
| Honesty about the fixture favouring BM25 | Page text unchanged and tested. The new repeatability wording is tested (`test_the_page_explains_hybrid_repeatability_and_how_ties_are_scored`) | ✅ |
| `make lint typecheck test` green (stop gate) | `EXIT=0 WALL=288s`, 1000 passed, 99.39 % | ✅ |
| `make check` green | ruff ✅, `128 files already formatted`, mypy `no issues found in 42 source files`, **1000 passed, 1 deselected**, **99.39 %**, `EXIT=0 WALL=278s` | ✅ |
| CI green on the new head | `gh pr checks 9`: quality pass, secret-scan pass, tests pass (1003 passed, 99.39 %), smoke-eval pass | ✅ |

## The faster tests still check real behaviour

- **Counts:** the suite went from 987 to **1000** tests, and coverage went from 99.36 % to **99.39 %**. `runner.py` is now 100 %, `ragas_judge.py` 100 %, `__main__.py` 98 %.
- **CLI tests on `CannedRun`:**
  - They still assert files, stdout and stderr, and exit codes.
  - They now also assert the exact arguments passed to the runner (modes, `attempt_llm`, `experiments`, `llm_questions_per_type`).
  - Two `..._for_real` tests plus `test_a_failed_llm_half_…` still drive the real path end to end.
  - The smoke-floor test asserts that a cached MRR is below 1.0 before expecting failure, so it cannot pass vacuously.
- **Runner tests:** the full 40-question run is kept for one mode (`n == 40`, latency `n == 40`, ingest). The other runner tests use the 8-question `subset`, with `n == 6` and `n == 8` asserted. `subset` is only reachable from `evaluate()` and `run()` arguments; the CLI never passes it (`__main__.py` does not reference it).
- **Integration:** the module-scoped `first_run` still feeds the floor test and the filter test. The reproducibility test still performs a genuine second run and asserts that `overall` and `by_type` are equal.

## Integrity checks

| Check | Result |
|---|---|
| `pragma: no cover`, `skip`/`skipif`/`xfail`/`pytest.skip(` added in `src/` or `tests/` (`git diff main...HEAD`) | none |
| Coverage `omit` / `fail_under` / `--cov-fail-under` changes | none |
| Protected files (`.github/`, `.claude/`, `CLAUDE.md`, `KICKOFF.md`, `docs/TOKEN_USAGE.md`, `docs/token_usage.jsonl`, `.env*`, `reports/qa/`, `docs/PR_TRACKER.md`) in `b8784d2..HEAD` | none |
| Secret-shaped strings in `src/` | none. The only match is the existing scrubber regex `observability.py:36` |
| Hard-coded models, URLs or retrieval params in new `src/` lines | none. `by_score_then_id` has no parameters, and the install hint is a command, not a tunable |
| Needs a credential | no. Everything ran without keys, both locally and on CI |
| Scope | in scope. The `search.py` sort is a tested response to QA m1 |

## Findings

### Blocker

None.

### Major

None.

### Minor

- **m4 (carried over, accepted).** A tie group cut at the retrieval depth is judged by its visible part only (`src/movie_rag/eval/metrics.py:12,107-118`). It is documented and does not affect the committed numbers.
- **m6 (new, nit). Stale follow-up text.**
  - Location: `docs/prs/PR-09.md`, "Follow-ups", which still lists "no deterministic tie-break in the retrieval layer".
  - The backlog entry has been reworded to "client-side fusion needed".
  - Fix: align the sentence with `docs/BACKLOG.md`.
- **m7 (new, nit). Assertion accepts either label.**
  - Location: `tests/unit/test_eval_cli.py:66`. `"not run" in out.replace("not requested", "not run")` passes for either label.
  - The CLI prints `not requested`, and `canned.calls[0]["attempt_llm"] is False` pins the behaviour, so nothing is lost.
  - Fix: assert `"not requested" in out` directly.

## Coverage and tests

- `make check` (local): **99.39 %** line+branch on `src/movie_rag` (gate 80 %), **1000 passed**, 1 deselected (live).
- `make lint typecheck test`, uncontended: 288 s wall, 1000 passed, 99.39 %.
- CI `tests` job: 1003 passed, 99.39 %, 202.9 s.

VERDICT: PASS
