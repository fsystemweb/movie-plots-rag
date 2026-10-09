# QA report: PR #2 (PR-02, Settings, errors, observability, data), round 2

Branch `pr/02-settings-data` @ `a574b98014d107a4fb1e2d96dcc0f4a6fd6b3bb8`, GitHub mode. Reviewer: qa-validator.
Round-2 diff reviewed: `0fd8dc6..a574b98`. Integrity was re-checked on the full diff `origin/main...origin/pr/02-settings-data`. Commit 7473ed9 there is the orchestrator's tracking commit, not the builder's work.
Skills applied: `pr-workflow`, `langsmith-tracing`.

## Summary

The builder rewrote the fixture generator and fixed all six round-1 minors in one commit (a574b98). The response is in `docs/prs/PR-02.md` under "QA response".

- **M1 is fixed.** I recomputed every distinctiveness metric with my own scripts, using the same method as round 1:
  - The share of a plot's sentences that are unique to it went from 0.13 to 0.91.
  - Shared premise sentences went from 16 to 0.
  - Films whose nearest neighbour has TF-IDF cosine ≥ 0.5 went from 208 to 0.
  - The memory-loss premise now belongs only to "The Forgetting Hour" (round 1: 8 lookalikes).
- **Sample review:** I read 15 generated plots chosen with a fixed seed. Every one has a premise and twist a human could turn into an unambiguous fuzzy-plot question.
- **The new tests have teeth.** Each new distinctiveness test fails on the round-1 fixture and passes on the new one.
- **m1–m6:** all fixed, each verified with my own command or test run. One side effect of the m3 fix narrowed the credential-key filter; it is a new minor.
- **Gates:** `make check` green: **391 passed, 100.00%** line+branch coverage. CI run 37989661959 succeeded on a574b98.
- **No credentials:** `make download` exits 2 with the documented hint and no traceback. `make doctor` exits 0.
- **Integrity:** no violations.

Result: 0 blockers, 0 majors and 3 new minors, none of which block. **PASS.**

## Round 1 history (0fd8dc6, VERDICT: FAIL)

- **M1 (major):** fixture plots were templated remixes. 249 of 291 films reused a genre-pool incident, 16 had identical setting, protagonist and incident, a plot's sentences were only 13% unique on average, and 8 noir films had the anchor's "no memory" premise. That undermined the PR-08 fuzzy questions.
- **m1:** hard-coded `*5` extraction limit.
- **m2:** hard-coded git timeout and Kaggle help URL.
- **m3:** the key filter rejected `total_tokens`.
- **m4:** `.env` was resolved relative to the current directory.
- **m5:** a real actor's name ("Arjun Kapoor") was in the "invented" fixture.
- **m6:** a fake key leaked into `os.environ` after two tests.

## Re-verification of round-1 findings

| Finding | Evidence (produced by QA at a574b98) | Status |
|---|---|---|
| **M1** fixture distinctiveness | See the section below: my own metrics, the old-vs-new test run and the 15-plot human read | ✅ fixed |
| m1 `*5` extraction limit | `download.py:120` now uses `settings.data.max_extracted_mb * 1024 * 1024`, and `config.yaml` has `max_extracted_mb: 1000`. `test_extraction_limit_comes_from_settings_not_a_multiple_of_the_download_limit` extracts 6 MB with download cap 1 and extracted cap 8, so the old `5×1 MB` rule would have failed it. The bomb test now sets `DATA__MAX_EXTRACTED_MB=5` | ✅ fixed |
| m2 git timeout, help URL | `observability.git_timeout_s: 5` is passed to `git_sha(timeout_s)`. `test_git_timeout_comes_from_settings` sets 7 and the test sees `[7.0]` in the subprocess call. The help URL comes from `data.kaggle_token_help_url`, and the fixture path in the help text from `data.fixture_path`. My grep of `src/` for `https?://`, model ids or `bm25` in added lines found nothing. My live `make download` prints the URL from config | ✅ fixed |
| m3 `total_tokens` rejected | `is_credential_like()` (`observability.py:35-52`). It accepts `total_tokens`, `prompt_tokens`, `completion_tokens`, `max_tokens`, `cache_key` and `sort_key`, and rejects `api_key`, `NEBIUS_API_KEY`, `KAGGLE_KEY`, `access_token`, `token`, `password` and `credentials` (both directions are parametrised). The fix also narrowed coverage: see new minor n1 | ✅ fixed (see n1) |
| m4 `.env` relative to CWD | `DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"` is used both in `Settings.model_config` and as the `load_settings` default. `test_default_load_reads_the_project_env_file_even_from_another_directory` runs from a subdirectory that has its own `.env`, and the project file wins | ✅ fixed |
| m5 real actor name | "Kapoor" is gone. Name pools now use invented surnames. The README and generator docstring say names "may coincide with real people". `test_cast_and_directors_avoid_a_denylist_of_real_surnames` passes. One residual coincidence is listed under n3 | ✅ fixed |
| m6 env leak | `conftest.py:40-43` now does setenv then delenv so monkeypatch remembers the variable was absent. I ran `pytest.main(test_observability.py, test_secrets.py)` in one process and checked `os.environ` before and after: `LANGSMITH_API_KEY`/`LANGSMITH_PROJECT` were `None` before and `None` after (54 passed) | ✅ fixed |

## M1 re-verification (independent)

**1. Metrics recomputed with my own scripts.** I copied the old and new CSVs to `logs/qa-pr2/` (since deleted) and ran the same code as in round 1. All figures are over the 291 kept films.

| Metric (my method) | Round 1 (0fd8dc6) | Round 2 (a574b98) |
|---|---|---|
| Mean share of a plot's sentences unique to it (two-word names masked) | 0.13 | **0.91** |
| Films with no unique sentence | 16 | **0** |
| Same share, but also masking the title/object words and every capitalised word (template-level) | 0.13, min 0.00 | **0.52**, min 0.33 |
| Films with fewer than 2 template-unique sentences | 268 | **0** |
| Premise (first) sentences shared by more than one film, fully masked | 16 | **0** |
| Films whose TF-IDF nearest neighbour (names stripped) is at cosine ≥ 0.3 / 0.4 / 0.5 / 0.6 | 284 / 279 / 208 / 53 | **16 / 0 / 0 / 0** |
| Highest nearest-neighbour cosine | 0.72 | **0.35** ("The Coffin on the Ferry" and "The Wrong Coffin") |
| Plots with a memory-loss *premise* | 9 (anchor + 8 noir) | **1**: only "The Forgetting Hour" |

For the last row I grepped for `memory|amnesia|recall|identity|remember|forget`. The other hits are incidental ("nobody remembers loading", "cannot remember who believes which version") and none of them is an amnesia premise.

**2. The new tests fail on the old fixture.** I loaded `tests/unit/test_fixture.py` in-process, pointed `FIXTURE` at the old CSV, and called each new test:

| Test | Old fixture | New fixture |
|---|---|---|
| `test_no_two_kept_films_share_a_premise_sentence` | FAIL | PASS |
| `test_most_of_each_plot_is_unique_to_its_film` | FAIL | PASS |
| `test_plots_are_lexically_distinct_nearest_neighbour_tfidf` | FAIL | PASS |
| `test_the_memory_loss_premise_belongs_to_the_anchor_alone` | FAIL | PASS |
| `test_cast_and_directors_avoid_a_denylist_of_real_surnames` | FAIL | PASS |
| `test_anchor_plots_do_not_overlap_any_other_plot` | FAIL (only because fewer than 30 anchor titles exist in the old file) | PASS |

**3. Human read of 15 generated plots** (`random.Random(7).sample`, non-anchor, kept). Each one has a concrete, unique premise and twist that supports an unambiguous paraphrase. Examples:

- a weather forecaster who is always wrong in a predictable direction gets hired as an oracle ("The Ruined Festival");
- a museum guard watches a painting change each night because a forger hides in the frame ("The Changing Painting");
- an emergency operator learns the whispering caller is already inside the operator's own house ("The Whispered Call");
- a pawnbroker's flute summons birds that refuse to let it be returned ("The Bird-Summoning Flute");
- a translator softens insults into compliments and becomes indispensable ("The Kinder Translation").

I would accept 15 of 15 as fuzzy-question targets.

The rest of each plot is still formulaic: 2–3 shared beat templates plus a genre finale. Some finales do not fit the plot, such as an ocean-trench film ending "toward a star nobody has named". `docs/BACKLOG.md` records this honestly. It is now adequate for §2 and PR-08; see n2 for the residual lexical bias.

**4. Story data is licence-safe and invented.** I spot-read the anchors "Glass Harvest" and "Atlas of Lost Birthdays" and the motifs in `motifs_1.yaml`; all read as original prose. Every YAML file has a header marking it SYNTHETIC. The `Wiki Page` values still end in `_(synthetic_film)`.

The anchors now cover 36 films, 17 genre labels including `adventure`, `war comedy` and `action`, and 17 origins including Chinese, Filipino, Punjabi, Egyptian and Telugu. Those are realistic Kaggle labels. Years run 1927–2017 across 10 decades, and 13 anchors have plots over 250 words. Cast and director names are invented-sounding; one residual coincidence is listed under n3.

## Acceptance criteria (round 2)

| Criterion | Evidence (QA) | Result |
|---|---|---|
| Cleaning rules tested on hand-built rows | `test_clean.py` is unchanged and its 53 tests pass. The fixture still drops 9 short plots (one of exactly 49 words) and keeps the one of exactly 50 | ✅ |
| id stable across runs | Unchanged code. `test_cleaning_twice_gives_identical_ids` and `test_ids_do_not_depend_on_other_rows_being_dropped` pass. I verified cross-process stability in round 1, and `clean.py` has not changed since | ✅ |
| Missing credentials: documented message, no stack trace | `env -u KAGGLE_USERNAME -u KAGGLE_KEY -u NEBIUS_API_KEY -u LANGSMITH_API_KEY make download` gives `set KAGGLE_USERNAME and KAGGLE_KEY — see docs/CREDENTIALS.md` plus the help text, exit 2, no traceback, no `data/`. `make doctor` prints the table (Nebius, LangSmith and Kaggle MISSING with next steps) and exits 0. New: an unreadable config gives "cannot load the configuration", exit 1, no traceback (`test_cli_reports_an_unreadable_configuration_without_a_traceback`) | ✅ |
| Secrets never logged (test) | I re-ran the 4 round-1 leak mutations on a574b98. Each one fails exactly one `test_secrets.py` test (`1 failed, 5 passed` ×4) | ✅ |
| Fixture per §2 (realistic, several genres/origins/decades, short and long plots, marked synthetic) | See the M1 re-verification above. 300 rows, exact schema, 17+ genres, 17 origins, 1910s–2010s, 9 short, 13 long, README marked SYNTHETIC | ✅ |
| config / errors / observability / download / clean / DATASET.md / doctor | Verified in round 1. The round-2 changes are limited to the minor fixes; `docs/DATASET.md:46-52` is updated for the new fixture | ✅ |
| `make check` green | `ruff check`: all checks passed. `ruff format --check`: 39 files formatted. `mypy src`: no issues in 8 files. **391 passed in 9.26s, TOTAL 586 stmts / 112 branches, 100.00%** | ✅ |
| CI | Run 37989661959 on head `a574b98`: quality, tests and secret-scan **success**; smoke-eval skipped (stub until PR-09) | ✅ |

## Integrity checks (full diff `origin/main...origin/pr/02-settings-data`)

| Check | Result |
|---|---|
| `pragma: no cover`, skip/xfail, `fail_under`/`omit`/`--cov-fail-under` changes | None in the builder's work. The only grep hits are prose inside `reports/qa/pr-1.md`, which arrived in the orchestrator's commit 7473ed9 |
| Protected files | The builder commits (73244a9, 0fd8dc6, a574b98) do not touch `.github/workflows/`, `.claude/`, `CLAUDE.md`, `KICKOFF.md`, `docs/TOKEN_USAGE.md`, `docs/token_usage.jsonl` or `.env*` |
| Lint config | `pyproject.toml` only *removes* the round-1 `E501` ignore for the generator. No new ignores |
| Secrets, `.env`, raw data | None. The only secret-shaped string is the fake `sk-abcdefghijklmnop` test input in `test_observability.py`, which existed in round 1. CI secret-scan succeeded |
| Hard-coded models, URLs or params in `src/` | None. My grep of the added `src/` lines for URLs, model ids and `bm25` found nothing |
| Meaningful tests and no credentials needed | Yes: see the mutation checks and the old-vs-new fixture test run |
| Scope | Fixes stay within PR-02 (fixture, config keys, minors). `docs/BACKLOG.md` gains one honest follow-up |
| `pr-workflow` | New commit on the same branch with no force-push (`git log` shows 73244a9 → 0fd8dc6 → a574b98). Conventional `fix:`. The QA response answers every finding point by point and was posted as PR comment 6089048435 |
| `langsmith-tracing` | Tracing off and no key: tested. Required metadata fields: present. No secrets in metadata: mutation-verified. Token counters can now be attached (m3) |

## Findings

### Blockers

None.

### Major

None.

### Minor (new in round 2, none blocking)

- **n1. The m3 fix narrowed the credential-key filter more than needed.** `src/movie_rag/observability.py:28-52` uses an allow-list of qualifiers. As a result, `is_credential_like()` now returns **False** for `github_token`, `hf_token`, `slack_token`, `private_token`, `openai_key`, `passphrase` and `secretkey`. The round-1 regex rejected all of these. The value check (configured secrets, `sk-`/`lsv2_` shapes) is still a backstop, so this is defence-in-depth only. Fix: reject by default when the last word is singular `token`, `key` or `secret`, or when the key contains `passphrase|secretkey|cookie|jwt`. Keep an explicit allow-list (`cache_key`, `sort_key`, `primary_key`, `idempotency_key`). Plural counters (`*_tokens`) already pass. Add these names to `CREDENTIAL_KEYS`.
- **n2. Lexical bias toward sparse retrieval** (note for PR-08/PR-09). Shared beats (`tests/fixtures/story_data/shared.yaml:4-18`) repeat each film's central object, which is also its title, 2–4 times per plot. A fuzzy question that names the object ("a changing painting") gives BM25 an easy match. Fix in PR-08: also paraphrase the object noun in the fuzzy questions, and note the bias in `docs/EVAL_SET.md`. Optionally reduce beats that embed `{obj}` to one per plot.
- **n3. One remaining real-name coincidence.** The anchor director "Boris Zaitsev" (`tests/fixtures/story_data/anchors_a.yaml:32`) is also the name of a well-known Russian writer (1881–1972). The README disclaimer covers coincidences, but renaming it costs nothing. Fix: pick an invented surname from the pool (for example "Boris Rostovtsev") and add "Zaitsev" to the denylist.
- **Observation (no action needed):** `test_anchor_plots_do_not_overlap_any_other_plot` (`tests/unit/test_fixture.py:173`) failed on the old fixture only because of its `checked >= 30` guard, not its overlap threshold. The other five new tests do discriminate, so coverage of M1 is adequate.

## Coverage

**100.00%** line+branch on `src/movie_rag` (586 statements, 112 branches, 0 missed). Gate is 80%. 391 tests passed.

VERDICT: PASS
