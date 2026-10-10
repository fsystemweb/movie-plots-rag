# QA report: PR #8, PR-08 Evaluation set

- Branch `pr/08-eval-set`, head `f32c205da3bdfe39ccd9b53fd4d41839cebd60f2`. Reviewed range `ae8af71..f32c205` (4 builder commits). `ae8af71` was written by the orchestrator (tracking docs, token log, `reports/qa/pr-7.md`), so it is out of scope.
- Skills loaded: `ragas-eval`, `langsmith-tracing`.
- Library versions: langsmith 0.14.7, qdrant-client 1.15.1. ragas is not installed and this PR does not touch it (the RAGAS wiring is PR-09).

## Summary

The PR adds the `src/movie_rag/eval/` package:
- `questions_v1.jsonl`: 40 hand-written questions, 10 of each type.
- `EvalQuestion`: the schema, with shape rules that depend on the question type.
- `load_eval_set()`: checks the structure (40 questions, 10 per type, unique ids, each film gold only once) and checks the set against the fixture (gold films exist, filtered gold films satisfy their filters, exact questions name their film, the 4-gram guard and a title check pass, unanswerable titles are absent from the fixture).
- `overlap.py`: the n-gram guard.
- `generate.py`: a seeded paraphrase generator. It writes only to `eval.generated_path` and raises `MissingCredentialError` at call time.
- `upload.py`: the LangSmith dataset upload. It skips and exits 0 when no key is set.
- `docs/EVAL_SET.md` and an `eval:` section in `config.yaml`.

`make check` is green with 99.50 % coverage. I reviewed all 40 questions one by one against the fixture. **0 rejected.** I found 7 minors and 4 nits, and no blockers or majors.

## Acceptance criteria

| Criterion | Evidence (produced by QA) | Status |
|---|---|---|
| Generator: fixed seed, paraphrase prompt, n-gram guard, runs once a key exists | `generate.py:117-118` shuffles with `random.Random(settings.ingest.random_seed)`. The versioned prompt is `prompts/paraphrase_v1.md` (`generate.py:65-71`). The guard and title refusal are in `generate.py:83-93`, with regeneration up to `eval.max_attempts`. Tests: same seed gives the same order, another seed gives another order, retry with feedback, rejection, title refusal (`test_eval_generate.py:107-163`). | ✅ |
| Generator raises `MissingCredentialError` at call time, never at import | `uv run python -c "import movie_rag.eval.generate, movie_rag.eval.upload"` printed `import ok`. `env -u NEBIUS_API_KEY uv run python -m movie_rag.eval.generate` printed `set NEBIUS_API_KEY — see docs/CREDENTIALS.md` with exit 2. Test: `test_without_a_key_generation_raises_the_credentials_error`. | ✅ |
| Generator never overwrites the committed file | The default output is `eval.generated_path` (`generate.py:173`, `config.yaml` `questions_v1_generated.jsonl`). The md5 of `questions_v1.jsonl` was `e1be5286…` before and after the CLI run, and no generated file was created. See minor m1 for the `--out` caveat. | ✅ |
| 40 questions from fixture films, 10 per type, each with type, gold `movie_id` and filters | `load_eval_set(load_settings(env_file=None))` returned 40 questions and raised nothing. `test_the_committed_set_has_forty_questions_ten_per_type` passes. | ✅ |
| Unanswerable questions use plausible films that are absent | Per-question review below. My keyword grep of the fixture found 0 hits each for jellyfish, figure skating, Rotterdam, samurai, sushi, Texas, zoo, Brazil, curling, Venus, baseball, whaling, vampire and dinosaur. No unanswerable question shares a 4-gram with any of the 291 plots (computed maximum: 0). | ✅ |
| `docs/EVAL_SET.md` | It contains the schema, the type definitions, how the set was written, the full 40-row table and an honest limitations section. `test_the_documentation_table_lists_every_question` passes. | ✅ |
| LangSmith upload skips without a key (exit 0) | `env -u LANGSMITH_API_KEY uv run python -m movie_rag.eval.upload` printed `skipped: no LANGSMITH_API_KEY (...)` with exit 0. No `.env` file exists, so it was not read. A mock client receives no calls (`test_without_a_key_nothing_is_called`). | ✅ |
| ✅ Overlap guard tested and passing for all fuzzy questions | I ran the guard myself. `shared_ngrams(q, gold.plot, 4)` returned `[]` for fuzzy-01 to fuzzy-10, and no fuzzy question contains its title. `tests/unit/test_eval_overlap.py` has 13 positive and negative tests. `test_every_fuzzy_question_passes_the_overlap_guard` passes. | ✅ |
| ✅ QA reviewed all 40 and listed rejections | See "Review of all 40 questions" below: 40 accepted, 0 rejected, so no regeneration is needed. | ✅ |
| Loader validation works | Mutation test: in a temporary file I copied plot text into fuzzy-01, gave filtered-01 a wrong filter and set an absent_title to a fixture title. The loader reported all three problems in one `EvalSetError`. A 39-line file was rejected with `unanswerable: 9 questions, expected 10`. | ✅ |
| Tunables in config | `config.yaml` `eval:` covers the paths, `per_type`, `ngram_size`, prompt version, `generate_count`, `max_attempts` and `dataset_name`. The seed comes from `ingest.random_seed`. The diff adds no URLs or model names to `src/` (grep was empty). | ✅ |

## Review of all 40 questions

Method:
- I dumped each gold film's full fixture record and compared it with the question.
- I grepped the 291 cleaned fixture films for each premise's key terms, to find competing films.
- I counted shared rare words (document frequency ≤ 3) and shared 3-grams.
- I ranked every question in dense, sparse and hybrid mode with the real FastEmbed models. Because of the in-memory hybrid problem described in the last section, the ranks were taken from a temporary collection on the real Qdrant service (`qa8_tmp_eval_check`, deleted afterwards).

### Fuzzy plot

| Id | Gold | Verdict | Reason |
|---|---|---|---|
| fuzzy-01 | The Ninth Bell of Varnholt | accept | The doctor, vanished predecessor, death-foretelling bell (as "chime from the steeple") and drawing of a ninth bell all match the plot. The only competing film is *The Thirteenth Chime* (a Turkish clockmaker's cuckoo clock), which is ruled out by the doctor, steeple and Yorkshire details. The bell is paraphrased. The cue word "Yorkshire" appears in 2 plots. |
| fuzzy-02 | The Forgetting Hour | accept | Amnesiac detective, harbour in the rain, head graze, suspects he is hunting himself: all faithful. It is the only amnesia premise; *The Wedding Planner's Alibi* only uses "remember" in passing. |
| fuzzy-03 | Glass Harvest | accept | Fog-harvesting glass towers ("transparent farming towers"), teenage engineer, mining company's secret water supply and blasting all match. The title words are avoided. Shared rare words: "towers" and "blasting" (each in 1 plot). |
| fuzzy-04 | Atlas of Lost Birthdays | accept | Lonely accountant, last train on a rainy night, hidden city of uncelebrated birthdays, strangers celebrating to return years: faithful. "Atlas" and "birthday" are avoided. "Anniversary of birth" is a little stilted but understandable. |
| fuzzy-05 | The Recycled Water Signal | accept | The water-reclamation engineer and the unknown language in the recycled water are paraphrased well. The plot says "desert colony ship", which the question reads as "across a desert planet" (nit n1). The sibling colony-ship film, *The Refrigerated Capsule*, is a stowaway clone story, so it does not fit. |
| fuzzy-06 | The Window-Cleaner's Witness | accept | High-rise window cleaner sees a murder, the killer sees her, the lawyer one floor below offers money for silence: faithful and unique. *Stage Fright at the Alhambra* has a window cleaner but no murder. "Window washer" keeps the title word "window" (minor m3). |
| fuzzy-07 | The Pre-Signed Guest Book | accept | Mountain hut warden finds a guest book already signed for tomorrow by a hiker who has not arrived. "Visitors' register" is a good paraphrase and no other film fits. |
| fuzzy-08 | The Song-Eating Hatchling | accept | Apprentice dragon-keeper, hatchling that eats only songs, choirmaster's hymn swallowed: faithful. "Dragon" and "songs" are avoided. The other dragon film, *The Burnt Birthday Cake*, does not fit. |
| fuzzy-09 | The Hollow Crown of Marsh End | accept | Farmer's daughter, crown of reeds found floating on the flood and put on as a joke, the marsh obeys her, she forgets her name: faithful and unique ("crown" appears in only one plot). |
| fuzzy-10 | Salvage Rights | accept | Insurer hires a deep-sea diver, wreck off the Great Barrier Reef ("famous Australian coral reef"), hull cut open from inside, hidden log, deliberate scuttling: faithful and unique. It is the most lexically keyed fuzzy question: "reef", "diver" and "hull" each appear in only 1 plot, and the shared 3-gram "cut open from" stops one word short of the 4-gram guard (minor m4). |

### Exact entity

The chunk prefix is `Title (Year) | Genre | Director` (`ingest/chunk.py:60-64`), so titles, years and directors are all in the chunk text. The fixture has no duplicate titles. Each of the 4 directors named has exactly 1 film in the fixture (counted).

| Id | Gold | Verdict | Reason |
|---|---|---|---|
| exact-01 | Counterfeit Spring | accept | The title is unique and does not appear in any other plot. Rank 1 in dense, sparse and hybrid. |
| exact-02 | Kimchi Wars | accept | Unique title. Rank 1 in all modes. |
| exact-03 | Nile Postman | accept | Unique title. The release year is in the chunk prefix, so "when was it released" is answerable. Rank 1 in all modes. |
| exact-04 | Pashmina | accept | The single-word title appears only in its own plot. Rank 1 in dense and hybrid, rank 2 in sparse. |
| exact-05 | Echo Chamber 9 | accept | Unique title containing a digit. Rank 1 in all modes. |
| exact-06 | Black Tulip Hotel | accept | Unique title. Rank 1 in all modes. |
| exact-07 | Rails Beneath the Snow (Gleb Ostrovsky-Marin) | accept | The director is unique and in the prefix. Rank 1 in sparse and hybrid, rank 4 in dense. |
| exact-08 | Midnight Ferry to Kowloon (Ling Wai-hung) | accept | The director is unique. Rank 1 in sparse, 2 in hybrid, and not in dense's top 8. A useful question for separating the modes. |
| exact-09 | Tin Soldiers of Avenue Dorval (Rejean Thibodault) | accept | The full name is unique. The surname "Thibodault" also appears for other directors and characters, which makes the question slightly harder. Rank 1 in sparse, 2 in hybrid. |
| exact-10 | Cold Storage (Maxime Delorimier-Haas) | accept | The director is unique. Rank 1 in sparse, 2 in hybrid, 8 in dense. |

### Filtered

I checked every gold film against its filters with `matches_filters`. All filter values (genres and origins) exist in the fixture and the retrieval layer applies them; the number of films that pass each filter matches what the question notes say.

| Id | Gold | Filters | Verdict | Reason |
|---|---|---|---|---|
| filtered-01 | Kerosene Summer | origin=telugu | accept | Faithful (village, tanker siphoning the well, landowner). The filter alone isolates the film, which is documented (minor m5). |
| filtered-02 | Operation Teacup | 1940-49, british | accept | Fake beach tea party at a seaside resort to mislead the invasion forecast: faithful and unique among the 5 films that pass the filter. The note says "two British horror films" but there are three (nit n2). |
| filtered-03 | The Drover's Hymn | western, australian | accept | Faithful. The filter alone isolates the film (m5). |
| filtered-04 | Monsoon Heist | 2000-09, crime, bollywood | accept | Faithful (six strangers, retired magician, vault, floods). The filter alone isolates the film (m5). |
| filtered-05 | A Song for the Sleeping Mill | musical, tamil | accept | Unique among 3 films (the other two are an opera janitor and a bird flute). The 3-gram "the owners family" is shared with the plot, which is under the 4-gram limit. |
| filtered-06 | Seven Days of Rain in Tsarskoye | 1920-29, russian | accept | Unique among 2 films. *The Suitcase in the Lock* has a lock keeper but no railway halt. Calling the stationmaster "keeper" is acceptable. |
| filtered-07 | The Walnut-Shell Boat | 1960-69, animation | accept | Unique among 2 films. "Chased by" a sponge pirate is not in the plot, where the pirate offers a map (nit n3). |
| filtered-08 | The Identical Umbrellas | thriller, hong kong | accept | Faithful and unique among 2 films. Tests an origin value that contains a space. |
| filtered-09 | Orbit of the Last Orchard | science fiction, russian | accept | Faithful except that "jettisoned" stands in for the plot's "evacuation of biological samples" (nit n3). Unique among 2 films. The anchor director is not named. |
| filtered-10 | Seven Stones for Hatsue | 1950-59, japanese | accept | Faithful: potter becomes "ceramicist", kiln becomes "workshop", seven matched tea bowls before the first snow. Unique among 6 films. |

### Unanswerable

None of the absent titles is a fixture title. Gold rank is "none" in every mode by construction.

| Id | Absent title | Verdict | Reason |
|---|---|---|---|
| unanswerable-01 | Raptor for President | accept | Nothing about dinosaurs, presidents or Mars politics in the fixture. Absurd but plausible as an animated comedy. Labelled "easy". |
| unanswerable-02 | The Jellyfish Court | accept | Near miss. *The Spiral Trench* and *The Submarine Pantry* share the trench and submarine setting, but no fixture plot mentions jellyfish. |
| unanswerable-03 | Triple Axel Heist | accept | Strong near miss. *Monsoon Heist* also has a retired professional who assembles a crew to rob a diamond merchant, but the figure skater, Rotterdam and the 1970s are absent, so abstaining is correct. |
| unanswerable-04 | Ronin Roll | accept | No samurai, sushi or Texas in the fixture. Easy. |
| unanswerable-05 | Frequency Zoo | accept | Near miss: 9 fixture plots mention radio, but none mentions a zoo or Brazil. |
| unanswerable-06 | Sweep Nation | accept | No curling and no mockumentary in the fixture. Easy. |
| unanswerable-07 | Hive Run | accept | Near miss: bee films and *The Drifting Capsule* (astronaut) exist, but nothing involves Venus or baseball. |
| unanswerable-08 | The Harpoon Winter | accept | The origin `norwegian` matches no film, so every mode returns 0 results and abstaining is trivial. This is documented. |
| unanswerable-09 | Beam of Devotion | accept | Near miss: there are 13 lighthouse plots, but none has a sentient or infatuated lighthouse (0 hits for lighthouse plus love or infatuation). |
| unanswerable-10 | Fangs and Fillings | accept | No film passes the filter (0 results). The only fixture dentist is in an American western (*The Jar of Teeth*), as the note says. |

**Rejections: none.** The set does not need to be regenerated.

## Lexical leakage and mode separation (builder's informal finding)

I re-measured retrieval on the fixture with the real models, top_k=8, on real Qdrant:

| Fuzzy questions (10) | Hit@1 | Hit@8 |
|---|---|---|
| sparse (BM25) | 10/10 | 10/10 |
| hybrid | 9/10 (fuzzy-01 at rank 2) | 10/10 |
| dense | 7/10 (ranks 7 and 4 for fuzzy-01/02, fuzzy-08 missed) | 9/10 |

This confirms the builder's claim (BM25 at rank 1 for all 10, dense 9/10 in the top 8).

What the questions leak:
- They contain no character names (no capitalised name tokens apart from "Yorkshire" and "Australian").
- They contain no title phrases, and none of them names the film's distinctive object.
- What they do keep is 0 to 4 occupation or setting nouns that are rare in a 291-plot corpus (counts are the number of plots containing the word):
  - "accountant" (2), "hymn" (2), "Yorkshire" (2), "farmers" (2)
  - "towers" (1), "blasting" (1)
  - "reef", "diver" and "hull" (1 each)
- fuzzy-07 keeps no word that rare and still ranks first in BM25.

My judgement: this is mainly a limitation of the fixture, not a defect in the question set.
- Every synthetic film has a one-of-a-kind occupation and premise, and the short generated plots repeat those nouns.
- Any faithful description that a human could pose will therefore share at least one rare noun, and in a 291-document corpus one such noun is enough for BM25.
- The KICKOFF guard (no shared 4-gram) is fully met.
- The limitation is documented honestly in `docs/EVAL_SET.md` ("Easy for lexical search") and in `docs/BACKLOG.md`. Both mark it as informal and point to PR-09 as the authoritative source.

The set still separates the modes, but in the direction that favours BM25 rather than dense. The PR-09 report should say so and not claim that hybrid or dense wins on fuzzy questions.

One question could be made harder without losing fidelity: fuzzy-10 (m4).

## Integrity checks

| Check | Result |
|---|---|
| `pragma: no cover`, `pytest.mark.skip`, `xfail`, `pytest.skip(` added | none (`git diff ae8af71..HEAD \| grep`, empty) |
| Coverage `omit` / `fail_under` / `--cov-fail-under` changes | none (no diff to `pyproject.toml` or `Makefile`) |
| Protected files (`.github/`, `.claude/`, `CLAUDE.md`, `KICKOFF.md`, `TOKEN_USAGE`, `token_usage.jsonl`, `.env*`) | not touched by the builder commits. `TOKEN_USAGE` and `token_usage.jsonl` changed only in the orchestrator commit `ae8af71`. |
| Secrets (`sk-…`, `lsv2_…`, literal keys) | none. The test key `test-langsmith-key-not-real` is in tests only. |
| Hard-coded models, URLs or retrieval parameters in `src/` | none. The upload uses `observability.langsmith_api_url` and the generator uses `make_chat_model(settings)`. |
| Network or credentials needed by tests | no. The generator tests use `ScriptedChatModel`, the upload tests use `MagicMock(spec=Client)`, and the CLIs are tested with `load_settings(env_file=None)`. |
| Tests assert something meaningful | yes. Exact messages and phrases, call arguments, counts and file round trips are checked. `test_no_example_carries_the_key` is weak (nit n4). |
| `.gitignore` change narrow and safe | `git check-ignore -v`: `data/raw/x.csv` and `src/movie_rag/ingest/data/x` are still ignored by `data/`. `…/eval/data/questions_v1_generated.jsonl` is ignored. `questions_v1.jsonl` is not ignored and is tracked. Only `src/movie_rag/eval/data/` is re-included. |
| Scope | Within the plan row. No Makefile target was added (the plan does not require one). The data file is at `src/movie_rag/eval/data/` and not the plan's `eval/data/`; this matches the KICKOFF §3 layout `src/movie_rag/ … eval/ (data/)` and is recorded in the PR. |
| `logging` instead of `print` | No `print(` in the `src/` diff. CLI output goes through `ingest.download.say`. |
| `uv.lock` | Unchanged; no new dependencies. |

## `make check`

`make check` at `f32c205`, log `logs/qa8-check.log`:
- ruff: All checks passed. ruff format: 105 files already formatted. mypy: no issues in 34 source files.
- pytest: **877 passed, 1 deselected** (the live test) in 500.85 s.
- Coverage: **99.50 %** line+branch (gate 80 %).
- New modules: `eval/overlap.py` 100 %, `eval/questions.py` 100 %, `eval/generate.py` 98 % (missing `78->76` and `186`, the `__main__` guard), `eval/upload.py` 97 % (missing `107`, the `__main__` guard).

## Skill checklists

**ragas-eval** (for the parts that apply to PR-08):
- [x] Overlap guard tested, with positive and negative cases, and all fuzzy questions pass.
- [x] Fixed seed and versioned paraphrase prompt.
- [n/a] ragas version and judge model, judge ≠ generator, deterministic metrics, report traceability: these are PR-09.

**langsmith-tracing:**
- [x] The upload command skips cleanly without a key (exit 0, no client calls).
- [x] No secrets in the payloads (examples contain only the question, filters, gold ids and question metadata).
- [x] Works with tracing off (conftest forces `LANGSMITH_TRACING=false`).
- [n/a] Run metadata: there are no new traced runs.

## Findings

### Blocker
None.

### Major
None.

### Minor
- **m1** `src/movie_rag/eval/generate.py:173`: `--out` accepts any path, including the committed `questions_v1.jsonl`, and no config validator enforces `generated_path != questions_path` (`config.py:125-133`). The only test of this (`test_eval_generate.py:222-224`) checks the committed config values. **Fix:** in `main`, refuse with `EXIT_USAGE` when `out.resolve() == cfg.resolve(cfg.questions_path).resolve()`, and add a `model_validator` on `EvalConfig` that rejects equal paths, with a test for each.
- **m2** `src/movie_rag/eval/upload.py:281-287`: `main` catches only `MovieRagError`. With a key set, an auth or network error from `Client.has_dataset`, `create_dataset` or `create_examples` (`langsmith.utils.LangSmithError`, `httpx`/`requests` errors) shows a traceback instead of a one-line message and exit 1. **Fix:** catch `LangSmithError` (and connection errors), print a message without the key and return `EXIT_FAILURE`, with a mocked-client test that raises.
- **m3** `questions_v1.jsonl` fuzzy-06: "window washer" keeps "window", one of the two main title words of *The Window-Cleaner's Witness* (the title phrase "window-cleaner's witness" appears twice in the plot). This is acceptable as natural speech, but it weakens the paraphrase of the distinctive object. **Fix (optional):** for example "someone who cleans the glass on the outside of a tower block".
- **m4** `questions_v1.jsonl` fuzzy-10 is the most lexically keyed question: "reef", "diver" and "hull" each appear in only one plot, and "hull was cut open from within" shares the 3-gram "cut open from" with the plot. **Fix (optional):** for example "an underwater salvage expert hired by insurers finds a sunken freighter off Queensland was opened from the inside".
- **m5** `questions_v1.jsonl` filtered-01, -03 and -04: the filter alone isolates the gold film (1 film passes), so in every mode the score depends on the filter and not on the semantic part of the question. This is documented in `docs/EVAL_SET.md`. **Fix (optional, could go to the backlog):** loosen one filter on each, for example drop the genre on filtered-04, so that 2 or more films pass.
- **m6** (process, not code) The PR description refers to the QA report as `reports/qa/pr-08.md`. In GitHub mode it is `reports/qa/pr-8.md`. **Fix:** update the link in `docs/prs/PR-08.md`.
- **m7** (informational, pre-existing, out of scope for PR-08, relevant to PR-09) With qdrant-client 1.15.1 in local `":memory:"` mode, `Retriever.search(mode="hybrid")` (prefetch + RRF + `query_points_groups`) returned the same top 10 for all 40 questions and ignored the filters (for example 10 results for `origin=telugu`, where only 1 film exists). On the real Qdrant service, hybrid behaves correctly. Hybrid numbers from an in-memory index are therefore meaningless, which may also explain why the builder's scratch check reported only BM25 and dense. **Fix:** the orchestrator should add a backlog item so that PR-09's eval runner uses the Qdrant service, or refuses in-memory, for hybrid mode.

### Nits
- **n1** fuzzy-05: "across a desert planet" goes beyond the plot's "desert colony ship".
- **n2** filtered-02 notes: there are three British horror films among the other four, not two.
- **n3** filtered-07 says "chased by" (in the plot the pirate offers a map) and filtered-09 says "jettisoned" (in the plot, "evacuation"). Both are small fidelity drifts; a person half-remembering a film could plausibly say either.
- **n4** `tests/unit/test_eval_upload.py:85-86`: `test_no_example_carries_the_key` is close to tautological because `to_example` never sees the settings. Assert on the actual `create_examples` and `create_dataset` call arguments instead.

## Verdict

0 blockers, 0 majors. All 40 questions were reviewed and accepted, with no rejections. All acceptance criteria have evidence that I produced myself. m1 and m2 are worth fixing in a follow-up; m3 to m5 are optional improvements to the question set; m7 is a backlog item for PR-09.

VERDICT: PASS
