# QA report: PR #1 (PR-01, Scaffold, tooling, CI), round 2

Branch `pr/01-scaffold` @ `32cdf45d0de70280c5e094ed8b861678f0fc9a9b`, GitHub mode. Reviewer: qa-validator.

## Summary

The builder fixed every round-1 finding in 6017590 and recorded the response in 32cdf45. Round 1 had 1 major and 5 minors.

- **Ruff exemptions:** `.claude/hooks/*.py` now ignores exactly the five rule codes that fire there. `tests/hooks` has no exemptions at all: it is fully linted and formatted.
- **Hook tests not weakened:** I compared the old and new hook test files as parsed Python. After renaming `O` to `ORCH` they are identical, and the 182 collected test IDs are the same before and after.
- **Gates:** `make setup && make check` is green: 182 passed, 100% line+branch coverage. CI is green on the head SHA.
- **Integrity:** no guardrail violations.

Zero blockers and zero majors remain.

## Round 1 history (ac625da, VERDICT: FAIL)

- **M1 (major):** the ruff `per-file-ignores` turned off whole rule families (9 entries for `tests/hooks`, including `B`), but only E741 fired there.
- **m1:** `tests/hooks` was permanently excluded from `ruff format`.
- **m2:** `"*.md"` was excluded from formatting across the whole repo.
- **m3:** the README `ci` row overstated the smoke eval.
- **m4:** `setup` used `[ -d .git ]`, which skips `pre-commit install` in git worktrees.
- **m5:** the write-up cited a stale count of 178 tests.

## Re-verification of round-1 findings

| Finding | Evidence (produced by QA at 32cdf45) | Status |
|---|---|---|
| M1 per-file ignores too wide | pyproject.toml:42 is `".claude/hooks/*.py" = ["E501", "SIM114", "RUF005", "UP017", "C420"]`, and there is no `tests/hooks` entry. Running `ruff check --isolated --select E,F,W,I,B,UP,SIM,C4,RUF` on `.claude/hooks/*.py` still flags exactly RUF005 ×2, C420 ×2, UP017, SIM114 and E501, so the ignore list matches what fires, code for code. The same isolated run on `tests/ src/` exits 0 with no hits. `ruff check .` prints "All checks passed!" | ✅ fixed |
| m1 tests/hooks format exclusion | pyproject.toml:47 is `exclude = [".claude/hooks/*.py", ".claude/**/*.md"]`. `ruff format --check .` reports "17 files already formatted", up from 2 in round 1 | ✅ fixed |
| m2 `*.md` repo-wide | The exclusion is narrowed to `.claude/**/*.md`, which is protected. README and docs markdown now go through the format check and pass | ✅ fixed |
| m3 README `ci` row | README.md:34 now reads "retrieval smoke eval (stub until PR-09) ... ready (smoke eval: PR-09)" | ✅ fixed |
| m4 `setup` in worktrees | Makefile:24 now uses `if git rev-parse --git-dir >/dev/null 2>&1; then ...`. `make setup` exits 0 and prints "pre-commit installed" | ✅ fixed |
| m5 stale count | docs/prs/PR-01.md:21-22 now cite `182 passed`, which matches my run | ✅ fixed |

## Hook tests: reformat and rename did not weaken them

The round-2 diff touches `tests/hooks/_hookutil.py`, `test_guardrails.py` and `test_track_tokens.py`. I checked it three ways:

- **Parsed code is identical.** I parsed each file in `tests/hooks/` at ac625da and at 32cdf45 (via `git show`) and compared `ast.dump`, renaming the old `O` to `ORCH` first. All 5 files (test_guardrails, test_track_tokens, _hookutil, conftest, test_stop_gates) came out *AST-identical*. The `assert` counts are unchanged (39/39, 33/33, 1/1, 0/0, 11/11), and no bare `O` remains. So the change is formatting plus one rename, with no change in behaviour.
- **Same tests are collected.** `pytest --collect-only -q tests` gives the same sorted list of test IDs at both commits ("COLLECTED IDS IDENTICAL"; 182). The parametrised counts are also the same, for example `test_git_and_gh_rules` 42, `test_env_rules` 20, `test_dangerous_shell` 19, `test_protected_paths_tool_write` 16, `test_content_rules` 14, `test_qa_scope` 11, `test_protected_paths_shell_write` 11, `test_orchestrator_write_scope` 7, `test_orchestrator_shell_scope` 5, `test_local_merge_verdict_gate` 4.
- **Protected hook scripts untouched.** No file under `.claude/` is in the diff.

## Acceptance criteria (re-run at 32cdf45)

| Criterion | Evidence | Result |
|---|---|---|
| uv project, pyproject (ruff, mypy strict, markers, coverage branch/source) | pyproject.toml has `strict = true`, `markers` live/integration with `--strict-markers`, `branch = true`, `source = ["src/movie_rag"]`. No `fail_under` and no `omit` | ✅ |
| Makefile per §4.7 | In a fresh clone at 32cdf45, `make ci` printed `ci ok: .claude/state/ci-01.ok = 32cdf45d...`. Unit tests check the stub targets | ✅ |
| pre-commit (ruff, ruff-format, check-yaml, end-of-file, gitleaks) | In the clone, `pre-commit run --all-files` passed all 6 hooks and left the tree clean | ✅ |
| docker-compose Qdrant (pinned, healthcheck, volume) | `docker-compose.yml` is unchanged since round 1. Re-verified: `docker compose up -d --wait qdrant` printed `Healthy`, `docker inspect` printed `healthy`, `/readyz` returned `all shards are ready`. Then `docker compose down` removed everything | ✅ |
| config.yaml skeleton, `src/movie_rag/__init__.py`, smoke test | Unchanged since round 1 (`git diff --quiet ac625da 32cdf45 -- config.yaml src tests/unit`). The 8 unit tests pass | ✅ |
| README skeleton | Quickstart, make targets and "Built autonomously" sections are present. Every link is tracked (verified in round 1; the only round-2 change is the `ci` row) | ✅ |
| `make setup && make check` green | `make setup` exits 0. `make check` exits 0: ruff "All checks passed!", "17 files already formatted", mypy "Success: no issues found in 1 source file", `182 passed in 2.51s`, "Total coverage: 100.00%" | ✅ |
| Hook tests still green | All 174 hook tests pass inside `make test`, and they are behaviourally identical to round 1 (see above) | ✅ |
| CI green in GitHub mode | `gh run view 37985322381` returned `headSha 32cdf45d...`, `conclusion success`. `gh pr checks 1`: quality pass, tests pass, secret-scan pass, smoke-eval skipping (gated on a label) | ✅ |

## Integrity checks (full diff `origin/main...origin/pr/01-scaffold`)

| Check | Result |
|---|---|
| `pragma: no cover`, `fail_under`, `omit`, `noqa`, skip/xfail/skipif/importorskip added | None (grep over the diff, excluding uv.lock, found nothing) |
| Protected paths (`.github/workflows/`, `.claude/`, `CLAUDE.md`, `KICKOFF.md`, token files, `.env*`) | None in the diff |
| Secrets / raw data | None. The grep found nothing, and the pre-commit gitleaks hook and CI secret-scan both pass |
| Hard-coded models, URLs or params in `src/` | None. `grep -rnE 'http\|qwen\|bge\|nebius\|6333' src/` is empty |
| Credential needed to pass | No |
| Scope | Within the plan row. The `tests/hooks` edits are formatting plus one rename that QA asked for, and the parsed code is unchanged |
| History rewrite between rounds | None. `git merge-base --is-ancestor ac625da 32cdf45` succeeds |

## Skills applied

- **pr-workflow**: Commits are conventional (`fix:`, `docs:`) and were added as new commits, with no force-push. docs/prs/PR-01.md has a "QA response" section answering each finding, and the fix sha 6017590 is cited.
- **qdrant-hybrid**: Only the infrastructure part applies. The image tag is pinned and identical in compose and CI, and the tunables are in config.yaml. The retrieval checklist items start in PR-03 and PR-04.

## Findings

### Blocker
None.

### Major
None.

### Minor
None new. All round-1 minors (m1 to m5) are fixed, as verified above.

## Coverage

Line+branch on `src/movie_rag`: **100.00%** (8 statements, 0 missed; gate 80%). Tests: **182 passed** (174 hook + 8 unit), 0 skipped, 0 failed.

VERDICT: PASS
