---
name: pr-workflow
description: Always load when starting or finishing a PR — branching, commits, make check, opening the PR in GitHub or local mode, and responding to QA findings.
---
# PR workflow

**Version first:** for any library you touch, `uv pip show <pkg>`; the installed version's docs win. Record versions +
doc URLs in the PR body.

## Steps
1. Branch `pr/<NN>-<slug>` from `main` (the orchestrator usually creates/checks it out for you — verify with
   `git branch --show-current`).
2. Conventional commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`, `refactor:`), small and focused.
3. `make check` green before opening (lint, format check, mypy strict, tests, coverage ≥ 80% line+branch).
4. Fill `.github/pull_request_template.md` completely: summary, plan row, acceptance checklist **with evidence**
   (command + output excerpt or file:line), library versions + doc URLs, decisions, coverage %, follow-ups (also
   appended to `docs/BACKLOG.md`).
5. Open the PR:
   - **GitHub mode** (`gh auth status && git remote get-url origin` both succeed):
     `git push -u origin pr/<NN>-<slug>` then `gh pr create --base main --title "<type>: <title> (PR-NN)" --body-file <file>`.
     Write the body file under `docs/prs/PR-NN.md` and commit it too.
   - **Local mode**: write `docs/prs/PR-NN.md` (same template) and commit it on the branch.
6. Fixes after CI/QA feedback: **new commits on the same branch**, never force-push, never a new PR. Push again in
   GitHub mode. Append a "QA response" section to `docs/prs/PR-NN.md` answering each finding point by point
   (fixed in <sha> / disagree because …), and in GitHub mode post it with `gh pr comment`.

## Never
Merge, approve, push to main, force-push, `--no-verify`, edit protected files (see CLAUDE.md), lower coverage gates.

## Review checklist (QA)
- [ ] Branch name and conventional commits
- [ ] Template fully filled; every acceptance criterion has evidence
- [ ] Versions + doc URLs recorded
- [ ] Follow-ups in `docs/BACKLOG.md`
- [ ] No force-push history rewrite between QA rounds
