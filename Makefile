# Movie Plots RAG: Makefile contract (KICKOFF.md section 4.7, CLAUDE.md "Makefile targets").
# Targets owned by later PRs print `not implemented yet (PR-NN)` and exit 0.

SHELL := /bin/bash
UV ?= $(shell command -v uv 2>/dev/null || echo $(HOME)/.local/bin/uv)
RUN := $(UV) run

# The coverage gate lives on the command line (never in pyproject.toml).
COV_FAIL_UNDER := 80
PYTEST_COV := --cov=src/movie_rag --cov-branch --cov-report=term-missing --cov-fail-under=$(COV_FAIL_UNDER)

BRANCH := $(shell git rev-parse --abbrev-ref HEAD 2>/dev/null)
PR_NUM := $(shell echo '$(BRANCH)' | sed -nE 's|^pr/([0-9]+)-.*|\1|p')
CI_ID := $(if $(PR_NUM),$(PR_NUM),$(subst /,-,$(BRANCH)))

.PHONY: help setup lint format typecheck test cov check ci up down download ingest serve ask ui \
        eval eval-smoke report doctor demo

help:
	@echo "targets: setup lint format typecheck test cov check ci up down download ingest serve ask ui eval eval-smoke report doctor demo"

setup:
	$(UV) sync --all-extras --dev
	@if [ -d .git ]; then $(RUN) pre-commit install; fi

lint:
	$(RUN) ruff check .

format:
	$(RUN) ruff check --fix .
	$(RUN) ruff format .

typecheck:
	$(RUN) mypy src

# Full suite, including tests/hooks, with the 80% line+branch gate on src/movie_rag.
test:
	$(RUN) pytest -m "not live" $(PYTEST_COV)

cov:
	$(RUN) pytest -m "not live" $(PYTEST_COV) --cov-report=html

check: lint
	$(RUN) ruff format --check .
	$(MAKE) --no-print-directory typecheck test

# Local-mode merge gate: check + gitleaks (if installed) + retrieval half of eval-smoke.
ci: check
	@if command -v gitleaks >/dev/null 2>&1; then gitleaks detect --no-banner --redact; \
	else echo "gitleaks not installed: skipping secret scan (CI runs it)"; fi
	@$(MAKE) --no-print-directory eval-smoke
	@mkdir -p .claude/state
	@git rev-parse HEAD > .claude/state/ci-$(CI_ID).ok
	@echo "ci ok: .claude/state/ci-$(CI_ID).ok = $$(cat .claude/state/ci-$(CI_ID).ok)"

up:
	docker compose up -d --wait qdrant

down:
	docker compose down

download:   ; @echo "not implemented yet (PR-02)"
doctor:     ; @echo "not implemented yet (PR-02)"
ingest:     ; @echo "not implemented yet (PR-03)"
demo:       ; @echo "not implemented yet (PR-04)"
serve:      ; @echo "not implemented yet (PR-05)"
ask:        ; @echo "not implemented yet (PR-06)"
ui:         ; @echo "not implemented yet (PR-07)"
eval:       ; @echo "not implemented yet (PR-09)"
eval-smoke: ; @echo "not implemented yet (PR-09)"
report:     ; @echo "not implemented yet (PR-09)"
