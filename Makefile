# Movie Plots RAG — Makefile contract (KICKOFF.md §4.7).
# Phase 0 bootstrap: only `test` (hook tests) is real; every other target is a stub until its PR lands.

SHELL := /bin/bash
UV ?= $(shell command -v uv 2>/dev/null || echo $(HOME)/.local/bin/uv)

.PHONY: help setup lint format typecheck test cov check ci up down download ingest serve ask ui \
        eval eval-smoke report doctor demo

help:
	@echo "targets: setup lint format typecheck test cov check ci up down download ingest serve ask ui eval eval-smoke report doctor demo"

setup:     ; @echo "not implemented yet (PR-01)"
lint:      ; @echo "not implemented yet (PR-01)"
format:    ; @echo "not implemented yet (PR-01)"
typecheck: ; @echo "not implemented yet (PR-01)"
cov:       ; @echo "not implemented yet (PR-01)"
check:     ; @echo "not implemented yet (PR-01)"
ci:        ; @echo "not implemented yet (PR-01)"
up:        ; @echo "not implemented yet (PR-01)"
down:      ; @echo "not implemented yet (PR-01)"
download:  ; @echo "not implemented yet (PR-02)"
doctor:    ; @echo "not implemented yet (PR-02)"
ingest:    ; @echo "not implemented yet (PR-03)"
demo:      ; @echo "not implemented yet (PR-04)"
serve:     ; @echo "not implemented yet (PR-05)"
ask:       ; @echo "not implemented yet (PR-06)"
ui:        ; @echo "not implemented yet (PR-07)"
eval:      ; @echo "not implemented yet (PR-09)"
eval-smoke: ; @echo "not implemented yet (PR-09)"
report:    ; @echo "not implemented yet (PR-09)"

# Hook tests run with an ephemeral pytest until PR-01 creates the uv project (then: full suite + coverage gate).
test:
	$(UV) run --no-project --with pytest python -m pytest tests/hooks -q -p no:cacheprovider
