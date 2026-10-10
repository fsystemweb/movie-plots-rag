# PR tracker

Orchestrator memory across restarts. Statuses: `todo · in-progress · merged · blocked · skipped`.

| ID | Title | Branch | Status | PR | CI | QA | Attempts | Merged (UTC) | Notes |
|---|---|---|---|---|---|---|---:|---|---|
| PR-01 | Scaffold, tooling, CI | pr/01-scaffold | merged | [#1](https://github.com/fsystemweb/movie-plots-rag/pull/1) | ✅ | ✅ | 1 | 2026-10-09 20:14 | QA FAIL r1 (ruff ignores too broad) → PASS r2; CI blocked first by workflow YAML bug, fixed on main 7101dca |
| PR-02 | Settings, errors, observability, data | pr/02-settings-data | merged | [#2](https://github.com/fsystemweb/movie-plots-rag/pull/2) | ✅ | ✅ | 1 | 2026-10-09 20:56 | QA FAIL r1 (synthetic fixture not distinctive) → PASS r2; fixture is synthetic (no licence-verified mirror) |
| PR-03 | Chunking, embeddings, ingestion | pr/03-ingestion | merged | [#3](https://github.com/fsystemweb/movie-plots-rag/pull/3) | ✅ | ✅ | 1 | 2026-10-09 22:14 | QA PASS r1 (4 minors → backlog); CI `tests` flaked on Docker Hub pull rate limit, green on delayed rerun |
| PR-04 | Hybrid retrieval | pr/04-retrieval | merged | [#4](https://github.com/fsystemweb/movie-plots-rag/pull/4) | ✅ | ✅ | 1 | 2026-10-10 08:39 | QA PASS r1 (4 minors → backlog); CI green first run; `rrf_k` dropped pending qdrant-client ≥1.16 |
| PR-05 | FastMCP server | pr/05-mcp-server | merged | [#5](https://github.com/fsystemweb/movie-plots-rag/pull/5) | ✅ | ✅ | 1 | 2026-10-10 09:14 | QA PASS r1 (5 minors → backlog); CI green first run; delivers PR-04 deferred tracing + title lookup |
| PR-06 | Agent + CLI | pr/06-agent-cli | merged | [#6](https://github.com/fsystemweb/movie-plots-rag/pull/6) | ✅ | ✅ | 1 | 2026-10-10 09:41 | QA PASS r1 (6 minors → backlog; m2 hallucinated titles can remain in answer text); CI green first run; live test never run |
| PR-07 | Streamlit test page | pr/07-streamlit-ui | merged | [#7](https://github.com/fsystemweb/movie-plots-rag/pull/7) | ✅ | ✅ | 1 | 2026-10-10 10:59 | QA PASS r1 (5 minors + 2 nits → backlog); CI green first run; screenshot captured headless |
| PR-08 | Evaluation set | pr/08-eval-set | in-progress | | | | 1 | | |
| PR-09 | Evaluation runner | pr/09-eval-runner | todo | | | | 0 | | |
| PR-10 | ADRs and README | pr/10-adrs-readme | todo | | | | 0 | | |
| PR-11 | Stakeholder overview, credentials guide, demo | pr/11-overview-demo | todo | | | | 0 | | |
| PR-12 | Stretch: reranker | pr/12-reranker | todo | | | | 0 | | |
