# PR tracker

Orchestrator memory across restarts. Statuses: `todo · in-progress · merged · blocked · skipped`.

| ID | Title | Branch | Status | PR | CI | QA | Attempts | Merged (UTC) | Notes |
|---|---|---|---|---|---|---|---:|---|---|
| PR-01 | Scaffold, tooling, CI | pr/01-scaffold | merged | [#1](https://github.com/fsystemweb/movie-plots-rag/pull/1) | ✅ | ✅ | 1 | 2026-10-09 20:14 | QA FAIL r1 (ruff ignores too broad) → PASS r2; CI blocked first by workflow YAML bug, fixed on main 7101dca |
| PR-02 | Settings, errors, observability, data | pr/02-settings-data | merged | [#2](https://github.com/fsystemweb/movie-plots-rag/pull/2) | ✅ | ✅ | 1 | 2026-10-09 20:56 | QA FAIL r1 (synthetic fixture not distinctive) → PASS r2; fixture is synthetic (no licence-verified mirror) |
| PR-03 | Chunking, embeddings, ingestion | pr/03-ingestion | merged | [#3](https://github.com/fsystemweb/movie-plots-rag/pull/3) | ✅ | ✅ | 1 | 2026-10-09 22:14 | QA PASS r1 (4 minors → backlog); CI `tests` flaked on Docker Hub pull rate limit, green on delayed rerun |
| PR-04 | Hybrid retrieval | pr/04-retrieval | in-progress | | | | 1 | | |
| PR-05 | FastMCP server | pr/05-mcp-server | todo | | | | 0 | | |
| PR-06 | Agent + CLI | pr/06-agent-cli | todo | | | | 0 | | |
| PR-07 | Streamlit test page | pr/07-streamlit-ui | todo | | | | 0 | | |
| PR-08 | Evaluation set | pr/08-eval-set | todo | | | | 0 | | |
| PR-09 | Evaluation runner | pr/09-eval-runner | todo | | | | 0 | | |
| PR-10 | ADRs and README | pr/10-adrs-readme | todo | | | | 0 | | |
| PR-11 | Stakeholder overview, credentials guide, demo | pr/11-overview-demo | todo | | | | 0 | | |
| PR-12 | Stretch: reranker | pr/12-reranker | todo | | | | 0 | | |
