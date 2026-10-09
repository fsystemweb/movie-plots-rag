---
name: langsmith-tracing
description: Use when adding tracing, run metadata, LangSmith datasets or experiments anywhere in the codebase, or reviewing observability.
---
# LangSmith tracing

**Version first:** `uv pip show langsmith`. The installed version's docs win; record version + doc URL in the PR.

## Docs
- Observability: https://docs.smith.langchain.com/observability
- Annotate code / traceable: https://docs.smith.langchain.com/observability/how_to_guides/annotate_code
- Evaluation: https://docs.smith.langchain.com/evaluation
- Datasets: https://docs.smith.langchain.com/evaluation/how_to_guides/manage_datasets_programmatically

## Working pattern
```python
from langsmith import traceable

@traceable(run_type="retriever", name="search", metadata=run_metadata(mode=mode))
def search(...) -> list[dict]:
    return [{"page_content": snippet, "type": "Document", "metadata": {...}} for ...]
```
- `observability.run_metadata(**extra)` returns git sha, prompt version, chat model, embedding model, retrieval mode,
  config hash. Attach it to every run.
- Tracing switch reads `LANGSMITH_TRACING`; without `LANGSMITH_API_KEY` everything is a no-op, same code path.
- Datasets/experiments: `Client().create_dataset(...)`, `create_examples(...)`, `client.evaluate(target, data=...,
  evaluators=[...], experiment_prefix=mode)`; without a key log "skipped: no LANGSMITH_API_KEY" and return.
- Never put secrets (API keys, headers, env dumps) into inputs/outputs/metadata. Use `hide_inputs`/filters if needed.

## Testing
- `tests/conftest.py` sets `LANGSMITH_TRACING=false`.
- Test `run_metadata()` keys; test functions run identically with tracing off; test that no metadata value contains a
  secret-shaped string even when a fake key is in the env.

## Review checklist
- [ ] Works with tracing off and no key (tests)
- [ ] Required metadata fields present
- [ ] No secrets in payloads/metadata
- [ ] Upload/experiment commands skip cleanly without a key
