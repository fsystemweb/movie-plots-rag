---
name: ragas-eval
description: Use when building or reviewing the evaluation set, deterministic metrics (Hit@k, MRR, abstention), RAGAS metrics, or the eval report.
---
# Evaluation with RAGAS + deterministic metrics

**Version first:** pin and check `uv pip show ragas`. The installed version's docs win; record version, judge model
and doc URL in the PR and in the report.

## Docs
- Metrics list: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/
- Custom / LangChain LLMs: https://docs.ragas.io/en/stable/howtos/customizations/customize_models/
- Evaluate API: https://docs.ragas.io/en/stable/references/evaluate/

## Working pattern
- 0.2/0.3-era API: `from ragas import evaluate, EvaluationDataset`; metrics `Faithfulness`, `ResponseRelevancy`,
  `LLMContextPrecisionWithReference`, `LLMContextRecall`; judge via `LangchainLLMWrapper(ChatOpenAI(model=cfg.judge_model, ...))`.
- Newer releases: `ragas.metrics.collections` + `llm_factory`. Use whatever the pinned version documents.
- **Judge ≠ generator** (assert in code: `cfg.judge_model != cfg.chat_model`).
- Deterministic first (no LLM): Hit@k (gold movie_id in top-k), MRR, abstention rate on unanswerable questions;
  latency p50/p95, tokens per question.
- Without `NEBIUS_API_KEY`: RAGAS step logs a clear skip and the report shows "pending credentials".
- Output `reports/eval_<mode>.json`; `make report` renders `reports/EVAL_RESULTS.md` from the JSON only (every number
  traceable).
- Question generation: fixed seed; paraphrase prompt; n-gram guard = no 4-word sequence shared with the source plot
  (lowercased, punctuation stripped).

## Testing
- Hit@k/MRR/abstention on hand-computed cases (e.g. ranks [1, 3, None] → MRR = (1 + 1/3 + 0)/3).
- Overlap guard: positive and negative examples; all fuzzy questions in `questions_v1.jsonl` pass it.
- RAGAS wiring tested with a fake judge or by asserting the skip path without a key.

## Review checklist
- [ ] ragas version + judge model recorded in report
- [ ] judge ≠ generator
- [ ] overlap guard tested
- [ ] deterministic metrics tested on hand-computed cases
- [ ] report numbers traceable to `reports/eval_<mode>.json`
