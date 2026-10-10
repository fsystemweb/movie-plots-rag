"""The evaluation report: the ``reports/eval_<mode>.json`` model and the ``reports/EVAL_RESULTS.md`` renderer.

The markdown is rendered from the JSON files only (never from a live run), so every number in it can be traced to a
file in ``reports/``. LLM-backed numbers that were not computed are printed as ``pending credentials`` (or ``not run``),
never as zero.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

from movie_rag.config import RetrievalMode
from movie_rag.errors import EvalError
from movie_rag.eval.agent_eval import AgentMetrics, AgentOutcome
from movie_rag.eval.langsmith_experiment import ExperimentResult
from movie_rag.eval.metrics import RetrievalMetrics, Summary
from movie_rag.eval.ragas_judge import METRIC_NAMES, MetricSummary, SampleScores
from movie_rag.eval.retrieval_eval import IndexInfo, QuestionRetrieval

SCHEMA_VERSION = 1
MODE_ORDER: tuple[RetrievalMode, ...] = ("dense", "sparse", "hybrid")
PENDING = "pending credentials"
LlmStatus = Literal["completed", "pending_credentials", "not_requested", "failed"]
STATUS_TEXT: dict[str, str] = {
    "pending_credentials": PENDING,
    "not_requested": "not run",
    "failed": "failed",
}
RAGAS_LABELS = {
    "faithfulness": "RAGAS faithfulness",
    "response_relevancy": "RAGAS response relevancy",
    "context_precision": "RAGAS context precision",
    "context_recall": "RAGAS context recall",
}


class QuestionSetInfo(BaseModel):
    path: str
    n: int
    per_type: dict[str, int]


class RetrievalSection(BaseModel):
    k_values: list[int]
    depth: int
    overall: RetrievalMetrics
    by_type: dict[str, RetrievalMetrics]
    latency_ms: Summary
    per_question: list[QuestionRetrieval]


class LlmSection(BaseModel):
    """The LLM-backed half. ``status`` says whether its numbers exist and, if not, why."""

    status: LlmStatus
    reason: str | None = None
    chat_model: str
    judge_model: str
    ragas_version: str
    agent: AgentMetrics | None = None
    ragas: dict[str, MetricSummary] | None = None
    agent_per_question: list[AgentOutcome] | None = None
    ragas_per_question: list[SampleScores] | None = None


class EvalReport(BaseModel):
    """Everything one ``make eval MODE=...`` run measured."""

    schema_version: int = SCHEMA_VERSION
    mode: RetrievalMode
    generated_at: str
    git_sha: str
    config_hash: str
    embedding_model: str
    sparse_model: str
    question_set: QuestionSetInfo
    index: IndexInfo
    retrieval: RetrievalSection
    llm: LlmSection
    langsmith: ExperimentResult


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def report_path(directory: Path, mode: str) -> Path:
    return directory / f"eval_{mode}.json"


def write_report(report: EvalReport, directory: Path) -> Path:
    """Write ``eval_<mode>.json`` (stable key order, trailing newline) and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = report_path(directory, report.mode)
    path.write_text(json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def read_reports(directory: Path) -> list[EvalReport]:
    """The ``eval_<mode>.json`` files of ``directory`` in dense, sparse, hybrid order. Raises if there are none."""
    reports: list[EvalReport] = []
    for mode in MODE_ORDER:
        path = report_path(directory, mode)
        if not path.is_file():
            continue
        try:
            reports.append(EvalReport.model_validate_json(path.read_text(encoding="utf-8")))
        except ValidationError as exc:
            raise EvalError(f"{path.name} is not a valid evaluation report ({exc.error_count()} errors)") from exc
    if not reports:
        raise EvalError(f"no eval_<mode>.json in {directory}: run `make eval` first")
    return reports


# --- markdown -----------------------------------------------------------------------------------------------


def _num(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _retrieval_row(label: str, metrics: RetrievalMetrics, ks: Sequence[int]) -> list[str]:
    return [label, str(metrics.n), *(_num(metrics.hit_at_k.get(k)) for k in ks), _num(metrics.mrr)]


def _llm_cell(report: EvalReport, value: str | None) -> str:
    llm = report.llm
    if llm.status != "completed" or value is None:
        return STATUS_TEXT.get(llm.status, "n/a")
    return value


def _fuzzy_sentence(reports: Sequence[EvalReport]) -> str:
    parts = []
    for r in reports:
        fuzzy = r.retrieval.by_type.get("fuzzy_plot")
        if fuzzy is not None and 1 in fuzzy.hit_at_k:
            parts.append(f"{r.mode} {_num(fuzzy.hit_at_k[1], 2)}")
    return "On the fuzzy plot questions Hit@1 is " + ", ".join(parts) + "." if parts else ""


def render_markdown(reports: Sequence[EvalReport]) -> str:
    """``EVAL_RESULTS.md`` for ``reports`` (one per mode)."""
    if not reports:
        raise EvalError("nothing to render: no evaluation reports")
    ks = reports[0].retrieval.k_values
    first = reports[0]
    out: list[str] = [
        "# Evaluation results",
        "",
        "Generated by `make report` from `reports/eval_<mode>.json` (every number below is in those files; "
        "regenerate them with `make eval`).",
        "",
        "## Read this first: what these numbers can and cannot show",
        "",
        f"The {first.question_set.n} questions were written against the **synthetic fixture** "
        f"(`tests/fixtures/movies_sample.csv`, {first.index.points} indexed chunks), not the real Wikipedia plots. "
        "That corpus is small and formulaic: each film's plot is built around a rare invented noun, so a keyword match "
        "(BM25) is almost always enough to find it. **The fixture favours BM25** and cannot show the advantage that "
        "hybrid retrieval is expected to have on real data (large vocabulary overlap between films, paraphrased "
        "questions, thousands of competing candidates). Do not read the mode ranking below as a ranking on the real "
        "dataset. " + _fuzzy_sentence(reports),
        "",
        "With 30 answerable questions one question is 3.3 percentage points: differences of one or two questions are "
        "noise. The gold film ids belong to the fixture, so the same questions cannot be reused on the full dataset "
        "(generate a new set with `python -m movie_rag.eval.generate`, which needs `NEBIUS_API_KEY`).",
        "",
        "## Run metadata",
        "",
        _table(
            ["Mode", "Generated (UTC)", "Git sha", "Config hash", "ragas", "Judge model", "Generator model"],
            [
                [
                    r.mode,
                    r.generated_at,
                    r.git_sha,
                    r.config_hash,
                    r.llm.ragas_version,
                    r.llm.judge_model,
                    r.llm.chat_model,
                ]
                for r in reports
            ],
        ),
        "",
        f"Embeddings: dense `{first.embedding_model}`, sparse `{first.sparse_model}`. Collection "
        f"`{first.index.collection}`. Retrieval depth {first.retrieval.depth}. The judge model is always different "
        "from the generator model (checked at run time).",
        "",
        "## Retrieval (deterministic, no credentials)",
        "",
        "Hit@k: the gold film is among the top k films. MRR: mean of 1/rank of the gold film (0 if not retrieved). "
        "Both are averaged over the questions that have a gold film (fuzzy plot, exact entity, filtered); the "
        "unanswerable ones are measured by abstention. The ranking is `search_movies` called through the project's "
        "MCP server. Films with equal scores (common in hybrid mode: reciprocal rank fusion ties a film that is first "
        "in one list and second in the other) are in no defined order, so Hit@k and MRR are the expectation over that "
        "order, which keeps the numbers reproducible (`docs/EVALUATION.md`).",
        "",
        _table(
            ["Mode", "n", *(f"Hit@{k}" for k in ks), "MRR", "latency p50 (ms)", "latency p95 (ms)"],
            [
                [
                    *_retrieval_row(r.mode, r.retrieval.overall, ks),
                    _ms(r.retrieval.latency_ms.p50),
                    _ms(r.retrieval.latency_ms.p95),
                ]
                for r in reports
            ],
        ),
        "",
        "### By question type",
        "",
        _table(
            ["Type", "Mode", "n", *(f"Hit@{k}" for k in ks), "MRR"],
            [
                [type_, *_retrieval_row(r.mode, metrics, ks)]
                for type_ in sorted({t for r in reports for t in r.retrieval.by_type})
                for r in reports
                if (metrics := r.retrieval.by_type.get(type_)) is not None
            ],
        ),
        "",
        "## Agent and RAGAS (need `NEBIUS_API_KEY`)",
        "",
        f"RAGAS {first.llm.ragas_version}, judge `{first.llm.judge_model}`, generator `{first.llm.chat_model}`. "
        "Rows marked *pending credentials* were not computed because no key was available when the reports were "
        "written; they are not zero.",
        "",
        _table(["Metric", *(r.mode for r in reports)], _llm_rows(reports)),
        "",
        _llm_notes(reports),
        "## LangSmith",
        "",
        _table(["Mode", "Experiment"], [[r.mode, r.langsmith.detail] for r in reports]),
        "",
    ]
    return "\n".join(out).rstrip("\n") + "\n"


def _agent_value(report: EvalReport, getter: str) -> str | None:
    agent = report.llm.agent
    if agent is None:
        return None
    values = {
        "correct_abstention": agent.abstention.correct_abstention_rate,
        "false_abstention": agent.abstention.false_abstention_rate,
        "citation_hit": agent.citation_hit_rate,
    }
    if getter in values:
        return _num(values[getter])
    summary = agent.tokens_per_question if getter.startswith("tokens") else agent.latency_ms
    value = summary.p50 if getter.endswith("p50") else summary.p95
    return _ms(value)


def _llm_rows(reports: Sequence[EvalReport]) -> list[list[str]]:
    rows: list[list[str]] = []
    for name in METRIC_NAMES:
        cells = []
        for r in reports:
            summary = r.llm.ragas.get(name) if r.llm.ragas else None
            text = (
                None
                if summary is None
                else f"{_num(summary.mean)} ({summary.n_scored}/{summary.n_scored + summary.n_failed})"
            )
            cells.append(_llm_cell(r, text))
        rows.append([RAGAS_LABELS[name], *cells])
    agent_rows = [
        ("Correct abstention rate (unanswerable questions, higher is better)", "correct_abstention"),
        ("False abstention rate (answerable questions, lower is better)", "false_abstention"),
        ("Citation hit rate (gold film cited)", "citation_hit"),
        ("Tokens per question p50", "tokens_p50"),
        ("Tokens per question p95", "tokens_p95"),
        ("Agent latency p50 (ms)", "latency_p50"),
        ("Agent latency p95 (ms)", "latency_p95"),
    ]
    for label, getter in agent_rows:
        rows.append([label, *(_llm_cell(r, _agent_value(r, getter)) for r in reports)])
    return rows


def _llm_notes(reports: Sequence[EvalReport]) -> str:
    notes = [
        f"* {r.mode}: {r.llm.status.replace('_', ' ')}" + (f" - {r.llm.reason}" if r.llm.reason else "")
        for r in reports
        if r.llm.status != "completed" or r.llm.reason
    ]
    explanation = (
        "Abstention is the agent's behaviour (an answer that cites no retrieved film), so it needs the LLM. There is "
        "no retrieval-only proxy: a score threshold would be a new tunable that is not comparable across dense, "
        "BM25 and RRF scores. Scores in parentheses are `scored/total` samples; ratios use the answerable questions "
        "for RAGAS (it needs a reference).\n"
    )
    return (("\n".join(notes) + "\n\n") if notes else "") + explanation
