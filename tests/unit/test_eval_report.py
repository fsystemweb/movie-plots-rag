"""The JSON report and the markdown rendered from it."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from eval_support import cached_reports
from movie_rag.config import Settings
from movie_rag.errors import EvalError
from movie_rag.eval.agent_eval import AgentMetrics
from movie_rag.eval.metrics import AbstentionMetrics, Summary
from movie_rag.eval.ragas_judge import MetricSummary
from movie_rag.eval.report import (
    EvalReport,
    LlmSection,
    read_reports,
    render_markdown,
    report_path,
    write_report,
)
from movie_rag.eval.runner import skipped_llm

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")


@pytest.fixture(scope="module")
def reports() -> list[EvalReport]:
    return cached_reports()


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings()


def with_llm(report: EvalReport, llm: LlmSection) -> EvalReport:
    return report.model_copy(update={"llm": llm})


def pending(settings: Settings, report: EvalReport) -> EvalReport:
    return with_llm(
        report, skipped_llm(settings, "pending_credentials", "set NEBIUS_API_KEY — see docs/CREDENTIALS.md")
    )


def completed(settings: Settings) -> LlmSection:
    summary = Summary(n=3, p50=900.0, p95=1500.0, mean=1000.0)
    return LlmSection(
        status="completed",
        reason="1 question(s) raised an error and are excluded",
        chat_model=settings.llm.chat_model,
        judge_model=settings.llm.judge_model,
        ragas_version="0.4.3",
        agent=AgentMetrics(
            n_questions=4,
            n_errors=1,
            abstention=AbstentionMetrics(
                n_unanswerable=1, n_answerable=2, correct_abstention_rate=1.0, false_abstention_rate=0.25
            ),
            citation_hit_rate=0.5,
            n_answerable_answered=2,
            latency_ms=summary,
            tokens_per_question=Summary(n=3, p50=1234.0, p95=2345.0, mean=1500.0),
        ),
        ragas={
            "faithfulness": MetricSummary(mean=0.91, n_scored=2, n_failed=1),
            "response_relevancy": MetricSummary(mean=0.82, n_scored=3, n_failed=0),
            "context_precision": MetricSummary(mean=None, n_scored=0, n_failed=3),
            "context_recall": MetricSummary(mean=0.5, n_scored=3, n_failed=0),
        },
    )


# --- JSON ----------------------------------------------------------------------------------------------------------


def test_a_report_round_trips_through_its_json_file(reports: list[EvalReport], tmp_path: Path) -> None:
    for report in reports:
        path = write_report(report, tmp_path / "nested")
        assert path == report_path(tmp_path / "nested", report.mode) and path.name == f"eval_{report.mode}.json"
        assert path.read_text().endswith("}\n")
        assert EvalReport.model_validate_json(path.read_text()) == report
    assert [r.mode for r in read_reports(tmp_path / "nested")] == ["dense", "sparse", "hybrid"]


def test_reading_returns_the_modes_that_exist_in_a_fixed_order(reports: list[EvalReport], tmp_path: Path) -> None:
    write_report(reports[2], tmp_path)
    write_report(reports[0], tmp_path)
    assert [r.mode for r in read_reports(tmp_path)] == ["dense", "hybrid"]


def test_reading_without_reports_says_what_to_run(tmp_path: Path) -> None:
    with pytest.raises(EvalError, match="run `make eval` first"):
        read_reports(tmp_path)


def test_a_malformed_report_names_the_file(tmp_path: Path) -> None:
    (tmp_path / "eval_dense.json").write_text(json.dumps({"mode": "dense"}))
    with pytest.raises(EvalError, match=r"eval_dense\.json is not a valid evaluation report"):
        read_reports(tmp_path)


# --- markdown ------------------------------------------------------------------------------------------------------


def test_pending_credentials_are_labelled_and_never_shown_as_numbers(
    settings: Settings, reports: list[EvalReport]
) -> None:
    text = render_markdown([pending(settings, r) for r in reports])
    llm_part = text.split("## Agent and RAGAS")[1].split("## LangSmith")[0]
    rows = [line for line in llm_part.splitlines() if line.startswith("| ") and "---" not in line][1:]
    assert len(rows) == 11  # four RAGAS metrics, three abstention/citation rows, tokens and latency p50/p95
    assert all(row.count("pending credentials") == 3 for row in rows)
    assert "RAGAS 0.4" in text and settings.llm.judge_model in text and settings.llm.chat_model in text
    assert "NEBIUS_API_KEY" in llm_part


def test_the_page_says_the_fixture_favours_bm25_and_cannot_show_hybrid_gains(
    settings: Settings, reports: list[EvalReport]
) -> None:
    text = render_markdown(reports)
    assert "The fixture favours BM25" in text
    assert "cannot show the advantage" in text and "hybrid" in text
    assert "On the fuzzy plot questions Hit@1 is dense" in text


def test_the_size_of_one_question_is_derived_from_the_data(reports: list[EvalReport]) -> None:
    n = reports[0].retrieval.overall.n
    assert n == 6  # the cached evaluation uses two questions of each type
    assert f"With {n} answerable questions one question is {100 / n:.1f} percentage points" in render_markdown(reports)
    thirty = reports[0].model_copy(
        update={
            "retrieval": reports[0].retrieval.model_copy(
                update={"overall": reports[0].retrieval.overall.model_copy(update={"n": 30})}
            )
        }
    )
    assert "With 30 answerable questions one question is 3.3 percentage points" in render_markdown([thirty])


def test_the_page_explains_hybrid_repeatability_and_how_ties_are_scored(
    reports: list[EvalReport],
) -> None:
    text = render_markdown(reports)
    assert "Equal final scores are listed by `movie_id`" in text
    assert "not exactly repeatable" in text and "expectation over all orders of a tie group" in text


def test_retrieval_numbers_in_the_page_are_the_numbers_of_the_json(reports: list[EvalReport]) -> None:
    text = render_markdown(reports)
    for report in reports:
        overall = report.retrieval.overall
        assert overall.mrr is not None
        row = next(line for line in text.splitlines() if line.startswith(f"| {report.mode} | {overall.n} |"))
        cells = [c.strip() for c in row.strip("|").split("|")]
        assert cells[2 : 2 + len(report.retrieval.k_values)] == [
            f"{overall.hit_at_k[k]:.3f}" for k in report.retrieval.k_values
        ]
        assert f"{overall.mrr:.3f}" in cells
    by_type = next(line for line in text.splitlines() if line.startswith("| fuzzy_plot | sparse |"))
    sparse = reports[1].retrieval.by_type["fuzzy_plot"]
    assert sparse.mrr is not None and f"{sparse.mrr:.3f}" in by_type


def test_completed_numbers_are_shown_with_how_many_samples_were_scored(
    settings: Settings, reports: list[EvalReport]
) -> None:
    text = render_markdown([with_llm(reports[0], completed(settings))])
    assert "| RAGAS faithfulness | 0.910 (2/3) |" in text
    assert "| RAGAS context precision | n/a (0/3) |" in text
    assert "| Correct abstention rate (unanswerable questions, higher is better) | 1.000 |" in text
    assert "| False abstention rate (answerable questions, lower is better) | 0.250 |" in text
    assert "| Citation hit rate (gold film cited) | 0.500 |" in text
    assert "| Tokens per question p50 | 1234 |" in text and "| Agent latency p95 (ms) | 1500 |" in text
    assert "1 question(s) raised an error" in text
    assert "pending credentials |" not in text.split("## Agent and RAGAS")[1].split("## LangSmith")[0].split("* ")[0]


def test_failed_and_not_requested_halves_are_not_zero(settings: Settings, reports: list[EvalReport]) -> None:
    failed = with_llm(reports[0], skipped_llm(settings, "failed", "RuntimeError: boom"))
    not_run = reports[1]
    text = render_markdown([failed, not_run])
    assert "| RAGAS faithfulness | failed | not run |" in text
    assert "* dense: failed - RuntimeError: boom" in text


def test_a_single_mode_page_renders(reports: list[EvalReport]) -> None:
    text = render_markdown(reports[:1])
    assert text.startswith("# Evaluation results") and text.endswith("\n") and not text.endswith("\n\n")
    assert "| sparse |" not in text


def test_nothing_to_render_is_an_error() -> None:
    with pytest.raises(EvalError, match="no evaluation reports"):
        render_markdown([])


def test_the_langsmith_state_is_listed_per_mode(reports: list[EvalReport]) -> None:
    text = render_markdown(reports)
    assert "| dense | not requested |" in text and "| hybrid | not requested |" in text
