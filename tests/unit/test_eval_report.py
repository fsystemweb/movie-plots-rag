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
    OVERVIEW_QUALITY_BEGIN,
    OVERVIEW_QUALITY_END,
    OVERVIEW_SPEED_BEGIN,
    OVERVIEW_SPEED_END,
    README_BEGIN,
    README_END,
    EvalReport,
    LlmSection,
    index_description,
    read_reports,
    render_markdown,
    render_overview_quality,
    render_overview_speed,
    render_readme_block,
    report_path,
    update_overview,
    update_readme,
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


# --- README block ------------------------------------------------------------------------------------------------


def test_the_readme_block_carries_the_json_numbers_between_the_markers(reports: list[EvalReport]) -> None:
    block = render_readme_block(reports)
    assert block.startswith(README_BEGIN) and block.endswith(README_END)
    for report in reports:
        overall = report.retrieval.overall
        assert overall.mrr is not None
        row = next(line for line in block.splitlines() if line.startswith(f"| {report.mode} | {overall.n} |"))
        cells = [c.strip() for c in row.strip("|").split("|")]
        ks = report.retrieval.k_values
        assert cells[2 : 2 + len(ks)] == [f"{overall.hit_at_k[k]:.3f}" for k in ks]
        assert cells[2 + len(ks)] == f"{overall.mrr:.3f}"
        assert report.git_sha in block and report.generated_at in block
    assert "reports/EVAL_RESULTS.md" in block


def test_the_readme_block_labels_pending_llm_metrics_instead_of_showing_numbers(
    settings: Settings, reports: list[EvalReport]
) -> None:
    block = render_readme_block([pending(settings, r) for r in reports])
    llm_rows = [
        line for line in block.splitlines() if line.startswith(("| RAGAS", "| Correct", "| False", "| Citation"))
    ]
    assert len(llm_rows) == 7 and all(row.count("pending credentials") == 3 for row in llm_rows)


def test_the_readme_block_shows_completed_llm_metrics(settings: Settings, reports: list[EvalReport]) -> None:
    block = render_readme_block([with_llm(r, completed(settings)) for r in reports])
    assert "| RAGAS faithfulness | 0.910 (2/3) |" in block and "pending credentials" not in block


def test_a_readme_block_needs_reports() -> None:
    with pytest.raises(EvalError, match="nothing to render"):
        render_readme_block([])


def test_update_readme_replaces_only_the_marked_region() -> None:
    text = f"before\n{README_BEGIN}\nold $1 \\1 numbers\n{README_END}\nafter\n"
    block = f"{README_BEGIN}\nnew $2 \\2\n{README_END}"
    assert update_readme(text, block) == f"before\n{block}\nafter\n"
    assert update_readme(update_readme(text, block), block) == update_readme(text, block)


@pytest.mark.parametrize(
    "text",
    ["no markers\n", f"{README_BEGIN}\nnever closed\n", f"{README_BEGIN}{README_END}\n{README_BEGIN}{README_END}"],
)
def test_update_readme_refuses_a_missing_or_duplicated_block(text: str) -> None:
    with pytest.raises(EvalError, match="exactly one results block"):
        update_readme(text, "x")


# --- stakeholder overview blocks ---------------------------------------------------------------------------------


def test_the_overview_quality_block_is_one_small_table_with_the_json_numbers(reports: list[EvalReport]) -> None:
    block = render_overview_quality(reports)
    assert block.startswith(OVERVIEW_QUALITY_BEGIN) and block.endswith(OVERVIEW_QUALITY_END)
    assert block.count("\n| Search method |") == 1  # one table
    for report, label in zip(
        reports, ("Meaning (dense)", "Exact words (sparse)", "Both combined (hybrid)"), strict=True
    ):
        overall = report.retrieval.overall
        row = next(line for line in block.splitlines() if line.startswith(f"| {label} |"))
        cells = [c.strip() for c in row.strip("|").split("|")]
        assert len(cells) == 5
        assert cells[1:3] == [f"{overall.hit_at_k[1]:.2f}", f"{overall.hit_at_k[3]:.2f}"]
    assert f"{reports[0].question_set.n} questions" in block and f"{reports[0].index.points} indexed passages" in block
    assert f"collection `{reports[0].index.collection}`" in block and "synthetic sample library" in block


def test_the_overview_marks_llm_measures_pending_then_shows_them(settings: Settings, reports: list[EvalReport]) -> None:
    pending_block = render_overview_quality([pending(settings, r) for r in reports])
    rows = [line for line in pending_block.splitlines() if line.startswith("| Meaning") or "| Both" in line[:8]]
    assert len(rows) == 2 and all(row.count("pending credentials") == 2 for row in rows)
    done = render_overview_quality([with_llm(r, completed(settings)) for r in reports])
    assert "| 0.910 (2/3) | 1.000 |" in done and "pending credentials" not in done


def test_the_index_is_described_from_the_data_not_assumed(reports: list[EvalReport]) -> None:
    fixture_only = reports[0].index.model_copy(update={"other_points": 0})
    mixed = fixture_only.model_copy(update={"other_points": 7, "collection": "movie_plots"})
    assert "synthetic sample library" in index_description(fixture_only)
    text = index_description(mixed)
    assert "7 of them not from the synthetic sample library" in text and "collection `movie_plots`" in text
    assert index_description(fixture_only, unit="chunks").count("chunks") == 1
    assert "not from the synthetic" in render_overview_quality([r.model_copy(update={"index": mixed}) for r in reports])
    assert "7 of them not from the synthetic" in render_readme_block(
        [r.model_copy(update={"index": mixed}) for r in reports]
    )


def test_the_overview_speed_block_converts_agent_milliseconds_to_seconds(
    settings: Settings, reports: list[EvalReport]
) -> None:
    block = render_overview_speed([pending(settings, r) for r in reports])
    assert block.startswith(OVERVIEW_SPEED_BEGIN) and block.endswith(OVERVIEW_SPEED_END)
    for report in reports:
        assert f"{report.retrieval.latency_ms.p50:.0f}" in block
    assert block.count("pending credentials") == 6
    done_reports = [with_llm(r, completed(settings)) for r in reports]
    agent = done_reports[0].llm.agent
    assert agent is not None and agent.latency_ms.p50 is not None
    seconds_row = next(
        line for line in render_overview_speed(done_reports).splitlines() if line.startswith("| Seconds per answer")
    )
    assert f"| {agent.latency_ms.p50 / 1000:.1f} |" in seconds_row


@pytest.mark.parametrize("render", [render_overview_quality, render_overview_speed])
def test_an_overview_block_needs_reports(render: Callable[[list[EvalReport]], str]) -> None:
    with pytest.raises(EvalError, match="nothing to render"):
        render([])


def test_update_overview_replaces_both_blocks_and_is_idempotent(reports: list[EvalReport]) -> None:
    text = (
        f"a\n{OVERVIEW_QUALITY_BEGIN}\nold $1\n{OVERVIEW_QUALITY_END}\nb\n{OVERVIEW_SPEED_BEGIN}\nold\n"
        f"{OVERVIEW_SPEED_END}\nc\n"
    )
    updated = update_overview(text, reports)
    assert updated == f"a\n{render_overview_quality(reports)}\nb\n{render_overview_speed(reports)}\nc\n"
    assert update_overview(updated, reports) == updated


@pytest.mark.parametrize(
    "text",
    [
        "no markers\n",
        f"{OVERVIEW_QUALITY_BEGIN}\n{OVERVIEW_QUALITY_END}\n",
        f"{OVERVIEW_SPEED_BEGIN}{OVERVIEW_SPEED_END}",
    ],
)
def test_update_overview_refuses_a_missing_block(text: str, reports: list[EvalReport]) -> None:
    with pytest.raises(EvalError, match="exactly one results block"):
        update_overview(text, reports)
