"""``python -m movie_rag.eval``: argument handling, output, files and exit codes.

Most tests replace ``runner.run`` with :class:`eval_support.CannedRun` (an instant result built from a cached
evaluation): they are about the command line, not about retrieval, which ``test_eval_runner.py`` covers. Two tests
(``..._for_real``) run the whole path on the in-memory engine.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from qdrant_client.http.exceptions import ResponseHandlingException

from eval_support import CannedRun, cached_reports, fake_deps
from fakes import ScriptedChatModel
from movie_rag.config import Settings
from movie_rag.eval import __main__ as cli
from movie_rag.eval.__main__ import RESULTS_NAME, build_parser, format_summary, main, smoke_failures
from movie_rag.eval.report import (
    OVERVIEW_QUALITY_BEGIN,
    OVERVIEW_QUALITY_END,
    OVERVIEW_SPEED_BEGIN,
    OVERVIEW_SPEED_END,
    README_BEGIN,
    README_END,
    render_overview_quality,
    render_overview_speed,
    render_readme_block,
    write_report,
)

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")


@pytest.fixture
def reports_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("EVAL__REPORTS_DIR", str(tmp_path / "reports"))
    readme = tmp_path / "README.md"  # never the real README: `report` rewrites its results block
    readme.write_text(f"# Title\n\n{README_BEGIN}\nold\n{README_END}\n\nafter\n")
    monkeypatch.setenv("EVAL__README_PATH", str(readme))
    overview = tmp_path / "SOLUTION_OVERVIEW.md"  # likewise: `report` rewrites its two generated blocks
    overview.write_text(
        f"# Overview\n\n{OVERVIEW_QUALITY_BEGIN}\nold\n{OVERVIEW_QUALITY_END}\n\ntext\n\n"
        f"{OVERVIEW_SPEED_BEGIN}\nold\n{OVERVIEW_SPEED_END}\n"
    )
    monkeypatch.setenv("EVAL__OVERVIEW_PATH", str(overview))
    return tmp_path / "reports"


@pytest.fixture
def canned(monkeypatch: pytest.MonkeyPatch) -> CannedRun:
    fake = CannedRun()
    monkeypatch.setattr(cli, "run", fake)
    return fake


# --- run -----------------------------------------------------------------------------------------------------------


def test_run_writes_one_json_per_mode_and_prints_the_summary(
    reports_dir: Path, canned: CannedRun, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["run", "--mode", "dense"]) == 0
    out, err = capsys.readouterr()
    assert [p.name for p in reports_dir.iterdir()] == ["eval_dense.json"]
    header = out.splitlines()[0].split()
    assert header == ["mode", "n", "Hit@1", "Hit@3", "Hit@5", "Hit@8", "MRR", "p50", "ms", "p95", "ms", "LLM", "half"]
    assert out.splitlines()[1].startswith("dense") and "pending credentials" in out
    assert "LLM half skipped: pending credentials" in err and "NEBIUS_API_KEY" in err
    assert "eval_dense.json" in err
    assert canned.calls == [{"modes": ("dense",), "attempt_llm": True, "deps": None}]


def test_run_without_the_llm_half_does_not_mention_credentials(
    reports_dir: Path, canned: CannedRun, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["run", "--mode", "sparse", "--no-llm"]) == 0
    out, err = capsys.readouterr()
    assert "not requested" in out and "pending" not in err and "pending" not in out
    assert canned.calls[0]["attempt_llm"] is False


def test_run_defaults_to_all_three_modes(reports_dir: Path, canned: CannedRun) -> None:
    assert main(["run", "--no-llm"]) == 0
    assert sorted(p.name for p in reports_dir.iterdir()) == ["eval_dense.json", "eval_hybrid.json", "eval_sparse.json"]
    assert canned.calls[0]["modes"] == ("dense", "sparse", "hybrid")


def test_a_failed_llm_half_makes_the_run_fail_after_the_files_are_written(
    reports_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key-not-real")

    def broken(*_: Any) -> dict[str, Any]:
        raise RuntimeError("judge down")

    deps = fake_deps(chat_model=ScriptedChatModel(replies=[AIMessage(content="none")]), scorer_factory=broken)
    assert main(["run", "--mode", "dense"], deps=deps) == 1  # for real: the failure comes from the runner
    assert (reports_dir / "eval_dense.json").is_file()


def test_run_for_real_on_the_in_memory_engine(reports_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "--mode", "dense", "--no-llm"], deps=fake_deps()) == 0
    assert "dense" in capsys.readouterr().out
    assert (reports_dir / "eval_dense.json").is_file()


# --- report --------------------------------------------------------------------------------------------------------


def test_report_renders_the_page_from_the_json_files(reports_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    for report in cached_reports():
        write_report(report, reports_dir)
    assert main(["report"]) == 0
    page = (reports_dir / RESULTS_NAME).read_text()
    assert page.startswith("# Evaluation results") and "| dense |" in page and "| hybrid |" in page
    assert RESULTS_NAME in capsys.readouterr().err


def test_report_rewrites_the_results_block_of_the_readme_and_nothing_else(
    reports_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reports = cached_reports()
    for report in reports:
        write_report(report, reports_dir)
    assert main(["report"]) == 0
    readme = (reports_dir.parent / "README.md").read_text()
    assert readme == f"# Title\n\n{render_readme_block(reports)}\n\nafter\n"
    assert "results block of" in capsys.readouterr().err


def test_report_rewrites_the_two_overview_blocks_and_nothing_else(
    reports_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reports = cached_reports()
    for report in reports:
        write_report(report, reports_dir)
    assert main(["report"]) == 0
    overview = (reports_dir.parent / "SOLUTION_OVERVIEW.md").read_text()
    assert overview == (
        f"# Overview\n\n{render_overview_quality(reports)}\n\ntext\n\n{render_overview_speed(reports)}\n"
    )
    assert "result blocks of" in capsys.readouterr().err


def test_report_fails_clearly_when_the_overview_has_no_blocks(
    reports_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for report in cached_reports():
        write_report(report, reports_dir)
    (reports_dir.parent / "SOLUTION_OVERVIEW.md").write_text("# no markers here\n")
    assert main(["report"]) == 1
    assert "exactly one results block" in capsys.readouterr().err


def test_report_fails_clearly_when_the_readme_has_no_results_block(
    reports_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for report in cached_reports():
        write_report(report, reports_dir)
    (reports_dir.parent / "README.md").write_text("# no markers here\n")
    assert main(["report"]) == 1
    assert "exactly one results block" in capsys.readouterr().err


def test_report_without_json_files_says_to_run_the_evaluation_first(
    reports_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["report"]) == 1
    assert "run `make eval` first" in capsys.readouterr().err


# --- smoke ---------------------------------------------------------------------------------------------------------


def test_smoke_prints_all_modes_writes_nothing_and_asks_for_no_experiments(
    reports_dir: Path, canned: CannedRun, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("EVAL__SMOKE_MIN_MRR", "0")
    assert main(["smoke"]) == 0
    out = capsys.readouterr().out
    assert [line.split()[0] for line in out.splitlines()[1:]] == ["dense", "sparse", "hybrid"]
    assert not reports_dir.exists()
    [call] = canned.calls
    assert (call["attempt_llm"], call["experiments"]) == (False, False)
    assert call["llm_questions_per_type"] > 0


def test_smoke_fails_when_a_mode_is_below_the_mrr_floor(
    reports_dir: Path, canned: CannedRun, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("EVAL__SMOKE_MIN_MRR", "1.0")
    reports = cached_reports()
    assert any(r.retrieval.overall.mrr is not None and r.retrieval.overall.mrr < 1.0 for r in reports)
    assert main(["smoke"]) == 1
    assert "smoke failed" in capsys.readouterr().err


def test_smoke_with_llm_but_no_key_skips_cleanly(
    reports_dir: Path, canned: CannedRun, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("EVAL__SMOKE_MIN_MRR", "0")
    assert main(["smoke", "--llm"]) == 0
    captured = capsys.readouterr()
    assert "LLM half skipped: pending credentials" in captured.err and "pending credentials" in captured.out
    assert canned.calls[0]["attempt_llm"] is True


def test_smoke_for_real_on_the_in_memory_engine(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("EVAL__SMOKE_MIN_MRR", "0")
    assert main(["smoke"], deps=fake_deps()) == 0
    assert [line.split()[0] for line in capsys.readouterr().out.splitlines()[1:]] == ["dense", "sparse", "hybrid"]


# --- failures ------------------------------------------------------------------------------------------------------


def test_an_unreachable_qdrant_is_a_clear_failure_not_a_traceback(
    reports_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    client = MagicMock()
    client.get_collections.side_effect = ResponseHandlingException(ConnectionError("refused"))
    assert main(["smoke"], deps=fake_deps(client=client, allow_local_qdrant=False)) == 1
    err = capsys.readouterr().err
    assert "evaluation failed: cannot reach Qdrant" in err and "make up" in err and "Traceback" not in err


def test_an_unreadable_configuration_is_reported_without_a_stack_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "config.yaml"
    bad.write_text("qdrant: {}\n")
    monkeypatch.setenv("MOVIE_RAG_CONFIG", str(bad))
    assert main(["report"]) == 1
    assert "cannot load the configuration" in capsys.readouterr().err


def test_an_unknown_mode_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(["run", "--mode", "fuzzy"])
    assert exit_info.value.code == 2


# --- helpers -------------------------------------------------------------------------------------------------------


def test_summary_table_and_smoke_floor_on_known_reports(make_settings: Callable[..., Settings]) -> None:
    settings = make_settings()
    reports = cached_reports()
    table = format_summary(reports).splitlines()
    assert len(table) == 4 and table[1].split()[0] == "dense"
    assert format_summary([]) == "(no reports)"
    mrr = reports[0].retrieval.overall.mrr
    assert mrr is not None
    low = make_settings(EVAL__SMOKE_MIN_MRR=str(min(1.0, mrr + 0.01)))
    assert smoke_failures(low, reports[:1])[0].startswith("dense: MRR")
    assert smoke_failures(make_settings(EVAL__SMOKE_MIN_MRR=str(max(0.0, mrr - 0.01))), reports[:1]) == []
    overall = reports[0].retrieval.overall.model_copy(update={"mrr": None})
    undefined = reports[0].model_copy(
        update={"retrieval": reports[0].retrieval.model_copy(update={"overall": overall})}
    )
    assert "MRR n/a is below" in smoke_failures(settings, [undefined])[0]
