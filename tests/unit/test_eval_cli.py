"""``python -m movie_rag.eval``: run, smoke and report, on the in-memory engine with the fake embedder."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from qdrant_client.http.exceptions import ResponseHandlingException

from eval_support import fake_deps, retrieval_only
from fakes import ScriptedChatModel
from movie_rag.config import Settings
from movie_rag.eval.__main__ import (
    RESULTS_NAME,
    build_parser,
    format_summary,
    main,
    smoke_failures,
)
from movie_rag.eval.report import write_report

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")


@pytest.fixture
def reports_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("EVAL__REPORTS_DIR", str(tmp_path / "reports"))
    return tmp_path / "reports"


def test_run_writes_one_json_per_mode_and_prints_the_summary(
    reports_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["run", "--mode", "dense"], deps=fake_deps()) == 0
    out, err = capsys.readouterr()
    assert [p.name for p in reports_dir.iterdir()] == ["eval_dense.json"]
    assert out.splitlines()[0].split() == [
        "mode",
        "n",
        "Hit@1",
        "Hit@3",
        "Hit@5",
        "Hit@8",
        "MRR",
        "p50",
        "ms",
        "p95",
        "ms",
        "LLM",
        "half",
    ]
    assert out.splitlines()[1].startswith("dense") and "pending credentials" in out
    assert "LLM half skipped: pending credentials" in err and "NEBIUS_API_KEY" in err
    assert "eval_dense.json" in err


def test_run_without_the_llm_half_does_not_mention_credentials(
    reports_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["run", "--mode", "sparse", "--no-llm"], deps=fake_deps()) == 0
    out, err = capsys.readouterr()
    assert "not requested" in out and "pending" not in err and "pending" not in out


def test_run_defaults_to_all_three_modes(reports_dir: Path) -> None:
    assert main(["run", "--no-llm"], deps=fake_deps()) == 0
    assert sorted(p.name for p in reports_dir.iterdir()) == ["eval_dense.json", "eval_hybrid.json", "eval_sparse.json"]


def test_a_failed_llm_half_makes_the_run_fail_after_the_files_are_written(
    reports_dir: Path, make_settings: Callable[..., Settings], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key-not-real")

    def broken(*_: Any) -> dict[str, Any]:
        raise RuntimeError("judge down")

    deps = fake_deps(chat_model=ScriptedChatModel(replies=[AIMessage(content="none")]), scorer_factory=broken)
    assert main(["run", "--mode", "dense"], deps=deps) == 1
    assert (reports_dir / "eval_dense.json").is_file()


def test_report_renders_the_page_from_the_json_files(
    reports_dir: Path, capsys: pytest.CaptureFixture[str], make_settings: Callable[..., Settings]
) -> None:
    for report in retrieval_only(make_settings()).reports:
        write_report(report, reports_dir)
    assert main(["report"]) == 0
    page = (reports_dir / RESULTS_NAME).read_text()
    assert page.startswith("# Evaluation results") and "| dense |" in page and "| sparse |" in page
    assert RESULTS_NAME in capsys.readouterr().err


def test_report_without_json_files_says_to_run_the_evaluation_first(
    reports_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["report"]) == 1
    assert "run `make eval` first" in capsys.readouterr().err


def test_smoke_prints_all_modes_writes_nothing_and_passes_above_the_floor(
    reports_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("EVAL__SMOKE_MIN_MRR", "0")
    assert main(["smoke"], deps=fake_deps()) == 0
    out = capsys.readouterr().out
    assert [line.split()[0] for line in out.splitlines()[1:]] == ["dense", "sparse", "hybrid"]
    assert not reports_dir.exists()


def test_smoke_fails_when_a_mode_is_below_the_mrr_floor(
    reports_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("EVAL__SMOKE_MIN_MRR", "1")
    assert main(["smoke"], deps=fake_deps()) == 1
    assert "smoke failed" in capsys.readouterr().err


def test_smoke_with_llm_but_no_key_skips_cleanly(
    reports_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("EVAL__SMOKE_MIN_MRR", "0")
    assert main(["smoke", "--llm"], deps=fake_deps()) == 0
    captured = capsys.readouterr()
    assert "LLM half skipped: pending credentials" in captured.err and "pending credentials" in captured.out


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
    reports = retrieval_only(settings).reports
    table = format_summary(reports).splitlines()
    assert len(table) == 3 and table[1].split()[0] == "dense"
    assert format_summary([]) == "(no reports)"
    mrr = reports[0].retrieval.overall.mrr
    assert mrr is not None
    low = make_settings(EVAL__SMOKE_MIN_MRR=str(min(1.0, mrr + 0.01)))
    assert smoke_failures(low, reports[:1])[0].startswith("dense: MRR")
    assert smoke_failures(make_settings(EVAL__SMOKE_MIN_MRR=str(max(0.0, mrr - 0.01))), reports[:1]) == []
    undefined = reports[0].model_copy(
        update={
            "retrieval": reports[0].retrieval.model_copy(
                update={"overall": reports[0].retrieval.overall.model_copy(update={"mrr": None})}
            )
        }
    )
    assert "MRR n/a is below" in smoke_failures(settings, [undefined])[0]
