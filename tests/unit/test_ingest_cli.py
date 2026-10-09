from __future__ import annotations

from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from fakes import FakeEmbedder
from movie_rag.ingest.__main__ import EXIT_FAILURE, EXIT_OK, build_parser, format_report, main
from movie_rag.ingest.pipeline import IngestReport

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")

REPORT = IngestReport(
    source="movies_sample.csv",
    collection="movie_plots",
    rows_read=300,
    rows_dropped=9,
    films=291,
    chunks_total=304,
    chunks_written=300,
    chunks_skipped=4,
    points_in_collection=304,
    elapsed_s=1.5,
)


@pytest.fixture(autouse=True)
def _isolated_dataset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No downloaded dataset, whatever the developer has in data/."""
    monkeypatch.setenv("DATA__RAW_DIR", str(tmp_path / "no-data"))


def test_format_report_lists_every_count() -> None:
    text = format_report(REPORT)
    for expected in ("movies_sample.csv", "300", "291", "304", "1.50s"):
        assert expected in text
    assert "rows read" in text and "rows dropped" in text and "chunks written" in text and "points in Qdrant" in text


def test_parser_flags_and_mutual_exclusion() -> None:
    parser = build_parser()
    args = parser.parse_args(["--fixture", "--recreate"])
    assert args.fixture and args.recreate and args.csv is None
    assert parser.parse_args(["--csv", "a.csv"]).csv == Path("a.csv")
    with pytest.raises(SystemExit):
        parser.parse_args(["--fixture", "--csv", "a.csv"])


def test_main_without_a_dataset_ingests_the_fixture_and_prints_the_summary(
    capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    client = QdrantClient(":memory:")
    assert main([], client=client, embedder=FakeEmbedder()) == EXIT_OK
    err = capsys.readouterr().err
    assert "ingested movies_sample.csv" in err and "rows read:            300" in err
    assert "make download" in caplog.text  # the fallback is announced
    assert client.count("movie_plots", exact=True).count > 291
    assert f"points in Qdrant:     {client.count('movie_plots', exact=True).count}" in err


def test_main_twice_is_idempotent(capsys: pytest.CaptureFixture[str]) -> None:
    client = QdrantClient(":memory:")
    assert main(["--fixture"], client=client, embedder=FakeEmbedder()) == EXIT_OK
    before = client.count("movie_plots", exact=True).count
    assert main(["--fixture"], client=client, embedder=FakeEmbedder()) == EXIT_OK
    assert client.count("movie_plots", exact=True).count == before
    assert "chunks written:       0" in capsys.readouterr().err


def test_main_recreate_flag_rebuilds(capsys: pytest.CaptureFixture[str]) -> None:
    client = QdrantClient(":memory:")
    main(["--fixture"], client=client, embedder=FakeEmbedder())
    capsys.readouterr()
    assert main(["--fixture", "--recreate"], client=client, embedder=FakeEmbedder()) == EXIT_OK
    assert "chunks skipped:       0" in capsys.readouterr().err


def test_main_missing_csv_is_a_friendly_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--csv", str(tmp_path / "nope.csv")], client=QdrantClient(":memory:"), embedder=FakeEmbedder())
    err = capsys.readouterr().err
    assert code == EXIT_FAILURE
    assert "ingest failed: dataset file not found" in err and "Traceback" not in err


def test_main_unreachable_qdrant_says_to_run_make_up(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("QDRANT_URL", "http://127.0.0.1:1")  # nothing listens there: connection refused
    code = main(["--fixture"], embedder=FakeEmbedder())
    err = capsys.readouterr().err
    assert code == EXIT_FAILURE
    assert "cannot talk to Qdrant at http://127.0.0.1:1" in err and "make up" in err and "Traceback" not in err


def test_main_unreadable_configuration_is_a_friendly_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "config.yaml"
    bad.write_text("qdrant: [not, a, mapping]\n", encoding="utf-8")
    monkeypatch.setenv("MOVIE_RAG_CONFIG", str(bad))
    assert main(["--fixture"], client=QdrantClient(":memory:"), embedder=FakeEmbedder()) == EXIT_FAILURE
    assert "cannot load the configuration" in capsys.readouterr().err


def test_main_builds_the_real_embedder_lazily(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without an injected embedder ``main`` builds a FastEmbedder; constructing it must not load any model."""
    from movie_rag.ingest import __main__ as cli

    created: list[object] = []
    real = cli.FastEmbedder

    class Spy(real):  # type: ignore[valid-type, misc]
        def __init__(self, config: object) -> None:
            super().__init__(config)
            created.append(self)

    monkeypatch.setattr(cli, "FastEmbedder", Spy)
    monkeypatch.setenv("QDRANT_URL", "http://127.0.0.1:1")
    assert main(["--fixture"]) == EXIT_FAILURE  # fails at the Qdrant call, before any model is needed
    assert len(created) == 1
    assert "cannot talk to Qdrant" in capsys.readouterr().err


@pytest.mark.filterwarnings("ignore:'movie_rag.ingest.__main__' found in sys.modules")
def test_module_entry_point_exits_through_main(monkeypatch: pytest.MonkeyPatch) -> None:
    import runpy
    import sys

    monkeypatch.setattr(sys, "argv", ["movie_rag.ingest", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("movie_rag.ingest", run_name="__main__", alter_sys=True)
    assert exit_info.value.code == 0
