from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from fakes import FakeEmbedder
from movie_rag.config import load_settings
from movie_rag.ingest.pipeline import ingest_csv
from movie_rag.retrieval.__main__ import ALL_MODES, EXIT_FAILURE, EXIT_OK, build_parser, emit, format_hits, main
from movie_rag.retrieval.search import MovieHit

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")


@pytest.fixture(scope="module")
def client() -> QdrantClient:
    qdrant = QdrantClient(":memory:")
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "movies_sample.csv"
    ingest_csv(load_settings(env_file=None), fixture, client=qdrant, embedder=FakeEmbedder(), recreate=True)
    return qdrant


def hit(**overrides: object) -> MovieHit:
    fields: dict[str, object] = {
        "movie_id": "the-heist-1999-7",
        "title": "The Heist",
        "release_year": 1999,
        "director": "Ann Lee",
        "genre": "thriller",
        "origin": "american",
        "wiki_url": "https://example.test/wiki/The_Heist",
        "score": 0.5,
        "chunk_idx": 1,
        "snippet": "A crew robs a bank.",
    }
    fields.update(overrides)
    return MovieHit.model_validate(fields)


def test_format_hits_lists_rank_title_year_score_metadata_citation_and_snippet() -> None:
    text = format_hits("hybrid", [hit(), hit(title="Second", release_year=None, director=None, wiki_url=None)])
    lines = text.splitlines()
    assert lines[0] == "== hybrid: 2 film(s) =="
    assert " 1. The Heist (1999)  score=0.5000  [thriller, american, Ann Lee]" in lines
    assert "    the-heist-1999-7  chunk 1  https://example.test/wiki/The_Heist" in lines
    assert "    A crew robs a bank." in lines
    assert " 2. Second (?)  score=0.5000  [thriller, american]" in lines


def test_format_hits_says_so_when_nothing_matches() -> None:
    assert format_hits("dense", []) == "== dense: 0 film(s) ==\n  (no match)"


def test_emit_writes_one_line_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    emit("hello\n\n")
    assert capsys.readouterr().out == "hello\n"


def test_parser_defaults_to_all_modes_and_the_configured_query() -> None:
    args = build_parser().parse_args([])
    assert (args.query, args.mode, args.top_k) == (None, ALL_MODES, None)
    args = build_parser().parse_args(["q", "--mode", "sparse", "--top-k", "3", "--year-from", "1950", "--genre", "x"])
    assert (args.query, args.mode, args.top_k, args.year_from, args.genre) == ("q", "sparse", 3, 1950, "x")
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--mode", "fuzzy"])


def test_without_arguments_the_demo_query_is_run_in_all_three_modes(
    client: QdrantClient, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = load_settings(env_file=None)
    assert main([], client=client, embedder=FakeEmbedder()) == EXIT_OK
    out = capsys.readouterr().out
    assert out.startswith(f"query: {settings.retrieval.demo_query}")
    for mode in ("dense", "sparse", "hybrid"):
        assert f"== {mode}: " in out


def test_a_single_mode_with_filters_and_top_k(client: QdrantClient, capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["a detective", "--mode", "dense", "--top-k", "2", "--year-from", "1980", "--year-to", "1990"]
    assert main(argv, client=client, embedder=FakeEmbedder()) == EXIT_OK
    out = capsys.readouterr().out
    assert out.splitlines()[0] == "query: a detective" and "== dense: 2 film(s) ==" in out
    assert "== sparse" not in out and "== hybrid" not in out
    years = [
        int(line.split("(")[-1].split(")")[0]) for line in out.splitlines() if line[:3].strip().rstrip(".").isdigit()
    ]
    assert len(years) == 2 and all(1980 <= y <= 1990 for y in years)


def test_invalid_filters_are_reported_without_a_traceback(
    client: QdrantClient, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["q", "--year-from", "2000", "--year-to", "1990"], client=client, embedder=FakeEmbedder())
    assert code == EXIT_FAILURE
    assert "invalid filters" in capsys.readouterr().err


def test_search_errors_are_reported_without_a_traceback(
    client: QdrantClient, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["q", "--top-k", "0"], client=client, embedder=FakeEmbedder()) == EXIT_FAILURE
    assert "search failed" in capsys.readouterr().err


@pytest.mark.parametrize(
    "error", [ResponseHandlingException(ConnectionError("down")), UnexpectedResponse(404, "Not Found", b"", None)]
)
def test_an_unreachable_or_missing_collection_points_at_make_up_and_ingest(
    error: Exception, capsys: pytest.CaptureFixture[str]
) -> None:
    broken = MagicMock(spec=QdrantClient)
    broken.query_points_groups.side_effect = error
    assert main(["q"], client=broken, embedder=FakeEmbedder()) == EXIT_FAILURE
    err = capsys.readouterr().err
    assert "make up" in err and "make ingest" in err


def test_an_unreadable_configuration_is_reported_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "config.yaml"
    bad.write_text("qdrant: [not, a, mapping]\n")
    monkeypatch.setenv("MOVIE_RAG_CONFIG", str(bad))
    assert main(["q"]) == EXIT_FAILURE
    assert "cannot load the configuration" in capsys.readouterr().err


def test_without_an_injected_client_main_connects_to_the_configured_url(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.invalid:6333")
    unreachable = MagicMock(spec=QdrantClient)
    unreachable.query_points_groups.side_effect = ResponseHandlingException(ConnectionError("refused"))
    factory = MagicMock(return_value=unreachable)  # no socket is ever opened
    monkeypatch.setattr("movie_rag.retrieval.search.QdrantClient", factory)
    assert main(["q", "--mode", "dense"], embedder=FakeEmbedder()) == EXIT_FAILURE
    assert factory.call_args.kwargs["url"] == "http://qdrant.invalid:6333"
    assert "cannot search Qdrant at http://qdrant.invalid:6333" in capsys.readouterr().err


@pytest.mark.filterwarnings("ignore:'movie_rag.retrieval.__main__' found in sys.modules")
def test_module_entry_point_exits_through_main(monkeypatch: pytest.MonkeyPatch) -> None:
    import runpy
    import sys

    monkeypatch.setattr(sys, "argv", ["movie_rag.retrieval", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("movie_rag.retrieval", run_name="__main__", alter_sys=True)
    assert exit_info.value.code == 0
