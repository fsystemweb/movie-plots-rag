"""The retrieval half of the evaluation on the in-memory engine with the fake embedder (plumbing, not accuracy).

qdrant-client's in-memory engine cannot do hybrid search (see tests/unit/test_retrieval.py), which is exactly why
``connect`` refuses it by default; these tests opt in with ``allow_local`` and use dense and sparse modes only.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException

from eval_support import real_looking_csv
from fakes import FakeEmbedder
from movie_rag.config import RetrievalMode, Settings, load_settings
from movie_rag.errors import EvalError, RetrievalError
from movie_rag.eval.questions import EvalQuestion, fixture_records, load_eval_set, matches_filters
from movie_rag.eval.retrieval_eval import (
    build_eval_server,
    connect,
    ensure_fixture_indexed,
    for_eval,
    run_retrieval,
    search_arguments,
)
from movie_rag.ingest.clean import MovieRecord
from movie_rag.ingest.pipeline import ingest_csv
from movie_rag.mcp_server import build_server

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings()


@pytest.fixture(scope="module")
def questions() -> list[EvalQuestion]:
    return load_eval_set(load_settings(env_file=None))


@pytest.fixture(scope="module")
def indexed(questions: list[EvalQuestion]) -> QdrantClient:
    settings = load_settings(env_file=None)
    client = QdrantClient(":memory:")
    ensure_fixture_indexed(settings, client, questions, embedder=FakeEmbedder())
    return client


# --- connect -------------------------------------------------------------------------------------------------------


def test_the_in_process_engine_is_refused_because_its_hybrid_search_is_wrong(settings: Settings) -> None:
    with pytest.raises(EvalError, match="needs a Qdrant service") as error:
        connect(settings, QdrantClient(":memory:"))
    assert "make up" in str(error.value)


def test_a_memory_url_is_refused_without_creating_a_client(make_settings: Callable[..., Settings]) -> None:
    with pytest.raises(EvalError, match="needs a Qdrant service"):
        connect(make_settings(QDRANT_URL=":memory:"))


def test_local_engine_can_be_allowed_for_plumbing_tests(settings: Settings) -> None:
    client = QdrantClient(":memory:")
    assert connect(settings, client, allow_local=True) is client


def test_an_unreachable_service_gives_an_actionable_message(settings: Settings) -> None:
    client = MagicMock()
    client.get_collections.side_effect = ResponseHandlingException(ConnectionError("refused"))
    with pytest.raises(EvalError, match=r"cannot reach Qdrant at http://localhost:6333.*make up"):
        connect(settings, client)


def test_a_reachable_service_client_is_returned_as_is(settings: Settings) -> None:
    client = MagicMock()
    assert connect(settings, client) is client
    client.get_collections.assert_called_once()


# --- the fixture is in the index -----------------------------------------------------------------------------------


def test_the_fixture_is_ingested_when_gold_films_are_missing_and_left_alone_otherwise(
    settings: Settings, questions: list[EvalQuestion]
) -> None:
    client = QdrantClient(":memory:")
    first = ensure_fixture_indexed(settings, client, questions, embedder=FakeEmbedder())
    assert first.fixture_ingested_now is True
    assert first.collection == settings.eval.collection != settings.qdrant.collection
    assert first.points == client.count(settings.eval.collection, exact=True).count > 0
    assert first.other_points == 0

    embedder = FakeEmbedder()
    second = ensure_fixture_indexed(settings, client, questions, embedder=embedder)
    assert second.fixture_ingested_now is False
    assert embedder.embedded_texts == []  # nothing was re-embedded
    assert second.points == first.points


def test_gold_films_that_the_fixture_cannot_provide_are_an_error(
    settings: Settings, questions: list[EvalQuestion]
) -> None:
    foreign = questions[0].model_copy(update={"gold_movie_ids": ["not-in-the-fixture-1999-0"]})
    with pytest.raises(EvalError, match="still missing after ingesting the fixture"):
        ensure_fixture_indexed(settings, QdrantClient(":memory:"), [foreign], embedder=FakeEmbedder())


# --- the evaluation never writes into the main collection ------------------------------------------------------------


def test_for_eval_points_a_copy_at_the_evaluation_collection(settings: Settings) -> None:
    copy = for_eval(settings)
    assert copy.qdrant.collection == settings.eval.collection != settings.qdrant.collection
    assert settings.qdrant.collection == "movie_plots"  # the original is untouched


def test_the_fixture_goes_to_the_eval_collection_and_the_main_collection_is_left_alone(
    settings: Settings, questions: list[EvalQuestion], tmp_path: Path
) -> None:
    client = QdrantClient(":memory:")
    main = settings.qdrant.collection
    ingest_csv(settings, real_looking_csv(tmp_path), client=client, embedder=FakeEmbedder())
    before = client.count(main, exact=True).count
    before_ids = {str(p.id) for p in client.scroll(main, limit=100, with_payload=False)[0]}

    info = ensure_fixture_indexed(settings, client, questions, embedder=FakeEmbedder())

    assert info.fixture_ingested_now is True and info.collection == settings.eval.collection
    assert client.count(main, exact=True).count == before > 0
    assert {str(p.id) for p in client.scroll(main, limit=100, with_payload=False)[0]} == before_ids
    assert client.count(settings.eval.collection, exact=True).count == info.points > before


def test_the_eval_server_searches_the_eval_collection_not_the_main_one(
    settings: Settings, questions: list[EvalQuestion], tmp_path: Path
) -> None:
    client = QdrantClient(":memory:")
    ingest_csv(settings, real_looking_csv(tmp_path), client=client, embedder=FakeEmbedder())
    ensure_fixture_indexed(settings, client, questions, embedder=FakeEmbedder())
    server = build_eval_server(settings, client, FakeEmbedder())
    found = asyncio.run(run_retrieval(server, questions[:3], "dense", 8))
    assert all(f.ranked_movie_ids and not any(m.startswith("real-film") for m in f.ranked_movie_ids) for f in found)


def test_a_non_empty_collection_with_other_films_is_never_filled_with_the_fixture(
    make_settings: Callable[..., Settings], questions: list[EvalQuestion], tmp_path: Path
) -> None:
    settings = make_settings(EVAL__COLLECTION="movie_plots")  # pointed at the real index by mistake
    client = QdrantClient(":memory:")
    ingest_csv(settings, real_looking_csv(tmp_path), client=client, embedder=FakeEmbedder())
    with pytest.raises(EvalError, match=r"holds 2 points that are not fixture films.*EVAL__COLLECTION"):
        ensure_fixture_indexed(settings, client, questions, embedder=FakeEmbedder())
    assert client.count("movie_plots", exact=True).count == 2  # nothing was written


def test_other_films_next_to_a_complete_fixture_are_reported_not_refused(
    make_settings: Callable[..., Settings], questions: list[EvalQuestion], tmp_path: Path
) -> None:
    settings = make_settings(EVAL__COLLECTION="movie_plots")
    client = QdrantClient(":memory:")
    ingest_csv(settings, settings.data.resolve(settings.data.fixture_path), client=client, embedder=FakeEmbedder())
    ingest_csv(settings, real_looking_csv(tmp_path), client=client, embedder=FakeEmbedder())
    info = ensure_fixture_indexed(settings, client, questions, embedder=FakeEmbedder())
    assert info.fixture_ingested_now is False and info.other_points == 2


# --- running the questions -----------------------------------------------------------------------------------------


def test_search_arguments_carry_the_question_the_mode_the_depth_and_the_filters(questions: list[EvalQuestion]) -> None:
    filtered = next(q for q in questions if q.type == "filtered")
    arguments = search_arguments(filtered, "sparse", 8)
    assert arguments["query"] == filtered.question
    assert (arguments["mode"], arguments["top_k"]) == ("sparse", 8)
    assert {k: v for k, v in arguments.items() if k in filtered.filters.model_dump(exclude_none=True)} == (
        filtered.filters.model_dump(exclude_none=True)
    )
    plain = search_arguments(next(q for q in questions if q.type == "fuzzy_plot"), "dense", 3)
    assert set(plain) == {"query", "mode", "top_k"}


@pytest.mark.parametrize("mode", ["dense", "sparse"])
async def test_every_question_is_ranked_through_the_mcp_search_tool(
    settings: Settings, questions: list[EvalQuestion], indexed: QdrantClient, mode: RetrievalMode
) -> None:
    server = build_eval_server(settings, indexed, FakeEmbedder())
    results = await run_retrieval(server, questions, mode, 8)
    assert [r.question_id for r in results] == [q.id for q in questions]
    for question, result in zip(questions, results, strict=True):
        assert result.gold_movie_ids == question.gold_movie_ids
        assert len(result.ranked_movie_ids) == len(result.scores) <= 8
        assert result.latency_ms >= 0
        if question.gold_movie_ids and result.rank is not None:
            assert result.ranked_movie_ids[result.rank - 1] in question.gold_movie_ids
        else:
            assert result.rank is None or question.gold_movie_ids
        assert result.as_ranked().scores == result.scores


async def test_filtered_questions_only_return_films_that_satisfy_their_filters(
    settings: Settings, questions: list[EvalQuestion], indexed: QdrantClient
) -> None:
    records: dict[str, MovieRecord] = {r.movie_id: r for r in fixture_records(settings)}
    filtered = [q for q in questions if q.type == "filtered"]
    server = build_eval_server(settings, indexed, FakeEmbedder())
    results = await run_retrieval(server, filtered, "dense", 8)
    for question, result in zip(filtered, results, strict=True):
        assert result.ranked_movie_ids
        assert all(matches_filters(question.filters, records[m]) for m in result.ranked_movie_ids)


async def test_no_questions_means_no_calls(settings: Settings, indexed: QdrantClient) -> None:
    server = build_eval_server(settings, indexed, FakeEmbedder())
    assert await run_retrieval(server, [], "dense", 8) == []


async def test_a_failing_search_aborts_the_run_instead_of_producing_partial_numbers(
    settings: Settings, questions: list[EvalQuestion]
) -> None:
    def broken() -> MagicMock:
        retriever = MagicMock()
        retriever.search.side_effect = RetrievalError("the query is empty")
        return retriever

    server = build_server(settings, broken)
    with pytest.raises(EvalError, match="search_movies failed"):
        await run_retrieval(server, questions[:1], "dense", 8)
