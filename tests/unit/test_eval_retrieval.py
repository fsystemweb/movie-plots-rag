"""The retrieval half of the evaluation on the in-memory engine with the fake embedder (plumbing, not accuracy).

qdrant-client's in-memory engine cannot do hybrid search (see tests/unit/test_retrieval.py), which is exactly why
``connect`` refuses it by default; these tests opt in with ``allow_local`` and use dense and sparse modes only.
"""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException

from fakes import FakeEmbedder
from movie_rag.config import RetrievalMode, Settings, load_settings
from movie_rag.errors import EvalError, RetrievalError
from movie_rag.eval.questions import EvalQuestion, fixture_records, load_eval_set, matches_filters
from movie_rag.eval.retrieval_eval import (
    build_eval_server,
    connect,
    ensure_fixture_indexed,
    run_retrieval,
    search_arguments,
)
from movie_rag.ingest.clean import MovieRecord
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
    assert first.points == client.count(settings.qdrant.collection, exact=True).count > 0

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
