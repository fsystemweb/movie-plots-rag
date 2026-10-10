"""The evaluation against the real FastEmbed models, and (with ``QDRANT_URL`` set) the real Qdrant service.

Without ``QDRANT_URL`` the in-process engine is used for dense and sparse only (it cannot do hybrid search, see
tests/integration/test_retrieval_qdrant.py); with it, all three modes run on a throw-away collection and the numbers
must be reproducible: Qdrant returns tied reciprocal-rank-fusion scores in varying order, which the tie-aware metrics
have to absorb.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest
from qdrant_client import QdrantClient

from movie_rag.config import RetrievalMode, Settings, load_settings
from movie_rag.eval.questions import fixture_records, load_eval_set, matches_filters
from movie_rag.eval.report import EvalReport
from movie_rag.eval.retrieval_eval import connect
from movie_rag.eval.runner import Dependencies, run
from movie_rag.ingest.embed import FastEmbedder

pytestmark = [pytest.mark.integration, pytest.mark.filterwarnings("ignore:Payload indexes have no effect")]

QDRANT_URL = os.environ.get("QDRANT_URL")  # read at import: the per-test environment scrubber would hide it later
MODES: tuple[RetrievalMode, ...] = ("dense", "sparse", "hybrid") if QDRANT_URL else ("dense", "sparse")


@pytest.fixture(scope="module")
def settings() -> Settings:
    os.environ["QDRANT__COLLECTION"] = f"movie_plots_eval_it_{uuid.uuid4().hex[:8]}"
    try:
        return load_settings(env_file=None)
    finally:
        del os.environ["QDRANT__COLLECTION"]


@pytest.fixture(scope="module")
def client(settings: Settings) -> Iterator[QdrantClient]:
    qdrant = QdrantClient(url=QDRANT_URL, timeout=60) if QDRANT_URL else QdrantClient(":memory:")
    yield qdrant
    if qdrant.collection_exists(settings.qdrant.collection):
        qdrant.delete_collection(settings.qdrant.collection)
    qdrant.close()


def evaluate(settings: Settings, client: QdrantClient) -> list[EvalReport]:
    deps = Dependencies(client=client, embedder=FastEmbedder(settings.embeddings), allow_local_qdrant=not QDRANT_URL)
    return run(settings, MODES, attempt_llm=False, experiments=False, deps=deps).reports


def test_the_evaluation_ingests_the_fixture_and_every_mode_clears_the_smoke_floor(
    settings: Settings, client: QdrantClient
) -> None:
    if QDRANT_URL:
        connect(settings, client)  # a real service is accepted
    reports = evaluate(settings, client)
    assert reports[0].index.fixture_ingested_now is True
    for report in reports:
        assert report.retrieval.overall.n == 30
        assert report.retrieval.overall.mrr is not None
        assert report.retrieval.overall.mrr >= settings.eval.smoke_min_mrr, report.mode
        assert report.retrieval.overall.hit_at_k[8] is not None and report.retrieval.overall.hit_at_k[8] >= 0.8


def test_a_second_run_reuses_the_index_and_gives_the_same_numbers(settings: Settings, client: QdrantClient) -> None:
    first = evaluate(settings, client)
    second = evaluate(settings, client)
    assert second[0].index.fixture_ingested_now is False
    for a, b in zip(first, second, strict=True):
        assert a.retrieval.overall == b.retrieval.overall, a.mode
        assert a.retrieval.by_type == b.retrieval.by_type, a.mode


def test_filtered_questions_stay_inside_their_filters_in_every_mode(settings: Settings, client: QdrantClient) -> None:
    records = {r.movie_id: r for r in fixture_records(settings)}
    filters = {q.id: q.filters for q in load_eval_set(settings) if q.type == "filtered"}
    for report in evaluate(settings, client):
        for found in report.retrieval.per_question:
            if found.question_id in filters:
                assert found.ranked_movie_ids, (report.mode, found.question_id)
                assert all(matches_filters(filters[found.question_id], records[m]) for m in found.ranked_movie_ids)
