"""Retrieval with the real FastEmbed models, and (with ``QDRANT_URL`` set) against the real Qdrant service.

Without ``QDRANT_URL`` the tests run on the in-process ``QdrantClient(":memory:")``. That engine (qdrant-client 1.15.1)
discards the prefetch queries and filters in ``query_points_groups`` (``set_prefetch_limit_recursively`` keeps only the
limit), so hybrid search is only exercised when ``QDRANT_URL`` points at a real server (CI, or ``make up`` and
``export QDRANT_URL=http://localhost:6333``).
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from fakes import FakeEmbedder
from movie_rag.config import RetrievalMode, Settings, load_settings
from movie_rag.ingest.clean import MovieRecord, clean_csv
from movie_rag.ingest.embed import FastEmbedder
from movie_rag.ingest.index import ensure_collection
from movie_rag.ingest.pipeline import ingest_csv, write_records
from movie_rag.retrieval import Retriever, SearchFilters

pytestmark = [pytest.mark.integration, pytest.mark.filterwarnings("ignore:Payload indexes have no effect")]

QDRANT_URL = os.environ.get("QDRANT_URL")  # read at import: the per-test environment scrubber would hide it later
FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "movies_sample.csv"
MODES: tuple[RetrievalMode, ...] = ("dense", "sparse", "hybrid") if QDRANT_URL else ("dense", "sparse")


def _settings(**overrides: str) -> Settings:
    for name, value in overrides.items():
        os.environ[name] = value
    try:
        return load_settings(env_file=None)
    finally:
        for name in overrides:
            del os.environ[name]


def _throwaway_collection() -> str:
    return f"movie_plots_it_{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="module")
def client() -> Iterator[QdrantClient]:
    qdrant = QdrantClient(url=QDRANT_URL, timeout=60) if QDRANT_URL else QdrantClient(":memory:")
    yield qdrant
    qdrant.close()


@pytest.fixture(scope="module")
def indexed(client: QdrantClient) -> Iterator[tuple[Retriever, Settings, list[MovieRecord]]]:
    """The fixture ingested with the real models into a throw-away collection."""
    settings = _settings(QDRANT__COLLECTION=_throwaway_collection())
    embedder = FastEmbedder(settings.embeddings)
    ingest_csv(settings, FIXTURE, client=client, embedder=embedder, recreate=True)
    records, _ = clean_csv(FIXTURE, settings.ingest.min_plot_words)
    yield Retriever(settings, client=client, embedder=embedder), settings, records
    client.delete_collection(settings.qdrant.collection)


def premise(record: MovieRecord) -> str:
    return ". ".join(record.plot.split(". ")[:2])


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_finds_a_film_from_its_premise(
    indexed: tuple[Retriever, Settings, list[MovieRecord]], mode: RetrievalMode
) -> None:
    retriever, _, records = indexed
    sample = records[:: len(records) // 15][:15]
    found = sum(r.movie_id in [h.movie_id for h in retriever.search(premise(r), mode=mode, top_k=3)] for r in sample)
    assert found >= len(sample) - 1, (mode, found, len(sample))


@pytest.mark.parametrize("mode", MODES)
def test_results_are_one_row_per_film_and_respect_filters(
    indexed: tuple[Retriever, Settings, list[MovieRecord]], mode: RetrievalMode
) -> None:
    retriever, _, records = indexed
    long_plot = max(records, key=lambda r: len(r.plot))
    hits = retriever.search(long_plot.plot, mode=mode, top_k=15)
    assert len({h.movie_id for h in hits}) == len(hits) == 15

    target = next(r for r in records if r.genre and r.origin and r.release_year > 1960)
    assert target.genre and target.origin
    filters = SearchFilters(year_from=target.release_year, year_to=target.release_year, genre=target.genre)
    filtered = retriever.search(premise(target), mode=mode, top_k=15, filters=filters)
    assert filtered and filtered[0].movie_id == target.movie_id
    assert {(h.release_year, h.genre) for h in filtered} == {(target.release_year, target.genre)}

    excluded = retriever.search(
        premise(target), mode=mode, top_k=15, filters=SearchFilters(year_from=target.release_year + 1)
    )
    assert excluded and target.movie_id not in {h.movie_id for h in excluded}


def test_get_movie_returns_the_whole_plot_of_a_split_film(
    indexed: tuple[Retriever, Settings, list[MovieRecord]],
) -> None:
    retriever, _, records = indexed
    longest = max(records, key=lambda r: len(r.plot))
    detail = retriever.get_movie(longest.movie_id)
    assert detail is not None and detail.plot == longest.plot and detail.n_chunks > 1


def _crowded_collection(client: QdrantClient) -> tuple[Settings, FakeEmbedder]:
    """Five strong matches the filter excludes, three weaker ones it keeps, and a prefetch limit of only 2."""
    settings = _settings(QDRANT__COLLECTION=_throwaway_collection(), RETRIEVAL__PREFETCH_LIMIT="2")
    embedder = FakeEmbedder()
    ensure_collection(client, settings)

    def record(movie_id: str, year: int, genre: str, plot: str) -> MovieRecord:
        return MovieRecord(
            movie_id=movie_id,
            title=movie_id.replace("-", " "),
            release_year=year,
            origin="american",
            director="Pat Doe",
            cast="A, B",
            genre=genre,
            wiki_url=f"https://example.test/{movie_id}",
            plot=plot,
        )

    strong = [record(f"strong-{i}", 2000, "drama", "zebra quartz " * 8) for i in range(5)]
    weak = [record(f"weak-{i}", 1990, "comedy", "zebra apple pear plum fig lime kiwi") for i in range(3)]
    write_records(client, embedder, settings, [*strong, *weak])
    return settings, embedder


@pytest.mark.parametrize("mode", MODES)
def test_the_filter_is_applied_before_the_candidate_cut_not_after_it(client: QdrantClient, mode: RetrievalMode) -> None:
    settings, embedder = _crowded_collection(client)
    try:
        assert settings.retrieval.prefetch_limit == 2
        retriever = Retriever(settings, client=client, embedder=embedder)
        hits = retriever.search("zebra quartz", mode=mode, top_k=3, filters=SearchFilters(year_to=1995))
        assert sorted(h.movie_id for h in hits) == ["weak-0", "weak-1", "weak-2"]
    finally:
        client.delete_collection(settings.qdrant.collection)


def test_exact_title_lookup_finds_the_film_and_its_whole_plot(
    indexed: tuple[Retriever, Settings, list[MovieRecord]],
) -> None:
    retriever, _, records = indexed
    record = records[10]
    assert [(f.movie_id, f.plot) for f in retriever.find_by_title(record.title)] == [(record.movie_id, record.plot)]
    assert retriever.find_by_title(record.title, year=record.release_year + 1) == []


def test_find_similar_returns_other_films_best_first(indexed: tuple[Retriever, Settings, list[MovieRecord]]) -> None:
    retriever, _, records = indexed
    hits = retriever.find_similar(records[0].movie_id, top_k=5)
    assert len(hits) == 5 and records[0].movie_id not in {h.movie_id for h in hits}
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_list_filters_uses_the_indexed_fields(indexed: tuple[Retriever, Settings, list[MovieRecord]]) -> None:
    retriever, _, records = indexed
    options = retriever.list_filters(limit=1000)
    assert set(options.genres) == {r.genre for r in records if r.genre}
    assert set(options.origins) == {r.origin for r in records if r.origin}
    assert (options.year_min, options.year_max) == (
        min(r.release_year for r in records),
        max(r.release_year for r in records),
    )
