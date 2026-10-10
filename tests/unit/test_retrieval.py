from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError
from qdrant_client import QdrantClient
from qdrant_client import models as m

from fakes import FakeEmbedder
from movie_rag.config import RetrievalMode, Settings, load_settings
from movie_rag.errors import MovieNotFoundError, RetrievalError
from movie_rag.ingest.clean import MovieRecord, clean_csv
from movie_rag.ingest.index import DENSE_VECTOR, SPARSE_VECTOR, ensure_collection
from movie_rag.ingest.pipeline import ingest_csv, write_records
from movie_rag.retrieval import (
    MovieDetail,
    Retriever,
    SearchFilters,
    build_filter,
    build_request,
    snippet_of,
)
from movie_rag.retrieval import search as search_module

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")

MakeSettings = Callable[..., Settings]
MODES: tuple[RetrievalMode, ...] = ("dense", "sparse", "hybrid")
# qdrant-client 1.15.1's in-memory engine drops the prefetch queries and filters of `query_points_groups`
# (local_collection.set_prefetch_limit_recursively keeps only the limit), so hybrid *behaviour* cannot be observed
# there. Hybrid is covered by request-shape tests below and, against the real service, by
# tests/integration/test_retrieval_qdrant.py.
LOCAL_MODES: tuple[RetrievalMode, ...] = ("dense", "sparse")


# --- filters ---------------------------------------------------------------------------------------------------


def test_no_filters_build_no_filter() -> None:
    assert build_filter(None) is None
    assert build_filter(SearchFilters()) is None


def test_year_range_genre_and_origin_become_must_conditions() -> None:
    flt = build_filter(SearchFilters(year_from=1950, year_to=1999, genre="Film Noir", origin=" British "))
    assert flt is not None and flt.must is not None
    conditions = {c.key: c for c in flt.must if isinstance(c, m.FieldCondition)}
    assert set(conditions) == {"release_year", "genre", "origin"}
    assert conditions["release_year"].range == m.Range(gte=1950, lte=1999)
    assert conditions["genre"].match == m.MatchValue(value="film noir")
    assert conditions["origin"].match == m.MatchValue(value="british")


def test_open_ended_year_range_sets_only_one_bound() -> None:
    flt = build_filter(SearchFilters(year_from=1980))
    assert flt is not None and flt.must is not None
    (condition,) = flt.must
    assert isinstance(condition, m.FieldCondition)
    assert condition.range == m.Range(gte=1980, lte=None)


def test_blank_genre_and_origin_mean_no_constraint() -> None:
    assert SearchFilters(genre="  ", origin="").genre is None
    assert build_filter(SearchFilters(genre="  ", origin="")) is None


def test_year_from_after_year_to_is_rejected() -> None:
    with pytest.raises(ValidationError, match="year_from"):
        SearchFilters(year_from=2000, year_to=1990)


def test_unknown_filter_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        SearchFilters(director="x")  # type: ignore[call-arg]


# --- snippets --------------------------------------------------------------------------------------------------


def test_short_text_is_returned_whole_with_whitespace_collapsed() -> None:
    assert snippet_of("one  two\nthree", 400) == "one two three"


def test_long_text_is_cut_at_a_word_boundary_within_the_limit() -> None:
    text = "alpha beta gamma delta epsilon zeta"
    out = snippet_of(text, 20)
    assert out == "alpha beta gamma…"
    assert len(out) <= 20


def test_a_cut_that_lands_exactly_on_a_space_keeps_the_last_word() -> None:
    out = snippet_of("alpha beta gamma delta", 11)  # budget 10 -> "alpha beta" then a space
    assert out == "alpha beta…"


def test_a_single_long_word_is_cut_hard() -> None:
    out = snippet_of("x" * 50, 10)
    assert out == "x" * 9 + "…" and len(out) == 10


# --- request shape ---------------------------------------------------------------------------------------------

DENSE_Q = [0.1, 0.2]
SPARSE_Q = m.SparseVector(indices=[1, 5], values=[1.0, 2.0])
FLT = m.Filter(must=[m.FieldCondition(key="genre", match=m.MatchValue(value="drama"))])


@pytest.fixture
def settings() -> Settings:
    return load_settings(env_file=None)


def test_hybrid_request_has_two_prefetches_each_carrying_the_filter_and_rrf_on_top(settings: Settings) -> None:
    req = build_request(settings, mode="hybrid", top_k=5, query_filter=FLT, dense=DENSE_Q, sparse=SPARSE_Q)
    dense_pf, sparse_pf = req["prefetch"]
    assert (dense_pf.using, sparse_pf.using) == (DENSE_VECTOR, SPARSE_VECTOR)
    assert dense_pf.query == DENSE_Q and sparse_pf.query == SPARSE_Q
    assert dense_pf.filter == FLT and sparse_pf.filter == FLT  # inside each prefetch
    assert dense_pf.limit == sparse_pf.limit == settings.retrieval.prefetch_limit
    assert req["query"] == m.FusionQuery(fusion=m.Fusion.RRF)
    assert "query_filter" not in req and "using" not in req
    assert (req["group_by"], req["group_size"], req["limit"]) == ("movie_id", 1, 5)
    assert req["collection_name"] == settings.qdrant.collection


def test_dense_and_sparse_requests_query_one_named_vector_with_the_filter(settings: Settings) -> None:
    dense = build_request(settings, mode="dense", top_k=3, query_filter=FLT, dense=DENSE_Q)
    assert (dense["query"], dense["using"], dense["query_filter"]) == (DENSE_Q, DENSE_VECTOR, FLT)
    sparse = build_request(settings, mode="sparse", top_k=3, query_filter=None, sparse=SPARSE_Q)
    assert (sparse["query"], sparse["using"], sparse["query_filter"]) == (SPARSE_Q, SPARSE_VECTOR, None)
    assert "prefetch" not in dense and "prefetch" not in sparse
    for req in (dense, sparse):  # same grouping and payload options in every mode
        assert (req["group_by"], req["group_size"], req["limit"], req["with_payload"]) == ("movie_id", 1, 3, True)


@pytest.mark.parametrize(
    ("mode", "kwargs"),
    [
        ("hybrid", {"dense": DENSE_Q}),
        ("hybrid", {"sparse": SPARSE_Q}),
        ("dense", {"sparse": SPARSE_Q}),
        ("sparse", {"dense": DENSE_Q}),
    ],
)
def test_a_request_missing_the_vector_its_mode_needs_is_rejected(
    settings: Settings, mode: RetrievalMode, kwargs: dict[str, Any]
) -> None:
    with pytest.raises(RetrievalError, match="query vector"):
        build_request(settings, mode=mode, top_k=3, query_filter=None, **kwargs)


# --- search over the fixture -----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fixture_index() -> tuple[QdrantClient, Settings, list[MovieRecord]]:
    settings = load_settings(env_file=None)
    client = QdrantClient(":memory:")
    csv_path = Path(__file__).resolve().parents[1] / "fixtures" / "movies_sample.csv"
    ingest_csv(settings, csv_path, client=client, embedder=FakeEmbedder(), recreate=True)
    records, _ = clean_csv(csv_path, settings.ingest.min_plot_words)
    return client, settings, records


@pytest.fixture
def retriever(fixture_index: tuple[QdrantClient, Settings, list[MovieRecord]]) -> Retriever:
    client, settings, _ = fixture_index
    return Retriever(settings, client=client, embedder=FakeEmbedder())


def first_sentence(record: MovieRecord) -> str:
    """The premise: the first two sentences of the plot."""
    return ". ".join(record.plot.split(". ")[:2])


@pytest.mark.parametrize("mode", LOCAL_MODES)
def test_a_film_is_found_by_its_own_premise_in_every_mode(
    retriever: Retriever, fixture_index: tuple[QdrantClient, Settings, list[MovieRecord]], mode: RetrievalMode
) -> None:
    _, _, records = fixture_index
    for record in records[:10]:
        hits = retriever.search(first_sentence(record), mode=mode, top_k=3)
        assert hits[0].movie_id == record.movie_id, (mode, record.title)


@pytest.mark.parametrize("mode", LOCAL_MODES)
def test_results_have_one_row_per_film_best_first_with_citation_fields(
    retriever: Retriever, fixture_index: tuple[QdrantClient, Settings, list[MovieRecord]], mode: RetrievalMode
) -> None:
    _, settings, records = fixture_index
    # A very generic query matches many chunks, several of them belonging to the same long film.
    long_plot = max(records, key=lambda r: len(r.plot)).plot
    hits = retriever.search(long_plot, mode=mode, top_k=20)
    ids = [h.movie_id for h in hits]
    assert len(ids) == len(set(ids)) == 20
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    by_id = {r.movie_id: r for r in records}
    for hit in hits:
        record = by_id[hit.movie_id]
        assert (hit.title, hit.release_year, hit.director) == (record.title, record.release_year, record.director)
        assert (hit.genre, hit.origin, hit.wiki_url) == (record.genre, record.origin, record.wiki_url)
        assert 0 < len(hit.snippet) <= settings.retrieval.snippet_max_chars


def test_a_multi_chunk_film_is_returned_once_through_its_best_chunk(
    retriever: Retriever, fixture_index: tuple[QdrantClient, Settings, list[MovieRecord]]
) -> None:
    client, settings, _ = fixture_index
    chunks_per_film = Counter(
        p.payload["movie_id"]
        for p in client.scroll(settings.qdrant.collection, limit=1000, with_payload=True)[0]
        if p.payload
    )
    multi = [movie_id for movie_id, n in chunks_per_film.items() if n > 1]
    assert multi, "the fixture should contain films that were split into several chunks"
    last_chunk = client.scroll(
        settings.qdrant.collection,
        scroll_filter=m.Filter(
            must=[
                m.FieldCondition(key="movie_id", match=m.MatchValue(value=multi[0])),
                m.FieldCondition(key="chunk_idx", match=m.MatchValue(value=chunks_per_film[multi[0]] - 1)),
            ]
        ),
        with_payload=True,
    )[0][0]
    assert last_chunk.payload is not None
    for mode in LOCAL_MODES:
        hits = retriever.search(last_chunk.payload["text"], mode=mode, top_k=5)
        assert [h.movie_id for h in hits].count(multi[0]) == 1
        top = hits[0]
        assert top.movie_id == multi[0] and top.chunk_idx == chunks_per_film[multi[0]] - 1


@pytest.mark.parametrize("mode", LOCAL_MODES)
def test_filters_are_respected_in_every_mode(
    retriever: Retriever, fixture_index: tuple[QdrantClient, Settings, list[MovieRecord]], mode: RetrievalMode
) -> None:
    _, _, records = fixture_index
    target = next(r for r in records if r.genre and r.origin and r.release_year > 1960)
    query = first_sentence(target)

    in_years = retriever.search(
        query, mode=mode, top_k=20, filters=SearchFilters(year_from=target.release_year, year_to=target.release_year)
    )
    assert in_years and {h.release_year for h in in_years} == {target.release_year}
    assert in_years[0].movie_id == target.movie_id

    assert target.genre is not None
    by_genre = retriever.search(query, mode=mode, top_k=20, filters=SearchFilters(genre=target.genre.upper()))
    assert by_genre and {h.genre for h in by_genre} == {target.genre}

    assert target.origin is not None
    by_origin = retriever.search(query, mode=mode, top_k=20, filters=SearchFilters(origin=target.origin))
    assert by_origin and {h.origin for h in by_origin} == {target.origin}


@pytest.mark.parametrize("mode", LOCAL_MODES)
def test_a_filter_that_excludes_the_best_film_never_returns_it(
    retriever: Retriever, fixture_index: tuple[QdrantClient, Settings, list[MovieRecord]], mode: RetrievalMode
) -> None:
    _, _, records = fixture_index
    target = records[3]
    other_years = SearchFilters(year_from=target.release_year + 1)
    hits = retriever.search(first_sentence(target), mode=mode, top_k=20, filters=other_years)
    assert hits and target.movie_id not in {h.movie_id for h in hits}
    assert all(h.release_year is not None and h.release_year > target.release_year for h in hits)


def test_combined_filters_with_no_match_return_an_empty_list(retriever: Retriever) -> None:
    for mode in LOCAL_MODES:
        assert retriever.search("anything", mode=mode, filters=SearchFilters(year_from=3000)) == []


# --- hybrid goes through the same Retriever.search ---------------------------------------------------------------


def test_hybrid_search_parses_groups_into_one_hit_per_film(settings: Settings) -> None:
    def point(movie_id: str, score: float, chunk_idx: int) -> m.ScoredPoint:
        payload = {
            "movie_id": movie_id,
            "title": movie_id.title(),
            "release_year": 1999,
            "director": None,
            "genre": "drama",
            "origin": "american",
            "wiki_page": f"https://example.test/{movie_id}",
            "chunk_idx": chunk_idx,
            "text": "word " * 200,
        }
        return m.ScoredPoint(id=movie_id, version=1, score=score, payload=payload)

    client = MagicMock(spec=QdrantClient)
    client.query_points_groups.return_value = m.GroupsResult(
        groups=[
            m.PointGroup(id="a", hits=[point("a", 0.9, 2)]),
            m.PointGroup(id="empty", hits=[]),
            m.PointGroup(id="b", hits=[point("b", 0.4, 0)]),
        ]
    )
    hits = Retriever(settings, client=client, embedder=FakeEmbedder()).search("q", mode="hybrid", top_k=2)
    assert [(h.movie_id, h.score, h.chunk_idx) for h in hits] == [("a", 0.9, 2), ("b", 0.4, 0)]
    assert hits[0].director is None and hits[0].wiki_url == "https://example.test/a"
    assert len(hits[0].snippet) == settings.retrieval.snippet_max_chars and hits[0].snippet.endswith("…")
    assert client.query_points_groups.call_args.kwargs["limit"] == 2


@pytest.mark.parametrize("server_order", [["b", "c", "a", "d"], ["d", "a", "c", "b"], ["c", "b", "d", "a"]])
def test_tied_scores_come_back_in_movie_id_order_whatever_order_the_server_used(
    settings: Settings, server_order: list[str]
) -> None:
    """RRF ties (a film first in one list and second in the other) arrive in a varying order; the list must not."""
    scores = {"a": 0.5, "b": 0.5, "c": 0.5, "d": 0.9}

    def point(movie_id: str) -> m.ScoredPoint:
        payload = {"movie_id": movie_id, "title": movie_id, "chunk_idx": 0, "text": "plot", "release_year": 2000}
        return m.ScoredPoint(id=movie_id, version=1, score=scores[movie_id], payload=payload)

    client = MagicMock(spec=QdrantClient)
    client.query_points_groups.return_value = m.GroupsResult(
        groups=[m.PointGroup(id=i, hits=[point(i)]) for i in server_order]
    )
    hits = Retriever(settings, client=client, embedder=FakeEmbedder()).search("q", mode="hybrid", top_k=4)
    assert [h.movie_id for h in hits] == ["d", "a", "b", "c"]  # best score first, then by movie_id


def test_find_similar_orders_ties_by_movie_id_too(settings: Settings) -> None:
    def point(movie_id: str) -> m.ScoredPoint:
        payload = {"movie_id": movie_id, "title": movie_id, "chunk_idx": 0, "text": "plot"}
        return m.ScoredPoint(id=movie_id, version=1, score=0.7, payload=payload)

    client = MagicMock(spec=QdrantClient)
    client.retrieve.return_value = [m.Record(id="anchor", payload={})]
    client.query_points_groups.return_value = m.GroupsResult(
        groups=[m.PointGroup(id=i, hits=[point(i)]) for i in ("z", "m", "a")]
    )
    hits = Retriever(settings, client=client, embedder=FakeEmbedder()).find_similar("x-1999-1", top_k=3)
    assert [h.movie_id for h in hits] == ["a", "m", "z"]


def test_hybrid_sends_the_filter_in_both_prefetches_to_the_client(settings: Settings) -> None:
    client = MagicMock(spec=QdrantClient)
    client.query_points_groups.return_value = m.GroupsResult(groups=[])
    Retriever(settings, client=client, embedder=FakeEmbedder()).search(
        "q", mode="hybrid", filters=SearchFilters(genre="drama")
    )
    kwargs = client.query_points_groups.call_args.kwargs
    flt = build_filter(SearchFilters(genre="drama"))
    assert [p.filter for p in kwargs["prefetch"]] == [flt, flt]
    assert "query_filter" not in kwargs


# --- Retriever behaviour ---------------------------------------------------------------------------------------


class SpyEmbedder(FakeEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    def embed_dense_query(self, text: str) -> list[float]:
        self.calls.append("dense")
        return super().embed_dense_query(text)

    def embed_sparse_query(self, text: str) -> m.SparseVector:
        self.calls.append("sparse")
        return super().embed_sparse_query(text)


@pytest.mark.parametrize(
    ("mode", "expected"), [("dense", ["dense"]), ("sparse", ["sparse"]), ("hybrid", ["dense", "sparse"])]
)
def test_only_the_query_embeddings_a_mode_needs_are_computed(
    fixture_index: tuple[QdrantClient, Settings, list[MovieRecord]], mode: RetrievalMode, expected: list[str]
) -> None:
    client, settings, _ = fixture_index
    spy = SpyEmbedder()
    Retriever(settings, client=client, embedder=spy).search("a query", mode=mode)
    assert spy.calls == expected
    assert spy.dense_batches == spy.sparse_batches == []  # queries never go through the document path


def test_mode_and_top_k_default_to_the_configuration(
    fixture_index: tuple[QdrantClient, Settings, list[MovieRecord]],
) -> None:
    client, settings, _ = fixture_index
    spy = SpyEmbedder()
    hits = Retriever(settings, client=client, embedder=spy).search("a film about a heist")
    assert settings.retrieval.default_mode == "hybrid" and spy.calls == ["dense", "sparse"]
    assert len(hits) == settings.retrieval.top_k


def test_top_k_is_honoured(retriever: Retriever) -> None:
    assert len(retriever.search("a film about a heist", top_k=2)) == 2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"query": "  "}, "empty"),
        ({"query": "x", "top_k": 0}, "top_k"),
        ({"query": "x", "mode": "fuzzy"}, "unknown mode"),
    ],
)
def test_invalid_requests_raise_retrieval_errors(retriever: Retriever, kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(RetrievalError, match=message):
        retriever.search(**kwargs)


def test_get_movie_reads_the_full_plot_from_chunk_zero(
    retriever: Retriever, fixture_index: tuple[QdrantClient, Settings, list[MovieRecord]]
) -> None:
    _, _, records = fixture_index
    record = max(records, key=lambda r: len(r.plot))  # a split film: the plot must still come back whole
    detail = retriever.get_movie(record.movie_id)
    assert isinstance(detail, MovieDetail)
    assert detail.plot == record.plot and detail.n_chunks > 1
    assert (detail.title, detail.release_year, detail.director, detail.cast) == (
        record.title,
        record.release_year,
        record.director,
        record.cast,
    )
    assert (detail.genre, detail.origin, detail.wiki_url) == (record.genre, record.origin, record.wiki_url)


def test_get_movie_of_an_unknown_id_is_none(retriever: Retriever) -> None:
    assert retriever.get_movie("no-such-film-1900-0") is None


def test_client_and_embedder_are_created_lazily_from_the_settings(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    qdrant_cls = MagicMock()
    embedder_cls = MagicMock()
    monkeypatch.setattr(search_module, "QdrantClient", qdrant_cls)
    monkeypatch.setattr(search_module, "FastEmbedder", embedder_cls)
    retriever = Retriever(settings)
    qdrant_cls.assert_not_called()
    embedder_cls.assert_not_called()
    assert retriever.client is retriever.client and retriever.embedder is retriever.embedder
    qdrant_cls.assert_called_once_with(url=settings.qdrant.url, timeout=settings.qdrant.timeout_s)
    embedder_cls.assert_called_once_with(settings.embeddings)


# --- title lookup, similar films, filter values, spans ----------------------------------------------------------


def test_find_by_title_matches_exactly_and_returns_the_full_plot(
    retriever: Retriever, fixture_index: tuple[QdrantClient, Settings, list[MovieRecord]]
) -> None:
    _, _, records = fixture_index
    record = records[4]
    (film,) = retriever.find_by_title(f" {record.title} ")
    assert (film.movie_id, film.plot, film.release_year) == (record.movie_id, record.plot, record.release_year)
    assert retriever.find_by_title(record.title.lower() + " ") == []  # exact, case-sensitive
    assert retriever.find_by_title("No Such Film") == []


def test_find_by_title_can_narrow_by_year_and_orders_remakes_by_year(
    fixture_index: tuple[QdrantClient, Settings, list[MovieRecord]],
) -> None:
    _client, settings, records = fixture_index
    original = records[4]
    scratch = QdrantClient(":memory:")
    settings = settings.model_copy(deep=True)
    ensure_collection(scratch, settings)
    remake = original.model_copy(update={"movie_id": "z-remake", "release_year": original.release_year - 20})
    write_records(scratch, FakeEmbedder(), settings, [original, remake])
    found = Retriever(settings, client=scratch, embedder=FakeEmbedder())
    assert [f.movie_id for f in found.find_by_title(original.title)] == ["z-remake", original.movie_id]
    assert [f.movie_id for f in found.find_by_title(original.title, year=original.release_year)] == [original.movie_id]
    assert len(found.find_by_title(original.title, limit=1)) == 1


def test_find_similar_excludes_the_film_and_unknown_ids_raise(
    retriever: Retriever, fixture_index: tuple[QdrantClient, Settings, list[MovieRecord]]
) -> None:
    _, settings, records = fixture_index
    hits = retriever.find_similar(records[0].movie_id, top_k=4)
    assert len(hits) == 4 and records[0].movie_id not in {h.movie_id for h in hits}
    assert len(retriever.find_similar(records[0].movie_id)) == settings.retrieval.top_k
    with pytest.raises(MovieNotFoundError, match="no-such-film"):
        retriever.find_similar("no-such-film")
    with pytest.raises(RetrievalError, match="top_k"):
        retriever.find_similar(records[0].movie_id, top_k=0)


def test_list_filters_reports_values_by_frequency_and_the_year_range(
    retriever: Retriever, fixture_index: tuple[QdrantClient, Settings, list[MovieRecord]]
) -> None:
    _, _, records = fixture_index
    options = retriever.list_filters(limit=1000)
    assert set(options.genres) == {r.genre for r in records if r.genre}
    assert set(options.origins) == {r.origin for r in records if r.origin}
    assert (options.year_min, options.year_max) == (
        min(r.release_year for r in records),
        max(r.release_year for r in records),
    )
    assert options.truncated is False
    capped = retriever.list_filters(limit=2)
    assert capped.genres == options.genres[:2] and capped.truncated is True


def test_list_filters_on_an_empty_collection_has_no_year_range() -> None:
    settings = load_settings(env_file=None)
    empty = QdrantClient(":memory:")
    ensure_collection(empty, settings)
    options = Retriever(settings, client=empty, embedder=FakeEmbedder()).list_filters(limit=5)
    assert (options.genres, options.origins, options.year_min, options.year_max) == ([], [], None, None)


def test_search_and_find_similar_open_retriever_spans(
    retriever: Retriever, fixture_index: tuple[QdrantClient, Settings, list[MovieRecord]]
) -> None:
    from fakes import RunCapture

    _, _, records = fixture_index
    with RunCapture() as capture:
        retriever.search("a heist", mode="sparse", top_k=2, filters=SearchFilters(genre="Drama", year_to=1990))
        retriever.find_similar(records[0].movie_id, top_k=2)
    search_run, similar_run = capture.runs["retriever.search"], capture.runs["retriever.find_similar"]
    assert search_run["run_type"] == similar_run["run_type"] == "retriever"
    meta = search_run["extra"]["metadata"]
    assert (meta["retrieval_mode"], meta["top_k"], meta["filters"]) == (
        "sparse",
        2,
        {"year_to": 1990, "genre": "drama"},
    )
    assert search_run["outputs"]["result_count"] <= 2 and search_run["outputs"]["latency_ms"] >= 0
    assert similar_run["outputs"]["result_count"] == 2 and similar_run["inputs"] == {"movie_id": records[0].movie_id}
