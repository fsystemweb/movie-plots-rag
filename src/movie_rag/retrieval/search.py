"""One search path for the three retrieval modes.

``dense``, ``sparse`` and ``hybrid`` all go through :meth:`Retriever.search`, which builds a single
``query_points_groups`` request with :func:`build_request`. The mode only decides the query part:

* ``dense`` / ``sparse``: the query vector on the ``dense`` / ``bm25`` named vector, filter as ``query_filter``;
* ``hybrid``: two prefetches (one per named vector, ``retrieval.prefetch_limit`` chunks each) fused with RRF
  in the same call. The metadata filter is attached to **each prefetch**, so every branch selects its candidates
  among matching chunks only (``prefetch_limit`` matching chunks per branch, never a cut that a later filter empties).
  The shape is explicit on purpose: it does not depend on the server pushing a top-level filter down into the
  prefetches (observed on Qdrant 1.15.4, but not documented).

Results are grouped by ``movie_id`` (``group_size=1``), so every film appears once, represented by its best chunk.
Queries are embedded with the *query* embedders (``embed_dense_query`` / ``embed_sparse_query``).

RRF parameters: qdrant-client 1.15.1 and Qdrant 1.15.x expose plain ``Fusion.RRF`` only (the server's constant is
fixed); a configurable ``k`` arrives with ``RrfQuery`` in qdrant-client 1.16 / Qdrant 1.16, so there is no ``rrf_k``
setting yet (see docs/BACKLOG.md).
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from qdrant_client import QdrantClient
from qdrant_client import models as m

from movie_rag.config import RetrievalMode, Settings
from movie_rag.errors import MovieNotFoundError, RetrievalError
from movie_rag.ingest.embed import Embedder, FastEmbedder
from movie_rag.ingest.index import DENSE_VECTOR, SPARSE_VECTOR, point_id
from movie_rag.observability import span

logger = logging.getLogger(__name__)

MODES: tuple[RetrievalMode, ...] = ("dense", "sparse", "hybrid")
ELLIPSIS = "…"


class SearchFilters(BaseModel):
    """Metadata filters; ``genre`` and ``origin`` match exactly but case-insensitively (stored lowercase)."""

    model_config = ConfigDict(extra="forbid")

    year_from: int | None = None
    year_to: int | None = None
    genre: str | None = None
    origin: str | None = None

    @field_validator("genre", "origin", mode="before")
    @classmethod
    def _normalise(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().lower() or None
        return value

    @model_validator(mode="after")
    def _years_in_order(self) -> SearchFilters:
        if self.year_from is not None and self.year_to is not None and self.year_from > self.year_to:
            raise ValueError(f"year_from ({self.year_from}) must not be after year_to ({self.year_to})")
        return self


class MovieHit(BaseModel):
    """One film in a result list: its best chunk's score and snippet plus the fields needed for a citation."""

    movie_id: str
    title: str
    release_year: int | None
    director: str | None
    genre: str | None
    origin: str | None
    wiki_url: str | None
    score: float
    chunk_idx: int
    snippet: str


class MovieDetail(BaseModel):
    """A film's metadata and its whole plot (stored on chunk 0)."""

    movie_id: str
    title: str
    release_year: int | None
    director: str | None
    cast: str | None
    genre: str | None
    origin: str | None
    wiki_url: str | None
    n_chunks: int
    plot: str


class FilterOptions(BaseModel):
    """The values the metadata filters accept: genres and origins (most frequent first) and the year range."""

    genres: list[str]
    origins: list[str]
    year_min: int | None
    year_max: int | None
    truncated: bool = False


def build_filter(filters: SearchFilters | None) -> m.Filter | None:
    """The Qdrant filter for ``filters`` (``None`` when nothing is constrained). Built once per search."""
    if filters is None:
        return None
    must: list[m.Condition] = []
    if filters.year_from is not None or filters.year_to is not None:
        must.append(m.FieldCondition(key="release_year", range=m.Range(gte=filters.year_from, lte=filters.year_to)))
    if filters.genre is not None:
        must.append(m.FieldCondition(key="genre", match=m.MatchValue(value=filters.genre)))
    if filters.origin is not None:
        must.append(m.FieldCondition(key="origin", match=m.MatchValue(value=filters.origin)))
    return m.Filter(must=must) if must else None


def snippet_of(text: str, max_chars: int) -> str:
    """``text`` cut to at most ``max_chars`` characters at a word boundary, with an ellipsis when cut."""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    budget = max_chars - len(ELLIPSIS)
    cut = text[:budget]
    if text[budget] != " " and " " in cut:  # do not end in the middle of a word
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:-") + ELLIPSIS


def by_score_then_id(hits: list[MovieHit]) -> list[MovieHit]:
    """Best score first; films with the same score are ordered by ``movie_id``, so identical calls give identical lists.

    Reciprocal rank fusion gives equal scores to a film that is first in one list and second in the other, and Qdrant
    returns tied films in a varying order. (Which tied films make the ``top_k`` cut is still the server's choice.)
    """
    return sorted(hits, key=lambda hit: (-hit.score, hit.movie_id))


def build_request(
    settings: Settings,
    *,
    mode: RetrievalMode,
    top_k: int,
    query_filter: m.Filter | None,
    dense: list[float] | None = None,
    sparse: m.SparseVector | None = None,
) -> dict[str, Any]:
    """Keyword arguments for ``QdrantClient.query_points_groups``; the same builder serves all three modes."""
    request: dict[str, Any] = {
        "collection_name": settings.qdrant.collection,
        "group_by": "movie_id",
        "group_size": 1,
        "limit": top_k,
        "with_payload": True,
    }
    prefetch_limit = settings.retrieval.prefetch_limit
    if mode == "hybrid":
        if dense is None or sparse is None:
            raise RetrievalError("hybrid search needs both a dense and a sparse query vector")
        request["prefetch"] = [
            m.Prefetch(query=dense, using=DENSE_VECTOR, limit=prefetch_limit, filter=query_filter),
            m.Prefetch(query=sparse, using=SPARSE_VECTOR, limit=prefetch_limit, filter=query_filter),
        ]
        request["query"] = m.FusionQuery(fusion=m.Fusion.RRF)
    elif mode == "dense":
        if dense is None:
            raise RetrievalError("dense search needs a dense query vector")
        request.update(query=dense, using=DENSE_VECTOR, query_filter=query_filter)
    else:
        if sparse is None:
            raise RetrievalError("sparse search needs a sparse query vector")
        request.update(query=sparse, using=SPARSE_VECTOR, query_filter=query_filter)
    return request


class Retriever:
    """Searches the movie collection. Models and the Qdrant connection are created on first use."""

    def __init__(
        self, settings: Settings, *, client: QdrantClient | None = None, embedder: Embedder | None = None
    ) -> None:
        self._settings = settings
        self._client = client
        self._embedder = embedder

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(url=self._settings.qdrant.url, timeout=self._settings.qdrant.timeout_s)
        return self._client

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = FastEmbedder(self._settings.embeddings)
        return self._embedder

    def search(
        self,
        query: str,
        *,
        mode: RetrievalMode | None = None,
        top_k: int | None = None,
        filters: SearchFilters | None = None,
    ) -> list[MovieHit]:
        """Top ``top_k`` films for ``query``, best first, one row per film.

        ``mode`` and ``top_k`` default to ``retrieval.default_mode`` and ``retrieval.top_k``.
        """
        cfg = self._settings.retrieval
        mode = mode or cfg.default_mode
        top_k = cfg.top_k if top_k is None else top_k
        query = query.strip()
        with span(
            "retriever.search",
            run_type="retriever",
            settings=self._settings,
            inputs={"query": query},
            retrieval_mode=mode,
            top_k=top_k,
            filters=filters.model_dump(exclude_none=True) if filters else {},
        ) as run:
            if mode not in MODES:
                raise RetrievalError(f"unknown mode {mode!r}; use one of {', '.join(MODES)}")
            if not query:
                raise RetrievalError("the query is empty")
            if top_k < 1:
                raise RetrievalError(f"top_k must be at least 1, got {top_k}")
            request = build_request(
                self._settings,
                mode=mode,
                top_k=top_k,
                query_filter=build_filter(filters),
                dense=self.embedder.embed_dense_query(query) if mode != "sparse" else None,
                sparse=self.embedder.embed_sparse_query(query) if mode != "dense" else None,
            )
            result = self.client.query_points_groups(**request)
            hits = by_score_then_id([self._to_hit(group.hits[0]) for group in result.groups if group.hits])
            run.set(result_count=len(hits), movie_ids=[h.movie_id for h in hits])
        logger.info("search mode=%s top_k=%d filters=%s -> %d films", mode, top_k, filters, len(hits))
        return hits

    def get_movie(self, movie_id: str) -> MovieDetail | None:
        """The film's metadata and full plot, read from chunk 0 by id (no scroll, no filter). ``None`` if unknown."""
        points = self.client.retrieve(self._settings.qdrant.collection, ids=[point_id(movie_id, 0)], with_payload=True)
        return _to_detail(points[0]) if points else None

    def find_by_title(self, title: str, *, year: int | None = None, limit: int = 10) -> list[MovieDetail]:
        """Films whose title equals ``title`` exactly (case-sensitive), optionally of release ``year``.

        Reads chunk 0 of each match with one filtered ``scroll`` (``title`` has no payload index, so Qdrant checks the
        payload; fine for chunk-0-only matches). Titles are not unique (remakes): up to ``limit`` films are returned,
        ordered by year then id so the answer is stable.
        """
        must: list[m.Condition] = [
            m.FieldCondition(key="title", match=m.MatchValue(value=title.strip())),
            m.FieldCondition(key="chunk_idx", match=m.MatchValue(value=0)),
        ]
        if year is not None:
            must.append(m.FieldCondition(key="release_year", match=m.MatchValue(value=year)))
        points, _ = self.client.scroll(
            self._settings.qdrant.collection, scroll_filter=m.Filter(must=must), limit=limit, with_payload=True
        )
        films = [_to_detail(p) for p in points]
        return sorted(films, key=lambda f: (f.release_year is None, f.release_year or 0, f.movie_id))

    def find_similar(self, movie_id: str, *, top_k: int | None = None) -> list[MovieHit]:
        """The ``top_k`` films closest to ``movie_id`` by dense plot vector, never including ``movie_id`` itself.

        The query is the film's chunk 0 point (Qdrant looks its dense vector up by id), grouped by film like a search.
        Raises :class:`MovieNotFoundError` for an unknown id.
        """
        top_k = self._settings.retrieval.top_k if top_k is None else top_k
        with span(
            "retriever.find_similar",
            run_type="retriever",
            settings=self._settings,
            inputs={"movie_id": movie_id},
            retrieval_mode="dense",
            top_k=top_k,
        ) as run:
            if top_k < 1:
                raise RetrievalError(f"top_k must be at least 1, got {top_k}")
            anchor = point_id(movie_id, 0)
            if not self.client.retrieve(self._settings.qdrant.collection, ids=[anchor], with_payload=False):
                raise MovieNotFoundError(f"no film with movie_id {movie_id!r}")
            result = self.client.query_points_groups(
                self._settings.qdrant.collection,
                query=anchor,
                using=DENSE_VECTOR,
                query_filter=m.Filter(must_not=[m.FieldCondition(key="movie_id", match=m.MatchValue(value=movie_id))]),
                group_by="movie_id",
                group_size=1,
                limit=top_k,
                with_payload=True,
            )
            hits = by_score_then_id([self._to_hit(group.hits[0]) for group in result.groups if group.hits])
            run.set(result_count=len(hits), movie_ids=[h.movie_id for h in hits])
        return hits

    def list_filters(self, *, limit: int) -> FilterOptions:
        """Genres and origins (at most ``limit`` each, most frequent first) and the release-year range.

        Uses Qdrant's facet counts on the indexed ``genre`` and ``origin`` fields and ``order_by`` on the indexed
        ``release_year``, so nothing is scanned client-side.
        """
        collection = self._settings.qdrant.collection
        genres = [str(h.value) for h in self.client.facet(collection, key="genre", limit=limit).hits]
        origins = [str(h.value) for h in self.client.facet(collection, key="origin", limit=limit).hits]
        bounds: dict[str, int | None] = {}
        for label, direction in (("year_min", m.Direction.ASC), ("year_max", m.Direction.DESC)):
            points, _ = self.client.scroll(
                collection,
                limit=1,
                order_by=m.OrderBy(key="release_year", direction=direction),
                with_payload=["release_year"],
            )
            bounds[label] = points[0].payload["release_year"] if points and points[0].payload else None
        return FilterOptions(
            genres=genres,
            origins=origins,
            year_min=bounds["year_min"],
            year_max=bounds["year_max"],
            truncated=limit in (len(genres), len(origins)),
        )

    def _to_hit(self, point: m.ScoredPoint) -> MovieHit:
        p = dict(point.payload or {})
        return MovieHit(
            movie_id=p["movie_id"],
            title=p["title"],
            release_year=p.get("release_year"),
            director=p.get("director"),
            genre=p.get("genre"),
            origin=p.get("origin"),
            wiki_url=p.get("wiki_page"),
            score=point.score,
            chunk_idx=p["chunk_idx"],
            snippet=snippet_of(p["text"], self._settings.retrieval.snippet_max_chars),
        )


def _to_detail(point: m.Record) -> MovieDetail:
    p = dict(point.payload or {})
    return MovieDetail(
        movie_id=p["movie_id"],
        title=p["title"],
        release_year=p.get("release_year"),
        director=p.get("director"),
        cast=p.get("cast"),
        genre=p.get("genre"),
        origin=p.get("origin"),
        wiki_url=p.get("wiki_page"),
        n_chunks=p["n_chunks"],
        plot=p["full_plot"],
    )
