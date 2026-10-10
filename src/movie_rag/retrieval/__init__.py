"""Hybrid retrieval over the Qdrant collection built by :mod:`movie_rag.ingest` (dense, BM25 sparse, RRF hybrid)."""

from movie_rag.retrieval.search import (
    FilterOptions,
    MovieDetail,
    MovieHit,
    Retriever,
    SearchFilters,
    build_filter,
    build_request,
    snippet_of,
)

__all__ = [
    "FilterOptions",
    "MovieDetail",
    "MovieHit",
    "Retriever",
    "SearchFilters",
    "build_filter",
    "build_request",
    "snippet_of",
]
