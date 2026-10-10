"""Pydantic return models of the MCP tools (their JSON schemas are part of the tool contract and snapshot-tested)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from movie_rag.config import RetrievalMode
from movie_rag.retrieval import MovieHit


class SearchResult(BaseModel):
    """Films matching a search, best first, one row per film."""

    query: str = Field(description="The query as searched (whitespace trimmed).")
    mode: RetrievalMode = Field(description="Retrieval mode that produced the ranking.")
    count: int = Field(description="Number of films returned.")
    results: list[MovieHit] = Field(description="Ranked films; cite each as 'title (release_year)' with its wiki_url.")
    note: str | None = Field(default=None, description="Hint when nothing matched (how to relax the search).")


class SimilarResult(BaseModel):
    """Films whose plots are closest to a given film."""

    movie_id: str = Field(description="The film the search started from (never part of the results).")
    count: int = Field(description="Number of films returned.")
    results: list[MovieHit] = Field(description="Most similar films first; cite each with its wiki_url.")
