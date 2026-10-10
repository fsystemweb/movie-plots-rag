"""Shared helpers for the evaluation tests: dependencies that need no service, no model download and no key."""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient

from fakes import FakeEmbedder
from movie_rag.config import RetrievalMode, Settings, load_settings
from movie_rag.eval.questions import load_eval_set
from movie_rag.eval.retrieval_eval import ensure_fixture_indexed
from movie_rag.eval.runner import Dependencies, RunResult, run

_shared: list[QdrantClient] = []


def shared_client() -> QdrantClient:
    """One in-memory index with the fixture, built once per test session (the evaluation only reads it)."""
    if not _shared:
        settings = load_settings(env_file=None)
        client = QdrantClient(":memory:")
        ensure_fixture_indexed(settings, client, load_eval_set(settings), embedder=FakeEmbedder())
        _shared.append(client)
    return _shared[0]


def fake_deps(*, fresh: bool = False, **overrides: Any) -> Dependencies:
    """In-memory Qdrant (plumbing only: its hybrid search is wrong) and the deterministic fake embedder.

    ``fresh`` gives an empty index, so the run has to ingest the fixture itself.
    """
    values: dict[str, Any] = {
        "client": QdrantClient(":memory:") if fresh else shared_client(),
        "embedder": FakeEmbedder(),
        "allow_local_qdrant": True,
    }
    values.update(overrides)
    return Dependencies(**values)


def retrieval_only(
    settings: Settings, modes: tuple[RetrievalMode, ...] = ("dense", "sparse"), *, fresh: bool = False
) -> RunResult:
    """Evaluate ``modes`` on the fixture without the LLM half and without LangSmith."""
    return run(settings, modes, attempt_llm=False, experiments=False, deps=fake_deps(fresh=fresh))
