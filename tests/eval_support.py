"""Shared helpers for the evaluation tests: dependencies that need no service, no model download and no key."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from qdrant_client import QdrantClient

from fakes import FakeEmbedder
from movie_rag.config import RetrievalMode, Settings, load_settings
from movie_rag.eval.questions import EvalQuestion, load_eval_set
from movie_rag.eval.report import EvalReport
from movie_rag.eval.retrieval_eval import ensure_fixture_indexed
from movie_rag.eval.runner import PENDING_REASON, Dependencies, RunResult, run, select_per_type, skipped_llm

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


def small(questions: Sequence[EvalQuestion]) -> Sequence[EvalQuestion]:
    """Two questions of each type (8 instead of 40): enough to exercise every code path in a fraction of the time."""
    return select_per_type(questions, 2)


def retrieval_only(
    settings: Settings,
    modes: tuple[RetrievalMode, ...] = ("dense", "sparse"),
    *,
    fresh: bool = False,
    subset: Callable[[Sequence[EvalQuestion]], Sequence[EvalQuestion]] | None = None,
) -> RunResult:
    """Evaluate ``modes`` on the fixture without the LLM half and without LangSmith."""
    return run(settings, modes, attempt_llm=False, experiments=False, deps=fake_deps(fresh=fresh), subset=subset)


_cached: list[list[EvalReport]] = []


def cached_reports() -> list[EvalReport]:
    """Retrieval-only reports for the three modes on the small question set, computed once per test session."""
    if not _cached:
        _cached.append(
            retrieval_only(load_settings(env_file=None), ("dense", "sparse", "hybrid"), subset=small).reports
        )
    return list(_cached[0])


class CannedRun:
    """Stands in for ``runner.run`` in CLI tests: same arguments, instant result built from :func:`cached_reports`.

    The LLM half is reported the way the real runner does without a model: ``pending_credentials`` when it was asked
    for and no key is set, ``not_requested`` when it was not asked for. ``calls`` records the keyword arguments.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, settings: Settings, modes: Sequence[RetrievalMode], **kwargs: Any) -> RunResult:
        self.calls.append({"modes": tuple(modes), **kwargs})
        by_mode = {r.mode: r for r in cached_reports()}
        llm = (
            skipped_llm(settings, "pending_credentials", PENDING_REASON)
            if kwargs["attempt_llm"]
            else skipped_llm(settings, "not_requested", None)
        )
        reports = [by_mode[mode].model_copy(update={"llm": llm}) for mode in modes]
        return RunResult(reports=reports, questions=[])
