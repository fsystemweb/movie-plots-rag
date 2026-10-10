"""The retrieval half of the evaluation: run every question through the MCP ``search_movies`` tool and rank the films.

The evaluator talks to the same FastMCP server the agent and the UI use, in process (``fastmcp.Client(server)``): no
``make serve`` is needed and the tool contract (arguments, filters, grouping by film) is the one under test. The
retriever behind it must be backed by a Qdrant *service*: qdrant-client's local engine ignores the prefetch queries and
filters of grouped hybrid queries, so numbers from it would be wrong; :func:`connect` refuses it.

The question set is written against the fixture films (``movie_id`` includes the CSV row index), so
:func:`ensure_fixture_indexed` makes sure the collection holds them, ingesting the fixture when it does not (ingestion
is idempotent: points that exist are skipped).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any

from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.local.qdrant_local import QdrantLocal

from movie_rag.config import RetrievalMode, Settings
from movie_rag.errors import EvalError
from movie_rag.eval.metrics import RankedQuestion, first_gold_rank
from movie_rag.eval.questions import EvalQuestion
from movie_rag.ingest.embed import Embedder, FastEmbedder
from movie_rag.ingest.index import point_id
from movie_rag.ingest.pipeline import ingest_csv
from movie_rag.mcp_server.models import SearchResult
from movie_rag.mcp_server.server import build_server
from movie_rag.retrieval import Retriever

logger = logging.getLogger(__name__)

SEARCH_TOOL = "search_movies"
LOCAL_ENGINE_MESSAGE = (
    "evaluation needs a Qdrant service, not the in-process engine: its hybrid search ignores prefetch queries and "
    "filters, so the numbers would be wrong. Start Qdrant with `make up` (QDRANT_URL, default {url})."
)


class QuestionRetrieval(BaseModel):
    """The ranked films one retrieval mode returned for one question."""

    question_id: str
    type: str
    question: str
    gold_movie_ids: list[str]
    ranked_movie_ids: list[str]
    scores: list[float]
    rank: int | None  # position of the best gold film in the order returned (metrics are tie-aware, see metrics.py)
    latency_ms: float

    def as_ranked(self) -> RankedQuestion:
        return RankedQuestion(
            question_id=self.question_id,
            type=self.type,
            gold_movie_ids=self.gold_movie_ids,
            ranked_movie_ids=self.ranked_movie_ids,
            scores=self.scores,
        )


class IndexInfo(BaseModel):
    """What the evaluation ran against."""

    collection: str
    points: int
    fixture_ingested_now: bool


def connect(settings: Settings, client: QdrantClient | None = None, *, allow_local: bool = False) -> QdrantClient:
    """A Qdrant client that is known to reach a *service* (``allow_local`` is for plumbing tests only).

    Raises :class:`EvalError` when the URL is the in-process engine, or the service does not answer.
    """
    url = settings.qdrant.url
    if client is None:
        if not url or url.startswith(":memory:"):
            raise EvalError(LOCAL_ENGINE_MESSAGE.format(url=url or "unset"))
        client = QdrantClient(url=url, timeout=settings.qdrant.timeout_s)
    if not allow_local and isinstance(getattr(client, "_client", None), QdrantLocal):
        raise EvalError(LOCAL_ENGINE_MESSAGE.format(url=url))
    try:
        client.get_collections()
    except (ResponseHandlingException, UnexpectedResponse, OSError) as exc:
        raise EvalError(f"cannot reach Qdrant at {url} ({type(exc).__name__}): start it with `make up`.") from exc
    return client


def _gold_ids(questions: Sequence[EvalQuestion]) -> list[str]:
    return sorted({g for q in questions for g in q.gold_movie_ids})


def _missing_gold(client: QdrantClient, settings: Settings, questions: Sequence[EvalQuestion]) -> list[str]:
    collection = settings.qdrant.collection
    gold = _gold_ids(questions)
    if not client.collection_exists(collection):
        return gold
    ids = {movie_id: point_id(movie_id, 0) for movie_id in gold}
    found = {str(p.id) for p in client.retrieve(collection, ids=list(ids.values()), with_payload=False)}
    return [movie_id for movie_id, pid in ids.items() if pid not in found]


def ensure_fixture_indexed(
    settings: Settings,
    client: QdrantClient,
    questions: Sequence[EvalQuestion],
    *,
    embedder: Embedder | None = None,
) -> IndexInfo:
    """Make sure every gold film is in the collection, ingesting the fixture if some are missing."""
    missing = _missing_gold(client, settings, questions)
    ingested = False
    if missing:
        logger.info("%d gold films are not indexed: ingesting the fixture", len(missing))
        ingest_csv(
            settings,
            settings.data.resolve(settings.data.fixture_path),
            client=client,
            embedder=embedder or FastEmbedder(settings.embeddings),
        )
        ingested = True
        still_missing = _missing_gold(client, settings, questions)
        if still_missing:
            raise EvalError(
                f"{len(still_missing)} gold films are still missing after ingesting the fixture "
                f"(first: {still_missing[0]}): the question set does not match {settings.data.fixture_path}."
            )
    points = client.count(settings.qdrant.collection, exact=True).count
    return IndexInfo(collection=settings.qdrant.collection, points=points, fixture_ingested_now=ingested)


def search_arguments(question: EvalQuestion, mode: RetrievalMode, depth: int) -> dict[str, Any]:
    """The ``search_movies`` arguments for ``question``: its text, its filters, the mode and the retrieval depth."""
    return {"query": question.question, "mode": mode, "top_k": depth, **question.filters.model_dump(exclude_none=True)}


def build_eval_server(settings: Settings, client: QdrantClient, embedder: Embedder | None = None) -> FastMCP:
    """The project's MCP server over ``client`` (the same tools the agent and the UI call)."""
    retriever = Retriever(settings, client=client, embedder=embedder)
    return build_server(settings, lambda: retriever)


async def run_retrieval(
    server: FastMCP, questions: Sequence[EvalQuestion], mode: RetrievalMode, depth: int
) -> list[QuestionRetrieval]:
    """Ask ``search_movies`` every question in ``mode`` and rank the films returned.

    One untimed warm-up call loads the embedding models first, so that latency measures searches, not model loading.
    Latency is the wall time of the in-process tool call (query embedding, Qdrant, serialisation).
    A failing tool call aborts the run: partial numbers would be silently wrong.
    """
    results: list[QuestionRetrieval] = []
    async with Client(server) as mcp:
        if questions:
            await _search(mcp, search_arguments(questions[0], mode, depth))
        for question in questions:
            started = time.perf_counter()
            found = await _search(mcp, search_arguments(question, mode, depth))
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            ranked = [hit.movie_id for hit in found.results]
            results.append(
                QuestionRetrieval(
                    question_id=question.id,
                    type=question.type,
                    question=question.question,
                    gold_movie_ids=question.gold_movie_ids,
                    ranked_movie_ids=ranked,
                    scores=[hit.score for hit in found.results],
                    rank=first_gold_rank(ranked, question.gold_movie_ids),
                    latency_ms=latency_ms,
                )
            )
    return results


async def _search(mcp: Client[Any], arguments: dict[str, Any]) -> SearchResult:
    try:
        result = await mcp.call_tool(SEARCH_TOOL, arguments)
    except ToolError as exc:
        raise EvalError(f"{SEARCH_TOOL} failed for {arguments['query']!r}: {exc}") from exc
    if result.structured_content is None:
        raise EvalError(f"{SEARCH_TOOL} returned no structured content")
    return SearchResult.model_validate(result.structured_content)
