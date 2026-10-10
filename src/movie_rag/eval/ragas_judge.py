"""RAGAS wiring: faithfulness, response relevancy, context precision and context recall, judged by a second model.

* **Judge != generator.** The judge is ``llm.judge_model`` (default ``openai/gpt-oss-120b``), the generator is
  ``llm.chat_model``; :func:`ensure_judge_differs` refuses to run when they are the same model, because a model grading
  its own answers inflates the scores.
* **Pinned API.** ragas 0.4.x: the ``ragas.metrics.collections`` metrics with ``llm_factory`` (instructor-based, over
  the OpenAI-compatible Nebius endpoint) and a ``BaseRagasEmbedding``. Docs:
  https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/ . The version in use is recorded in every report.
* **Local embeddings.** Response relevancy needs an embedding model; the project's own FastEmbed dense model is used,
  so no second credential or paid endpoint is involved.
* **Lazy and optional.** ragas is the ``eval`` extra and is imported only when a metric is built (``import ragas``
  costs seconds and the retrieval half of the evaluation does not need it); :func:`require_ragas` explains how to
  install it when it is missing. :func:`install_ragas_compat` first works around ragas 0.4.3 importing a module that
  ``langchain-community`` 0.4 removed (see its docstring).
* **Failure isolation.** A metric that raises or returns NaN for one sample is recorded as failed for that sample; the
  other samples and metrics still count, and the report says how many were scored.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.metadata
import logging
import math
import os
import sys
import types
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from movie_rag.config import Settings
from movie_rag.errors import EvalError
from movie_rag.ingest.embed import Embedder

logger = logging.getLogger(__name__)

METRIC_NAMES = ("faithfulness", "response_relevancy", "context_precision", "context_recall")
_VERTEXAI_MODULE = "langchain_community.chat_models.vertexai"


class RagasSample(BaseModel):
    """One answered question as RAGAS sees it."""

    question_id: str
    user_input: str
    response: str
    retrieved_contexts: list[str]
    reference: str


class SampleScores(BaseModel):
    """The four scores of one sample (``None``: the metric could not be computed, see ``errors``)."""

    question_id: str
    scores: dict[str, float | None]
    errors: dict[str, str]


class MetricSummary(BaseModel):
    """Mean of one metric over the samples it could score."""

    mean: float | None
    n_scored: int
    n_failed: int


Scorer = Callable[[RagasSample], Awaitable[Any]]


def ragas_version() -> str:
    """The installed ragas version (read from package metadata, so ragas itself is not imported)."""
    try:
        return importlib.metadata.version("ragas")
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def install_ragas_compat() -> None:
    """Make ``import ragas`` work with ``langchain-community`` >= 0.4 and switch ragas' telemetry off.

    ragas 0.4.3 (the latest release) does ``from langchain_community.chat_models.vertexai import ChatVertexAI`` at
    import time; ``langchain-community`` 0.4 removed that module (LangChain 1.x needs community >= 0.4, so it cannot
    be pinned back). ragas only lists the class among the models that support multiple completions, so an inert
    stand-in class is registered when the real module is missing. Remove this once ragas drops the import.
    """
    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
    if _VERTEXAI_MODULE in sys.modules:
        return
    try:
        importlib.import_module(_VERTEXAI_MODULE)
    except ImportError:
        stub = types.ModuleType(_VERTEXAI_MODULE)
        stub.ChatVertexAI = type("ChatVertexAI", (), {})  # type: ignore[attr-defined]
        sys.modules[_VERTEXAI_MODULE] = stub
        logger.debug("registered a stand-in for %s (ragas compatibility)", _VERTEXAI_MODULE)


def require_ragas() -> None:
    """Apply the compatibility step and make sure ragas (the optional ``eval`` extra) can be imported.

    Raises :class:`EvalError` with the command to install it when it is missing.
    """
    install_ragas_compat()
    try:
        importlib.import_module("ragas")
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] != "ragas":
            raise
        raise EvalError(
            "the RAGAS metrics need the optional `eval` extra, which is not installed: run `uv sync --extra eval` "
            "(or `make setup`, which installs all extras)."
        ) from exc


def ensure_judge_differs(settings: Settings) -> None:
    """Raise :class:`EvalError` unless the judge model differs from the chat (generator) model."""
    judge, chat = settings.llm.judge_model.strip(), settings.llm.chat_model.strip()
    if judge.casefold() == chat.casefold():
        raise EvalError(
            f"llm.judge_model and llm.chat_model are both {chat!r}: the judge must be a different model than the "
            "generator (set llm.judge_model in config.yaml or LLM__JUDGE_MODEL)."
        )


def make_judge(settings: Settings) -> Any:
    """The RAGAS judge LLM over the Nebius endpoint. Needs ``NEBIUS_API_KEY`` (``MissingCredentialError`` otherwise)."""
    ensure_judge_differs(settings)
    api_key = settings.require_nebius_api_key()
    require_ragas()
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    client = AsyncOpenAI(
        base_url=settings.llm.base_url, api_key=api_key.get_secret_value(), timeout=settings.eval.judge_timeout_s
    )
    return llm_factory(settings.llm.judge_model, client=client, temperature=settings.llm.temperature)


def make_embeddings(embedder: Embedder) -> Any:
    """A RAGAS embedding model backed by the project's own (local) dense embedder."""
    require_ragas()
    from ragas.embeddings.base import BaseRagasEmbedding

    class LocalDenseEmbeddings(BaseRagasEmbedding):
        def embed_text(self, text: str, **kwargs: Any) -> list[float]:
            return embedder.embed_dense_query(text)

        async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
            return await asyncio.to_thread(embedder.embed_dense_query, text)

    return LocalDenseEmbeddings()


def build_scorers(llm: Any, embeddings: Any, *, strictness: int) -> dict[str, Scorer]:
    """The four metrics as ``{name: async scorer(sample)}`` (``llm`` and ``embeddings`` from the factories above)."""
    require_ragas()
    from ragas.metrics.collections import AnswerRelevancy, ContextPrecisionWithReference, ContextRecall, Faithfulness

    faithfulness = Faithfulness(llm=llm)
    relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings, strictness=strictness)
    precision = ContextPrecisionWithReference(llm=llm)
    recall = ContextRecall(llm=llm)
    return {
        "faithfulness": lambda s: faithfulness.ascore(
            user_input=s.user_input, response=s.response, retrieved_contexts=s.retrieved_contexts
        ),
        "response_relevancy": lambda s: relevancy.ascore(user_input=s.user_input, response=s.response),
        "context_precision": lambda s: precision.ascore(
            user_input=s.user_input, reference=s.reference, retrieved_contexts=s.retrieved_contexts
        ),
        "context_recall": lambda s: recall.ascore(
            user_input=s.user_input, retrieved_contexts=s.retrieved_contexts, reference=s.reference
        ),
    }


def _value_of(result: Any) -> float:
    value = float(getattr(result, "value", result))
    if math.isnan(value):
        raise ValueError("the metric returned NaN (nothing to score)")
    return value


async def score_samples(
    samples: Sequence[RagasSample], scorers: Mapping[str, Scorer], *, concurrency: int
) -> list[SampleScores]:
    """Score every sample with every metric, ``concurrency`` calls at a time; failures are recorded, not raised."""
    gate = asyncio.Semaphore(concurrency)

    async def one(sample: RagasSample, name: str) -> tuple[float | None, str | None]:
        async with gate:
            try:
                return _value_of(await scorers[name](sample)), None
            except Exception as exc:  # a judge outage or malformed output must not discard the other scores
                logger.warning("ragas %s failed for %s: %s", name, sample.question_id, type(exc).__name__)
                return (
                    None,
                    f"{type(exc).__name__}: {str(exc).strip().splitlines()[0][:200] if str(exc).strip() else ''}",
                )

    names = list(scorers)
    outcomes = await asyncio.gather(*(one(s, n) for s in samples for n in names))
    results: list[SampleScores] = []
    for index, sample in enumerate(samples):
        chunk = outcomes[index * len(names) : (index + 1) * len(names)]
        results.append(
            SampleScores(
                question_id=sample.question_id,
                scores={n: value for n, (value, _) in zip(names, chunk, strict=True)},
                errors={n: err for n, (_, err) in zip(names, chunk, strict=True) if err},
            )
        )
    return results


def summarise_scores(results: Sequence[SampleScores], names: Sequence[str] = METRIC_NAMES) -> dict[str, MetricSummary]:
    """Mean per metric over the samples it scored, with how many were scored and how many failed."""
    summary: dict[str, MetricSummary] = {}
    for name in names:
        values = [r.scores[name] for r in results if r.scores.get(name) is not None]
        scored = [v for v in values if v is not None]
        summary[name] = MetricSummary(
            mean=sum(scored) / len(scored) if scored else None,
            n_scored=len(scored),
            n_failed=len(results) - len(scored),
        )
    return summary
