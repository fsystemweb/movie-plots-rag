"""One LangSmith experiment per retrieval mode, built from results that were already computed.

The target of ``Client.evaluate`` returns the ranking the run produced for the example's question (no second
retrieval pass), and the evaluators recompute Hit@k and the reciprocal rank from the example's ``gold_movie_ids``, so
the experiment shows the same numbers as ``reports/eval_<mode>.json``. The dataset is the one uploaded by
``python -m movie_rag.eval.upload`` (created here when it does not exist yet).

Without ``LANGSMITH_API_KEY`` nothing happens and the result says so. A LangSmith outage never fails an evaluation:
the result is ``failed`` with the error type.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

from langsmith import Client
from pydantic import BaseModel

from movie_rag.config import RetrievalMode, Settings
from movie_rag.eval.metrics import expected_hit_at_k, expected_reciprocal_rank
from movie_rag.eval.questions import EvalQuestion
from movie_rag.eval.retrieval_eval import QuestionRetrieval
from movie_rag.eval.upload import make_client, upload_questions
from movie_rag.observability import run_metadata, scrub

logger = logging.getLogger(__name__)

SKIP_DETAIL = "skipped: no LANGSMITH_API_KEY"


class ExperimentResult(BaseModel):
    """What happened to the LangSmith experiment of one mode."""

    status: Literal["skipped", "uploaded", "failed"]
    detail: str
    experiment_name: str | None = None


def make_target(results: Sequence[QuestionRetrieval]) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """The experiment target: looks up the ranking computed for ``inputs["question_id"]``."""
    by_id: Mapping[str, QuestionRetrieval] = {r.question_id: r for r in results}

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        found = by_id[inputs["question_id"]]
        return {"ranked_movie_ids": found.ranked_movie_ids, "scores": found.scores, "latency_ms": found.latency_ms}

    return target


def make_evaluators(k_values: Sequence[int]) -> list[Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]]:
    """``hit_at_<k>`` for every k and ``reciprocal_rank``; unanswerable examples (no gold) score nothing."""

    def evaluator_for(k: int) -> Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]:
        def hit(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> dict[str, Any]:
            gold = reference_outputs.get("gold_movie_ids") or []
            if not gold:
                return {"key": f"hit_at_{k}", "score": None, "comment": "unanswerable: no gold film"}
            score = expected_hit_at_k(outputs["ranked_movie_ids"], gold, k, outputs.get("scores"))
            return {"key": f"hit_at_{k}", "score": score}

        return hit

    def rr(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> dict[str, Any]:
        gold = reference_outputs.get("gold_movie_ids") or []
        if not gold:
            return {"key": "reciprocal_rank", "score": None, "comment": "unanswerable: no gold film"}
        score = expected_reciprocal_rank(outputs["ranked_movie_ids"], gold, outputs.get("scores"))
        return {"key": "reciprocal_rank", "score": score}

    return [*(evaluator_for(k) for k in k_values), rr]


def run_experiment(
    settings: Settings,
    mode: RetrievalMode,
    questions: Sequence[EvalQuestion],
    results: Sequence[QuestionRetrieval],
    *,
    client: Client | None = None,
) -> ExperimentResult:
    """Create the experiment ``<mode>-...`` on the question dataset; skip cleanly without a LangSmith key."""
    if client is None and settings.langsmith_api_key is None:
        return ExperimentResult(status="skipped", detail=SKIP_DETAIL)
    try:
        ls = client or make_client(settings)
        upload_questions(settings, questions, ls)
        outcome = ls.evaluate(
            make_target(results),
            data=settings.eval.dataset_name,
            evaluators=make_evaluators(settings.eval.k_values),
            experiment_prefix=mode,
            description=f"Retrieval-only evaluation of the {mode} mode (Hit@k, reciprocal rank)",
            metadata=scrub(run_metadata(settings, retrieval_mode=mode), settings),
            max_concurrency=0,
        )
    except Exception as exc:  # LangSmith is observability: its outage must not fail an evaluation
        logger.warning("LangSmith experiment for %s failed: %s", mode, type(exc).__name__)
        return ExperimentResult(status="failed", detail=f"failed: {type(exc).__name__}")
    name = str(getattr(outcome, "experiment_name", "") or "") or None
    return ExperimentResult(status="uploaded", detail=f"experiment {name or mode} uploaded", experiment_name=name)
