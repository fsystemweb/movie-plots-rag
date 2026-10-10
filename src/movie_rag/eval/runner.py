"""Run the evaluation: retrieval metrics for each mode, the LLM half when a key exists, one report per mode.

Order of work for a run: load and validate the question set, connect to the Qdrant service (refusing the in-process
engine), make sure the fixture films are indexed, then per mode: rank every question through the MCP ``search_movies``
tool, compute Hit@k / MRR / latency, optionally run the agent and RAGAS, optionally record a LangSmith experiment.

The LLM half is *attempted* when asked for (``attempt_llm``). Without ``NEBIUS_API_KEY`` it is recorded as
``pending_credentials`` and the run still succeeds: the retrieval numbers do not depend on it.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from fastmcp import FastMCP
from langchain_core.language_models import BaseChatModel
from langsmith import Client
from qdrant_client import QdrantClient

from movie_rag.config import RetrievalMode, Settings
from movie_rag.errors import EvalError, MovieRagError
from movie_rag.eval.agent_eval import agent_metrics, build_samples, run_agent
from movie_rag.eval.langsmith_experiment import ExperimentResult, run_experiment
from movie_rag.eval.metrics import retrieval_metrics, retrieval_metrics_by_type, summarise
from movie_rag.eval.questions import EvalQuestion, fixture_records, load_eval_set
from movie_rag.eval.ragas_judge import (
    Scorer,
    build_scorers,
    ensure_judge_differs,
    make_embeddings,
    make_judge,
    ragas_version,
    score_samples,
    summarise_scores,
)
from movie_rag.eval.report import EvalReport, LlmSection, QuestionSetInfo, RetrievalSection, utc_now
from movie_rag.eval.retrieval_eval import (
    IndexInfo,
    build_eval_server,
    connect,
    ensure_fixture_indexed,
    run_retrieval,
)
from movie_rag.ingest.embed import Embedder, FastEmbedder
from movie_rag.observability import git_sha, scrub

logger = logging.getLogger(__name__)

MODES: tuple[RetrievalMode, ...] = ("dense", "sparse", "hybrid")
PENDING_REASON = "set NEBIUS_API_KEY — see docs/CREDENTIALS.md"

ScorerFactory = Callable[[Settings, Embedder], dict[str, Scorer]]


def default_scorers(settings: Settings, embedder: Embedder) -> dict[str, Scorer]:
    """The four RAGAS metrics with the Nebius judge and the local embedder."""
    return build_scorers(make_judge(settings), make_embeddings(embedder), strictness=settings.eval.relevancy_strictness)


@dataclass
class Dependencies:
    """Everything a run needs from the outside; the defaults are the production ones, tests inject fakes."""

    client: QdrantClient | None = None
    embedder: Embedder | None = None
    chat_model: BaseChatModel | None = None
    scorer_factory: ScorerFactory = default_scorers
    langsmith_client: Client | None = None
    allow_local_qdrant: bool = False  # tests only: the in-process engine cannot do hybrid search


def select_per_type(questions: Sequence[EvalQuestion], per_type: int) -> list[EvalQuestion]:
    """The first ``per_type`` questions of every type, in file order (a small, stable sample for the smoke run)."""
    seen: Counter[str] = Counter()
    chosen: list[EvalQuestion] = []
    for q in questions:
        if seen[q.type] < per_type:
            seen[q.type] += 1
            chosen.append(q)
    return chosen


def question_set_info(settings: Settings, questions: Sequence[EvalQuestion]) -> QuestionSetInfo:
    return QuestionSetInfo(
        path=str(settings.eval.questions_path),
        n=len(questions),
        per_type=dict(sorted(Counter(q.type for q in questions).items())),
    )


def skipped_llm(settings: Settings, status: str, reason: str | None) -> LlmSection:
    """An LLM section without numbers (``pending_credentials`` or ``not_requested``), versions still recorded."""
    return LlmSection.model_validate(
        {
            "status": status,
            "reason": reason,
            "chat_model": settings.llm.chat_model,
            "judge_model": settings.llm.judge_model,
            "ragas_version": ragas_version(),
        }
    )


async def run_llm_half(
    settings: Settings,
    server: FastMCP,
    questions: Sequence[EvalQuestion],
    mode: RetrievalMode,
    deps: Dependencies,
    embedder: Embedder,
) -> LlmSection:
    """The agent answers the questions, RAGAS judges the answers. Judge == generator is refused before any call."""
    ensure_judge_differs(settings)
    records = {r.movie_id: r for r in fixture_records(settings)}
    try:
        scorers = deps.scorer_factory(settings, embedder)  # before the agent runs: a missing extra fails fast
        outcomes = await run_agent(settings, server, questions, mode, model=deps.chat_model)
        samples = build_samples(questions, outcomes, records)
        scored = await score_samples(samples, scorers, concurrency=settings.eval.ragas_concurrency)
    except MovieRagError:
        raise
    except Exception as exc:  # judge or model outage: keep the retrieval numbers, mark this half failed
        logger.warning("the LLM half failed for %s: %s", mode, type(exc).__name__)
        return skipped_llm(settings, "failed", str(scrub(f"{type(exc).__name__}: {exc}"[:300], settings)))
    metrics = agent_metrics(questions, outcomes)
    return LlmSection(
        status="completed",
        reason=None if metrics.n_errors == 0 else f"{metrics.n_errors} question(s) raised an error and are excluded",
        chat_model=settings.llm.chat_model,
        judge_model=settings.llm.judge_model,
        ragas_version=ragas_version(),
        agent=metrics,
        ragas=summarise_scores(scored),
        agent_per_question=outcomes,
        ragas_per_question=scored,
    )


@dataclass
class RunResult:
    """Reports of a run plus the questions they cover."""

    reports: list[EvalReport]
    questions: list[EvalQuestion]
    failed: bool = False


async def evaluate(
    settings: Settings,
    modes: Sequence[RetrievalMode],
    *,
    attempt_llm: bool,
    llm_questions_per_type: int | None = None,
    experiments: bool = True,
    deps: Dependencies | None = None,
    subset: Callable[[Sequence[EvalQuestion]], Sequence[EvalQuestion]] | None = None,
) -> RunResult:
    """Evaluate ``modes``. ``llm_questions_per_type`` limits the LLM half to a sample (smoke run).

    ``subset`` narrows the validated question set before anything runs (tests use it to stay fast; the CLI never does).
    """
    deps = deps or Dependencies()
    questions = list(load_eval_set(settings))
    if subset is not None:
        questions = list(subset(questions))
    client = connect(settings, deps.client, allow_local=deps.allow_local_qdrant)
    embedder = deps.embedder or FastEmbedder(settings.embeddings)
    index = ensure_fixture_indexed(settings, client, questions, embedder=embedder)
    server = build_eval_server(settings, client, embedder)
    depth = max(settings.eval.k_values)
    if depth > settings.mcp.max_top_k:
        raise EvalError(f"eval.k_values reach {depth} but mcp.max_top_k is {settings.mcp.max_top_k}")
    result = RunResult(reports=[], questions=questions)
    for mode in modes:
        report = await _evaluate_mode(
            settings,
            server,
            questions,
            mode,
            depth,
            index,
            deps,
            embedder,
            attempt_llm,
            llm_questions_per_type,
            experiments,
        )
        result.failed = result.failed or report.llm.status == "failed"
        result.reports.append(report)
    return result


async def _evaluate_mode(
    settings: Settings,
    server: FastMCP,
    questions: list[EvalQuestion],
    mode: RetrievalMode,
    depth: int,
    index: IndexInfo,
    deps: Dependencies,
    embedder: Embedder,
    attempt_llm: bool,
    llm_questions_per_type: int | None,
    experiments: bool,
) -> EvalReport:
    ks = settings.eval.k_values
    found = await run_retrieval(server, questions, mode, depth)
    ranked = [f.as_ranked() for f in found]
    if not attempt_llm:
        llm = skipped_llm(settings, "not_requested", None)
    elif settings.nebius_api_key is None:
        logger.info("LLM half skipped for %s: %s", mode, PENDING_REASON)
        llm = skipped_llm(settings, "pending_credentials", PENDING_REASON)
    else:
        sample = select_per_type(questions, llm_questions_per_type) if llm_questions_per_type else questions
        llm = await run_llm_half(settings, server, sample, mode, deps, embedder)
    langsmith = (
        run_experiment(settings, mode, questions, found, client=deps.langsmith_client)
        if experiments
        else ExperimentResult(status="skipped", detail="not requested")
    )
    return EvalReport(
        mode=mode,
        generated_at=utc_now(),
        git_sha=git_sha(settings.observability.git_timeout_s),
        config_hash=settings.config_hash(),
        embedding_model=settings.embeddings.dense_model,
        sparse_model=settings.embeddings.sparse_model,
        question_set=question_set_info(settings, questions),
        index=index,
        retrieval=RetrievalSection(
            k_values=list(ks),
            depth=depth,
            overall=retrieval_metrics(ranked, ks),
            by_type=retrieval_metrics_by_type(ranked, ks),
            latency_ms=summarise([f.latency_ms for f in found]),
            per_question=found,
        ),
        llm=llm,
        langsmith=langsmith,
    )


def run(settings: Settings, modes: Sequence[RetrievalMode], **kwargs: Any) -> RunResult:
    """Synchronous entry point for the CLI."""
    return asyncio.run(evaluate(settings, modes, **kwargs))
