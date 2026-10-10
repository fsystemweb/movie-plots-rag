"""The LLM half of the evaluation: the agent answers every question, then RAGAS judges the answers.

The agent gets the same tools as in production, served in process by the project's FastMCP server (no ``make serve``),
with the retrieval mode pinned to the mode under evaluation so that the three modes are compared on the same agent.

Deterministic numbers computed from the agent's behaviour (no judge involved):

* **abstention**: the agent abstains when its answer cites no retrieved film. On the unanswerable questions that is the
  right behaviour (``correct_abstention_rate``, higher is better); on answerable questions it is a miss
  (``false_abstention_rate``, lower is better);
* **citation hit rate**: on answerable questions, the share whose gold film is among the cited films;
* latency and tokens per question (p50 / p95 / mean).

A question whose run raised (model or network error) is recorded with its error and left out of every rate: a failed
call is neither an abstention nor an answer.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

from fastmcp import Client, FastMCP
from langchain_core.language_models import BaseChatModel
from langchain_mcp_adapters.tools import load_mcp_tools as load_session_tools
from pydantic import BaseModel

from movie_rag.agent import MovieAgent
from movie_rag.agent.agent import mode_interceptor
from movie_rag.agent.models import Answer
from movie_rag.config import RetrievalMode, Settings
from movie_rag.errors import MissingCredentialError
from movie_rag.eval.metrics import Abstention, AbstentionMetrics, Summary, abstention_metrics, summarise
from movie_rag.eval.questions import EvalQuestion
from movie_rag.eval.ragas_judge import RagasSample
from movie_rag.ingest.clean import MovieRecord

logger = logging.getLogger(__name__)


class AgentOutcome(BaseModel):
    """What the agent did with one question."""

    question_id: str
    type: str
    answer: str = ""
    abstained: bool = False
    cited_movie_ids: list[str] = []
    retrieved_contexts: list[str] = []
    tool_calls: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    error: str | None = None


class AgentMetrics(BaseModel):
    """Deterministic metrics of one agent run over the question set."""

    n_questions: int
    n_errors: int
    abstention: AbstentionMetrics
    citation_hit_rate: float | None
    n_answerable_answered: int
    latency_ms: Summary
    tokens_per_question: Summary


def outcome_from_answer(question: EvalQuestion, answer: Answer) -> AgentOutcome:
    """The evaluation record of an agent :class:`Answer` (contexts are what the tools showed the agent)."""
    return AgentOutcome(
        question_id=question.id,
        type=question.type,
        answer=answer.text,
        abstained=answer.abstained,
        cited_movie_ids=[c.movie_id for c in answer.citations],
        retrieved_contexts=[
            f"{film.label} [{film.movie_id}]: {film.snippet}" for film in answer.retrieved if film.snippet
        ],
        tool_calls=len(answer.tool_calls),
        total_tokens=answer.usage.total_tokens,
        latency_ms=answer.latency_ms,
    )


async def run_agent(
    settings: Settings,
    server: FastMCP,
    questions: Sequence[EvalQuestion],
    mode: RetrievalMode,
    *,
    model: BaseChatModel | None = None,
) -> list[AgentOutcome]:
    """Ask the agent every question with the retrieval mode pinned to ``mode``.

    ``model`` replaces the Nebius chat model (tests). ``MissingCredentialError`` propagates; any other failure of a
    single question is recorded on its outcome.
    """
    outcomes: list[AgentOutcome] = []
    async with Client(server) as mcp:
        tools = await load_session_tools(mcp.session, tool_interceptors=[mode_interceptor(mode)])
        agent = MovieAgent(settings, model=model, tools=tools)
        for question in questions:
            try:
                answer = await agent.ask(question.question, mode=mode)
            except MissingCredentialError:
                raise
            except Exception as exc:  # one failed call (rate limit, timeout) must not discard the whole run
                logger.warning("agent failed on %s: %s", question.id, type(exc).__name__)
                outcomes.append(AgentOutcome(question_id=question.id, type=question.type, error=type(exc).__name__))
                continue
            outcomes.append(outcome_from_answer(question, answer))
    return outcomes


def agent_metrics(questions: Sequence[EvalQuestion], outcomes: Sequence[AgentOutcome]) -> AgentMetrics:
    """Abstention, citation hit rate, latency and tokens over the questions the agent answered without error."""
    by_id: Mapping[str, EvalQuestion] = {q.id: q for q in questions}
    ok = [o for o in outcomes if o.error is None]
    answerable = [o for o in ok if by_id[o.question_id].gold_movie_ids]
    hits = sum(bool(set(by_id[o.question_id].gold_movie_ids) & set(o.cited_movie_ids)) for o in answerable)
    return AgentMetrics(
        n_questions=len(outcomes),
        n_errors=len(outcomes) - len(ok),
        abstention=abstention_metrics(
            Abstention(should_abstain=not by_id[o.question_id].gold_movie_ids, abstained=o.abstained) for o in ok
        ),
        citation_hit_rate=hits / len(answerable) if answerable else None,
        n_answerable_answered=len(answerable),
        latency_ms=summarise([o.latency_ms for o in ok]),
        tokens_per_question=summarise([float(o.total_tokens) for o in ok]),
    )


def reference_text(record: MovieRecord) -> str:
    """The ground truth RAGAS compares against: the gold film's title, year and whole plot."""
    return f"{record.title} ({record.release_year}). {record.plot}"


def build_samples(
    questions: Sequence[EvalQuestion], outcomes: Sequence[AgentOutcome], records: Mapping[str, MovieRecord]
) -> list[RagasSample]:
    """RAGAS samples for the answerable questions the agent answered without error.

    Unanswerable questions have no reference, so context precision/recall are undefined for them; they are measured by
    abstention instead.
    """
    by_id: Mapping[str, EvalQuestion] = {q.id: q for q in questions}
    samples: list[RagasSample] = []
    for outcome in outcomes:
        question = by_id[outcome.question_id]
        if outcome.error is not None or not question.gold_movie_ids:
            continue
        gold = records.get(question.gold_movie_ids[0])
        if gold is None:
            continue
        samples.append(
            RagasSample(
                question_id=question.id,
                user_input=question.question,
                response=outcome.answer,
                retrieved_contexts=outcome.retrieved_contexts,
                reference=reference_text(gold),
            )
        )
    return samples
