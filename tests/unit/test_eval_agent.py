"""The LLM half's agent run: a scripted chat model over the project's MCP server (in memory), no key, no network."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatResult

from fakes import FakeEmbedder, ScriptedChatModel, tool_call_message
from index_fixture import Index, premise
from movie_rag.config import Settings
from movie_rag.errors import MissingCredentialError
from movie_rag.eval.agent_eval import (
    AgentOutcome,
    agent_metrics,
    build_samples,
    reference_text,
    run_agent,
)
from movie_rag.eval.questions import EvalQuestion
from movie_rag.eval.retrieval_eval import build_eval_server
from movie_rag.retrieval import SearchFilters

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")


@pytest.fixture(scope="module")
def index() -> Index:
    return Index()


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings()


def fuzzy(qid: str, film: Any) -> EvalQuestion:
    return EvalQuestion(id=qid, type="fuzzy_plot", question=premise(film), gold_movie_ids=[film.movie_id])


def unanswerable(qid: str) -> EvalQuestion:
    return EvalQuestion(
        id=qid, type="unanswerable", question="a film about a submarine opera singer", absent_title="Deep Aria"
    )


class ExplodingModel(ScriptedChatModel):
    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise RuntimeError("model unavailable")


async def test_the_agent_answers_every_question_with_the_mode_pinned_and_citations_from_tool_results(
    index: Index, settings: Settings
) -> None:
    film = index.records[3]
    questions = [fuzzy("fuzzy-01", film), unanswerable("unanswerable-01")]
    search = ("search_movies", {"query": questions[0].question, "mode": "hybrid"})  # the model asks for hybrid
    model = ScriptedChatModel(
        replies=[
            tool_call_message(search, tokens=100),
            AIMessage(content=f"{film.title} ({film.release_year}) fits."),
            tool_call_message(("search_movies", {"query": questions[1].question}), tokens=10),
            AIMessage(content="Nothing in the index fits."),
        ]
    )
    server = build_eval_server(index.settings, index.client, FakeEmbedder())

    outcomes = await run_agent(index.settings, server, questions, "dense", model=model)

    first, second = outcomes
    assert (first.question_id, first.type, first.error) == ("fuzzy-01", "fuzzy_plot", None)
    assert first.cited_movie_ids == [film.movie_id] and first.abstained is False
    assert first.tool_calls == 1 and first.total_tokens > 100 and first.latency_ms > 0
    assert first.retrieved_contexts and film.title in first.retrieved_contexts[0]
    assert second.abstained is True and second.cited_movie_ids == []
    tool_results = [m.text for prompt in model.prompts for m in prompt if isinstance(m, ToolMessage)]
    assert tool_results and all('"mode":"dense"' in t.replace(" ", "") for t in tool_results)


async def test_a_failing_question_is_recorded_and_the_run_goes_on(index: Index, settings: Settings) -> None:
    questions = [fuzzy("fuzzy-01", index.records[3]), unanswerable("unanswerable-01")]
    server = build_eval_server(index.settings, index.client, FakeEmbedder())
    outcomes = await run_agent(
        index.settings, server, questions, "dense", model=ExplodingModel(replies=[AIMessage(content="x")])
    )
    assert [o.error for o in outcomes] == ["RuntimeError", "RuntimeError"]
    assert all(o.answer == "" and o.cited_movie_ids == [] for o in outcomes)


async def test_a_missing_key_is_not_swallowed_per_question(index: Index) -> None:
    server = build_eval_server(index.settings, index.client, FakeEmbedder())
    with pytest.raises(MissingCredentialError, match="NEBIUS_API_KEY"):
        await run_agent(index.settings, server, [unanswerable("unanswerable-01")], "dense")


# --- metrics -------------------------------------------------------------------------------------------------------


def outcome(qid: str, type_: str, **kwargs: Any) -> AgentOutcome:
    return AgentOutcome(question_id=qid, type=type_, **kwargs)


def test_agent_metrics_on_a_hand_computed_run() -> None:
    questions = [
        EvalQuestion(id="fuzzy-01", type="fuzzy_plot", question="a long enough question", gold_movie_ids=["g1"]),
        EvalQuestion(id="exact-01", type="exact_entity", question="a long enough question", gold_movie_ids=["g2"]),
        EvalQuestion(
            id="filtered-01",
            type="filtered",
            question="a long enough question",
            gold_movie_ids=["g3"],
            filters=SearchFilters(year_from=1990),
        ),
        EvalQuestion(id="unanswerable-01", type="unanswerable", question="a long enough question", absent_title="A"),
        EvalQuestion(id="unanswerable-02", type="unanswerable", question="a long enough question", absent_title="B"),
    ]
    outcomes = [
        outcome("fuzzy-01", "fuzzy_plot", cited_movie_ids=["g1", "x"], total_tokens=100, latency_ms=1000),
        outcome("exact-01", "exact_entity", abstained=True, total_tokens=50, latency_ms=2000),
        outcome("filtered-01", "filtered", error="RuntimeError"),
        outcome("unanswerable-01", "unanswerable", abstained=True, total_tokens=10, latency_ms=3000),
        outcome("unanswerable-02", "unanswerable", cited_movie_ids=["x"], total_tokens=20, latency_ms=4000),
    ]
    metrics = agent_metrics(questions, outcomes)
    assert (metrics.n_questions, metrics.n_errors, metrics.n_answerable_answered) == (5, 1, 2)
    assert metrics.citation_hit_rate == 0.5  # g1 cited, g2 not (abstained)
    assert metrics.abstention.correct_abstention_rate == 0.5  # 1 of 2 unanswerable questions abstained
    assert metrics.abstention.false_abstention_rate == 0.5  # 1 of 2 answered answerable questions abstained
    assert metrics.latency_ms.p50 == 2500.0 and metrics.tokens_per_question.p50 == 35.0


def test_agent_metrics_without_answerable_questions_have_no_hit_rate() -> None:
    question = EvalQuestion(
        id="unanswerable-01", type="unanswerable", question="a long enough question", absent_title="A"
    )
    metrics = agent_metrics([question], [outcome("unanswerable-01", "unanswerable", abstained=True)])
    assert metrics.citation_hit_rate is None and metrics.abstention.correct_abstention_rate == 1.0


def test_ragas_samples_cover_answered_answerable_questions_with_the_gold_plot_as_reference(index: Index) -> None:
    film, other = index.records[3], index.records[4]
    questions = [
        fuzzy("fuzzy-01", film),
        fuzzy("fuzzy-02", other),
        fuzzy("fuzzy-03", index.records[5]),
        unanswerable("unanswerable-01"),
    ]
    outcomes = [
        outcome("fuzzy-01", "fuzzy_plot", answer="It is X.", retrieved_contexts=["X (1999) [id]: snippet"]),
        outcome("fuzzy-02", "fuzzy_plot", error="RuntimeError"),
        outcome("fuzzy-03", "fuzzy_plot", answer="Y"),
        outcome("unanswerable-01", "unanswerable", abstained=True),
    ]
    records = {r.movie_id: r for r in index.records if r.movie_id != index.records[5].movie_id}
    [only] = build_samples(questions, outcomes, records)  # errored, unanswerable and unknown-gold are left out
    assert only.question_id == "fuzzy-01" and only.response == "It is X."
    assert only.retrieved_contexts == ["X (1999) [id]: snippet"]
    assert only.reference == reference_text(film) and film.plot in only.reference
