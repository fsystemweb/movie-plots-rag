"""The evaluation runner end to end on the in-memory engine, the fake embedder and a scripted agent model."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from eval_support import fake_deps, retrieval_only
from fakes import ScriptedChatModel
from movie_rag.config import Settings
from movie_rag.errors import EvalError
from movie_rag.eval.questions import EvalQuestion, load_eval_set
from movie_rag.eval.ragas_judge import RagasSample
from movie_rag.eval.runner import PENDING_REASON, evaluate, run, select_per_type

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings()


async def constant_score(_: RagasSample) -> float:
    return 0.8


def fake_scorers(*_: Any) -> dict[str, Any]:
    return dict.fromkeys(("faithfulness", "response_relevancy", "context_precision", "context_recall"), constant_score)


def test_a_retrieval_only_run_measures_every_mode_and_records_what_it_ran_against(settings: Settings) -> None:
    result = retrieval_only(settings, fresh=True)
    assert [r.mode for r in result.reports] == ["dense", "sparse"]
    dense = result.reports[0]
    assert dense.question_set.n == 40 and dense.question_set.per_type == {
        "exact_entity": 10,
        "filtered": 10,
        "fuzzy_plot": 10,
        "unanswerable": 10,
    }
    assert dense.retrieval.overall.n == 30  # the unanswerable questions have no gold film
    assert dense.retrieval.depth == max(settings.eval.k_values) and dense.retrieval.k_values == [1, 3, 5, 8]
    assert set(dense.retrieval.by_type) == {"exact_entity", "filtered", "fuzzy_plot"}
    assert len(dense.retrieval.per_question) == 40
    assert dense.retrieval.latency_ms.n == 40 and dense.retrieval.latency_ms.p95 is not None
    assert dense.index.fixture_ingested_now is True and dense.index.points > 0
    assert result.reports[1].index == dense.index  # one index for the whole run
    assert dense.embedding_model == settings.embeddings.dense_model
    assert dense.config_hash == settings.config_hash()
    assert dense.llm.status == "not_requested" and dense.langsmith.status == "skipped"


def test_without_a_key_the_llm_half_is_pending_credentials_and_the_run_succeeds(settings: Settings) -> None:
    result = run(settings, ("dense",), attempt_llm=True, deps=fake_deps())
    [report] = result.reports
    assert result.failed is False
    assert report.llm.status == "pending_credentials" and report.llm.reason == PENDING_REASON
    assert report.llm.agent is None and report.llm.ragas is None
    assert report.llm.judge_model == settings.llm.judge_model != report.llm.chat_model == settings.llm.chat_model
    assert report.llm.ragas_version.startswith("0.4")
    assert report.retrieval.overall.n == 30  # the retrieval numbers do not depend on the key


def test_with_a_key_the_agent_answers_and_ragas_scores_the_answers(make_settings: Callable[..., Settings]) -> None:
    settings = make_settings(NEBIUS_API_KEY="test-key-not-real")
    model = ScriptedChatModel(replies=[AIMessage(content="No film in the index fits that.")])
    deps = fake_deps(chat_model=model, scorer_factory=fake_scorers)
    result = run(settings, ("dense",), attempt_llm=True, llm_questions_per_type=1, experiments=False, deps=deps)
    [report] = result.reports
    llm = report.llm
    assert llm.status == "completed" and llm.reason is None and result.failed is False
    assert llm.agent is not None and llm.agent.n_questions == 4  # one question of each type
    assert llm.agent.abstention.correct_abstention_rate == 1.0  # the scripted model never cites a film
    assert llm.agent.abstention.false_abstention_rate == 1.0
    assert llm.agent.citation_hit_rate == 0.0
    assert llm.ragas is not None
    assert {n: s.mean for n, s in llm.ragas.items()} == pytest.approx(dict.fromkeys(llm.ragas, 0.8))
    assert all(s.n_scored == 3 and s.n_failed == 0 for s in llm.ragas.values())  # unanswerable has no reference
    assert llm.agent_per_question is not None and len(llm.agent_per_question) == 4
    assert llm.ragas_per_question is not None and len(llm.ragas_per_question) == 3
    assert report.retrieval.overall.n == 30  # retrieval always covers the whole set


def test_the_judge_may_not_be_the_generator_even_with_a_key(make_settings: Callable[..., Settings]) -> None:
    settings = make_settings(NEBIUS_API_KEY="test-key-not-real", LLM__JUDGE_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507")
    model = ScriptedChatModel(replies=[AIMessage(content="x")])
    with pytest.raises(EvalError, match="different model"):
        run(settings, ("dense",), attempt_llm=True, deps=fake_deps(chat_model=model, scorer_factory=fake_scorers))
    assert model.prompts == []  # nothing was asked of the model


def test_a_judge_outage_marks_the_llm_half_failed_but_keeps_the_retrieval_numbers(
    make_settings: Callable[..., Settings],
) -> None:
    settings = make_settings(NEBIUS_API_KEY="test-key-not-real")

    def broken(*_: Any) -> dict[str, Any]:
        raise RuntimeError("judge exploded with test-key-not-real")

    deps = fake_deps(chat_model=ScriptedChatModel(replies=[AIMessage(content="none")]), scorer_factory=broken)
    result = run(settings, ("dense",), attempt_llm=True, llm_questions_per_type=1, experiments=False, deps=deps)
    [report] = result.reports
    assert result.failed is True and report.llm.status == "failed"
    assert report.llm.reason is not None and "RuntimeError" in report.llm.reason
    assert "test-key-not-real" not in report.llm.reason
    assert report.retrieval.overall.n == 30


def test_a_langsmith_experiment_is_recorded_per_mode_when_a_key_exists(
    make_settings: Callable[..., Settings],
) -> None:
    settings = make_settings(LANGSMITH_API_KEY="lsv2_not_a_real_key_000")
    client = MagicMock()
    client.evaluate.return_value.experiment_name = "dense-1234"
    result = run(settings, ("dense",), attempt_llm=False, deps=fake_deps(langsmith_client=client))
    assert result.reports[0].langsmith.status == "uploaded"
    assert result.reports[0].langsmith.experiment_name == "dense-1234"
    assert client.evaluate.call_args.kwargs["experiment_prefix"] == "dense"


def test_the_retrieval_depth_must_fit_the_mcp_tool(make_settings: Callable[..., Settings]) -> None:
    settings = make_settings(MCP__MAX_TOP_K="4")
    with pytest.raises(EvalError, match=r"mcp\.max_top_k"):
        run(settings, ("dense",), attempt_llm=False, deps=fake_deps())


async def test_evaluate_refuses_the_in_process_engine_by_default(settings: Settings) -> None:
    with pytest.raises(EvalError, match="needs a Qdrant service"):
        await evaluate(settings, ("hybrid",), attempt_llm=False, deps=fake_deps(allow_local_qdrant=False))


def test_select_per_type_takes_the_first_questions_of_each_type_in_file_order(settings: Settings) -> None:
    questions = load_eval_set(settings)
    chosen = select_per_type(questions, 2)
    assert len(chosen) == 8
    assert [q.id for q in chosen if q.type == "fuzzy_plot"] == ["fuzzy-01", "fuzzy-02"]
    assert {q.type for q in chosen} == {q.type for q in questions}
    assert select_per_type(questions, 100) == questions
    assert isinstance(chosen[0], EvalQuestion)
