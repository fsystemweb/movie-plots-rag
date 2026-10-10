"""The LangSmith experiment: skipped without a key, built from the computed rankings with one.

``Client.evaluate`` is replaced by a mock; the target and evaluators it receives are called for real.
"""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import MagicMock

import pytest

from movie_rag.config import Settings
from movie_rag.eval.langsmith_experiment import SKIP_DETAIL, make_evaluators, make_target, run_experiment
from movie_rag.eval.questions import EvalQuestion
from movie_rag.eval.retrieval_eval import QuestionRetrieval

QUESTION = EvalQuestion(id="fuzzy-01", type="fuzzy_plot", question="a ferry captain hides a bell", gold_movie_ids=["g"])
LOST = EvalQuestion(id="unanswerable-01", type="unanswerable", question="a submarine opera singer", absent_title="X")


def result(question: EvalQuestion, ranked: list[str], scores: list[float]) -> QuestionRetrieval:
    return QuestionRetrieval(
        question_id=question.id,
        type=question.type,
        question=question.question,
        gold_movie_ids=question.gold_movie_ids,
        ranked_movie_ids=ranked,
        scores=scores,
        rank=None,
        latency_ms=12.5,
    )


RESULTS = [result(QUESTION, ["x", "g"], [0.5, 0.5]), result(LOST, ["x"], [0.4])]


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings(LANGSMITH_API_KEY="lsv2_not_a_real_key_000")


def test_without_a_key_nothing_is_called_and_the_result_says_so(make_settings: Callable[..., Settings]) -> None:
    outcome = run_experiment(make_settings(), "dense", [QUESTION], RESULTS)
    assert (outcome.status, outcome.detail) == ("skipped", SKIP_DETAIL)


def test_the_target_returns_the_ranking_computed_for_the_examples_question() -> None:
    target = make_target(RESULTS)
    assert target({"question": QUESTION.question, "filters": {}}) == {
        "ranked_movie_ids": ["x", "g"],
        "scores": [0.5, 0.5],
        "latency_ms": 12.5,
    }


def test_evaluators_score_hit_at_k_and_reciprocal_rank_with_ties() -> None:
    outputs = {"ranked_movie_ids": ["x", "g"], "scores": [0.5, 0.5]}
    reference = {"gold_movie_ids": ["g"], "expect_abstention": False}
    scores = [e(outputs, reference) for e in make_evaluators([1, 2])]
    assert [(s["key"], s["score"]) for s in scores] == [("hit_at_1", 0.5), ("hit_at_2", 1.0), ("reciprocal_rank", 0.75)]


def test_unanswerable_examples_get_no_retrieval_score() -> None:
    outputs = {"ranked_movie_ids": ["x"], "scores": [0.4]}
    reference = {"gold_movie_ids": [], "expect_abstention": True}
    assert all(e(outputs, reference)["score"] is None for e in make_evaluators([1, 3]))


def test_with_a_key_the_experiment_is_created_on_the_question_dataset(settings: Settings) -> None:
    client = MagicMock()
    client.has_dataset.return_value = True
    client.evaluate.return_value.experiment_name = "hybrid-abc123"
    outcome = run_experiment(settings, "hybrid", [QUESTION, LOST], RESULTS, client=client)
    assert (outcome.status, outcome.experiment_name) == ("uploaded", "hybrid-abc123")
    kwargs = client.evaluate.call_args.kwargs
    assert kwargs["data"] == settings.eval.dataset_name and kwargs["experiment_prefix"] == "hybrid"
    assert [
        e({"ranked_movie_ids": ["g"], "scores": [1.0]}, {"gold_movie_ids": ["g"]})["key"] for e in kwargs["evaluators"]
    ] == [
        "hit_at_1",
        "hit_at_3",
        "hit_at_5",
        "hit_at_8",
        "reciprocal_rank",
    ]
    assert kwargs["metadata"]["retrieval_mode"] == "hybrid" and "lsv2_" not in str(kwargs["metadata"])


def test_the_dataset_is_created_when_it_does_not_exist_yet(settings: Settings) -> None:
    client = MagicMock()
    client.has_dataset.return_value = False
    run_experiment(settings, "dense", [QUESTION, LOST], RESULTS, client=client)
    client.create_dataset.assert_called_once()
    client.create_examples.assert_called_once()


def test_a_langsmith_outage_never_fails_the_evaluation(settings: Settings) -> None:
    client = MagicMock()
    client.has_dataset.side_effect = ConnectionError("down")
    outcome = run_experiment(settings, "dense", [QUESTION], RESULTS, client=client)
    assert (outcome.status, outcome.detail) == ("failed", "failed: ConnectionError")


def test_an_experiment_without_a_reported_name_falls_back_to_the_mode(settings: Settings) -> None:
    client = MagicMock()
    client.has_dataset.return_value = True
    client.evaluate.return_value = object()
    outcome = run_experiment(settings, "sparse", [QUESTION], RESULTS, client=client)
    assert outcome.status == "uploaded" and outcome.experiment_name is None and "sparse" in outcome.detail
