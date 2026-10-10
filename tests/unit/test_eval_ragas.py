"""RAGAS wiring with a fake judge: no network, no key. The metrics are the real ragas 0.4 classes."""

from __future__ import annotations

import importlib
import importlib.metadata
import math
import sys
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from fakes import FakeEmbedder
from movie_rag.config import Settings
from movie_rag.errors import EvalError, MissingCredentialError
from movie_rag.eval.ragas_judge import (
    METRIC_NAMES,
    RagasSample,
    SampleScores,
    build_scorers,
    ensure_judge_differs,
    install_ragas_compat,
    make_embeddings,
    make_judge,
    ragas_version,
    score_samples,
    summarise_scores,
)
from movie_rag.eval.runner import default_scorers

VERTEXAI = "langchain_community.chat_models.vertexai"
QUESTION = "a ferry captain hides a stolen bell"


def sample(qid: str = "fuzzy-01") -> RagasSample:
    return RagasSample(
        question_id=qid,
        user_input=QUESTION,
        response="Midnight Ferry (1988) hides a bell.",
        retrieved_contexts=["Midnight Ferry (1988): a captain hides a bell.", "Other Film (1999): unrelated."],
        reference="Midnight Ferry (1988). A captain hides a stolen bell.",
    )


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings(NEBIUS_API_KEY="test-key-not-real")


@pytest.fixture
def restore_vertexai_module() -> Iterator[None]:
    saved = sys.modules.pop(VERTEXAI, None)
    yield
    sys.modules.pop(VERTEXAI, None)
    if saved is not None:
        sys.modules[VERTEXAI] = saved


# --- compatibility and versions ------------------------------------------------------------------------------------


def test_ragas_imports_after_the_compat_step_and_telemetry_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAGAS_DO_NOT_TRACK", raising=False)
    install_ragas_compat()
    import ragas

    assert ragas.__version__ == ragas_version()
    assert sys.modules  # keep the import observable
    import os

    assert os.environ["RAGAS_DO_NOT_TRACK"] == "true"


def test_a_stand_in_is_registered_only_when_the_removed_module_is_really_missing(
    monkeypatch: pytest.MonkeyPatch, restore_vertexai_module: None
) -> None:
    def missing(name: str, package: str | None = None) -> Any:
        raise ImportError(name)

    monkeypatch.setattr(importlib, "import_module", missing)
    install_ragas_compat()
    assert sys.modules[VERTEXAI].ChatVertexAI.__name__ == "ChatVertexAI"
    stand_in = sys.modules[VERTEXAI]
    install_ragas_compat()  # a second call leaves the registered module alone
    assert sys.modules[VERTEXAI] is stand_in


def test_the_real_module_is_used_when_it_exists(monkeypatch: pytest.MonkeyPatch, restore_vertexai_module: None) -> None:
    calls: list[str] = []
    monkeypatch.setattr(importlib, "import_module", lambda name, package=None: calls.append(name))
    install_ragas_compat()
    assert calls == [VERTEXAI] and VERTEXAI not in sys.modules


def test_the_ragas_version_is_read_from_the_package_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    assert ragas_version().startswith("0.4")
    monkeypatch.setattr(importlib.metadata, "version", _raise_not_found)
    assert ragas_version() == "not installed"


def _raise_not_found(name: str) -> str:
    raise importlib.metadata.PackageNotFoundError(name)


# --- judge != generator --------------------------------------------------------------------------------------------


def test_the_committed_judge_is_not_the_generator(make_settings: Callable[..., Settings]) -> None:
    ensure_judge_differs(make_settings())


@pytest.mark.parametrize("judge", ["Qwen/Qwen3-30B-A3B-Instruct-2507", "  qwen/qwen3-30b-a3b-instruct-2507 "])
def test_a_judge_that_is_the_generator_is_refused(make_settings: Callable[..., Settings], judge: str) -> None:
    settings = make_settings(LLM__JUDGE_MODEL=judge)
    with pytest.raises(EvalError, match="must be a different model"):
        ensure_judge_differs(settings)


def test_the_judge_is_checked_before_the_key_and_before_any_client_exists(
    make_settings: Callable[..., Settings],
) -> None:
    settings = make_settings(LLM__JUDGE_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507", NEBIUS_API_KEY="test-key-not-real")
    with pytest.raises(EvalError, match="different model"):
        make_judge(settings)
    with pytest.raises(EvalError, match="different model"):
        default_scorers(settings, FakeEmbedder())


def test_the_judge_needs_the_nebius_key_at_call_time(make_settings: Callable[..., Settings]) -> None:
    with pytest.raises(MissingCredentialError, match="NEBIUS_API_KEY"):
        make_judge(make_settings())


def test_the_judge_is_the_configured_judge_model_on_the_nebius_endpoint(settings: Settings) -> None:
    llm = make_judge(settings)
    assert llm.model == settings.llm.judge_model != settings.llm.chat_model
    assert str(llm.client.base_url).rstrip("/") == settings.llm.base_url.rstrip("/")
    assert llm.client.timeout == settings.eval.judge_timeout_s


# --- the four metrics ----------------------------------------------------------------------------------------------


def fake_judge() -> Any:
    install_ragas_compat()
    from ragas.llms.base import InstructorBaseRagasLLM

    replies: dict[str, dict[str, Any]] = {
        "StatementGeneratorOutput": {"statements": ["It hides a bell.", "It is set on a ferry."]},
        "NLIStatementOutput": {
            "statements": [
                {"statement": "It hides a bell.", "reason": "stated", "verdict": 1},
                {"statement": "It is set on a ferry.", "reason": "not stated", "verdict": 0},
            ]
        },
        "AnswerRelevanceOutput": {"question": QUESTION, "noncommittal": 0},
        "ContextPrecisionOutput": {"reason": "useful", "verdict": 1},
        "ContextRecallOutput": {
            "classifications": [
                {"statement": "A captain hides a bell.", "reason": "in context", "attributed": 1},
                {"statement": "It was stolen.", "reason": "not in context", "attributed": 0},
            ]
        },
    }

    class FakeJudge(InstructorBaseRagasLLM):  # type: ignore[misc]
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def generate(self, prompt: str, response_model: Any) -> Any:
            raise NotImplementedError

        async def agenerate(self, prompt: str, response_model: Any) -> Any:
            self.prompts.append(prompt)
            return response_model.model_validate(replies[response_model.__name__])

    return FakeJudge()


async def test_the_four_metrics_score_a_sample_with_the_expected_values() -> None:
    judge = fake_judge()
    scorers = build_scorers(judge, make_embeddings(FakeEmbedder()), strictness=2)
    assert list(scorers) == list(METRIC_NAMES)
    [result] = await score_samples([sample()], scorers, concurrency=2)
    assert result.errors == {}
    assert result.scores["faithfulness"] == 0.5  # 1 of 2 statements supported
    assert result.scores["response_relevancy"] == pytest.approx(1.0)  # the generated question is the asked one
    assert result.scores["context_precision"] == pytest.approx(1.0)  # both contexts judged useful
    assert result.scores["context_recall"] == 0.5  # 1 of 2 reference statements attributable
    assert any("ferry" in p for p in judge.prompts)  # the judge saw the sample


async def test_default_scorers_use_the_judge_and_the_local_embedder(settings: Settings) -> None:
    scorers = default_scorers(settings, FakeEmbedder())
    assert list(scorers) == list(METRIC_NAMES)


async def test_local_embeddings_come_from_the_project_embedder() -> None:
    embeddings = make_embeddings(FakeEmbedder())
    sync = embeddings.embed_text(QUESTION)
    assert len(sync) == 384
    assert await embeddings.aembed_text(QUESTION) == sync


# --- failure isolation and summaries -------------------------------------------------------------------------------


async def test_one_failing_metric_does_not_discard_the_other_scores() -> None:
    async def fine(_: RagasSample) -> float:
        return 0.75

    async def boom(_: RagasSample) -> float:
        raise TimeoutError("judge timed out\nsecond line")

    async def nan(_: RagasSample) -> float:
        return math.nan

    results = await score_samples(
        [sample("a"), sample("b")],
        {"faithfulness": fine, "context_recall": boom, "context_precision": nan},
        concurrency=1,
    )
    assert [r.question_id for r in results] == ["a", "b"]
    for r in results:
        assert r.scores == {"faithfulness": 0.75, "context_recall": None, "context_precision": None}
        assert r.errors["context_recall"] == "TimeoutError: judge timed out"
        assert "NaN" in r.errors["context_precision"]


def test_summaries_average_the_scored_samples_and_count_the_failures() -> None:
    results = [
        SampleScores(question_id="a", scores={"faithfulness": 1.0, "context_recall": None}, errors={}),
        SampleScores(question_id="b", scores={"faithfulness": 0.5, "context_recall": None}, errors={}),
        SampleScores(question_id="c", scores={"faithfulness": None, "context_recall": None}, errors={}),
    ]
    summary = summarise_scores(results, ["faithfulness", "context_recall"])
    assert (summary["faithfulness"].mean, summary["faithfulness"].n_scored, summary["faithfulness"].n_failed) == (
        0.75,
        2,
        1,
    )
    assert (summary["context_recall"].mean, summary["context_recall"].n_scored, summary["context_recall"].n_failed) == (
        None,
        0,
        3,
    )
