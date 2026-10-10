"""The LangSmith upload: skips without a key, creates the dataset once with a mocked client."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langsmith import Client

from movie_rag.config import Settings, load_settings
from movie_rag.eval import upload
from movie_rag.eval.questions import EvalQuestion, load_eval_set
from movie_rag.eval.upload import SKIP_MESSAGE, make_client, to_example, upload_questions

KEY = "test-langsmith-key-not-real"


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings(LANGSMITH_API_KEY=KEY)


@pytest.fixture
def questions(make_settings: Callable[..., Settings]) -> list[EvalQuestion]:
    return load_eval_set(make_settings())


def mock_client(*, exists: bool = False) -> MagicMock:
    client = MagicMock(spec=Client)
    client.has_dataset.return_value = exists
    client.create_dataset.return_value.id = "dataset-id"
    return client


def test_without_a_key_nothing_is_called(make_settings: Callable[..., Settings], questions: list[EvalQuestion]) -> None:
    client = mock_client()
    result = upload_questions(make_settings(), questions, client)
    assert result.status == "skipped"
    assert result.describe() == SKIP_MESSAGE
    assert client.method_calls == []


def test_with_a_key_the_dataset_and_all_examples_are_created(settings: Settings, questions: list[EvalQuestion]) -> None:
    client = mock_client()
    result = upload_questions(settings, questions, client)
    assert result.status == "created"
    assert result.examples == 40
    client.has_dataset.assert_called_once_with(dataset_name=settings.eval.dataset_name)
    assert client.create_dataset.call_args.args == (settings.eval.dataset_name,)
    kwargs = client.create_examples.call_args.kwargs
    assert kwargs["dataset_id"] == "dataset-id"
    assert [e["metadata"]["question_id"] for e in kwargs["examples"]] == [x.id for x in questions]
    assert "created LangSmith dataset" in result.describe()


def test_an_existing_dataset_is_left_alone(settings: Settings, questions: list[EvalQuestion]) -> None:
    client = mock_client(exists=True)
    result = upload_questions(settings, questions, client)
    assert result.status == "exists"
    client.create_dataset.assert_not_called()
    client.create_examples.assert_not_called()
    assert "already exists" in result.describe()


def test_example_shapes(questions: list[EvalQuestion]) -> None:
    by_type = {x.type: to_example(x) for x in reversed(questions)}
    filtered = next(to_example(x) for x in questions if x.id == "filtered-04")
    assert filtered["inputs"]["filters"] == {
        "year_from": 2000,
        "year_to": 2009,
        "genre": "crime",
        "origin": "bollywood",
    }
    assert filtered["outputs"] == {"gold_movie_ids": ["monsoon-heist-2005-280"], "expect_abstention": False}
    fuzzy = by_type["fuzzy_plot"]
    assert fuzzy["inputs"]["filters"] == {}
    assert fuzzy["inputs"]["question_id"] == "fuzzy-01"  # experiment targets key on the id, not the text
    assert fuzzy["metadata"] == {"question_id": "fuzzy-01", "type": "fuzzy_plot"}
    unanswerable = by_type["unanswerable"]
    assert unanswerable["outputs"] == {"gold_movie_ids": [], "expect_abstention": True}
    assert unanswerable["metadata"]["absent_title"]


def test_no_example_carries_the_key(settings: Settings, questions: list[EvalQuestion]) -> None:
    assert KEY not in str([to_example(x) for x in questions])


def test_the_client_is_built_from_the_key_and_the_configured_url(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    constructed = MagicMock()
    monkeypatch.setattr(upload, "Client", constructed)
    make_client(settings)
    constructed.assert_called_once_with(api_key=KEY, api_url=settings.observability.langsmith_api_url)


def test_a_client_needs_a_key(make_settings: Callable[..., Settings]) -> None:
    with pytest.raises(ValueError, match="LANGSMITH_API_KEY"):
        make_client(make_settings())


@pytest.fixture
def isolated_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(upload, "load_settings", lambda: load_settings(env_file=None))


@pytest.mark.usefixtures("isolated_cli")
def test_the_command_prints_the_skip_and_exits_0_without_a_key(capsys: pytest.CaptureFixture[str]) -> None:
    assert upload.main() == upload.EXIT_OK
    assert SKIP_MESSAGE in capsys.readouterr().err


@pytest.mark.usefixtures("isolated_cli")
def test_the_command_uploads_with_a_key(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", KEY)
    client = mock_client()
    assert upload.main(client=client) == upload.EXIT_OK
    assert "created LangSmith dataset 'movie-plots-questions-v1' with 40 example(s)" in capsys.readouterr().err
    client.create_examples.assert_called_once()


@pytest.mark.usefixtures("isolated_cli")
def test_the_command_exits_1_when_the_question_file_is_unreadable(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("EVAL__QUESTIONS_PATH", str(tmp_path / "missing.jsonl"))
    assert upload.main(client=mock_client()) == upload.EXIT_FAILURE
    assert "question file not found" in capsys.readouterr().err
