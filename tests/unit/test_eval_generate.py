"""The paraphrase generator, run against a scripted chat model (no key, no network)."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from fakes import ScriptedChatModel
from movie_rag.config import Settings, load_settings
from movie_rag.errors import EvalSetError, MissingCredentialError
from movie_rag.eval import generate
from movie_rag.eval.generate import (
    clean_model_text,
    generate_questions,
    load_paraphrase_prompt,
    refusal_reasons,
)
from movie_rag.eval.overlap import passes_overlap_guard
from movie_rag.eval.questions import fixture_records, read_questions
from movie_rag.ingest.clean import MovieRecord

CLEAN = "Somebody goes looking for something that nobody else seems to care about at all."


def reply(text: str) -> AIMessage:
    return AIMessage(content=text)


def film(movie_id: str, title: str, plot: str) -> MovieRecord:
    return MovieRecord(
        movie_id=movie_id,
        title=title,
        release_year=1999,
        origin=None,
        director=None,
        cast=None,
        genre=None,
        wiki_url=None,
        plot=plot,
    )


FILMS = [
    film(f"f{i}-1999-{i}", f"Film Number {i}", f"A baker named Dolores {i} finds a ladder in the cellar.")
    for i in range(6)
]


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings(EVAL__GENERATE_COUNT="3", EVAL__MAX_ATTEMPTS="2")


def test_the_prompt_is_versioned_and_states_the_guard_size(settings: Settings) -> None:
    prompt = load_paraphrase_prompt(settings)
    assert "{" not in prompt
    assert "4 or more consecutive words" in prompt


def test_an_unknown_prompt_version_is_an_error(make_settings: Callable[..., Settings]) -> None:
    with pytest.raises(EvalSetError, match="unknown paraphrase prompt version 'paraphrase_v0'"):
        load_paraphrase_prompt(make_settings(EVAL__PARAPHRASE_PROMPT_VERSION="paraphrase_v0"))


def test_model_text_is_reduced_to_one_unquoted_line() -> None:
    assert clean_model_text('\n  "Which film is it?"  \nExtra commentary') == "Which film is it?"
    assert clean_model_text("“Curly quotes”") == "Curly quotes"
    assert clean_model_text("  \n  ") == ""


def test_refusal_reasons_list_copied_phrases_and_titles() -> None:
    plot = "A baker named Dolores finds a ladder in the cellar."
    target = film("x-1999-0", "The Ladder", plot)
    assert refusal_reasons("A baker named Dolores wants help", target, 4) == [
        'it copies 4-word phrases from the plot: "a baker named dolores"'
    ]
    assert refusal_reasons("Who climbs the ladder? The Ladder it is", target, 4) == ["it contains the film's title"]
    assert refusal_reasons("", target, 4) == ["the reply was empty"]
    assert refusal_reasons("A pastry chef stumbles on a way down", target, 4) == []


def test_a_clean_reply_becomes_a_fuzzy_question_with_the_films_id(settings: Settings) -> None:
    model = ScriptedChatModel(replies=[reply(CLEAN)])
    report = generate_questions(settings, FILMS, model)
    assert [x.id for x in report.questions] == ["gen-fuzzy-01", "gen-fuzzy-02", "gen-fuzzy-03"]
    assert {x.type for x in report.questions} == {"fuzzy_plot"}
    assert all(x.question == CLEAN and len(x.gold_movie_ids) == 1 for x in report.questions)
    assert report.model_calls == 3
    assert report.rejected == []
    assert "prompt paraphrase_v1" in report.questions[0].notes


def test_the_prompt_and_the_plot_reach_the_model(settings: Settings) -> None:
    model = ScriptedChatModel(replies=[reply(CLEAN)])
    generate_questions(settings, FILMS[:1], model)
    system, human = model.prompts[0]
    assert isinstance(system, SystemMessage)
    assert system.content == load_paraphrase_prompt(settings)
    assert isinstance(human, HumanMessage)
    assert str(human.content).startswith("Plot:\nA baker named Dolores 0")


def test_same_seed_same_films_in_the_same_order(settings: Settings) -> None:
    first = generate_questions(settings, FILMS, ScriptedChatModel(replies=[reply(CLEAN)]))
    second = generate_questions(settings, FILMS, ScriptedChatModel(replies=[reply(CLEAN)]))
    assert [x.gold_movie_ids for x in first.questions] == [x.gold_movie_ids for x in second.questions]
    other = generate_questions(make_other_seed(settings), FILMS, ScriptedChatModel(replies=[reply(CLEAN)]))
    assert [x.gold_movie_ids for x in other.questions] != [x.gold_movie_ids for x in first.questions]


def make_other_seed(settings: Settings) -> Settings:
    return settings.model_copy(update={"ingest": settings.ingest.model_copy(update={"random_seed": 7})})


def test_a_rejected_question_is_regenerated_with_the_offending_phrase_listed(settings: Settings) -> None:
    copying = "Which baker named Dolores 0 finds a ladder in the cellar?"
    one = settings.model_copy(update={"eval": settings.eval.model_copy(update={"generate_count": 1})})
    model = ScriptedChatModel(replies=[reply(copying), reply(CLEAN)])
    report = generate_questions(one, FILMS, model)
    assert report.model_calls == 2
    assert report.questions[0].question == CLEAN
    assert "attempt 2" in report.questions[0].notes
    retry_human = model.prompts[1][1]
    assert "was refused because it copies 4-word phrases" in str(retry_human.content)
    assert "finds a ladder in the" in str(retry_human.content)


def test_a_film_that_never_passes_is_rejected_and_the_next_one_is_tried(settings: Settings) -> None:
    one = settings.model_copy(update={"eval": settings.eval.model_copy(update={"generate_count": 1})})
    first_film = (
        generate_questions(one, FILMS, ScriptedChatModel(replies=[reply(CLEAN)])).questions[0].gold_movie_ids[0]
    )
    copying = f"{FILMS[int(first_film[1])].plot}"
    model = ScriptedChatModel(replies=[reply(copying), reply(copying), reply(CLEAN)])
    report = generate_questions(one, FILMS, model)
    assert [r.movie_id for r in report.rejected] == [first_film]
    assert report.rejected[0].attempts == 2
    assert "copies 4-word phrases" in report.rejected[0].reasons[0]
    assert report.questions[0].gold_movie_ids[0] != first_film
    assert report.model_calls == 3


def test_a_question_naming_the_title_is_refused(settings: Settings) -> None:
    one = settings.model_copy(update={"eval": settings.eval.model_copy(update={"generate_count": 1})})
    titled = "Is there something like Film Number 0 or Film Number 1 or Film Number 2 or Film Number 3?"
    report = generate_questions(one, FILMS[:1], ScriptedChatModel(replies=[reply(titled)]))
    assert report.questions == []
    assert report.rejected[0].reasons == ["it contains the film's title"]


def test_the_run_ends_when_the_films_run_out(settings: Settings) -> None:
    report = generate_questions(settings, FILMS[:2], ScriptedChatModel(replies=[reply(CLEAN)]))
    assert len(report.questions) == 2


def test_every_accepted_question_passes_the_guard_against_its_plot(settings: Settings) -> None:
    plots = {r.movie_id: r.plot for r in FILMS}
    report = generate_questions(settings, FILMS, ScriptedChatModel(replies=[reply(CLEAN)]))
    assert all(passes_overlap_guard(x.question, plots[x.gold_movie_ids[0]], 4) for x in report.questions)


def test_without_a_key_generation_raises_the_credentials_error(settings: Settings) -> None:
    with pytest.raises(MissingCredentialError, match=re.escape("set NEBIUS_API_KEY — see docs/CREDENTIALS.md")):
        generate_questions(settings, FILMS)


# --- the command line --------------------------------------------------------------------------------------


@pytest.fixture
def isolated_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generate, "load_settings", lambda: load_settings(env_file=None))


@pytest.mark.usefixtures("isolated_cli")
def test_the_cli_prints_the_hint_and_exits_2_without_a_key(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    assert generate.main(["--out", str(out)]) == generate.EXIT_USAGE
    assert "set NEBIUS_API_KEY — see docs/CREDENTIALS.md" in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.usefixtures("isolated_cli")
def test_the_cli_writes_the_questions_for_the_fixture(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, make_settings: Callable[..., Settings]
) -> None:
    out = tmp_path / "gen" / "out.jsonl"
    model = ScriptedChatModel(replies=[reply(CLEAN)])
    assert generate.main(["--out", str(out)], model=model) == generate.EXIT_OK
    written = read_questions(out)
    expected = make_settings().eval.generate_count
    assert len(written) == expected
    known = {r.movie_id for r in fixture_records(make_settings())}
    assert {x.gold_movie_ids[0] for x in written} <= known
    assert f"wrote {expected} question(s)" in capsys.readouterr().err


@pytest.mark.usefixtures("isolated_cli")
def test_the_cli_reports_rejected_films(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    model = ScriptedChatModel(replies=[reply("")] * 3 + [reply(CLEAN)])
    assert generate.main(["--out", str(out)], model=model) == generate.EXIT_OK
    err = capsys.readouterr().err
    assert "1 film(s) rejected after 3 attempts each" in err
    assert "the reply was empty" in err


@pytest.mark.usefixtures("isolated_cli")
def test_the_cli_exits_1_on_a_configuration_error(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVAL__PARAPHRASE_PROMPT_VERSION", "paraphrase_v0")
    model = ScriptedChatModel(replies=[reply(CLEAN)])
    assert generate.main(["--out", str(tmp_path / "o.jsonl")], model=model) == generate.EXIT_FAILURE
    assert "unknown paraphrase prompt version" in capsys.readouterr().err


def test_the_default_output_is_the_configured_generated_path_not_the_committed_set(settings: Settings) -> None:
    assert settings.eval.generated_path != settings.eval.questions_path
    assert "generated" in settings.eval.generated_path.name
