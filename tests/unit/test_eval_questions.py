"""The question schema, the JSONL loader, the corpus checks and the committed 40-question set."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from movie_rag.config import Settings
from movie_rag.errors import EvalSetError
from movie_rag.eval.overlap import passes_overlap_guard
from movie_rag.eval.questions import (
    QUESTION_TYPES,
    EvalQuestion,
    check_against_corpus,
    fixture_records,
    load_eval_set,
    matches_filters,
    read_questions,
    validate_questions,
    write_questions,
)
from movie_rag.ingest.clean import MovieRecord
from movie_rag.retrieval.search import SearchFilters


def film(movie_id: str = "alpha-1999-0", title: str = "Alpha", **overrides: Any) -> MovieRecord:
    fields: dict[str, Any] = {
        "movie_id": movie_id,
        "title": title,
        "release_year": 1999,
        "origin": "british",
        "director": "Jane Roe",
        "cast": None,
        "genre": "drama",
        "wiki_url": None,
        "plot": "A lighthouse keeper discovers a hidden staircase beneath the old kitchen floor.",
    }
    return MovieRecord(**{**fields, **overrides})


def q(type_: str = "fuzzy_plot", **overrides: Any) -> EvalQuestion:
    base: dict[str, Any] = {
        "id": "fuzzy-01",
        "type": type_,
        "question": "Which film has a keeper who finds a secret way down?",
        "gold_movie_ids": ["alpha-1999-0"],
    }
    if type_ == "unanswerable":
        base.update(id="unanswerable-01", gold_movie_ids=[], absent_title="Nowhere Man")
    return EvalQuestion(**{**base, **overrides})


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings()


@pytest.fixture
def committed(settings: Settings) -> list[EvalQuestion]:
    return load_eval_set(settings)


# --- the committed set ------------------------------------------------------------------------------------


def test_the_committed_set_has_forty_questions_ten_per_type(committed: list[EvalQuestion]) -> None:
    assert len(committed) == 40
    assert Counter(x.type for x in committed) == dict.fromkeys(QUESTION_TYPES, 10)


def test_ids_are_unique_and_every_gold_film_is_gold_once(committed: list[EvalQuestion]) -> None:
    assert len({x.id for x in committed}) == 40
    gold = [g for x in committed for g in x.gold_movie_ids]
    assert len(gold) == len(set(gold)) == 30


def test_gold_ids_exist_in_the_fixture(committed: list[EvalQuestion], settings: Settings) -> None:
    known = {r.movie_id for r in fixture_records(settings)}
    assert {g for x in committed for g in x.gold_movie_ids} <= known


def test_every_fuzzy_question_passes_the_overlap_guard(committed: list[EvalQuestion], settings: Settings) -> None:
    plots = {r.movie_id: r.plot for r in fixture_records(settings)}
    fuzzy = [x for x in committed if x.type == "fuzzy_plot"]
    assert len(fuzzy) == 10
    for x in fuzzy:
        assert passes_overlap_guard(x.question, plots[x.gold_movie_ids[0]], settings.eval.ngram_size), x.id


def test_the_guard_and_the_title_check_also_hold_for_filtered_questions(
    committed: list[EvalQuestion], settings: Settings
) -> None:
    by_id = {r.movie_id: r for r in fixture_records(settings)}
    for x in (x for x in committed if x.type == "filtered"):
        assert passes_overlap_guard(x.question, by_id[x.gold_movie_ids[0]].plot, settings.eval.ngram_size), x.id
        assert by_id[x.gold_movie_ids[0]].title.lower() not in x.question.lower()


def test_filtered_gold_films_satisfy_their_filters_and_the_filters_supported_by_retrieval(
    committed: list[EvalQuestion], settings: Settings
) -> None:
    by_id = {r.movie_id: r for r in fixture_records(settings)}
    for x in (x for x in committed if x.type == "filtered"):
        assert x.filters.model_dump(exclude_none=True)
        assert all(matches_filters(x.filters, by_id[g]) for g in x.gold_movie_ids), x.id


def test_exact_questions_name_the_title_or_director(committed: list[EvalQuestion], settings: Settings) -> None:
    by_id = {r.movie_id: r for r in fixture_records(settings)}
    exact = [x for x in committed if x.type == "exact_entity"]
    named = ["title" if by_id[x.gold_movie_ids[0]].title.lower() in x.question.lower() else "director" for x in exact]
    assert Counter(named) == {"title": 6, "director": 4}


def test_unanswerable_questions_have_no_gold_and_their_films_are_absent(
    committed: list[EvalQuestion], settings: Settings
) -> None:
    titles = {r.title.lower() for r in fixture_records(settings)}
    unanswerable = [x for x in committed if x.type == "unanswerable"]
    assert len(unanswerable) == 10
    for x in unanswerable:
        assert x.gold_movie_ids == []
        assert x.absent_title is not None
        assert x.absent_title.lower() not in titles


def test_unanswerable_questions_that_carry_filters_match_no_film(
    committed: list[EvalQuestion], settings: Settings
) -> None:
    records = fixture_records(settings)
    with_filters = [x for x in committed if x.type == "unanswerable" and x.filters.model_dump(exclude_none=True)]
    assert len(with_filters) == 2
    for x in with_filters:
        assert not any(matches_filters(x.filters, r) for r in records), x.id


def test_the_questions_file_is_where_the_configuration_says(settings: Settings) -> None:
    path = settings.eval.resolve(settings.eval.questions_path)
    assert path.is_file()
    assert read_questions(path) == load_eval_set(settings)


def test_every_committed_question_carries_a_note(committed: list[EvalQuestion]) -> None:
    assert all(x.notes.strip() for x in committed)


# --- the record -------------------------------------------------------------------------------------------


def test_filters_are_normalised_like_the_retrieval_layer() -> None:
    question = q("filtered", id="filtered-01", filters={"genre": " Film Noir ", "year_from": 1940})
    assert question.filters == SearchFilters(genre="film noir", year_from=1940)


@pytest.mark.parametrize(
    ("type_", "overrides", "message"),
    [
        ("fuzzy_plot", {"gold_movie_ids": []}, "exactly one gold"),
        ("fuzzy_plot", {"gold_movie_ids": ["a-1-0", "b-1-1"]}, "exactly one gold"),
        ("fuzzy_plot", {"filters": {"genre": "drama"}}, "no filters"),
        ("exact_entity", {"id": "exact-01", "filters": {"origin": "british"}}, "no filters"),
        ("filtered", {"id": "filtered-01"}, "at least one filter"),
        ("filtered", {"id": "filtered-01", "gold_movie_ids": [], "filters": {"genre": "drama"}}, "at least one gold"),
        ("unanswerable", {"gold_movie_ids": ["alpha-1999-0"]}, "no gold"),
        ("unanswerable", {"absent_title": None}, "absent_title is required"),
        ("fuzzy_plot", {"absent_title": "Alpha"}, "only for them"),
        ("fuzzy_plot", {"gold_movie_ids": ["a-1-0", "a-1-0"]}, "duplicate"),
        ("fuzzy_plot", {"id": "Fuzzy 1"}, "pattern"),
        ("fuzzy_plot", {"question": "short"}, "at least 10"),
        ("fuzzy_plot", {"surprise": 1}, "Extra inputs"),
        ("fuzzy_plot", {"type": "trivia"}, "Input should be"),
    ],
)
def test_a_record_of_the_wrong_shape_is_rejected(type_: str, overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        q(type_, **overrides)


def test_a_filtered_question_needs_a_gold_film_and_a_filter_and_accepts_both() -> None:
    assert q("filtered", id="filtered-01", filters={"year_to": 1950}).gold_movie_ids == ["alpha-1999-0"]


# --- the file ---------------------------------------------------------------------------------------------


def test_write_then_read_round_trips_and_omits_unset_filters(tmp_path: Path) -> None:
    questions = [q(), q("filtered", id="filtered-01", filters={"genre": "drama"}), q("unanswerable")]
    path = tmp_path / "nested" / "qs.jsonl"
    write_questions(questions, path)
    assert read_questions(path) == questions
    first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert first["filters"] == {}
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_blank_lines_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "qs.jsonl"
    write_questions([q()], path)
    path.write_text("\n" + path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")
    assert len(read_questions(path)) == 1


def test_a_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(EvalSetError, match="not found"):
        read_questions(tmp_path / "absent.jsonl")


def test_a_bad_line_is_reported_with_its_number(tmp_path: Path) -> None:
    path = tmp_path / "qs.jsonl"
    good = json.dumps(q().model_dump(mode="json"))
    path.write_text(good + "\n{not json\n", encoding="utf-8")
    with pytest.raises(EvalSetError, match=r"qs\.jsonl line 2"):
        read_questions(path)
    path.write_text(good + "\n" + json.dumps({"id": "fuzzy-02"}) + "\n", encoding="utf-8")
    with pytest.raises(EvalSetError, match=r"line 2"):
        read_questions(path)


# --- set-level validation ---------------------------------------------------------------------------------


def balanced(per_type: int = 1) -> list[EvalQuestion]:
    out: list[EvalQuestion] = []
    for i in range(per_type):
        out += [
            q(id=f"fuzzy-{i + 1:02d}", gold_movie_ids=[f"a-{i}"]),
            q("exact_entity", id=f"exact-{i + 1:02d}", gold_movie_ids=[f"b-{i}"]),
            q("filtered", id=f"filtered-{i + 1:02d}", gold_movie_ids=[f"c-{i}"], filters={"genre": "drama"}),
            q("unanswerable", id=f"unanswerable-{i + 1:02d}"),
        ]
    return out


def test_a_balanced_set_validates() -> None:
    validate_questions(balanced(2), per_type=2)


def test_wrong_counts_duplicate_ids_and_repeated_gold_are_all_reported() -> None:
    questions = balanced(1)
    questions.append(q(id="fuzzy-01", gold_movie_ids=["a-0"]))
    with pytest.raises(EvalSetError) as caught:
        validate_questions(questions, per_type=1)
    message = str(caught.value)
    assert "duplicate id fuzzy-01" in message
    assert "fuzzy_plot: 2 questions, expected 1" in message
    assert "a-0 is the gold answer of 2 questions" in message


def test_a_missing_type_is_reported() -> None:
    with pytest.raises(EvalSetError, match="unanswerable: 0 questions, expected 1"):
        validate_questions(balanced(1)[:3], per_type=1)


# --- corpus checks ----------------------------------------------------------------------------------------


def test_a_consistent_set_has_no_problems() -> None:
    records = [film()]
    questions = [
        q(),
        q("exact_entity", id="exact-01", question="Tell me about the film Alpha please"),
        q("filtered", id="filtered-01", filters={"genre": "drama", "year_from": 1990}),
        q("unanswerable"),
    ]
    assert check_against_corpus(questions, records, 4) == []


def test_an_unknown_gold_id_is_reported() -> None:
    problems = check_against_corpus([q(gold_movie_ids=["ghost-1-1"])], [film()], 4)
    assert problems == ["fuzzy-01: gold movie_id ghost-1-1 is not in the dataset"]


def test_a_fuzzy_question_copying_the_plot_is_reported_with_the_phrase() -> None:
    copied = q(question="Which film has a hidden staircase beneath the old kitchen?")
    (problem,) = check_against_corpus([copied], [film()], 4)
    assert "fuzzy-01" in problem
    assert "hidden staircase beneath the" in problem


def test_a_fuzzy_question_naming_the_title_is_reported() -> None:
    (problem,) = check_against_corpus([q(question="I want to see Alpha, what is it about?")], [film()], 4)
    assert "names the title of alpha-1999-0" in problem


def test_an_exact_question_must_name_the_title_or_the_director() -> None:
    vague = q("exact_entity", id="exact-01", question="Which film has a keeper who finds a secret way down?")
    assert "does not name the title or director" in check_against_corpus([vague], [film()], 4)[0]
    by_director = q("exact_entity", id="exact-01", question="What did Jane Roe direct, exactly?")
    assert check_against_corpus([by_director], [film()], 4) == []
    no_director = film(director=None)
    assert check_against_corpus([by_director], [no_director], 4) != []


def test_a_filtered_gold_film_must_satisfy_the_filters() -> None:
    wrong = q("filtered", id="filtered-01", filters={"origin": "japanese"})
    (problem,) = check_against_corpus([wrong], [film()], 4)
    assert "does not satisfy" in problem


def test_an_unanswerable_film_must_be_absent_from_the_dataset() -> None:
    present = q("unanswerable", absent_title="  alpha ")
    (problem,) = check_against_corpus([present], [film()], 4)
    assert "is a film in the dataset" in problem


def test_an_unanswerable_question_must_not_copy_any_plot() -> None:
    copying = q("unanswerable", question="Find the one with a hidden staircase beneath the old kitchen floor")
    (problem,) = check_against_corpus([copying], [film()], 4)
    assert "alpha-1999-0" in problem


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        (SearchFilters(), True),
        (SearchFilters(year_from=1999, year_to=1999), True),
        (SearchFilters(year_from=2000), False),
        (SearchFilters(year_to=1998), False),
        (SearchFilters(genre="drama", origin="british"), True),
        (SearchFilters(genre="comedy"), False),
        (SearchFilters(origin="american"), False),
    ],
)
def test_filter_matching_follows_the_retrieval_semantics(filters: SearchFilters, expected: bool) -> None:
    assert matches_filters(filters, film()) is expected


def test_load_eval_set_reports_every_problem_in_a_broken_file(settings: Settings, tmp_path: Path) -> None:
    questions = balanced(10)
    questions[0] = q(gold_movie_ids=["ghost-1-1"])
    path = tmp_path / "qs.jsonl"
    write_questions(questions, path)
    with pytest.raises(EvalSetError, match=r"disagrees with the fixture.*ghost-1-1"):
        load_eval_set(settings, path)


def test_load_eval_set_rejects_a_set_of_the_wrong_size(settings: Settings, tmp_path: Path) -> None:
    path = tmp_path / "qs.jsonl"
    write_questions(balanced(1), path)
    with pytest.raises(EvalSetError, match="expected 10"):
        load_eval_set(settings, path)


def test_the_documentation_table_lists_every_question(committed: list[EvalQuestion]) -> None:
    doc = (Path(__file__).resolve().parents[2] / "docs" / "EVAL_SET.md").read_text(encoding="utf-8")
    for x in committed:
        assert f"`{x.id}`" in doc
        assert x.question.replace("|", "/") in doc
        for movie_id in x.gold_movie_ids:
            assert f"`{movie_id}`" in doc
        if x.absent_title:
            assert x.absent_title in doc
