"""The evaluation question record, its JSONL file and the checks that keep the set honest.

One question per line in ``questions_v1.jsonl``. Four types, ten each (``eval.per_type``):

* ``fuzzy_plot``: describes one film's plot in other words (no ``eval.ngram_size``-word run copied from the plot);
* ``exact_entity``: names a film's title or director; the gold film is the one named;
* ``filtered``: a plot description plus metadata filters (year range, genre, origin); the gold film satisfies them;
* ``unanswerable``: asks for a plausible film that is not in the index; there is no gold film and the right answer is
  to abstain.

:func:`check_against_corpus` verifies a question set against the films it was written from (gold ids exist, filters
hold for the gold film, exact questions name their film, the overlap guard passes, unanswerable films are absent).
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from movie_rag.config import Settings, load_settings
from movie_rag.errors import EvalSetError
from movie_rag.eval.overlap import contains_phrase, shared_ngrams
from movie_rag.ingest.clean import MovieRecord, clean_csv
from movie_rag.retrieval.search import SearchFilters

QuestionType = Literal["fuzzy_plot", "exact_entity", "filtered", "unanswerable"]
QUESTION_TYPES: tuple[QuestionType, ...] = ("fuzzy_plot", "exact_entity", "filtered", "unanswerable")
ID_PATTERN = r"^[a-z]+(-[a-z]+)*-[0-9]{2,}$"


class EvalQuestion(BaseModel):
    """One evaluation question with its gold answer."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=ID_PATTERN, description="Unique, for example 'fuzzy-03'.")
    type: QuestionType
    question: str = Field(min_length=10, description="What the user asks, in the user's words.")
    gold_movie_ids: list[str] = Field(
        default_factory=list,
        description="movie_id of the film that answers the question (empty for unanswerable questions).",
    )
    filters: SearchFilters = Field(
        default_factory=SearchFilters, description="Metadata filters the question carries (year range, genre, origin)."
    )
    absent_title: str | None = Field(
        default=None, description="Unanswerable only: the plausible film that is NOT in the index."
    )
    notes: str = Field(default="", description="Why the question exists and what it is meant to test.")

    @model_validator(mode="after")
    def _shape_matches_type(self) -> EvalQuestion:
        gold, has_filters = self.gold_movie_ids, bool(self.filters.model_dump(exclude_none=True))
        if len(set(gold)) != len(gold):
            raise ValueError("gold_movie_ids contains a duplicate")
        if self.type in ("fuzzy_plot", "exact_entity") and (len(gold) != 1 or has_filters):
            raise ValueError(f"{self.type} needs exactly one gold movie_id and no filters")
        if self.type == "filtered" and (not gold or not has_filters):
            raise ValueError("filtered needs at least one gold movie_id and at least one filter")
        if self.type == "unanswerable" and gold:
            raise ValueError("unanswerable questions have no gold movie_id")
        if (self.type == "unanswerable") != (self.absent_title is not None):
            raise ValueError("absent_title is required for unanswerable questions and only for them")
        return self


def read_questions(path: Path) -> list[EvalQuestion]:
    """Parse a JSONL file (blank lines ignored). Errors name the file and the line."""
    if not path.is_file():
        raise EvalSetError(f"question file not found: {path}")
    questions: list[EvalQuestion] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            questions.append(EvalQuestion.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise EvalSetError(f"{path.name} line {number}: {_first_line(exc)}") from exc
    return questions


def write_questions(questions: Iterable[EvalQuestion], path: Path) -> None:
    """Write one compact JSON object per line (unset filters omitted), creating the folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(q.model_dump(mode="json", exclude_none=True), ensure_ascii=False) for q in questions]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _first_line(error: Exception) -> str:
    return str(error).strip().splitlines()[0] if str(error).strip() else type(error).__name__


def validate_questions(questions: Sequence[EvalQuestion], per_type: int) -> None:
    """Structural checks: unique ids, exactly ``per_type`` questions of each type, no film gold twice."""
    problems: list[str] = []
    ids = Counter(q.id for q in questions)
    problems += [f"duplicate id {i}" for i, count in ids.items() if count > 1]
    counts = Counter(q.type for q in questions)
    problems += [f"{t}: {counts[t]} questions, expected {per_type}" for t in QUESTION_TYPES if counts[t] != per_type]
    gold = Counter(g for q in questions for g in q.gold_movie_ids)
    problems += [f"movie_id {g} is the gold answer of {c} questions" for g, c in gold.items() if c > 1]
    if problems:
        raise EvalSetError("invalid question set: " + "; ".join(problems))


def matches_filters(filters: SearchFilters, record: MovieRecord) -> bool:
    """Whether ``record`` passes ``filters`` the way the retrieval layer applies them."""
    if filters.year_from is not None and record.release_year < filters.year_from:
        return False
    if filters.year_to is not None and record.release_year > filters.year_to:
        return False
    if filters.genre is not None and record.genre != filters.genre:
        return False
    return filters.origin is None or record.origin == filters.origin


def check_against_corpus(
    questions: Sequence[EvalQuestion], records: Sequence[MovieRecord], ngram_size: int
) -> list[str]:
    """Problems found when the questions are held against the films they are about (empty list: all good)."""
    by_id: Mapping[str, MovieRecord] = {r.movie_id: r for r in records}
    titles = {" ".join(r.title.lower().split()) for r in records}
    problems: list[str] = []
    for q in questions:
        golds: list[MovieRecord] = []
        for movie_id in q.gold_movie_ids:
            if movie_id in by_id:
                golds.append(by_id[movie_id])
            else:
                problems.append(f"{q.id}: gold movie_id {movie_id} is not in the dataset")
        for film in golds:
            problems += _check_gold(q, film, ngram_size)
        if q.type == "unanswerable":
            problems += _check_absent(q, records, titles, ngram_size)
    return problems


def _check_gold(q: EvalQuestion, film: MovieRecord, ngram_size: int) -> list[str]:
    problems: list[str] = []
    if q.type == "exact_entity":
        names = [film.title] + ([film.director] if film.director else [])
        if not any(contains_phrase(q.question, name) for name in names):
            problems.append(f"{q.id}: does not name the title or director of {film.movie_id}")
    else:  # fuzzy and filtered questions describe the plot, so they must not name the film
        if contains_phrase(q.question, film.title):
            problems.append(f"{q.id}: names the title of {film.movie_id}")
        copied = shared_ngrams(q.question, film.plot, ngram_size)
        if copied:
            problems.append(f"{q.id}: shares {ngram_size}-word phrases with the plot of {film.movie_id}: {copied}")
    if q.type == "filtered" and not matches_filters(q.filters, film):
        problems.append(f"{q.id}: gold film {film.movie_id} does not satisfy the question's filters")
    return problems


def _check_absent(q: EvalQuestion, records: Sequence[MovieRecord], titles: set[str], ngram_size: int) -> list[str]:
    problems: list[str] = []
    if q.absent_title is not None and " ".join(q.absent_title.lower().split()) in titles:
        problems.append(f"{q.id}: absent_title {q.absent_title!r} is a film in the dataset")
    for record in records:
        copied = shared_ngrams(q.question, record.plot, ngram_size)
        if copied:
            problems.append(f"{q.id}: shares {ngram_size}-word phrases with the plot of {record.movie_id}: {copied}")
    return problems


def fixture_records(settings: Settings) -> list[MovieRecord]:
    """The cleaned fixture films the committed question set was written from."""
    path = settings.data.resolve(settings.data.fixture_path)
    records, _ = clean_csv(path, settings.ingest.min_plot_words)
    return records


def load_eval_set(settings: Settings | None = None, path: Path | None = None) -> list[EvalQuestion]:
    """The committed question set, fully validated: structure, and consistency with the fixture films.

    Raises :class:`~movie_rag.errors.EvalSetError` listing every problem found.
    """
    settings = settings or load_settings()
    cfg = settings.eval
    questions = read_questions(path or cfg.resolve(cfg.questions_path))
    validate_questions(questions, cfg.per_type)
    problems = check_against_corpus(questions, fixture_records(settings), cfg.ngram_size)
    if problems:
        raise EvalSetError("question set disagrees with the fixture: " + "; ".join(problems))
    return questions
