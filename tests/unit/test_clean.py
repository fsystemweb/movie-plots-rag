from __future__ import annotations

import csv
from pathlib import Path

import pytest

from movie_rag.errors import DataError
from movie_rag.ingest.clean import (
    RAW_COLUMNS,
    CleanStats,
    MovieRecord,
    clean_csv,
    clean_row,
    clean_rows,
    count_words,
    make_movie_id,
    normalise_label,
    parse_year,
    read_raw_rows,
    sample_records,
    slugify,
)

MIN_WORDS = 50


def words(n: int) -> str:
    return " ".join(f"word{i}" for i in range(n))


def row(**overrides: str | None) -> dict[str, str | None]:
    base: dict[str, str | None] = {
        "Release Year": "1999",
        "Title": "The Matrix Has Eyes",
        "Origin/Ethnicity": "American",
        "Director": "A. Director",
        "Cast": "X, Y",
        "Genre": "Science Fiction",
        "Wiki Page": "https://en.wikipedia.org/wiki/Example",
        "Plot": words(60),
    }
    base.update(overrides)
    return base


# --- slug and id ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("title", "slug"),
    [
        ("The Matrix", "the-matrix"),
        ("  Amélie!  ", "amelie"),
        ("Dr. Strangelove: Or, How I Learned...", "dr-strangelove-or-how-i-learned"),
        ("Se7en", "se7en"),
        ("!!!", "untitled"),
        ("東京物語", "untitled"),
        ("", "untitled"),
    ],
)
def test_slugify(title: str, slug: str) -> None:
    assert slugify(title) == slug


def test_movie_id_is_slug_year_and_row_index() -> None:
    assert make_movie_id("The Matrix", 1999, 42) == "the-matrix-1999-42"


def test_movie_id_is_stable_and_distinguishes_remakes_and_duplicates() -> None:
    assert make_movie_id("Ben-Hur", 1959, 7) == make_movie_id("Ben-Hur", 1959, 7)
    assert make_movie_id("Ben-Hur", 1925, 3) != make_movie_id("Ben-Hur", 1959, 7)
    assert make_movie_id("Ben-Hur", 1959, 7) != make_movie_id("Ben-Hur", 1959, 8)


# --- labels, years, words ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Drama", "drama"),
        ("  SCIENCE   Fiction ", "science fiction"),
        ("Unknown", None),
        ("UNKNOWN", None),
        ("  unknown ", None),
        ("", None),
        ("   ", None),
        (None, None),
        ("unknown genre", "unknown genre"),
    ],
)
def test_normalise_label(raw: str | None, expected: str | None) -> None:
    assert normalise_label(raw) == expected


@pytest.mark.parametrize(
    ("raw", "year"),
    [("1999", 1999), (" 2017 ", 2017), ("99", None), ("19999", None), ("abc", None), ("", None), (None, None)],
)
def test_parse_year(raw: str | None, year: int | None) -> None:
    assert parse_year(raw) == year


def test_count_words_splits_on_any_whitespace() -> None:
    assert count_words("a  b\nc\td") == 4
    assert count_words("") == 0


# --- row cleaning --------------------------------------------------------------------------------------


def test_clean_row_normalises_fields() -> None:
    record = clean_row(row(Genre="Film Noir", **{"Origin/Ethnicity": "British"}), 5, MIN_WORDS)
    assert isinstance(record, MovieRecord)
    assert record.movie_id == "the-matrix-has-eyes-1999-5"
    assert record.genre == "film noir"
    assert record.origin == "british"
    assert record.release_year == 1999
    assert record.director == "A. Director"  # case kept
    assert record.wiki_url == "https://en.wikipedia.org/wiki/Example"
    assert record.plot == words(60)


def test_unknown_genre_origin_director_cast_become_null() -> None:
    record = clean_row(
        row(Genre="unknown", Director="Unknown", Cast="unknown", **{"Origin/Ethnicity": "Unknown"}), 0, MIN_WORDS
    )
    assert isinstance(record, MovieRecord)
    assert (record.genre, record.origin, record.director, record.cast) == (None, None, None, None)


def test_missing_optional_cells_become_null() -> None:
    record = clean_row(row(Director=None, Cast=None, **{"Wiki Page": None}), 0, MIN_WORDS)
    assert isinstance(record, MovieRecord)
    assert (record.director, record.cast, record.wiki_url) == (None, None, None)


def test_blank_wiki_page_becomes_null() -> None:
    record = clean_row(row(**{"Wiki Page": " "}), 0, MIN_WORDS)
    assert isinstance(record, MovieRecord)
    assert record.wiki_url is None


@pytest.mark.parametrize(("n_words", "kept"), [(0, False), (49, False), (50, True), (51, True)])
def test_plot_length_boundary(n_words: int, kept: bool) -> None:
    result = clean_row(row(Plot=words(n_words)), 0, MIN_WORDS)
    assert isinstance(result, MovieRecord) is kept
    if not kept:
        assert result == "short_plot"


def test_threshold_comes_from_the_argument() -> None:
    assert clean_row(row(Plot=words(10)), 0, 5) != "short_plot"
    assert clean_row(row(Plot=words(10)), 0, 11) == "short_plot"


def test_plot_whitespace_is_collapsed_before_counting() -> None:
    padded = ("word  \n\n " * 49).strip()  # 49 words with lots of whitespace
    assert clean_row(row(Plot=padded), 0, MIN_WORDS) == "short_plot"
    record = clean_row(row(Plot=words(60).replace(" ", "\r\n  ")), 0, MIN_WORDS)
    assert isinstance(record, MovieRecord)
    assert "\n" not in record.plot
    assert record.plot == words(60)


@pytest.mark.parametrize(
    "bad",
    [
        {"Title": ""},
        {"Title": "   "},
        {"Title": None},
        {"Release Year": ""},
        {"Release Year": "n/a"},
        {"Release Year": None},
    ],
)
def test_rows_without_title_or_year_are_invalid(bad: dict[str, str | None]) -> None:
    assert clean_row(row(**bad), 0, MIN_WORDS) == "invalid"


def test_missing_columns_in_a_mapping_are_invalid_not_crashes() -> None:
    assert clean_row({}, 0, MIN_WORDS) == "invalid"


# --- batches -------------------------------------------------------------------------------------------


def test_clean_rows_counts_every_outcome_and_keeps_raw_row_indexes() -> None:
    rows = [
        row(Title="Alpha"),
        row(Title="Beta", Plot=words(3)),
        row(Title="Gamma", **{"Release Year": "oops"}),
        row(Title="Delta"),
    ]
    records, stats = clean_rows(rows, MIN_WORDS)
    assert [r.movie_id for r in records] == ["alpha-1999-0", "delta-1999-3"]
    assert stats == CleanStats(rows_read=4, rows_kept=2, dropped_short_plot=1, dropped_invalid=1)
    assert stats.rows_dropped == 2


def test_ids_do_not_depend_on_other_rows_being_dropped() -> None:
    kept_alone, _ = clean_rows([row(Title="Alpha"), row(Title="Delta")], MIN_WORDS)
    kept_among_junk, _ = clean_rows([row(Title="Alpha"), row(Title="Junk", Plot=""), row(Title="Delta")], MIN_WORDS)
    assert kept_alone[1].movie_id == "delta-1999-1"
    assert kept_among_junk[1].movie_id == "delta-1999-2"  # position in the raw file, not in the kept list


def test_cleaning_twice_gives_identical_ids() -> None:
    rows = [row(Title=f"Film {i}", **{"Release Year": str(1950 + i)}) for i in range(20)]
    first, _ = clean_rows(rows, MIN_WORDS)
    second, _ = clean_rows(rows, MIN_WORDS)
    assert [r.movie_id for r in first] == [r.movie_id for r in second]
    assert len({r.movie_id for r in first}) == 20


def test_duplicate_titles_and_years_still_get_unique_ids() -> None:
    records, _ = clean_rows([row(Title="Same"), row(Title="Same")], MIN_WORDS)
    assert records[0].movie_id != records[1].movie_id


# --- files ---------------------------------------------------------------------------------------------


def write_csv(path: Path, rows: list[dict[str, str | None]], columns: tuple[str, ...] = RAW_COLUMNS) -> Path:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_read_raw_rows_round_trips_multiline_quoted_fields(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "m.csv", [row(Plot="line one\nline two, with comma")])
    assert read_raw_rows(path)[0]["Plot"] == "line one\nline two, with comma"


def test_missing_file_gets_an_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="make download"):
        read_raw_rows(tmp_path / "nope.csv")


def test_missing_columns_are_named(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "m.csv", [row()], columns=("Title", "Plot"))
    with pytest.raises(DataError, match="Release Year"):
        read_raw_rows(path)


def test_clean_csv_reads_cleans_and_logs(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = write_csv(tmp_path / "m.csv", [row(Title="Alpha"), row(Title="Beta", Plot=words(2))])
    with caplog.at_level("INFO", logger="movie_rag.ingest.clean"):
        records, stats = clean_csv(path, MIN_WORDS)
    assert [r.title for r in records] == ["Alpha"]
    assert stats.rows_read == 2
    assert "rows_read=2 kept=1 dropped_short_plot=1 dropped_invalid=0" in caplog.text


# --- sampling ------------------------------------------------------------------------------------------


def test_sampling_is_deterministic_for_a_seed_and_keeps_input_order() -> None:
    records, _ = clean_rows([row(Title=f"Film {i}") for i in range(50)], MIN_WORDS)
    a = sample_records(records, 10, seed=42)
    b = sample_records(records, 10, seed=42)
    c = sample_records(records, 10, seed=43)
    assert a == b
    assert a != c
    assert len(a) == len({r.movie_id for r in a}) == 10
    positions = [records.index(r) for r in a]
    assert positions == sorted(positions)


def test_sampling_more_than_available_returns_everything() -> None:
    records, _ = clean_rows([row(Title="Only")], MIN_WORDS)
    assert sample_records(records, 5, seed=1) == records
