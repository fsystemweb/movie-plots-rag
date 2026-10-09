"""The committed synthetic fixture is part of the contract: later PRs (ingest, retrieval, eval, UI) depend on it."""

from __future__ import annotations

import csv
import importlib.util
from collections import Counter
from pathlib import Path
from types import ModuleType

from movie_rag.ingest.clean import RAW_COLUMNS, clean_csv, count_words, read_raw_rows

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "movies_sample.csv"
GENERATOR = ROOT / "tests" / "fixtures" / "build_movies_sample.py"
LONG_PLOT_WORDS = 250  # a plot of this many words is well over 250 tokens and must be chunked later


def load_generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_movies_sample", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixture_has_exact_kaggle_schema_and_about_300_rows() -> None:
    with FIXTURE.open(encoding="utf-8", newline="") as fh:
        assert tuple(csv.DictReader(fh).fieldnames or ()) == RAW_COLUMNS
    assert len(read_raw_rows(FIXTURE)) == 300


def test_fixture_is_byte_identical_to_the_generator_output(tmp_path: Path) -> None:
    generator = load_generator()
    regenerated = tmp_path / "regenerated.csv"
    generator.write_csv(regenerated)
    assert regenerated.read_bytes() == FIXTURE.read_bytes()


def test_fixture_is_clearly_synthetic() -> None:
    rows = read_raw_rows(FIXTURE)
    assert all(str(r["Wiki Page"]).endswith("_(synthetic_film)") for r in rows)
    readme = (FIXTURE.parent / "README.md").read_text()
    assert "synthetic" in readme.lower()


def test_fixture_covers_genres_origins_and_decades() -> None:
    records, _ = clean_csv(FIXTURE, min_plot_words=50)
    assert len({r.genre for r in records if r.genre}) >= 10
    assert len({r.origin for r in records if r.origin}) >= 8
    assert len({r.release_year // 10 for r in records}) >= 8
    assert any(r.genre is None for r in records)
    assert any(r.origin is None for r in records)


def test_fixture_has_short_plots_that_cleaning_drops() -> None:
    raw = read_raw_rows(FIXTURE)
    short = [r for r in raw if count_words(str(r["Plot"])) < 50]
    assert 5 <= len(short) <= 15
    records, stats = clean_csv(FIXTURE, min_plot_words=50)
    assert stats.rows_read == 300
    assert stats.dropped_short_plot == len(short)
    assert stats.dropped_invalid == 0
    assert len(records) == 300 - len(short)


def test_fixture_exercises_the_exact_word_boundary() -> None:
    lengths = Counter(count_words(str(r["Plot"])) for r in read_raw_rows(FIXTURE))
    assert lengths[49] >= 1  # dropped
    assert lengths[50] >= 1  # kept


def test_fixture_has_long_plots_for_chunking() -> None:
    records, _ = clean_csv(FIXTURE, min_plot_words=50)
    long_ones = [r for r in records if count_words(r.plot) >= LONG_PLOT_WORDS]
    assert len(long_ones) >= 5


def test_fixture_ids_are_unique_and_titles_are_unique() -> None:
    records, _ = clean_csv(FIXTURE, min_plot_words=50)
    assert len({r.movie_id for r in records}) == len(records)
    assert len({r.title for r in records}) == len(records)


def test_fixture_contains_the_demo_anchor_films() -> None:
    records, _ = clean_csv(FIXTURE, min_plot_words=50)
    by_title = {r.title: r for r in records}
    memory = by_title["The Forgetting Hour"]
    assert memory.genre == "film noir"
    assert "no memory" in memory.plot


def test_generator_is_deterministic() -> None:
    generator = load_generator()
    assert generator.build_rows() == generator.build_rows()
