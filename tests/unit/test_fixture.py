"""The committed synthetic fixture is part of the contract: later PRs (ingest, retrieval, eval, UI) depend on it."""

from __future__ import annotations

import csv
import importlib.util
import math
import re
from collections import Counter
from pathlib import Path
from types import ModuleType

from movie_rag.ingest.clean import RAW_COLUMNS, MovieRecord, clean_csv, count_words, read_raw_rows

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
    assert len(long_ones) >= 12


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


# --- distinctiveness: the evaluation set (PR-08) is written from these films -----------------------------------------

FORGETTING_HOUR = "The Forgetting Hour"
MASK_CAPITALISED = re.compile(r"\b[A-Z][\w'-]*")
WORD = re.compile(r"[a-z]{3,}")
REAL_PEOPLE_DENYLIST = ("Kapoor", "Brennan", "Bachchan")  # real surnames that must not appear in cast or director


def kept_records() -> list[MovieRecord]:
    return clean_csv(FIXTURE, min_plot_words=50)[0]


def masked_sentences(plot: str) -> list[str]:
    """Sentences with every capitalised word (names, sentence starts) masked, so names cannot fake uniqueness."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", plot) if s.strip()]
    return [MASK_CAPITALISED.sub("X", s) for s in sentences]


def word_ngrams(plot: str, n: int = 4) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z']+", plot.lower())
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def tfidf_nearest_neighbour_cosines(plots: list[str]) -> list[float]:
    docs = [Counter(WORD.findall(p.lower())) for p in plots]
    document_frequency = Counter(w for d in docs for w in d)
    vectors: list[dict[str, float]] = []
    for d in docs:
        weights = {w: (1 + math.log(c)) * math.log(len(docs) / document_frequency[w]) for w, c in d.items()}
        norm = math.sqrt(sum(x * x for x in weights.values()))
        vectors.append({w: x / norm for w, x in weights.items()})
    return [
        max(sum(x * other.get(w, 0.0) for w, x in v.items()) for j, other in enumerate(vectors) if j != i)
        for i, v in enumerate(vectors)
    ]


def test_generator_ships_at_least_thirty_hand_written_anchors_across_genres_origins_and_decades() -> None:
    anchors = load_generator()._load_anchors()
    assert len(anchors) >= 30
    assert len({a["genre"] for a in anchors}) >= 14
    assert len({a["origin"] for a in anchors}) >= 15
    assert len({a["year"] // 10 for a in anchors}) >= 9
    assert len({a["title"] for a in anchors}) == len(anchors)
    long_anchors = [a for a in anchors if count_words(a["plot"]) >= LONG_PLOT_WORDS]
    assert len(long_anchors) >= 12


def test_no_two_kept_films_share_a_premise_sentence() -> None:
    premises = Counter(masked_sentences(r.plot)[0] for r in kept_records())
    assert [p for p, c in premises.items() if c > 1] == []


def test_titles_are_unique_across_the_whole_file() -> None:
    titles = [str(r["Title"]) for r in read_raw_rows(FIXTURE)]
    assert len(set(titles)) == len(titles)


def test_most_of_each_plot_is_unique_to_its_film() -> None:
    """Only the closing sentence (and rarely a beat) may repeat between films; premise, twist and beats cannot."""
    sentences = [masked_sentences(r.plot) for r in kept_records()]
    occurrences = Counter(s for per_film in sentences for s in set(per_film))
    shares = [sum(occurrences[s] == 1 for s in per_film) / len(per_film) for per_film in sentences]
    assert sum(shares) / len(shares) >= 0.75
    assert min(shares) >= 0.5  # at most half of any plot is shared connective text


def test_plots_are_lexically_distinct_nearest_neighbour_tfidf() -> None:
    cosines = tfidf_nearest_neighbour_cosines([r.plot for r in kept_records()])
    assert max(cosines) < 0.4
    assert sum(c < 0.3 for c in cosines) / len(cosines) >= 0.95


def test_anchor_plots_do_not_overlap_any_other_plot() -> None:
    anchor_titles = {a["title"] for a in load_generator()._load_anchors()}
    records = kept_records()
    grams = [word_ngrams(r.plot) for r in records]
    checked = 0
    for i, record in enumerate(records):
        if record.title in anchor_titles:
            checked += 1
            overlap = max(len(grams[i] & grams[j]) / len(grams[i]) for j in range(len(records)) if j != i)
            assert overlap < 0.1, record.title
    assert checked >= 30


def test_the_memory_loss_premise_belongs_to_the_anchor_alone() -> None:
    hits = [r.title for r in kept_records() if re.search(r"\b(memory|amnesia|amnesiac)\b", r.plot, re.IGNORECASE)]
    assert hits == [FORGETTING_HOUR]


def test_cast_and_directors_avoid_a_denylist_of_real_surnames() -> None:
    people = " ".join(f"{r['Director']} {r['Cast']}" for r in read_raw_rows(FIXTURE))
    assert [name for name in REAL_PEOPLE_DENYLIST if name in people] == []
