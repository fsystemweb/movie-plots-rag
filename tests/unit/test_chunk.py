from __future__ import annotations

import random
from itertools import pairwise
from pathlib import Path

import pytest

from movie_rag.ingest.chunk import (
    HEADER_TEXT_SEPARATOR,
    Chunk,
    build_header,
    chunk_record,
    chunk_text,
    split_sentences,
)
from movie_rag.ingest.clean import MovieRecord, clean_csv


def words(text: str) -> int:
    """Test token counter: one token per whitespace-separated word (exact, so budgets can be asserted tightly)."""
    return len(text.split())


def make_plot(n_sentences: int, words_per_sentence: int = 10) -> str:
    """``n_sentences`` sentences of ``words_per_sentence`` words each; sentence i is recognisable as ``s{i}``."""
    return " ".join(
        " ".join([f"S{i}"] + [f"w{i}x{j}" for j in range(words_per_sentence - 2)] + ["end."])
        for i in range(n_sentences)
    )


def make_record(plot: str, **overrides: object) -> MovieRecord:
    fields: dict[str, object] = {
        "movie_id": "the-heist-1999-7",
        "title": "The Heist",
        "release_year": 1999,
        "origin": "american",
        "director": "Ann Lee",
        "cast": "A, B",
        "genre": "thriller",
        "wiki_url": "https://example.test/wiki/The_Heist",
        "plot": plot,
    }
    fields.update(overrides)
    return MovieRecord.model_validate(fields)


# --- sentence splitting ----------------------------------------------------------------------------------


def test_split_sentences_basic() -> None:
    assert split_sentences("He ran. She hid! Who knew? Nobody.") == ["He ran.", "She hid!", "Who knew?", "Nobody."]


def test_split_sentences_keeps_quotes_and_brackets_with_the_sentence() -> None:
    text = 'He said "Run." Then (as planned) he left.) The end.'
    assert split_sentences(text) == ['He said "Run."', "Then (as planned) he left.)", "The end."]


@pytest.mark.parametrize(
    "text",
    [
        "Dr. Smith met Mr. Jones in St. Louis.",
        "J. R. Ewing arrived at 3.5 million dollars.",
        "It happened in the U.S. Army camp near Fort Dix.",
        "They waited approx. three years.",
    ],
)
def test_split_sentences_does_not_split_on_abbreviations_initials_and_decimals(text: str) -> None:
    assert split_sentences(text) == [text]


def test_split_sentences_without_terminal_punctuation_and_blank_input() -> None:
    assert split_sentences("no punctuation at all") == ["no punctuation at all"]
    assert split_sentences("   ") == []


def test_split_sentences_ellipsis_followed_by_capital_splits() -> None:
    assert split_sentences("He waited... Then he left.") == ["He waited...", "Then he left."]


# --- chunk_text ------------------------------------------------------------------------------------------


def test_short_plot_is_a_single_whole_chunk() -> None:
    plot = make_plot(5)
    assert chunk_text(plot, 250, 40, words) == [plot]


def test_plot_exactly_at_the_budget_stays_whole_and_one_over_is_split() -> None:
    at_budget = make_plot(25)  # 250 words
    assert words(at_budget) == 250
    assert chunk_text(at_budget, 250, 40, words) == [at_budget]
    assert len(chunk_text(make_plot(26), 250, 40, words)) > 1


def test_empty_plot_has_no_chunks() -> None:
    assert chunk_text("   ", 250, 40, words) == []


def test_long_plot_is_cut_only_at_sentence_boundaries_and_respects_the_budget() -> None:
    plot = make_plot(60)  # 600 words
    chunks = chunk_text(plot, 250, 40, words)
    assert len(chunks) > 2
    for chunk in chunks:
        assert words(chunk) <= 250
        assert chunk.startswith("S")  # starts at the first word of a sentence
        assert chunk.endswith("end.")  # ends with a complete sentence
        assert len(split_sentences(chunk)) == words(chunk) // 10


def test_chunks_overlap_by_whole_sentences_within_the_overlap_budget() -> None:
    chunks = chunk_text(make_plot(60), 250, 40, words)
    for previous, following in pairwise(chunks):
        previous_sentences, following_sentences = split_sentences(previous), split_sentences(following)
        shared = [s for s in following_sentences if s in previous_sentences]
        assert shared, "consecutive chunks must share their boundary sentences"
        assert following_sentences[: len(shared)] == shared  # the overlap leads the next chunk
        assert previous_sentences[-len(shared) :] == shared  # and trails the previous one
        assert sum(words(s) for s in shared) <= 40
        assert len(following_sentences) > len(shared)  # progress: every chunk adds a new sentence


def test_chunks_cover_every_sentence_in_order() -> None:
    plot = make_plot(47)
    original = split_sentences(plot)
    seen: list[str] = []
    for chunk in chunk_text(plot, 250, 40, words):
        for sentence in split_sentences(chunk):
            if sentence not in seen:
                seen.append(sentence)
    assert seen == original


def test_zero_overlap_shares_nothing() -> None:
    chunks = chunk_text(make_plot(60), 250, 0, words)
    sentences = [s for c in chunks for s in split_sentences(c)]
    assert len(sentences) == len(set(sentences)) == 60


def test_overlap_smaller_than_one_sentence_shares_nothing() -> None:
    chunks = chunk_text(make_plot(60), 250, 9, words)  # sentences have 10 words
    sentences = [s for c in chunks for s in split_sentences(c)]
    assert len(sentences) == len(set(sentences))


def test_a_sentence_longer_than_the_budget_is_split_on_words() -> None:
    giant = " ".join(f"g{i}" for i in range(620)) + "."
    plot = "Before it began. " + giant + " After it ended."
    chunks = chunk_text(plot, 250, 40, words)
    assert all(words(c) <= 250 for c in chunks)
    assert "g0" in chunks[0] and "g619." in " ".join(chunks)
    assert chunks[-1].endswith("After it ended.")


def test_a_single_word_larger_than_the_budget_still_terminates() -> None:
    counter = lambda text: sum(10 if w == "huge" else 1 for w in text.split())  # noqa: E731
    chunks = chunk_text("Start now. huge huge. End now.", 5, 0, counter)
    assert [c for c in chunks if "huge" in c]
    assert "End now." in chunks[-1]


@pytest.mark.parametrize("seed", range(20))
def test_chunking_terminates_and_covers_random_plots(seed: int) -> None:
    rng = random.Random(seed)
    sentences = [
        " ".join(f"T{seed}_{i}_{j}" for j in range(rng.randint(1, 40))) + "." for i in range(rng.randint(1, 80))
    ]
    plot = " ".join(sentences)
    max_tokens, overlap = rng.randint(5, 120), rng.randint(0, 60)
    chunks = chunk_text(plot, max_tokens, overlap, words)
    covered = {s for c in chunks for s in split_sentences(c)}
    long_ones = [s for s in sentences if words(s) > max_tokens]
    assert all(s in covered for s in sentences if s not in long_ones)
    if not long_ones:
        assert all(words(c) <= max_tokens for c in chunks)


# --- headers and records ---------------------------------------------------------------------------------


def test_build_header_format() -> None:
    assert build_header(make_record("x")) == "The Heist (1999) | thriller | Ann Lee"


def test_build_header_omits_missing_genre_and_director() -> None:
    assert build_header(make_record("x", genre=None)) == "The Heist (1999) | Ann Lee"
    assert build_header(make_record("x", director=None)) == "The Heist (1999) | thriller"
    assert build_header(make_record("x", genre=None, director=None)) == "The Heist (1999)"


def test_every_chunk_carries_the_header_and_its_position() -> None:
    record = make_record(make_plot(60))
    chunks = chunk_record(record, 250, 40, words)
    assert len(chunks) > 2
    assert [c.chunk_idx for c in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        assert chunk.header == "The Heist (1999) | thriller | Ann Lee"
        assert chunk.n_chunks == len(chunks)
        assert chunk.movie_id == record.movie_id
        assert chunk.embed_text == f"{chunk.header}{HEADER_TEXT_SEPARATOR}{chunk.text}"
        assert chunk.embed_text.startswith("The Heist (1999) | thriller | Ann Lee\n")


def test_short_plot_record_has_one_chunk_with_the_header() -> None:
    (chunk,) = chunk_record(make_record(make_plot(3)), 250, 40, words)
    assert (chunk.chunk_idx, chunk.n_chunks) == (0, 1)
    assert chunk.text == make_plot(3)
    assert chunk.embed_text.splitlines()[0] == "The Heist (1999) | thriller | Ann Lee"


def test_chunk_is_immutable() -> None:
    chunk = Chunk(movie_id="a", chunk_idx=0, n_chunks=1, header="h", text="t")
    with pytest.raises(ValueError, match="frozen"):
        chunk.text = "other"  # type: ignore[misc]


def test_fixture_long_plots_are_chunked_and_short_ones_stay_whole(fixture_csv: Path) -> None:
    records, _ = clean_csv(fixture_csv, 50)
    per_film = {r.movie_id: chunk_record(r, 250, 40, words) for r in records}
    multi = [chunks for chunks in per_film.values() if len(chunks) > 1]
    assert len(multi) == 13  # the fixture's 13 plots over 250 words
    assert all(len(chunks) == 1 for r in records if words(r.plot) <= 250 for chunks in [per_film[r.movie_id]])
    for chunks in per_film.values():
        assert all(c.embed_text.startswith(c.header) for c in chunks)
