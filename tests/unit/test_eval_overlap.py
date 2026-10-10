"""The n-gram overlap guard: positive and negative examples, normalisation, boundaries."""

from __future__ import annotations

import pytest

from movie_rag.eval.overlap import contains_phrase, ngrams, normalise_words, passes_overlap_guard, shared_ngrams

PLOT = "Hollis Merrick, an auction house porter, is the only one to notice the swapped lot. The mayor's plan fails."


def test_words_are_lowercased_and_stripped_of_punctuation() -> None:
    assert normalise_words("The Mayor's plan -- fails!") == ["the", "mayors", "plan", "fails"]


def test_hyphens_and_underscores_separate_words_and_accents_survive() -> None:
    assert normalise_words("window-cleaner_s Yaşarel") == ["window", "cleaner", "s", "yaşarel"]


def test_curly_apostrophes_are_dropped_like_straight_ones() -> None:
    assert normalise_words("mayor\u2019s") == normalise_words("mayor's") == ["mayors"]


def test_ngrams_of_a_short_text_are_empty() -> None:
    assert ngrams(["a", "b", "c"], 4) == set()
    assert ngrams([], 1) == set()


def test_ngrams_are_consecutive_runs() -> None:
    assert ngrams(["a", "b", "c", "d"], 3) == {("a", "b", "c"), ("b", "c", "d")}


def test_ngram_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        ngrams(["a"], 0)


def test_a_copied_four_word_run_is_found_whatever_its_case_and_punctuation() -> None:
    question = "Which film has AN AUCTION-HOUSE porter who finds a fake?"
    assert shared_ngrams(question, PLOT, 4) == ["an auction house porter"]
    assert not passes_overlap_guard(question, PLOT, 4)


def test_three_shared_words_are_allowed_at_n_equals_four() -> None:
    question = "An auction house clerk spots a forgery"
    assert shared_ngrams(question, PLOT, 4) == []
    assert passes_overlap_guard(question, PLOT, 4)
    assert not passes_overlap_guard(question, PLOT, 3)  # the size really is a parameter


def test_scattered_shared_words_are_not_an_overlap() -> None:
    assert passes_overlap_guard("the porter notices the only lot swapped", PLOT, 4)


def test_a_phrase_spanning_a_sentence_boundary_counts() -> None:
    assert shared_ngrams("the swapped lot the mayors", PLOT, 4) == ["swapped lot the mayors", "the swapped lot the"]


def test_apostrophes_do_not_hide_a_copied_phrase() -> None:
    assert not passes_overlap_guard("Why does the mayors plan fail so badly", PLOT, 3)


def test_an_empty_question_passes() -> None:
    assert passes_overlap_guard("", PLOT, 4)


def test_contains_phrase_matches_whole_words_in_order() -> None:
    assert contains_phrase("Tell me about THE FORGETTING HOUR, please", "The Forgetting Hour")
    assert not contains_phrase("an hour of forgetting", "The Forgetting Hour")
    assert not contains_phrase("the forgetting hours", "The Forgetting Hour")
    assert not contains_phrase("anything", "!!!")
