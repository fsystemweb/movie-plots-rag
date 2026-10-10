"""The n-gram overlap guard for fuzzy questions.

A fuzzy question must describe a film in its own words. If it shares a run of ``n`` consecutive words with the film's
plot (``n`` is ``eval.ngram_size``, 4 in ``config.yaml``), a lexical retriever such as BM25 could find the film by
matching the copied phrase, which would measure string matching and not understanding. Words are compared lowercased
with punctuation stripped (apostrophes are dropped, so ``mayor's`` becomes ``mayors``; every other non-alphanumeric
character separates words).
"""

from __future__ import annotations

import re
from collections.abc import Sequence

_APOSTROPHES = re.compile("['\u2019\u2018`]")
_WORD = re.compile(r"[^\W_]+")


def normalise_words(text: str) -> list[str]:
    """Lowercase words of ``text`` without punctuation: ``"The mayor's plan!"`` -> ``["the", "mayors", "plan"]``."""
    return _WORD.findall(_APOSTROPHES.sub("", text.lower()))


def ngrams(words: Sequence[str], n: int) -> set[tuple[str, ...]]:
    """Every run of ``n`` consecutive words (empty when there are fewer than ``n`` words)."""
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def shared_ngrams(question: str, plot: str, n: int) -> list[str]:
    """The ``n``-word phrases that appear in both ``question`` and ``plot``, sorted, as plain strings."""
    common = ngrams(normalise_words(question), n) & ngrams(normalise_words(plot), n)
    return sorted(" ".join(gram) for gram in common)


def passes_overlap_guard(question: str, plot: str, n: int) -> bool:
    """True when ``question`` shares no run of ``n`` words with ``plot``."""
    return not shared_ngrams(question, plot, n)


def contains_phrase(text: str, phrase: str) -> bool:
    """True when the words of ``phrase`` occur consecutively in ``text`` (case and punctuation ignored)."""
    wanted = normalise_words(phrase)
    words = normalise_words(text)
    if not wanted:
        return False
    return any(words[i : i + len(wanted)] == wanted for i in range(len(words) - len(wanted) + 1))
