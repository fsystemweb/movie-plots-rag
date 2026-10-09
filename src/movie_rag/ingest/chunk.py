"""Sentence-aware chunking of film plots (KICKOFF section 1, "Ingestion").

Definitions
-----------
* A **token** is whatever the ``TokenCounter`` passed in says. Production uses the dense model's own WordPiece
  tokenizer without its two special tokens (see :meth:`movie_rag.ingest.embed.FastEmbedder.count_tokens`), so
  ``ingest.chunk_tokens`` is measured in the units the embedding model actually sees.
* The **budget** (``ingest.chunk_tokens``, default 250) applies to the plot text of a chunk. The one-line header
  ``Title (Year) | Genre | Director`` is prepended to *every* chunk on top of that (about 15 more tokens), which keeps a
  chunk far below the 512-token limit of the dense model. Sums are taken per sentence, so a chunk is "about" the
  budget: it can overshoot by the whitespace/rounding difference between summing sentences and counting the joined text.
* A plot whose total is at most the budget stays **whole** (one chunk).
* Longer plots are cut at **sentence boundaries**. Consecutive chunks overlap by the trailing whole sentences of the
  previous chunk that fit in ``ingest.chunk_overlap`` tokens (possibly none when the last sentence is longer than the
  overlap). Every chunk contains at least one sentence that its predecessor did not, so chunking always terminates.
* A single sentence longer than the budget (rare) is split at word boundaries.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict

from movie_rag.ingest.clean import MovieRecord

TokenCounter = Callable[[str], int]

HEADER_SEPARATOR = " | "
HEADER_TEXT_SEPARATOR = "\n"

# Sentence end: ., ! or ? (optionally followed by closing quotes/brackets), whitespace, then something that can start a
# sentence. Capital-letter lookahead keeps "approx. three years" and "3.5 million" intact.
_BOUNDARY = re.compile(r"""([.!?]+["')\]]*)\s+(?=["'(\[]?[A-Z0-9])""")
_INITIALISM = re.compile(r"^(?:[A-Za-z]\.)*[A-Za-z]$")
_ABBREVIATIONS = frozenset(
    {"mr", "mrs", "ms", "dr", "st", "jr", "sr", "vs", "etc", "no", "inc", "ltd", "co", "col", "gen", "capt", "lt"}
    | {"sgt", "mt", "prof", "rev", "hon", "gov", "sen", "rep", "fig", "approx", "dept", "est"}
)


class Chunk(BaseModel):
    """One passage of a film's plot, ready to embed."""

    model_config = ConfigDict(frozen=True)

    movie_id: str
    chunk_idx: int
    n_chunks: int
    header: str
    text: str

    @property
    def embed_text(self) -> str:
        """What gets embedded (dense and BM25): the header, a newline, then the plot passage."""
        return f"{self.header}{HEADER_TEXT_SEPARATOR}{self.text}"


def build_header(record: MovieRecord) -> str:
    """``Title (Year) | Genre | Director``; a missing genre or director is left out rather than printed as ``None``."""
    parts = [f"{record.title} ({record.release_year})"]
    parts.extend(part for part in (record.genre, record.director) if part)
    return HEADER_SEPARATOR.join(parts)


def _is_abbreviation(preceding: str) -> bool:
    last_word = preceding.rsplit(None, 1)[-1] if preceding.strip() else ""
    return last_word.lower() in _ABBREVIATIONS or bool(_INITIALISM.match(last_word))


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences, keeping the end punctuation. Never returns empty strings."""
    sentences: list[str] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        punctuation = match.group(1)
        if punctuation == "." and _is_abbreviation(text[start : match.start(1)]):
            continue
        sentences.append(text[start : match.end(1)])
        start = match.end()
    sentences.append(text[start:])
    return [s.strip() for s in sentences if s.strip()]


def _split_long_sentence(sentence: str, max_tokens: int, count: TokenCounter) -> list[str]:
    """Split an over-budget sentence on word boundaries into pieces that each fit (a lone huge word stays)."""
    pieces: list[str] = []
    current: list[str] = []
    used = 0
    for word in sentence.split():
        size = count(word)
        if current and used + size > max_tokens:
            pieces.append(" ".join(current))
            current, used = [], 0
        current.append(word)
        used += size
    pieces.append(" ".join(current))  # the sentence is over budget, hence non-empty
    return pieces


def chunk_text(plot: str, max_tokens: int, overlap: int, count: TokenCounter) -> list[str]:
    """Split ``plot`` into passages of about ``max_tokens`` tokens with up to ``overlap`` tokens of shared sentences."""
    plot = plot.strip()
    if not plot:
        return []
    if count(plot) <= max_tokens:
        return [plot]
    sentences: list[str] = []
    for sentence in split_sentences(plot):
        if count(sentence) > max_tokens:
            sentences.extend(_split_long_sentence(sentence, max_tokens, count))
        else:
            sentences.append(sentence)
    sizes = [count(s) for s in sentences]

    chunks: list[str] = []
    start, total = 0, len(sentences)
    while True:  # ends via the break: every pass consumes at least one new sentence
        end, used = start, 0
        while end < total and (end == start or used + sizes[end] <= max_tokens):
            used += sizes[end]
            end += 1
        chunks.append(" ".join(sentences[start:end]))
        if end >= total:
            break
        next_start, shared = end, 0
        while next_start - 1 > start and shared + sizes[next_start - 1] <= overlap:
            next_start -= 1
            shared += sizes[next_start]
        start = next_start
    return chunks


def chunk_record(record: MovieRecord, max_tokens: int, overlap: int, count: TokenCounter) -> list[Chunk]:
    """All chunks of one film, indexed from 0, each carrying the film header."""
    texts = chunk_text(record.plot, max_tokens, overlap, count)
    header = build_header(record)
    return [
        Chunk(movie_id=record.movie_id, chunk_idx=i, n_chunks=len(texts), header=header, text=text)
        for i, text in enumerate(texts)
    ]
