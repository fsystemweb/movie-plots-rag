"""Deterministic evaluation metrics: Hit@k, MRR, abstention and latency/token summaries. No LLM, no I/O.

Conventions (all covered by hand-computed tests):

* A *rank* is 1-based: the position of the best-placed gold film in the ranked list of ``movie_id`` values, or ``None``
  when no gold film is in the list. With several gold films the best-placed one counts. A film listed twice is ranked
  at its first position and the repeat does not push later films down.
* **Ties.** Films with the same score are in no defined order (Qdrant's reciprocal rank fusion gives equal scores to a
  film that is first in one list and second in the other, and returns the tied films in a varying order), so a
  tie-aware rank is used when scores are given: the best-placed gold film may be at any position of its tie group with
  equal probability, and Hit@k and the reciprocal rank are the expectations over that. Without ties the rank is exact
  and nothing changes. A tie group cut by the end of the list is judged by the part that is visible.
* ``Hit@k`` is 1 when the rank is at most ``k`` (``k`` is inclusive: rank 3 is a hit at k=3, a miss at k=2).
* ``MRR`` is the mean of ``1/rank`` (0 for a miss) over the questions that have a gold film.
* Questions without a gold film (unanswerable ones) are excluded from Hit@k and MRR, and are what abstention is about.
* An undefined metric (no question to average over) is ``None``, never 0.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence

from pydantic import BaseModel, Field

TIE_REL_TOL = 1e-9  # scores closer than this (relative) are the same score: float noise, not a ranking


class RankedQuestion(BaseModel):
    """What a retrieval run returned for one question, with the answer key."""

    question_id: str
    type: str
    gold_movie_ids: list[str]
    ranked_movie_ids: list[str]
    scores: list[float] | None = None  # same length as ranked_movie_ids; enables tie-aware ranks


class RetrievalMetrics(BaseModel):
    """Hit@k and MRR over the questions that have a gold film."""

    n: int = Field(description="Questions averaged over (those with at least one gold film).")
    hit_at_k: dict[int, float | None]
    mrr: float | None


class Abstention(BaseModel):
    """What the system did on one question and what it should have done."""

    should_abstain: bool
    abstained: bool


class AbstentionMetrics(BaseModel):
    """Abstention behaviour over unanswerable and answerable questions."""

    n_unanswerable: int
    n_answerable: int
    correct_abstention_rate: float | None = Field(description="Abstained on unanswerable questions (higher is better).")
    false_abstention_rate: float | None = Field(description="Abstained on answerable questions (lower is better).")


class Summary(BaseModel):
    """p50 / p95 / mean of a series (``None`` for an empty series)."""

    n: int
    p50: float | None
    p95: float | None
    mean: float | None


def first_gold_rank(ranked_movie_ids: Sequence[str], gold_movie_ids: Iterable[str]) -> int | None:
    """1-based rank of the best-placed gold film in ``ranked_movie_ids`` (``None`` if absent or there is no gold)."""
    gold = set(gold_movie_ids)
    seen: set[str] = set()
    position = 0
    for movie_id in ranked_movie_ids:
        if movie_id in seen:
            continue
        seen.add(movie_id)
        position += 1
        if movie_id in gold:
            return position
    return None


def rank_distribution(
    ranked_movie_ids: Sequence[str], gold_movie_ids: Iterable[str], scores: Sequence[float] | None = None
) -> list[tuple[int, float]]:
    """Where the best-placed gold film is, as ``(rank, probability)`` pairs summing to 1 (empty: not retrieved).

    Without ``scores`` (or without ties) this is ``[(rank, 1.0)]``. With a tie group of ``g`` films that holds ``m``
    gold films and starts at rank ``a``, the best gold film is at rank ``a + j`` with probability
    ``C(g - j - 1, m - 1) / C(g, m)`` (the tied films come in uniformly random order).
    """
    if scores is not None and len(scores) != len(ranked_movie_ids):
        raise ValueError("scores and ranked_movie_ids must have the same length")
    gold = set(gold_movie_ids)
    seen: set[str] = set()
    films: list[tuple[str, float | None]] = []
    for index, movie_id in enumerate(ranked_movie_ids):
        if movie_id not in seen:
            seen.add(movie_id)
            films.append((movie_id, None if scores is None else scores[index]))
    start = 0
    while start < len(films):
        end = start + 1
        score = films[start][1]
        while score is not None and end < len(films) and _tied(score, films[end][1]):
            end += 1
        group_gold = sum(movie_id in gold for movie_id, _ in films[start:end])
        if group_gold:
            size = end - start
            total = math.comb(size, group_gold)
            return [
                (start + j + 1, math.comb(size - j - 1, group_gold - 1) / total) for j in range(size - group_gold + 1)
            ]
        start = end
    return []


def _tied(a: float, b: float | None) -> bool:
    return b is not None and math.isclose(a, b, rel_tol=TIE_REL_TOL, abs_tol=0.0)


def expected_hit_at_k(
    ranked_movie_ids: Sequence[str],
    gold_movie_ids: Iterable[str],
    k: int,
    scores: Sequence[float] | None = None,
) -> float:
    """Tie-aware Hit@k of one question: the probability that the best gold film is in the top ``k``."""
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    return sum(p for rank, p in rank_distribution(ranked_movie_ids, gold_movie_ids, scores) if rank <= k)


def expected_reciprocal_rank(
    ranked_movie_ids: Sequence[str], gold_movie_ids: Iterable[str], scores: Sequence[float] | None = None
) -> float:
    """Tie-aware reciprocal rank of one question (0 when no gold film was retrieved)."""
    return sum(p / rank for rank, p in rank_distribution(ranked_movie_ids, gold_movie_ids, scores))


def hit_at_k(rank: int | None, k: int) -> bool:
    """Whether a film at ``rank`` is among the top ``k``."""
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    return rank is not None and rank <= k


def reciprocal_rank(rank: int | None) -> float:
    """``1/rank``, or 0 when the gold film was not retrieved."""
    if rank is not None and rank < 1:
        raise ValueError(f"ranks start at 1, got {rank}")
    return 0.0 if rank is None else 1.0 / rank


def retrieval_metrics(questions: Iterable[RankedQuestion], k_values: Sequence[int]) -> RetrievalMetrics:
    """Hit@k for every ``k`` in ``k_values`` and MRR, over the questions with a gold film."""
    answerable = [q for q in questions if q.gold_movie_ids]
    n = len(answerable)
    if n == 0:
        return RetrievalMetrics(n=0, hit_at_k=dict.fromkeys(k_values), mrr=None)
    return RetrievalMetrics(
        n=n,
        hit_at_k={
            k: sum(expected_hit_at_k(q.ranked_movie_ids, q.gold_movie_ids, k, q.scores) for q in answerable) / n
            for k in k_values
        },
        mrr=sum(expected_reciprocal_rank(q.ranked_movie_ids, q.gold_movie_ids, q.scores) for q in answerable) / n,
    )


def retrieval_metrics_by_type(
    questions: Iterable[RankedQuestion], k_values: Sequence[int]
) -> dict[str, RetrievalMetrics]:
    """:func:`retrieval_metrics` per question type (types without gold films, such as unanswerable, are omitted)."""
    groups: dict[str, list[RankedQuestion]] = defaultdict(list)
    for q in questions:
        if q.gold_movie_ids:
            groups[q.type].append(q)
    return {type_: retrieval_metrics(group, k_values) for type_, group in sorted(groups.items())}


def abstention_metrics(outcomes: Iterable[Abstention]) -> AbstentionMetrics:
    """Correct abstentions among questions that should be abstained, false abstentions among the others."""
    items = list(outcomes)
    unanswerable = [o for o in items if o.should_abstain]
    answerable = [o for o in items if not o.should_abstain]
    return AbstentionMetrics(
        n_unanswerable=len(unanswerable),
        n_answerable=len(answerable),
        correct_abstention_rate=_rate(sum(o.abstained for o in unanswerable), len(unanswerable)),
        false_abstention_rate=_rate(sum(o.abstained for o in answerable), len(answerable)),
    )


def _rate(count: int, total: int) -> float | None:
    return count / total if total else None


def percentile(values: Sequence[float], q: float) -> float | None:
    """The ``q`` quantile (0..1) by linear interpolation between order statistics; ``None`` for no values."""
    if not 0 <= q <= 1:
        raise ValueError(f"q must be between 0 and 1, got {q}")
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarise(values: Sequence[float]) -> Summary:
    """p50, p95 and mean of ``values`` (latencies in ms, tokens per question)."""
    return Summary(
        n=len(values),
        p50=percentile(values, 0.5),
        p95=percentile(values, 0.95),
        mean=sum(values) / len(values) if values else None,
    )
