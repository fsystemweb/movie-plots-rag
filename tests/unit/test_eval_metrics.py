"""Hit@k, MRR, abstention and percentiles on hand-computed cases (no model, no I/O)."""

from __future__ import annotations

import pytest

from movie_rag.eval.metrics import (
    Abstention,
    RankedQuestion,
    abstention_metrics,
    expected_hit_at_k,
    expected_reciprocal_rank,
    first_gold_rank,
    hit_at_k,
    percentile,
    rank_distribution,
    reciprocal_rank,
    retrieval_metrics,
    retrieval_metrics_by_type,
    summarise,
)


def q(
    qid: str, ranked: list[str], gold: list[str], type_: str = "fuzzy_plot", scores: list[float] | None = None
) -> RankedQuestion:
    return RankedQuestion(question_id=qid, type=type_, gold_movie_ids=gold, ranked_movie_ids=ranked, scores=scores)


# --- rank ----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ranked", "gold", "expected"),
    [
        (["a", "b", "c"], ["a"], 1),
        (["a", "b", "c"], ["c"], 3),
        (["a", "b", "c"], ["x"], None),
        ([], ["a"], None),
        (["a", "b", "c"], [], None),  # no gold film (unanswerable): there is no rank
        (["a", "b", "c"], ["c", "b"], 2),  # several gold films: the best-placed one counts
        (["a", "a", "b"], ["b"], 2),  # a film listed twice is ranked once, the repeat does not push b down
    ],
)
def test_first_gold_rank(ranked: list[str], gold: list[str], expected: int | None) -> None:
    assert first_gold_rank(ranked, gold) == expected


def test_hit_at_k_boundaries_are_inclusive() -> None:
    assert hit_at_k(3, 3) is True
    assert hit_at_k(3, 2) is False
    assert hit_at_k(1, 1) is True
    assert hit_at_k(None, 100) is False
    with pytest.raises(ValueError, match="at least 1"):
        hit_at_k(1, 0)


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(1) == 1.0
    assert reciprocal_rank(3) == pytest.approx(1 / 3)
    assert reciprocal_rank(None) == 0.0
    with pytest.raises(ValueError, match="start at 1"):
        reciprocal_rank(0)


# --- aggregates ----------------------------------------------------------------------------------------------------


def test_mrr_and_hit_at_k_over_ranks_1_3_and_a_miss() -> None:
    """Ranks [1, 3, None]: MRR = (1 + 1/3 + 0) / 3 = 4/9; Hit@1 = 1/3; Hit@3 = Hit@5 = 2/3."""
    questions = [
        q("q1", ["a", "x"], ["a"]),
        q("q2", ["x", "y", "b"], ["b"]),
        q("q3", ["x", "y", "z"], ["c"]),
    ]
    metrics = retrieval_metrics(questions, [1, 3, 5])
    assert metrics.n == 3
    assert metrics.mrr == pytest.approx(4 / 9)
    assert metrics.hit_at_k == pytest.approx({1: 1 / 3, 3: 2 / 3, 5: 2 / 3})


def test_unanswerable_questions_are_left_out_of_hit_and_mrr() -> None:
    questions = [q("q1", ["a"], ["a"]), q("u1", ["a", "b"], [], type_="unanswerable")]
    metrics = retrieval_metrics(questions, [1])
    assert metrics.n == 1
    assert metrics.mrr == 1.0
    assert metrics.hit_at_k == {1: 1.0}


def test_no_question_with_a_gold_film_means_undefined_not_zero() -> None:
    metrics = retrieval_metrics([q("u1", ["a"], [], type_="unanswerable")], [1, 3])
    assert metrics.n == 0 and metrics.mrr is None
    assert metrics.hit_at_k == {1: None, 3: None}
    assert retrieval_metrics([], [1]).mrr is None


def test_multi_gold_question_counts_the_best_placed_gold_film_once() -> None:
    metrics = retrieval_metrics([q("q1", ["x", "g2", "g1"], ["g1", "g2"])], [1, 2])
    assert metrics.hit_at_k == {1: 0.0, 2: 1.0}
    assert metrics.mrr == 0.5


def test_k_larger_than_the_list_is_fine() -> None:
    metrics = retrieval_metrics([q("q1", ["a", "b"], ["b"])], [8])
    assert metrics.hit_at_k == {8: 1.0}


def test_metrics_by_type_group_only_types_with_gold_films() -> None:
    questions = [
        q("f1", ["a"], ["a"], "fuzzy_plot"),
        q("f2", ["x", "b"], ["b"], "fuzzy_plot"),
        q("e1", ["x", "y"], ["c"], "exact_entity"),
        q("u1", ["a"], [], "unanswerable"),
    ]
    by_type = retrieval_metrics_by_type(questions, [1])
    assert list(by_type) == ["exact_entity", "fuzzy_plot"]
    assert by_type["fuzzy_plot"].mrr == pytest.approx(0.75)
    assert by_type["exact_entity"].mrr == 0.0


# --- ties ----------------------------------------------------------------------------------------------------------


def test_without_scores_or_ties_the_rank_is_exact() -> None:
    assert rank_distribution(["a", "b", "c"], ["b"]) == [(2, 1.0)]
    assert rank_distribution(["a", "b", "c"], ["b"], [0.9, 0.5, 0.1]) == [(2, 1.0)]
    assert rank_distribution(["a", "b"], ["z"], [0.9, 0.5]) == []


def test_a_gold_film_tied_with_one_other_is_first_or_second_with_equal_probability() -> None:
    """[x, g, y] scored [0.5, 0.5, 0.1]: g is rank 1 or 2 with probability 1/2 each.

    Hit@1 = 1/2, Hit@2 = 1, reciprocal rank = 1/2 * 1 + 1/2 * 1/2 = 3/4.
    """
    ranked, scores = ["x", "g", "y"], [0.5, 0.5, 0.1]
    assert rank_distribution(ranked, ["g"], scores) == [(1, 0.5), (2, 0.5)]
    assert expected_hit_at_k(ranked, ["g"], 1, scores) == 0.5
    assert expected_hit_at_k(ranked, ["g"], 2, scores) == 1.0
    assert expected_reciprocal_rank(ranked, ["g"], scores) == pytest.approx(0.75)


def test_the_tie_group_is_judged_by_where_it_starts() -> None:
    """[x, y, g, z] scored [.9, .5, .5, .1]: the tie group holds ranks 2 and 3."""
    assert rank_distribution(["x", "y", "g", "z"], ["g"], [0.9, 0.5, 0.5, 0.1]) == [(2, 0.5), (3, 0.5)]


def test_two_gold_films_in_one_tie_group_of_three() -> None:
    """Choosing 2 of 3 tied slots for the gold films: the best one is first with probability 2/3, second with 1/3."""
    dist = rank_distribution(["a", "b", "c"], ["a", "c"], [0.5, 0.5, 0.5])
    assert dist == [(1, pytest.approx(2 / 3)), (2, pytest.approx(1 / 3))]
    assert sum(p for _, p in dist) == pytest.approx(1.0)


def test_scores_that_differ_only_by_float_noise_are_tied() -> None:
    a, b = 1 / 62 + 1 / 63, 1 / 63 + 1 / 62 + 1e-18
    assert rank_distribution(["x", "g"], ["g"], [a, b]) == [(1, 0.5), (2, 0.5)]


def test_a_tie_that_does_not_involve_the_gold_film_changes_nothing() -> None:
    assert rank_distribution(["x", "y", "g"], ["g"], [0.5, 0.5, 0.1]) == [(3, 1.0)]


def test_a_repeated_film_does_not_make_a_tie_group_larger() -> None:
    assert rank_distribution(["x", "x", "g"], ["g"], [0.5, 0.5, 0.5]) == [(1, 0.5), (2, 0.5)]


def test_scores_must_line_up_with_the_films() -> None:
    with pytest.raises(ValueError, match="same length"):
        rank_distribution(["a", "b"], ["a"], [1.0])
    with pytest.raises(ValueError, match="at least 1"):
        expected_hit_at_k(["a"], ["a"], 0)


def test_retrieval_metrics_use_the_tie_aware_expectation() -> None:
    """One question solved at rank 1, one tied for ranks 1-2: Hit@1 = (1 + 1/2) / 2 = 3/4, MRR = (1 + 3/4) / 2."""
    questions = [
        q("q1", ["a", "b"], ["a"], scores=[0.9, 0.1]),
        q("q2", ["x", "g"], ["g"], scores=[0.5, 0.5]),
    ]
    metrics = retrieval_metrics(questions, [1, 2])
    assert metrics.hit_at_k == pytest.approx({1: 0.75, 2: 1.0})
    assert metrics.mrr == pytest.approx(0.875)


# --- abstention ----------------------------------------------------------------------------------------------------


def test_abstention_rates() -> None:
    """3 unanswerable (2 abstained) and 2 answerable (1 abstained): correct 2/3, false 1/2."""
    outcomes = [
        Abstention(should_abstain=True, abstained=True),
        Abstention(should_abstain=True, abstained=True),
        Abstention(should_abstain=True, abstained=False),
        Abstention(should_abstain=False, abstained=False),
        Abstention(should_abstain=False, abstained=True),
    ]
    metrics = abstention_metrics(outcomes)
    assert (metrics.n_unanswerable, metrics.n_answerable) == (3, 2)
    assert metrics.correct_abstention_rate == pytest.approx(2 / 3)
    assert metrics.false_abstention_rate == 0.5


def test_abstention_is_undefined_without_questions_of_a_kind() -> None:
    only_answerable = abstention_metrics([Abstention(should_abstain=False, abstained=False)])
    assert only_answerable.correct_abstention_rate is None
    assert only_answerable.false_abstention_rate == 0.0
    nothing = abstention_metrics([])
    assert nothing.correct_abstention_rate is None and nothing.false_abstention_rate is None


# --- percentiles ---------------------------------------------------------------------------------------------------


def test_percentiles_interpolate_between_order_statistics() -> None:
    values = [40.0, 10.0, 30.0, 20.0]
    assert percentile(values, 0.5) == 25.0  # between 20 and 30
    assert percentile(values, 0.95) == pytest.approx(38.5)  # position 2.85: 30 + 0.85 * 10
    assert percentile(values, 0) == 10.0 and percentile(values, 1) == 40.0
    assert percentile([7.0], 0.95) == 7.0
    assert percentile([], 0.5) is None
    with pytest.raises(ValueError, match="between 0 and 1"):
        percentile(values, 1.5)


def test_summary_of_a_series() -> None:
    summary = summarise([10.0, 20.0, 30.0, 40.0])
    assert (summary.n, summary.p50, summary.mean) == (4, 25.0, 25.0)
    assert summary.p95 == pytest.approx(38.5)
    empty = summarise([])
    assert (empty.n, empty.p50, empty.p95, empty.mean) == (0, None, None, None)
