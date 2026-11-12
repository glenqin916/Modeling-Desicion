import numpy as np
import pytest

from src import config
from src.metric import (_patient_index, _resample_rows, _top_k_hits, bootstrap_compare,
                        called_flags, weekly_precision)


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    y = (rng.random(4000) < 0.11).astype(int)
    g = rng.integers(0, 2500, 4000)
    return y, g, rng


def test_top_k_hits_hand_example():
    # one week of 5, budget 2: the two highest scores are rows 3 and 0 (y = 1, 0)
    y = np.array([0, 1, 0, 1, 0])
    s = np.array([0.8, 0.1, 0.2, 0.9, 0.3])
    hits = _top_k_hits(y, s, np.random.default_rng(0), week_n=5, budget=2)
    assert hits.tolist() == [1]


def test_perfect_and_inverse_rankers(toy):
    y, _, _ = toy
    perfect = weekly_precision(y, y.astype(float))
    worst = weekly_precision(y, -y.astype(float))
    # a week of 200 has about 22 positives, so a perfect ranker calls all of them: ~22/40
    assert perfect == pytest.approx(y.mean() * config.WEEK_N / config.BUDGET, rel=0.1)
    assert worst == 0.0


def test_constant_scores_give_base_rate(toy):
    y, _, _ = toy
    assert weekly_precision(y, np.zeros(len(y))) == pytest.approx(y.mean(), abs=0.015)


def test_ties_are_broken_at_random_not_by_row_order():
    # positives all at the end; ties broken by row order would never call them
    y = np.r_[np.zeros(3000), np.ones(1000)].astype(int)
    assert weekly_precision(y, np.zeros(len(y))) == pytest.approx(0.25, abs=0.02)


def test_precision_is_independent_of_row_order(toy):
    y, _, rng = toy
    s = rng.random(len(y))
    perm = rng.permutation(len(y))
    a = weekly_precision(y, s, n_shuffles=200)
    b = weekly_precision(y[perm], s[perm], n_shuffles=200)
    assert a == pytest.approx(b, abs=0.01)


@pytest.mark.parametrize("budget,week_n", [(41, 40), (0, 200)])
def test_invalid_budget_raises(budget, week_n):
    y = np.zeros(1000, int)
    with pytest.raises(ValueError):
        weekly_precision(y, np.zeros(1000), week_n=week_n, budget=budget)


def test_too_few_rows_for_one_week_raises():
    with pytest.raises(ValueError):
        weekly_precision(np.zeros(50, int), np.zeros(50), week_n=200)


def test_resample_keeps_patients_whole(toy):
    _, g, rng = toy
    order, starts, counts = _patient_index(g)
    rows = _resample_rows(rng, order, starts, counts)
    size = np.bincount(np.unique(g, return_inverse=True)[1])
    # every sampled patient appears with all of its rows, a whole number of times
    picked = g[rows]
    for p in np.unique(picked)[:200]:
        n_rows = (picked == p).sum()
        assert n_rows % size[np.searchsorted(np.unique(g), p)] == 0
    assert len(np.unique(rows)) <= len(g)


def test_bootstrap_identical_scorers_have_zero_difference(toy):
    y, g, rng = toy
    s = rng.random(len(y))
    r = bootstrap_compare(y, g, {"a": s, "b": s.copy()}, "a", n_boot=50)
    assert r["b"]["diff_lo"] == r["b"]["diff_hi"] == r["b"]["diff_vs_ref"] == 0


def test_bootstrap_ci_brackets_point_and_orders_scorers(toy):
    y, g, rng = toy
    good = y + rng.normal(0, 0.8, len(y))
    r = bootstrap_compare(y, g, {"good": good, "rand": rng.random(len(y))}, "rand", n_boot=200)
    for k in r:
        assert r[k]["lo"] <= r[k]["precision"] <= r[k]["hi"]
        assert r[k]["caught_per_week"] == pytest.approx(r[k]["precision"] * config.BUDGET)
    assert r["good"]["diff_lo"] > 0


def test_bootstrap_is_reproducible(toy):
    y, g, rng = toy
    s = rng.random(len(y))
    a = bootstrap_compare(y, g, {"s": s}, "s", n_boot=30)
    b = bootstrap_compare(y, g, {"s": s}, "s", n_boot=30)
    assert a == b


def test_called_flags_counts_and_agree_with_metric(toy):
    y, _, rng = toy
    s = rng.random(len(y))
    called, in_week = called_flags(y, s)
    weeks = len(y) // config.WEEK_N
    assert called.sum() == weeks * config.BUDGET
    assert in_week.sum() == weeks * config.WEEK_N
    assert not (called & ~in_week).any()
    # same seed and one shuffle -> the same calls as the metric
    assert y[called].mean() == pytest.approx(weekly_precision(y, s, n_shuffles=1))
