"""The locked metric: precision at 40 per simulated week.

The data has no dates, so the test set is cut at random into "weeks" of N discharges. In each
week the 40 highest scores are called (ties broken at random), and precision is the share of
those 40 who are readmitted within 30 days. Intervals come from a patient-level bootstrap: whole
patients are resampled with all their encounters, because a patient's rows are correlated.
"""
import numpy as np

from . import config


def _check(n, week_n, budget):
    if not 1 <= budget <= week_n:
        raise ValueError(f"budget must be between 1 and week_n ({week_n}), got {budget}")
    if n < week_n:
        raise ValueError(f"need at least one full week: {n} rows < week_n = {week_n}")


def _top_k_hits(y, score, rng, week_n, budget):
    """One random week assignment -> array (weeks,) of readmissions among the called."""
    n = len(y)
    _check(n, week_n, budget)
    w = n // week_n
    perm = rng.permutation(n)[: w * week_n]
    week = np.repeat(np.arange(w), week_n)
    s, t, tie = score[perm], y[perm], rng.random(w * week_n)
    order = np.lexsort((tie, -s, week))          # by week, then score desc, ties random
    called = order.reshape(w, week_n)[:, :budget]
    return t[called].sum(axis=1)


def weekly_precision(y, score, week_n=config.WEEK_N, budget=config.BUDGET,
                     n_shuffles=config.N_WEEK_SHUFFLES, seed=config.SEED):
    """Point estimate: mean precision at budget, averaged over random week assignments."""
    rng = np.random.default_rng(seed)
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    hits = np.concatenate([_top_k_hits(y, score, rng, week_n, budget) for _ in range(n_shuffles)])
    return hits.mean() / budget


def _patient_index(groups):
    codes, uniq = _factorize(groups)
    order = np.argsort(codes, kind="stable")
    counts = np.bincount(codes, minlength=len(uniq))
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return order, starts, counts


def _factorize(a):
    uniq, codes = np.unique(np.asarray(a), return_inverse=True)
    return codes, uniq


def _resample_rows(rng, order, starts, counts):
    pick = rng.integers(0, len(counts), len(counts))
    c = counts[pick]
    base = np.repeat(starts[pick], c)
    offset = np.arange(c.sum()) - np.repeat(np.cumsum(c) - c, c)
    return order[base + offset]


def bootstrap_compare(y, groups, scores: dict, reference: str, week_n=config.WEEK_N,
                      budget=config.BUDGET, n_boot=None, seed=config.SEED):
    """Precision at budget for several scorers on identical resamples and weeks.

    Returns {name: {precision, lo, hi, caught_per_week, diff_vs_ref, diff_lo, diff_hi}}.
    Differences are paired: every scorer sees the same patients and the same weeks.
    """
    n_boot = n_boot or config.N_BOOT
    y = np.asarray(y)
    scores = {k: np.asarray(v, dtype=float) for k, v in scores.items()}
    order, starts, counts = _patient_index(groups)
    rng = np.random.default_rng(seed)
    boot = {k: np.empty(n_boot) for k in scores}
    for b in range(n_boot):
        rows = _resample_rows(rng, order, starts, counts)
        week_seed = rng.integers(2**32)
        for k, s in scores.items():
            r = np.random.default_rng(week_seed)          # same weeks for every scorer
            boot[k][b] = _top_k_hits(y[rows], s[rows], r, week_n, budget).mean() / budget

    out = {}
    for k, s in scores.items():
        p = weekly_precision(y, s, week_n, budget, seed=seed)
        d = boot[k] - boot[reference]
        out[k] = {
            "precision": p,
            "lo": np.quantile(boot[k], 0.025),
            "hi": np.quantile(boot[k], 0.975),
            "caught_per_week": p * budget,
            "diff_vs_ref": p - weekly_precision(y, scores[reference], week_n, budget, seed=seed),
            "diff_lo": np.quantile(d, 0.025),
            "diff_hi": np.quantile(d, 0.975),
        }
    return out


def called_flags(y, score, week_n=config.WEEK_N, budget=config.BUDGET, seed=config.SEED):
    """One fixed week assignment -> boolean 'was called' per row (for slicing and misses).

    Rows left over after the last full week are never called; they are marked as not in a
    week and excluded from slice statistics.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    _check(n, week_n, budget)
    w = n // week_n
    perm = rng.permutation(n)[: w * week_n]
    week = np.repeat(np.arange(w), week_n)
    s, tie = np.asarray(score, dtype=float)[perm], rng.random(w * week_n)
    order = np.lexsort((tie, -s, week))
    called_pos = order.reshape(w, week_n)[:, :budget].ravel()
    called = np.zeros(n, dtype=bool)
    called[perm[called_pos]] = True
    in_week = np.zeros(n, dtype=bool)
    in_week[perm] = True
    return called, in_week
