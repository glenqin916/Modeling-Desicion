"""R5: failure analysis by slice, each with its sample size and a patient-level interval.

For one fixed week assignment, each test encounter is either called or not. Per slice:
  recall_at_budget    = readmissions in the slice that got a call / readmissions in the slice
  precision_of_called = readmissions among the slice's called patients / slice's called patients
The worst slice is the planned, non-small slice with the lowest recall at budget: those are the
readmissions the nurse never hears about.
"""
import warnings

import numpy as np
import pandas as pd

from . import config
from .metric import _patient_index, _resample_rows


def _ratio_ci(num, den, groups, n_boot=None, seed=config.SEED):
    n_boot = n_boot or config.N_BOOT
    order, starts, counts = _patient_index(groups)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        r = _resample_rows(rng, order, starts, counts)
        d = den[r].sum()
        vals[b] = num[r].sum() / d if d else np.nan
    if np.isnan(vals).all():
        return np.nan, np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanquantile(vals, 0.025), np.nanquantile(vals, 0.975)


def slice_table(test: pd.DataFrame, called: np.ndarray, in_week: np.ndarray, model_name: str):
    df = test.loc[in_week].copy()
    df["called"] = called[in_week]
    df["hit"] = df["called"] & (df["y"] == 1)
    rows = []
    variables = [(v, "planned") for v in config.PLANNED_SLICES] + \
                [(v, "exploratory") for v in config.EXPLORATORY_SLICES if v in df]
    for var, kind in variables:
        for level, s in df.groupby(var):
            pos, cal, hit = int(s["y"].sum()), int(s["called"].sum()), int(s["hit"].sum())
            g = s[config.GROUP_COL].to_numpy()
            r_lo, r_hi = _ratio_ci(s["hit"].to_numpy(int), s["y"].to_numpy(int), g)
            p_lo, p_hi = _ratio_ci(s["hit"].to_numpy(int), s["called"].to_numpy(int), g)
            rows.append({
                "model": model_name, "variable": var, "level": level, "kind": kind,
                "n_encounters": len(s), "n_patients": s[config.GROUP_COL].nunique(),
                "readmissions": pos, "called": cal, "caught": hit,
                "recall_at_budget": hit / pos if pos else np.nan, "recall_lo": r_lo, "recall_hi": r_hi,
                "precision_of_called": hit / cal if cal else np.nan, "precision_lo": p_lo, "precision_hi": p_hi,
                "small": pos < config.MIN_SLICE_POSITIVES,
            })
    overall = {"recall_at_budget": df["hit"].sum() / df["y"].sum(),
               "precision_of_called": df["hit"].sum() / df["called"].sum()}
    return pd.DataFrame(rows), overall


def worst_slice(table: pd.DataFrame):
    """The planned, non-small slice with the lowest recall at budget, or None if there is none."""
    eligible = table[(table["kind"] == "planned") & (~table["small"])].dropna(subset=["recall_at_budget"])
    if eligible.empty:
        return None
    return eligible.loc[eligible["recall_at_budget"].idxmin()]
