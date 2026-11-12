import numpy as np
import pandas as pd
import pytest

from src import config, slices, unsupervised
from src.metric import called_flags


def test_choose_k_and_assign(split_frames, monkeypatch):
    tr, te = split_frames
    monkeypatch.setattr(config, "KMEANS_K_RANGE", range(2, 5))
    k, table = unsupervised.choose_k(tr)
    assert k in (2, 3, 4) and list(table["k"]) == [2, 3, 4]
    km = unsupervised.fit_clusters(tr, k)
    a, b = unsupervised.assign(km, te), unsupervised.assign(km, te)
    assert (a == b).all() and a.nunique() <= k


def test_cluster_profile(split_frames):
    tr, _ = split_frames
    tr = tr.copy()
    km = unsupervised.fit_clusters(tr, 3)
    tr["comorbidity_cluster"] = unsupervised.assign(km, tr)
    prof = unsupervised.profile(tr)
    assert prof["n_train"].sum() == len(tr)
    assert prof["readmit_rate_train"].between(0, 1).all()


@pytest.fixture
def slice_inputs(split_frames):
    _, te = split_frames
    te = te.copy()
    te["comorbidity_cluster"] = "0"
    y = te["y"].to_numpy()
    score = te["number_inpatient"].to_numpy(float)
    called, in_week = called_flags(y, score)
    return te, called, in_week


def test_slice_counts_are_consistent(slice_inputs):
    te, called, in_week = slice_inputs
    table, overall = slices.slice_table(te, called, in_week, "m")
    for var, t in table.groupby("variable"):
        assert t["n_encounters"].sum() == in_week.sum()
        assert t["readmissions"].sum() == te["y"].to_numpy()[in_week].sum()
        assert t["called"].sum() == called.sum()
    assert (table["caught"] <= table["readmissions"]).all()
    assert np.allclose(table["recall_at_budget"], table["caught"] / table["readmissions"])
    ok = table["recall_lo"].isna() | (table["recall_lo"] <= table["recall_at_budget"] + 1e-9)
    assert ok.all()
    assert 0 <= overall["recall_at_budget"] <= 1


def test_slice_with_no_calls_gives_nan_precision(slice_inputs):
    te, called, in_week = slice_inputs
    table, _ = slices.slice_table(te, called, in_week, "m")
    zero = table[(table["variable"] == "prior_inpatient") & (table["level"] == "0")].iloc[0]
    assert zero["called"] == 0 and np.isnan(zero["precision_of_called"])


def test_worst_slice_ignores_small_and_exploratory():
    t = pd.DataFrame({
        "kind": ["planned", "planned", "exploratory", "planned"],
        "small": [False, True, False, False],
        "recall_at_budget": [0.4, 0.0, 0.1, 0.3],
        "level": ["a", "b", "c", "d"],
    })
    assert slices.worst_slice(t)["level"] == "d"


def test_worst_slice_with_nothing_eligible():
    t = pd.DataFrame({"kind": ["planned"], "small": [True], "recall_at_budget": [0.1], "level": ["a"]})
    assert slices.worst_slice(t) is None
