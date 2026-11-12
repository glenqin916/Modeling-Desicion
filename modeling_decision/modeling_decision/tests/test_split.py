import pandas as pd
import pytest

from src import split


def test_no_patient_on_both_sides(split_frames):
    tr, te = split_frames
    assert not set(tr["patient_nbr"]) & set(te["patient_nbr"])


def test_test_share_and_rate(df, split_frames):
    tr, te = split_frames
    assert len(tr) + len(te) == len(df)
    assert 0.15 < len(te) / len(df) < 0.25
    assert abs(te["y"].mean() - tr["y"].mean()) < 0.02


def test_split_is_deterministic(df):
    a, b = split.make_test_patients(df), split.make_test_patients(df)
    assert (a == b).all()


def test_split_file_is_reused(df, tmp_path):
    f = tmp_path / "s.csv"
    _, te1 = split.get_split(df, f)
    saved = pd.read_csv(f)
    _, te2 = split.get_split(df, f)
    assert set(saved["patient_nbr"]) == set(te1["patient_nbr"])
    assert te1["encounter_id"].tolist() == te2["encounter_id"].tolist()


def test_stale_split_file_is_rejected(df, tmp_path):
    """A split file written for other data must not silently produce a tiny test set."""
    f = tmp_path / "s.csv"
    pd.DataFrame({"patient_nbr": [-1, -2, -3]}).to_csv(f, index=False)
    with pytest.raises(ValueError):
        split.get_split(df, f)


def test_cv_folds_grouped_and_cover_train(split_frames):
    tr, _ = split_frames
    folds = split.cv_folds(tr)
    seen = []
    for a, b in folds:
        assert not set(tr["patient_nbr"].iloc[a]) & set(tr["patient_nbr"].iloc[b])
        seen.extend(b)
    assert sorted(seen) == list(range(len(tr)))
