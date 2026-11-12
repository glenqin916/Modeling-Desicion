"""The locked split: 20% of patients held out, grouped on patient_nbr.

The first run writes data/splits/test_patients.csv. Every later run reads that file instead of
re-splitting, so the test set cannot drift if code or library versions change. Commit the file.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from . import config


def make_test_patients(df: pd.DataFrame, seed: int = config.SEED) -> np.ndarray:
    sgkf = StratifiedGroupKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=seed)
    _, test_idx = next(sgkf.split(df, df["y"], groups=df[config.GROUP_COL]))
    return np.sort(df[config.GROUP_COL].iloc[test_idx].unique())


def get_split(df: pd.DataFrame, split_file=config.SPLIT_FILE):
    """Return (train, test) frames, creating the locked split file on first use."""
    if split_file.exists():
        test_patients = pd.read_csv(split_file)[config.GROUP_COL].to_numpy()
    else:
        test_patients = make_test_patients(df)
        split_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({config.GROUP_COL: test_patients}).to_csv(split_file, index=False)
        print(f"Wrote locked split to {split_file} ({len(test_patients)} test patients)")

    is_test = df[config.GROUP_COL].isin(test_patients)
    found = np.isin(test_patients, df[config.GROUP_COL].unique()).mean()
    share = is_test.mean()
    if found < 0.99 or not 0.1 < share < 0.3:
        raise ValueError(
            f"{split_file} does not match this data: {found:.0%} of its patients found, "
            f"test share {share:.1%}. Use --split-file to point at the right file.")
    train, test = df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)
    assert not set(train[config.GROUP_COL]) & set(test[config.GROUP_COL]), "patient leak"
    return train, test


def cv_folds(train: pd.DataFrame, seed: int = config.SEED):
    """Grouped, stratified folds inside train for tuning."""
    sgkf = StratifiedGroupKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=seed)
    return list(sgkf.split(train, train["y"], groups=train[config.GROUP_COL]))
