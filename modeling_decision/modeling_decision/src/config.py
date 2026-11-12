"""Every number the proposal locks lives here. Change one only with a note in the report."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
# The UCI dataset folder, kept in the repo root under its UCI download name (CC BY 4.0)
RAW_DIR = ROOT / "diabetes+130-us+hospitals+for+years+1999-2008"
RAW_CSV = RAW_DIR / "diabetic_data.csv"
SPLIT_FILE = DATA_DIR / "splits" / "test_patients.csv"
RESULTS_DIR = ROOT / "results"

SEED = 0

# --- Locked: the split -------------------------------------------------------
N_FOLDS = 5            # test = fold 0 of StratifiedGroupKFold(5) -> 20% of patients
GROUP_COL = "patient_nbr"

# --- Locked: the metric ------------------------------------------------------
BUDGET = 40            # calls per week
WEEK_N = 200           # discharges per simulated week (assumption, varied below)
SENSITIVITY_N = (100, 400)
N_BOOT = 1000          # patient-level bootstrap resamples
N_WEEK_SHUFFLES = 20   # random week assignments averaged inside each estimate

# --- Data cleaning -----------------------------------------------------------
# discharge_disposition_id: 11 expired, 13/14 hospice, 19/20/21 expired (Medicare codes).
# These patients cannot be readmitted, so they are out of scope for the call list.
EXCLUDE_DISPOSITION = (11, 13, 14, 19, 20, 21)

# --- Tuning grids (both models get a comparable, small grid) -----------------
LOGIT_C_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
GBM_GRID = {
    "learning_rate": (0.03, 0.1),
    "max_leaf_nodes": (15, 31),
    "l2_regularization": (0.0, 1.0),
}

# --- Unsupervised ------------------------------------------------------------
KMEANS_K_RANGE = range(2, 9)

# --- Slices (named in the proposal) ------------------------------------------
PLANNED_SLICES = ("payer", "prior_inpatient")
EXPLORATORY_SLICES = ("age_band", "comorbidity_cluster")
MIN_SLICE_POSITIVES = 30   # a slice with fewer readmissions is reported but flagged "small"

# --- Smoke-test settings (python -m src.run --quick) -----------------------
QUICK = {
    "N_BOOT": 100,
    "LOGIT_C_GRID": (0.03, 0.3),
    "GBM_GRID": {"learning_rate": (0.1,), "max_leaf_nodes": (15,), "l2_regularization": (0.0,)},
    "KMEANS_K_RANGE": range(2, 5),
}


class overrides:
    """Temporarily replace settings in this module; restores them on exit, even on error."""

    def __init__(self, values):
        self.values, self.saved = dict(values), {}

    def __enter__(self):
        g = globals()
        for k, v in self.values.items():
            self.saved[k] = g[k]
            g[k] = v
        return self

    def __exit__(self, *exc):
        globals().update(self.saved)
        return False
