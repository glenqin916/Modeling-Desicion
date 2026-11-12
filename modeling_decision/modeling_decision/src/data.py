"""Load, clean and describe the UCI Diabetes 130-US hospitals data.

Leakage rule enforced here: every feature is a property of the encounter itself, known at
discharge. Nothing is computed across a patient's other rows (a 30-day readmission is itself
another row, so "number of encounters in the dataset" would leak the label).
"""
import numpy as np
import pandas as pd

from . import config

MEDS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
]

NUMERIC = [
    "age_years", "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency", "number_inpatient",
    "number_diagnoses",
]

DX_GROUPS = [
    "circulatory", "respiratory", "digestive", "diabetes", "injury",
    "musculoskeletal", "genitourinary", "neoplasms",
]
COMORB = [f"comorb_{g}" for g in DX_GROUPS]

CATEGORICAL = [
    "race", "gender", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "payer_code", "medical_specialty", "primary_dx",
    "max_glu_serum", "A1Cresult", "change", "diabetesMed",
] + MEDS

FEATURES = NUMERIC + COMORB + CATEGORICAL

# Columns that must never be features. Checked by tests/test_leakage.py.
FORBIDDEN = {"encounter_id", "patient_nbr", "readmitted", "y", "weight",
             "first_encounter", "n_encounters"}


def icd9_group(code) -> str:
    """Map an ICD-9 code to one of Strack et al.'s (2014) disease groups."""
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return "missing"
    code = str(code).strip()
    if code in ("", "?"):
        return "missing"
    if code[0] in "VE":
        return "other"
    try:
        x = float(code)
    except ValueError:
        return "other"
    c = int(x)   # the 3-digit category: 459.81 belongs to 390-459
    if c == 250:
        return "diabetes"
    if 390 <= c <= 459 or c == 785:
        return "circulatory"
    if 460 <= c <= 519 or c == 786:
        return "respiratory"
    if 520 <= c <= 579 or c == 787:
        return "digestive"
    if 800 <= c <= 999:
        return "injury"
    if 710 <= c <= 739:
        return "musculoskeletal"
    if 580 <= c <= 629 or c == 788:
        return "genitourinary"
    if 140 <= c <= 239:
        return "neoplasms"
    return "other"


def load_raw(path=config.RAW_CSV) -> pd.DataFrame:
    # keep_default_na=False: the literal string "None" in A1Cresult / max_glu_serum means
    # "not measured", which is information, not missingness. Only "?" is missing.
    return pd.read_csv(path, na_values=["?"], keep_default_na=False, low_memory=False)


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """Exclusions, target, and encounter-level features."""
    df = raw[~raw["discharge_disposition_id"].isin(config.EXCLUDE_DISPOSITION)].copy()
    df["y"] = (df["readmitted"] == "<30").astype(int)

    # age comes as "[70-80)": use the band midpoint
    lo = df["age"].str.extract(r"\[(\d+)-")[0].astype(int)
    df["age_years"] = lo + 5

    groups = df[["diag_1", "diag_2", "diag_3"]].apply(lambda c: c.map(icd9_group))
    df["primary_dx"] = groups["diag_1"]
    for g in DX_GROUPS:
        df[f"comorb_{g}"] = (groups == g).any(axis=1).astype(int)

    for c in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        df[c] = df[c].astype(str)
    for c in ["race", "payer_code", "medical_specialty"]:
        df[c] = df[c].fillna("Missing")

    # Evaluation-only helpers (never features): Strack et al.'s first encounter per patient
    df["first_encounter"] = df["encounter_id"] == df.groupby("patient_nbr")["encounter_id"].transform("min")

    # Slice variables
    payer_map = {"MC": "Medicare", "MD": "Medicaid", "HM": "HMO", "SP": "Self-pay",
                 "BC": "Blue Cross", "Missing": "Missing"}
    df["payer"] = df["payer_code"].map(payer_map).fillna("Other")
    df["prior_inpatient"] = pd.cut(df["number_inpatient"], [-1, 0, 1, 2, np.inf],
                                   labels=["0", "1", "2", "3+"]).astype(str)
    df["age_band"] = pd.cut(df["age_years"], [0, 50, 70, 80, 200], right=False,
                            labels=["<50", "50-69", "70-79", "80+"]).astype(str)
    return df.reset_index(drop=True)


def load() -> pd.DataFrame:
    return prepare(load_raw())
