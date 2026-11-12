"""Synthetic data with the exact schema of diabetic_data.csv, for tests and smoke runs.

The numbers mean nothing. Readmission is made to depend on prior inpatient visits, age and a
circulatory diagnosis so the pipeline has a signal to find.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import MEDS  # noqa: E402

COLUMNS = ["encounter_id", "patient_nbr", "race", "gender", "age", "weight",
           "admission_type_id", "discharge_disposition_id", "admission_source_id",
           "time_in_hospital", "payer_code", "medical_specialty", "num_lab_procedures",
           "num_procedures", "num_medications", "number_outpatient", "number_emergency",
           "number_inpatient", "diag_1", "diag_2", "diag_3", "number_diagnoses",
           "max_glu_serum", "A1Cresult"] + MEDS + ["change", "diabetesMed", "readmitted"]


def make(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    n_pat = int(n / 1.4)
    pat = rng.integers(100000, 100000 + n_pat, n)
    codes = ["428", "250.83", "414", "486", "786", "V57", "E909", "715", "599", "820", "174", "560"]
    age_lo = rng.choice(np.arange(0, 100, 10), n, p=[.002, .007, .016, .037, .095, .17, .22, .256, .17, .027])
    inpatient = rng.poisson(0.6, n)
    d1 = rng.choice(codes, n)
    logit = -2.6 + 0.45 * inpatient + 0.01 * (age_lo - 60) + 0.3 * (d1 == "428")
    y = rng.random(n) < 1 / (1 + np.exp(-logit))
    other = rng.random(n) < 0.35
    df = pd.DataFrame({
        "encounter_id": rng.permutation(np.arange(1, n + 1) * 7),
        "patient_nbr": pat,
        "race": rng.choice(["Caucasian", "AfricanAmerican", "Hispanic", "Other", "?"], n),
        "gender": rng.choice(["Female", "Male"], n),
        "age": [f"[{a}-{a + 10})" for a in age_lo],
        "weight": "?",
        "admission_type_id": rng.integers(1, 9, n),
        "discharge_disposition_id": rng.choice([1, 1, 1, 3, 6, 11, 13, 18, 22], n),
        "admission_source_id": rng.choice([1, 7, 4, 6, 17], n),
        "time_in_hospital": rng.integers(1, 15, n),
        "payer_code": rng.choice(["MC", "MD", "HM", "SP", "BC", "?", "?", "UN"], n),
        "medical_specialty": rng.choice(["InternalMedicine", "Cardiology", "?", "?", "Surgery-General", "Family/GeneralPractice"], n),
        "num_lab_procedures": rng.integers(1, 100, n),
        "num_procedures": rng.integers(0, 7, n),
        "num_medications": rng.integers(1, 60, n),
        "number_outpatient": rng.poisson(0.4, n),
        "number_emergency": rng.poisson(0.2, n),
        "number_inpatient": inpatient,
        "diag_1": d1, "diag_2": rng.choice(codes + ["?"], n), "diag_3": rng.choice(codes + ["?"], n),
        "number_diagnoses": rng.integers(1, 17, n),
        "max_glu_serum": rng.choice(["None", "None", "None", ">200", ">300", "Norm"], n),
        "A1Cresult": rng.choice(["None", "None", "None", ">7", ">8", "Norm"], n),
    })
    for m in MEDS:
        df[m] = rng.choice(["No", "No", "No", "Steady", "Up", "Down"], n)
    df["change"] = rng.choice(["No", "Ch"], n)
    df["diabetesMed"] = rng.choice(["Yes", "No"], n)
    df["readmitted"] = np.where(y, "<30", np.where(other, ">30", "NO"))
    return df[COLUMNS]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--n", type=int, default=20000)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    make(a.n).to_csv(a.out, index=False)
    print(f"wrote {a.out}")
