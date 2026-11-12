import numpy as np
import pandas as pd
import pytest

from src import config, data


@pytest.mark.parametrize("code,group", [
    ("250", "diabetes"), ("250.83", "diabetes"), ("250.01", "diabetes"),
    ("390", "circulatory"), ("428", "circulatory"), ("459", "circulatory"),
    ("459.81", "circulatory"),          # decimal at the top of a range
    ("785", "circulatory"),
    ("460", "respiratory"), ("519.8", "respiratory"), ("786", "respiratory"),
    ("520", "digestive"), ("579.9", "digestive"), ("787", "digestive"),
    ("800", "injury"), ("999.9", "injury"),
    ("710", "musculoskeletal"), ("739.5", "musculoskeletal"),
    ("580", "genitourinary"), ("629.9", "genitourinary"), ("788", "genitourinary"),
    ("140", "neoplasms"), ("239.5", "neoplasms"),
    ("V57", "other"), ("E909", "other"), ("001", "other"), ("389.9", "other"),
    ("?", "missing"), ("", "missing"), (np.nan, "missing"), (None, "missing"),
])
def test_icd9_group(code, group):
    assert data.icd9_group(code) == group


def test_icd9_group_garbage_is_other():
    assert data.icd9_group("abc") == "other"


def test_load_raw_keeps_none_as_a_level(raw_csv):
    raw = data.load_raw(raw_csv)
    assert "None" in set(raw["A1Cresult"]) and "None" in set(raw["max_glu_serum"])
    assert raw["payer_code"].isna().any()          # "?" is the only missing marker


def test_exclusions(df):
    excluded = {str(i) for i in config.EXCLUDE_DISPOSITION}
    assert not df["discharge_disposition_id"].isin(excluded).any()


def test_target_matches_label(df):
    assert set(df["y"]) <= {0, 1}
    assert (df["y"] == (df["readmitted"] == "<30")).all()


def test_age_is_band_midpoint(df):
    lo = df["age"].str.extract(r"\[(\d+)-")[0].astype(int)
    assert (df["age_years"] == lo + 5).all()


def test_comorbidity_flags_match_codes(df):
    for _, row in df.sample(200, random_state=0).iterrows():
        groups = {data.icd9_group(row[c]) for c in ("diag_1", "diag_2", "diag_3")}
        for g in data.DX_GROUPS:
            assert row[f"comorb_{g}"] == int(g in groups)
        assert row["primary_dx"] == data.icd9_group(row["diag_1"])


def test_one_first_encounter_per_patient(df):
    assert (df.groupby("patient_nbr")["first_encounter"].sum() == 1).all()


def test_slice_variables(df):
    assert set(df["payer"]) <= {"Medicare", "Medicaid", "HMO", "Self-pay", "Blue Cross", "Missing", "Other"}
    assert set(df["prior_inpatient"]) <= {"0", "1", "2", "3+"}
    assert set(df["age_band"]) <= {"<50", "50-69", "70-79", "80+"}
    ok = (df["prior_inpatient"] == "3+") == (df["number_inpatient"] >= 3)
    assert ok.all()


def test_no_missing_in_categoricals(df):
    assert not df[data.CATEGORICAL].isna().any().any()


def test_features_exist_and_are_allowed(df):
    assert set(data.FEATURES) <= set(df.columns)
    assert not set(data.FEATURES) & data.FORBIDDEN
    assert len(data.FEATURES) == len(set(data.FEATURES))


def test_prepare_does_not_mutate_input(raw_csv):
    raw = data.load_raw(raw_csv)
    before = raw.copy()
    data.prepare(raw)
    pd.testing.assert_frame_equal(raw, before)
