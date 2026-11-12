"""Full pipeline on synthetic data: outputs exist, are complete, and are reproducible."""
import io
import json
import zipfile

import pandas as pd
import pytest

from src import config, run
from tests.make_fake_data import make


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e")
    csv = root / "diabetic_data.csv"
    make(12000, seed=3).to_csv(csv, index=False)
    outs = []
    for i in (1, 2):
        out = root / f"results{i}"
        run.main(["--quick", "--data", str(csv), "--split-file", str(root / "split.csv"),
                  "--out", str(out)])
        outs.append(out)
    return outs


EXPECTED = ["headline.json", "headline.md", "slices.csv", "cv_logistic.csv", "cv_gbm.csv",
            "cv_logistic_plus_cluster.csv", "logistic_coefficients.csv",
            "kmeans_choose_k.csv", "kmeans_profile.csv"]


def test_all_outputs_written(e2e):
    for name in EXPECTED:
        assert (e2e[0] / name).exists(), name


def test_headline_contents(e2e):
    h = json.loads((e2e[0] / "headline.json").read_text())
    for n in (config.WEEK_N, *config.SENSITIVITY_N):
        assert f"N={n}" in h["precision_at_40"]
    main = h["precision_at_40"][f"N={config.WEEK_N}"]
    assert set(main) == {"baseline_prior_visits", "baseline_oldest", "random", "logistic_l1", "gbm"}
    assert main["baseline_prior_visits"]["diff_vs_ref"] == 0
    assert main["logistic_l1"]["precision"] > main["random"]["precision"]
    assert h["split"]["train_encounters"] + h["split"]["test_encounters"] == \
        h["split"]["encounters_after_exclusions"]


def test_runs_are_identical(e2e):
    a = json.loads((e2e[0] / "headline.json").read_text())
    b = json.loads((e2e[1] / "headline.json").read_text())
    assert a == b
    pd.testing.assert_frame_equal(pd.read_csv(e2e[0] / "slices.csv"), pd.read_csv(e2e[1] / "slices.csv"))


def test_quick_mode_does_not_leak_into_config(e2e):
    """--quick must not change module-level settings for later code in the same process."""
    assert config.N_BOOT == 1000
    assert config.LOGIT_C_GRID == (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)


def test_extra_and_figures_accept_paths(e2e, tmp_path):
    """extra.py and figures.py must run against any data/results folder, not only the defaults."""
    from src import extra, figures
    root = e2e[0].parent
    extra.main(["--data", str(root / "diabetic_data.csv"), "--split-file", str(root / "split.csv"),
                "--out", str(e2e[0]), "--quick"])
    e = json.loads((e2e[0] / "extra.json").read_text())
    assert {"gbm_minus_logistic", "home_only_precision", "budget_curve", "logistic_sparse"} <= set(e)
    figures.main(["--out", str(e2e[0])])
    for f in ("fig1_headline.png", "fig2_budget.png", "fig3_slices.png"):
        assert (e2e[0] / f).stat().st_size > 1000


def test_fetch_data_from_local_zip(tmp_path, monkeypatch):
    from data import fetch_data
    buf = io.BytesIO()
    df = make(500, seed=0)
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("folder/diabetic_data.csv", df.to_csv(index=False))
        z.writestr("folder/IDS_mapping.csv", "a,b\n")
    zp = tmp_path / "d.zip"
    zp.write_bytes(buf.getvalue())
    monkeypatch.setattr(fetch_data, "RAW", tmp_path / "raw")
    monkeypatch.setattr(fetch_data, "EXPECTED_ROWS", 500)
    monkeypatch.setattr("sys.argv", ["fetch_data.py", "--zip", str(zp)])
    with pytest.raises(SystemExit) as e:
        fetch_data.main()
    assert e.value.code == 0
    assert (tmp_path / "raw" / "diabetic_data.csv").exists()
