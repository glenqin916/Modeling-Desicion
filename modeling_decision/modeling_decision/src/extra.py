"""Follow-up analyses for the report (run after src.run; reads results/headline.json).

    python -m src.extra

Refits both models with the tuned hyperparameters (identical to GridSearchCV's refit, same
seed) and writes results/extra.json:
  - gbm vs logistic, paired difference in precision at 40 (R3, R6)
  - who gets called: share of calls going to patients discharged to another facility
    rather than home (an unintended proxy found while reading the coefficients; exploratory)
  - precision at 40 if only home discharges are eligible for a call (R6 reversal check)
  - the prior_inpatient = 0 slice: share of all readmissions, base rate, share of calls (R5)
  - the sparse logistic models (C = 0.01), including a refit with the intercept penalty made
    negligible; perfect-ranker precision; AUC for every scorer (reference only)
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import config, data, models, slices, split
from sklearn.metrics import roc_auc_score

from .metric import _patient_index, _resample_rows, bootstrap_compare, called_flags, weekly_precision

# discharge_disposition_id 1 = home, 6 = home with home health service, 8 = home under IV care
HOME = {"1", "6", "8"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=config.RAW_CSV)
    ap.add_argument("--split-file", type=Path, default=config.SPLIT_FILE)
    ap.add_argument("--out", type=Path, default=config.RESULTS_DIR, help="folder holding headline.json")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    with config.overrides(config.QUICK if args.quick else {}):
        return _main(args)


def _main(args):
    out_dir = args.out
    h = json.loads((out_dir / "headline.json").read_text())
    df = data.prepare(data.load_raw(args.data))
    train, test = split.get_split(df, args.split_file)
    feats = data.FEATURES

    logit = models.logistic_pipeline().set_params(**h["tuned"]["logistic_l1"]).fit(train[feats], train["y"])
    gbm = models.gbm_pipeline().set_params(**h["tuned"]["gbm"]).fit(train[feats], train["y"])
    s = {"logistic_l1": logit.predict_proba(test[feats])[:, 1],
         "gbm": gbm.predict_proba(test[feats])[:, 1],
         "baseline_prior_visits": test["number_inpatient"].to_numpy(float)}
    y, g = test["y"].to_numpy(), test["patient_nbr"].to_numpy()
    out = {}

    r = bootstrap_compare(y, g, s, "logistic_l1")
    out["gbm_minus_logistic"] = {k: r["gbm"][k] for k in ("diff_vs_ref", "diff_lo", "diff_hi")}
    out["check_logistic_precision"] = r["logistic_l1"]["precision"]

    home = test["discharge_disposition_id"].isin(HOME).to_numpy()
    out["share_test_discharged_home"] = float(home.mean())
    out["readmit_rate_home"] = float(y[home].mean())
    out["readmit_rate_not_home"] = float(y[~home].mean())
    calls = {}
    for k, sc in s.items():
        called, _ = called_flags(y, sc)
        calls[k] = {"share_calls_not_home": float((~home[called]).mean()),
                    "calls": int(called.sum())}
    out["calls_by_destination"] = calls

    # Home-only eligibility: weeks are formed from home discharges only
    rh = bootstrap_compare(y[home], g[home], {k: v[home] for k, v in s.items()}, "baseline_prior_visits")
    out["home_only_precision"] = {k: {m: v[m] for m in ("precision", "lo", "hi", "diff_vs_ref", "diff_lo", "diff_hi")}
                                  for k, v in rh.items()}

    zero = (test["number_inpatient"] == 0).to_numpy()
    out["no_prior_inpatient"] = {
        "share_encounters": float(zero.mean()),
        "share_of_readmissions": float(y[zero].sum() / y.sum()),
        "readmit_rate": float(y[zero].mean()),
        "readmit_rate_others": float(y[~zero].mean()),
    }
    for k, sc in s.items():
        called, _ = called_flags(y, sc)
        out["no_prior_inpatient"][f"share_calls_{k}"] = float(zero[called].mean())

    # How sparse is the logistic model along its C grid? (readability vs CV score)
    out["logistic_nonzero_by_C"] = {}
    for c in config.LOGIT_C_GRID:
        m = models.logistic_pipeline().set_params(clf__C=c).fit(train[feats], train["y"])
        out["logistic_nonzero_by_C"][str(c)] = int((m.named_steps["clf"].coef_ != 0).sum())

    # Budget curve: readmitted patients reached per week as the call budget changes (R6)
    rnd = np.zeros(len(test))
    curve = {}
    for k, sc in {**s, "random": rnd}.items():
        curve[k] = {}
        for b in range(10, 101, 10):
            p = bootstrap_compare(y, g, {k: sc}, k, budget=b, n_boot=min(200, config.N_BOOT))[k]
            curve[k][str(b)] = {"reached": p["precision"] * b, "precision": p["precision"],
                                "lo": p["lo"], "hi": p["hi"]}
    out["budget_curve"] = curve

    # The readable logistic (C = 0.01, chosen as the sparsest grid point within 0.005 of the
    # best CV score), declared as a sensitivity result, not a re-selection on test
    sparse = models.logistic_pipeline().set_params(clf__C=0.01).fit(train[feats], train["y"])
    s2 = {**s, "logistic_sparse": sparse.predict_proba(test[feats])[:, 1]}
    rs = bootstrap_compare(y, g, s2, "baseline_prior_visits")
    rl = bootstrap_compare(y, g, s2, "logistic_l1")
    out["logistic_sparse"] = {
        "precision": rs["logistic_sparse"]["precision"], "lo": rs["logistic_sparse"]["lo"],
        "hi": rs["logistic_sparse"]["hi"],
        "diff_vs_baseline": [rs["logistic_sparse"][m] for m in ("diff_vs_ref", "diff_lo", "diff_hi")],
        "diff_vs_full_logistic": [rl["logistic_sparse"][m] for m in ("diff_vs_ref", "diff_lo", "diff_hi")],
    }
    models.logistic_coefficients(sparse).to_csv(out_dir / "logistic_sparse_coefficients.csv", index=False)

    # The same sparse model with the intercept effectively unpenalized. liblinear penalizes the
    # intercept, which pushed part of it onto near-constant "drug = No" columns; a large
    # intercept_scaling makes that penalty negligible. Declared check, same C.
    clean = models.logistic_pipeline().set_params(clf__C=0.01, clf__intercept_scaling=100.0)
    clean.fit(train[feats], train["y"])
    s3 = {**s2, "logistic_sparse_clean": clean.predict_proba(test[feats])[:, 1]}
    rc = bootstrap_compare(y, g, s3, "baseline_prior_visits")
    rcs = bootstrap_compare(y, g, s3, "logistic_sparse")
    rvc = bootstrap_compare(y, g, s3, "logistic_sparse_clean")   # every model vs the recommended one
    out["vs_logistic_sparse_clean"] = {
        k: [rvc[k][m] for m in ("diff_vs_ref", "diff_lo", "diff_hi")] for k in ("logistic_l1", "gbm")}
    coef_clean = models.logistic_coefficients(clean)
    coef_clean.to_csv(out_dir / "logistic_sparse_clean_coefficients.csv", index=False)
    out["logistic_sparse_clean"] = {
        "precision": rc["logistic_sparse_clean"]["precision"], "lo": rc["logistic_sparse_clean"]["lo"],
        "hi": rc["logistic_sparse_clean"]["hi"], "n_coefficients": int(len(coef_clean)),
        "diff_vs_baseline": [rc["logistic_sparse_clean"][m] for m in ("diff_vs_ref", "diff_lo", "diff_hi")],
        "diff_vs_sparse": [rcs["logistic_sparse_clean"][m] for m in ("diff_vs_ref", "diff_lo", "diff_hi")],
    }

    # Calibration of the headline: a perfect ranker, and AUC for every scorer (for reference only;
    # AUC averages over all thresholds, the decision uses one)
    out["perfect_ranker_precision"] = weekly_precision(y, y.astype(float))
    order, starts, counts = _patient_index(g)
    rng = np.random.default_rng(config.SEED)
    boots = {k: [] for k in s3}
    for _ in range(min(200, config.N_BOOT)):
        r = _resample_rows(rng, order, starts, counts)
        for k, v in s3.items():
            boots[k].append(roc_auc_score(y[r], v[r]))
    out["auc"] = {k: {"auc": float(roc_auc_score(y, v)),
                      "lo": float(np.quantile(boots[k], 0.025)), "hi": float(np.quantile(boots[k], 0.975))}
                  for k, v in s3.items()}

    # The recommended (19-coefficient) model under the discharge-destination analysis (Section 6)
    called_c, _ = called_flags(y, s3["logistic_sparse_clean"])
    rhc = bootstrap_compare(y[home], g[home], {"logistic_sparse_clean": s3["logistic_sparse_clean"][home],
                                               "baseline_prior_visits": s["baseline_prior_visits"][home]},
                            "baseline_prior_visits")["logistic_sparse_clean"]
    out["logistic_sparse_clean_destination"] = {
        "share_calls_not_home": float((~home[called_c]).mean()),
        "home_only": {m: rhc[m] for m in ("precision", "lo", "hi", "diff_vs_ref", "diff_lo", "diff_hi")},
    }

    # The recommended model through the same checks as the full models (R3, R5, R6)
    sc_clean = s3["logistic_sparse_clean"]
    called_r, in_week_r = called_flags(y, sc_clean)
    t_r, overall_r = slices.slice_table(test.assign(comorbidity_cluster="all"), called_r, in_week_r,
                                        "logistic_sparse_clean")
    pi = t_r[t_r["variable"] == "prior_inpatient"].set_index("level")
    misses = (pi["readmissions"] - pi["caught"])
    rob = {}
    for n in (100, 400):
        r = bootstrap_compare(y, g, {"clean": sc_clean, "base": s["baseline_prior_visits"]}, "base", week_n=n)["clean"]
        rob[f"N={n}"] = [r[m] for m in ("diff_vs_ref", "diff_lo", "diff_hi")]
    bud = {}
    for b in (10, 100):
        r = bootstrap_compare(y, g, {"clean": sc_clean, "base": s["baseline_prior_visits"]}, "base",
                              budget=b, n_boot=min(200, config.N_BOOT))
        bud[str(b)] = {"clean_reached": r["clean"]["precision"] * b, "baseline_reached": r["base"]["precision"] * b}
    out["logistic_sparse_clean_checks"] = {
        "prior_inpatient_recall": {lv: {k: float(pi.loc[lv, k]) for k in
                                        ("readmissions", "caught", "recall_at_budget", "recall_lo", "recall_hi")}
                                   for lv in pi.index},
        "overall_recall": float(overall_r["recall_at_budget"]),
        "misses_total": int(misses.sum()), "misses_no_prior_inpatient": int(misses.get("0", 0)),
        "robustness_diff_vs_baseline": rob, "budget": bud,
    }

    # Readmission rate by discharge destination, all encounters after exclusions (Section 6)
    rates = df.groupby("discharge_disposition_id")["y"].agg(["size", "mean"])
    out["readmit_rate_by_disposition"] = {
        k: {"n": int(r["size"]), "rate": float(r["mean"])} for k, r in rates.iterrows()}

    (out_dir / "extra.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
