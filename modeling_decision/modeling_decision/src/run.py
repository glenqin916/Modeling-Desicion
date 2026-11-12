"""Reproduce every number in the report.

    python -m src.run            # full run (headline numbers)
    python -m src.run --quick    # small grids and bootstrap, for a smoke test
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, data, models, slices, split, unsupervised
from .metric import bootstrap_compare, called_flags


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fmt(r):
    return f"{r['precision']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=config.RAW_CSV)
    ap.add_argument("--split-file", type=Path, default=config.SPLIT_FILE)
    ap.add_argument("--out", type=Path, default=config.RESULTS_DIR)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    with config.overrides(config.QUICK if args.quick else {}):
        return _main(args)


def _main(args):
    np.random.seed(config.SEED)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    # ---- data and locked split ------------------------------------------------------------
    df = data.prepare(data.load_raw(args.data))
    train, test = split.get_split(df, args.split_file)
    summary = {
        "encounters_after_exclusions": len(df), "patients": int(df["patient_nbr"].nunique()),
        "share_encounters_from_repeat_patients":
            float((df.groupby("patient_nbr")["patient_nbr"].transform("size") > 1).mean()),
        "train_encounters": len(train), "test_encounters": len(test),
        "test_patients": int(test["patient_nbr"].nunique()),
        "train_readmit_rate": float(train["y"].mean()), "test_readmit_rate": float(test["y"].mean()),
    }
    log(f"split: {json.dumps(summary)}")
    folds = split.cv_folds(train)

    # ---- R4: comorbidity clusters (fit on train only) -------------------------------------
    k, k_table = unsupervised.choose_k(train)
    km = unsupervised.fit_clusters(train, k)
    train["comorbidity_cluster"] = unsupervised.assign(km, train)
    test["comorbidity_cluster"] = unsupervised.assign(km, test)
    k_table.to_csv(out / "kmeans_choose_k.csv", index=False)
    unsupervised.profile(train).to_csv(out / "kmeans_profile.csv")
    log(f"k-means: k={k} (max silhouette)")

    # ---- R3: two models, same split, same metric, same tuning procedure -------------------
    feats = data.FEATURES
    assert not set(feats) & data.FORBIDDEN
    logit, logit_params, logit_cv = models.fit_logistic(train, folds, feats)
    log(f"logistic tuned: {logit_params}")
    gbm, gbm_params, gbm_cv = models.fit_gbm(train, folds, feats)
    log(f"gbm tuned: {gbm_params}")
    _, logit_k_params, logit_k_cv = models.fit_logistic(
        train, folds, feats + ["comorbidity_cluster"], extra_cat=["comorbidity_cluster"])
    cluster_cv = {
        "logistic_cv": float(logit_cv["mean_test_score"].max()),
        "logistic_plus_cluster_cv": float(logit_k_cv["mean_test_score"].max()),
    }
    log(f"cluster as feature, CV on train: {cluster_cv}")
    for name, cv in [("logistic", logit_cv), ("gbm", gbm_cv), ("logistic_plus_cluster", logit_k_cv)]:
        cv.assign(params=cv["params"].astype(str)).to_csv(out / f"cv_{name}.csv", index=False)
    models.logistic_coefficients(logit).to_csv(out / "logistic_coefficients.csv", index=False)

    # ---- R2: baselines and headline on the locked test set --------------------------------
    scores = {
        "baseline_prior_visits": test["number_inpatient"].to_numpy(float),
        "baseline_oldest": test["age_years"].to_numpy(float),
        "random": np.zeros(len(test)),
        "logistic_l1": logit.predict_proba(test[feats])[:, 1],
        "gbm": gbm.predict_proba(test[feats])[:, 1],
    }
    ref = "baseline_prior_visits"
    y, g = test["y"].to_numpy(), test["patient_nbr"].to_numpy()

    results = {}
    for n in (config.WEEK_N, *config.SENSITIVITY_N):
        results[f"N={n}"] = bootstrap_compare(y, g, scores, ref, week_n=n)
        log(f"N={n}: " + "; ".join(f"{k} {fmt(v)}" for k, v in results[f"N={n}"].items()))
    first = test["first_encounter"].to_numpy()
    results["first_encounter_only_N=200"] = bootstrap_compare(
        y[first], g[first], {k: v[first] for k, v in scores.items()}, ref)

    # ---- R5: slices ----------------------------------------------------------------------
    tables, overall = [], {}
    worst = {}
    for name in ["baseline_prior_visits", "logistic_l1", "gbm"]:
        called, in_week = called_flags(y, scores[name])
        t, overall[name] = slices.slice_table(test, called, in_week, name)
        tables.append(t)
        w = slices.worst_slice(t)
        worst[name] = None if w is None else w.to_dict()
    slice_df = pd.concat(tables, ignore_index=True)
    slice_df.to_csv(out / "slices.csv", index=False)

    # ---- write everything -----------------------------------------------------------------
    headline = {
        "split": summary, "kmeans_k": k, "cluster_cv": cluster_cv,
        "tuned": {"logistic_l1": logit_params, "gbm": gbm_params},
        "precision_at_40": results, "slice_overall": overall, "worst_slice": worst,
        "config": {"seed": config.SEED, "budget": config.BUDGET, "week_n": config.WEEK_N,
                   "n_boot": config.N_BOOT, "quick": args.quick},
    }
    (out / "headline.json").write_text(json.dumps(headline, indent=2, default=float))

    main_res = results[f"N={config.WEEK_N}"]
    lines = [f"Precision at {config.BUDGET} per simulated week of {config.WEEK_N} discharges "
             f"(test set, patient bootstrap 95% CI, {config.N_BOOT} resamples)", "",
             "| Scorer | Precision@40 [95% CI] | Caught / week | vs prior-visits baseline [95% CI] |",
             "|---|---|---|---|"]
    for k_, r in main_res.items():
        lines.append(f"| {k_} | {fmt(r)} | {r['caught_per_week']:.1f} | "
                     f"{r['diff_vs_ref']:+.3f} [{r['diff_lo']:+.3f}, {r['diff_hi']:+.3f}] |")
    lines += ["", "Worst planned slice (lowest recall at budget, >= "
              f"{config.MIN_SLICE_POSITIVES} readmissions):"]
    for k_, w in worst.items():
        if w is None:
            lines.append(f"- {k_}: no planned slice with enough readmissions")
            continue
        lines.append(f"- {k_}: {w['variable']}={w['level']}: recall {w['recall_at_budget']:.3f} "
                     f"[{w['recall_lo']:.3f}, {w['recall_hi']:.3f}], n={w['n_encounters']} encounters, "
                     f"{w['readmissions']} readmissions (overall recall {overall[k_]['recall_at_budget']:.3f})")
    (out / "headline.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
