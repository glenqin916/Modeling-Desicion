"""Figures for the report (run after src.run and src.extra).

    python -m src.figures      # writes results/fig1_headline.png, fig2_budget.png, fig3_slices.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402

BLUE, ORANGE, GRAY, LIGHT = "#2a78d6", "#eb6834", "#52514e", "#9a9993"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"logistic_l1": BLUE, "gbm": ORANGE, "baseline_prior_visits": GRAY, "random": LIGHT,
          "baseline_oldest": LIGHT}
NAMES = {"logistic_l1": "L1 logistic", "gbm": "Gradient boosting",
         "baseline_prior_visits": "Baseline: most prior visits",
         "baseline_oldest": "Oldest 40", "random": "Random 40"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "figure.dpi": 200, "savefig.bbox": "tight",
})


def fig_headline(h, out):
    r = h["precision_at_40"][f"N={config.WEEK_N}"]
    order = ["random", "baseline_oldest", "baseline_prior_visits", "logistic_l1", "gbm"]
    fig, ax = plt.subplots(figsize=(6.2, 2.2))
    for i, k in enumerate(order):
        v = r[k]
        ax.plot([v["lo"] * 40, v["hi"] * 40], [i, i], color=COLORS[k], lw=2, solid_capstyle="round")
        ax.plot(v["precision"] * 40, i, "o", ms=7, color=COLORS[k], mec="white", mew=1.5)
        ax.text(v["hi"] * 40 + 0.15, i, f"{v['precision'] * 40:.1f}", va="center", color=INK, fontsize=8.5)
    ax.set_yticks(range(len(order)), [NAMES[k] for k in order])
    ax.set_xlabel("Readmitted patients reached per week, out of 40 calls (95% CI)")
    ax.set_xlim(3.5, 10.5)
    ax.grid(axis="y", visible=False)
    fig.savefig(out / "fig1_headline.png")
    plt.close(fig)


def fig_budget(e, out):
    c = e["budget_curve"]
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    for k in ["random", "baseline_prior_visits", "logistic_l1", "gbm"]:
        b = sorted(int(x) for x in c[k])
        y = [c[k][str(x)]["reached"] for x in b]
        ax.plot(b, y, color=COLORS[k], lw=2, marker="o", ms=4, mec="white", mew=1)
        dy = {"gbm": 0.55, "logistic_l1": -0.55}.get(k, 0)
        label = "Random" if k == "random" else NAMES[k]
        ax.text(b[-1] + 2, y[-1] + dy, label, va="center", color=INK, fontsize=8)
    ax.axvline(40, color=MUTED, lw=0.8, ls=(0, (3, 3)))
    ax.text(41, 1.5, "today's budget", color=MUTED, fontsize=7.5)
    ax.set_xlabel("Calls per week (out of 200 discharges)")
    ax.set_ylabel("Readmitted patients reached")
    ax.set_xlim(5, 135)
    ax.set_ylim(0, 17.5)
    fig.savefig(out / "fig2_budget.png")
    plt.close(fig)


def fig_slices(s, out):
    s = s[s["variable"] == "prior_inpatient"]
    levels = ["0", "1", "2", "3+"]
    models = ["baseline_prior_visits", "logistic_l1", "gbm"]
    fig, ax = plt.subplots(figsize=(6.8, 2.5))
    w = 0.26
    for j, m in enumerate(models):
        for i, lv in enumerate(levels):
            row = s[(s["model"] == m) & (s["level"].astype(str) == lv)].iloc[0]
            x = i + (j - 1) * w
            ax.bar(x, row["recall_at_budget"], width=w - 0.03, color=COLORS[m],
                   label=NAMES[m] if i == 0 else None)
            if row["recall_hi"] > row["recall_lo"]:
                ax.plot([x, x], [row["recall_lo"], row["recall_hi"]], color=INK, lw=1)
    n = {lv: s[(s["level"].astype(str) == lv)].iloc[0] for lv in levels}
    ax.text(-w, 0.02, "0", ha="center", color=INK, fontsize=8)
    ax.set_xticks(range(4), [f"{lv} prior stays\n(n = {n[lv]['readmissions']:,} readm.)" for lv in levels],
                  fontsize=8)
    ax.set_ylabel("Share of readmissions\nthat got a call")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.savefig(out / "fig3_slices.png")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=config.RESULTS_DIR, help="results folder to read and write")
    out = ap.parse_args(argv).out
    h = json.loads((out / "headline.json").read_text())
    e = json.loads((out / "extra.json").read_text())
    s = pd.read_csv(out / "slices.csv")
    fig_headline(h, out)
    fig_budget(e, out)
    fig_slices(s, out)
    print("wrote figures to", out)


if __name__ == "__main__":
    main()
