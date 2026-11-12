# Readmission triage: which 40 patients should the discharge nurse call?

Each week a discharge nurse decides which 40 recently discharged diabetic patients to phone.
Missing a patient who is then readmitted within 30 days costs an avoidable inpatient stay.
Calling one who is not readmitted costs about 10 minutes of nurse time.

This repo implements the locked proposal: a patient-grouped split, precision at 40 per simulated
week as the metric, and "call the 40 with the most prior inpatient visits" as the baseline.

## Reproduce the headline number

Requires Python 3.11 or newer (the pinned pandas and numpy versions need it).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.run                # about 6 minutes on 2 cores; writes results/
python -m src.extra              # follow-up analyses for the report (about 3 minutes)
python -m src.figures            # results/fig*.png
```

On Windows (Command Prompt):

```
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m src.run
python -m src.extra
python -m src.figures
```

The data is already in the repo, so there is no download step.

The headline table is printed and saved to `results/headline.md`. Everything is seeded
(`SEED = 0` in `src/config.py`), and two runs give byte-identical `results/headline.json`.

`make all` runs the same steps. `make test` runs the leakage and metric checks, and
`make smoke` runs the full pipeline on synthetic data with the real schema in under a minute.

## The locked choices and where they live

| Proposal item | Code | What it does |
|---|---|---|
| Split | `src/split.py` | `StratifiedGroupKFold(5)` on `patient_nbr`, seed 0; fold 0 = test (20% of patients). The first run writes `data/splits/test_patients.csv`; later runs read it. **Commit that file.** |
| Metric | `src/metric.py` | Test set cut into random "weeks" of 200 discharges; top 40 per week called; precision averaged over 20 week assignments. 95% CI from a patient-level bootstrap (1,000 resamples). Model-vs-baseline differences are paired: same patients, same weeks. Sensitivity at N = 100 and 400. |
| Baseline | `src/run.py` | Prior inpatient visits (`number_inpatient`), with oldest-40 and random alongside. Ties are broken at random. |
| Exclusions | `src/config.py` | Discharge dispositions 11, 13, 14, 19, 20, 21 (died or hospice) are dropped before splitting. |

## Leakage rules (checked in `tests/`)

- **No patient on both sides.** The test set and every CV fold are grouped on `patient_nbr`.
- **No features from a patient's other rows.** Every feature belongs to the encounter itself and is known at discharge. `number_inpatient`, `number_emergency` and `number_outpatient` count visits in the year *before* the encounter, so they qualify. Counting a patient's rows in the dataset does not: a 30-day readmission is itself another row. `data.FORBIDDEN` lists the columns that can never be features.
- **Preprocessing fit on train only.** One-hot and ordinal encoders (rare levels pooled at < 50), the scaler and k-means are fit inside training folds via sklearn Pipelines. ICD-9 grouping is a fixed rule.
- **"None" is not missing.** In `A1Cresult` and `max_glu_serum`, "None" means "not measured", so only `?` is read as missing.

## What the run produces (`results/`)

| File | Requirement | Contents |
|---|---|---|
| `headline.md`, `headline.json` | R2, R3 | Precision at 40 with CI for both baselines, random, L1 logistic and gradient boosting; the paired difference from the baseline; N sensitivity; first-encounter-only robustness check; tuned hyperparameters |
| `cv_*.csv` | R3 | Grouped-CV score for every grid point (both models tuned on the same folds with the locked metric) |
| `logistic_coefficients.csv` | R3 | Non-zero L1 coefficients and odds ratios (numeric features per standard deviation) |
| `kmeans_choose_k.csv`, `kmeans_profile.csv` | R4 | k chosen by silhouette on train; cluster sizes, comorbidity mix and train readmission rate |
| `headline.json` → `cluster_cv` | R4 | Whether adding the cluster id to the logistic model raises CV precision on train (the decision is made on CV, not test) |
| `extra.json`, `logistic_sparse*_coefficients.csv` | R3, R5, R6 | GBM vs logistic paired difference; the 35- and 19-coefficient sparse logistic models (the 19-coefficient refit makes the intercept penalty negligible); perfect-ranker precision and AUC for every scorer; share of calls going to patients not discharged home; precision if only home discharges are eligible; the no-prior-inpatient slice; readmission rate by discharge destination; logistic sparsity along the C grid |
| `slices.csv` | R5 | Per slice: encounters, patients, readmissions, called, caught, recall at budget and precision among the called, each with a patient-bootstrap 95% CI. Insurance and prior-inpatient count are **planned**; age band and comorbidity cluster are marked **exploratory**. Slices with < 30 readmissions are flagged `small`. |

The worst slice is the planned, non-small slice with the lowest recall at budget, meaning the
readmissions the nurse never hears about. Expect patients with **zero prior inpatient visits**
to come out worst for any scorer that leans on prior visits: they are the largest group, and
almost none of them make the top 40.

## Tests

```bash
python -m pytest -q          # 79 tests, about 30 seconds, synthetic data only (no download needed)
```

| File | What it checks |
|---|---|
| `tests/test_data.py` | ICD-9 grouping at every range edge, exclusions, target, "None" kept as a level, comorbidity flags, slice variables, no forbidden features |
| `tests/test_split.py` | No patient on both sides, 20% test share, determinism, the split file is reused, a split file from other data is rejected, CV folds are grouped |
| `tests/test_metric.py` | Hand-checked example, perfect/inverse/constant scorers, random tie-breaking, invalid budgets, patient-level resampling, paired bootstrap, reproducibility |
| `tests/test_models.py` | Both pipelines fit and predict, unseen categories at predict time, the planted signal is learned, tuning uses the locked metric |
| `tests/test_unsupervised_and_slices.py` | k-means selection and assignment, slice counts add up, slices with no calls, worst-slice rule |
| `tests/test_end_to_end.py` | The full pipeline on synthetic data: every output written, two runs identical, `--quick` leaves settings untouched, `src.extra` and `src.figures` run on any folder, `fetch_data.py --zip` |

## Repo layout

```
diabetes+130-us+hospitals+for+years+1999-2008/   the UCI data (diabetic_data.csv, IDS_mapping.csv)
data/fetch_data.py        re-download the data + check row/column counts
data/splits/              the locked test patient list (committed)
src/config.py             every locked number: seed, budget, N, grids, exclusions, slices
src/data.py               cleaning, ICD-9 groups, feature lists
src/split.py              locked split + grouped CV folds
src/metric.py             precision at 40 per simulated week, patient bootstrap
src/models.py             L1 logistic and gradient boosting pipelines, grouped-CV tuning
src/unsupervised.py       k-means on the comorbidity block
src/slices.py             failure analysis by slice
src/run.py                runs everything and writes results/
tests/                    79 pytest checks (data, split, metric, models, slices, end-to-end); synthetic data generator
```

## Data

[UCI Diabetes 130-US Hospitals for Years 1999–2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008),
CC BY 4.0. Strack et al. (2014), *BioMed Research International*, Article ID 781670.
The data ships with the repo in `diabetes+130-us+hospitals+for+years+1999-2008/` (`diabetic_data.csv` and `IDS_mapping.csv`, as downloaded from UCI), which CC BY 4.0 allows with the citation above. To re-download it, run `python data/fetch_data.py` (or `--zip path/to/file.zip`).
