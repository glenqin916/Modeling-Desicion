Precision at 40 per simulated week of 200 discharges (test set, patient bootstrap 95% CI, 1000 resamples)

| Scorer | Precision@40 [95% CI] | Caught / week | vs prior-visits baseline [95% CI] |
|---|---|---|---|
| baseline_prior_visits | 0.198 [0.181, 0.214] | 7.9 | +0.000 [+0.000, +0.000] |
| baseline_oldest | 0.127 [0.115, 0.138] | 5.1 | -0.071 [-0.088, -0.053] |
| random | 0.114 [0.104, 0.125] | 4.6 | -0.083 [-0.098, -0.069] |
| logistic_l1 | 0.221 [0.204, 0.235] | 8.8 | +0.023 [+0.012, +0.032] |
| gbm | 0.227 [0.211, 0.242] | 9.1 | +0.029 [+0.020, +0.038] |

Worst planned slice (lowest recall at budget, >= 30 readmissions):
- baseline_prior_visits: prior_inpatient=0: recall 0.000 [0.000, 0.000], n=13204 encounters, 1132 readmissions (overall recall 0.350)
- logistic_l1: prior_inpatient=0: recall 0.139 [0.119, 0.158], n=13204 encounters, 1132 readmissions (overall recall 0.385)
- gbm: prior_inpatient=0: recall 0.148 [0.128, 0.169], n=13204 encounters, 1132 readmissions (overall recall 0.395)
