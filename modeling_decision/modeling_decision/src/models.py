"""The two models across the interpretability axis, tuned the same way.

Simple:   L1-regularized logistic regression (sparse coefficients).
Flexible: histogram gradient boosting.

Both are sklearn Pipelines, so every preprocessing step (one-hot / ordinal encoding with rare
levels pooled, scaling) is fit inside each training fold and never sees the test set. Both are
tuned by grouped CV on train with the locked metric as the score.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from . import config
from .data import CATEGORICAL, COMORB, NUMERIC
from .metric import weekly_precision


def precision_at_budget_scorer(estimator, X, y):
    """GridSearchCV scorer: the locked metric on a validation fold (5 week shuffles)."""
    return weekly_precision(y, estimator.predict_proba(X)[:, 1], n_shuffles=5)


def logistic_pipeline(extra_cat=()):
    cat = list(CATEGORICAL) + list(extra_cat)
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50), cat),
        ("num", StandardScaler(), NUMERIC),
        ("comorb", "passthrough", COMORB),
    ])
    clf = LogisticRegression(l1_ratio=1.0, solver="liblinear", max_iter=2000,
                             random_state=config.SEED)
    return Pipeline([("pre", pre), ("clf", clf)])


def gbm_pipeline(extra_cat=()):
    cat = list(CATEGORICAL) + list(extra_cat)
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                               encoded_missing_value=np.nan, min_frequency=50), cat),
        ("num", "passthrough", NUMERIC + COMORB),
    ])
    is_cat = [True] * len(cat) + [False] * (len(NUMERIC) + len(COMORB))
    clf = HistGradientBoostingClassifier(categorical_features=is_cat, max_iter=500,
                                         early_stopping=True, random_state=config.SEED)
    return Pipeline([("pre", pre), ("clf", clf)])


def tune(pipeline, grid, train: pd.DataFrame, folds, features):
    search = GridSearchCV(pipeline, grid, scoring=precision_at_budget_scorer, cv=folds,
                          refit=True, n_jobs=-1)
    search.fit(train[features], train["y"])
    cv = pd.DataFrame(search.cv_results_)[["params", "mean_test_score", "std_test_score"]]
    return search.best_estimator_, search.best_params_, cv


def fit_logistic(train, folds, features, extra_cat=()):
    grid = {"clf__C": list(config.LOGIT_C_GRID)}
    return tune(logistic_pipeline(extra_cat), grid, train, folds, features)


def fit_gbm(train, folds, features, extra_cat=()):
    grid = {f"clf__{k}": list(v) for k, v in config.GBM_GRID.items()}
    return tune(gbm_pipeline(extra_cat), grid, train, folds, features)


def logistic_coefficients(model) -> pd.DataFrame:
    names = model.named_steps["pre"].get_feature_names_out()
    coef = model.named_steps["clf"].coef_.ravel()
    out = pd.DataFrame({"feature": names, "coef": coef, "odds_ratio": np.exp(coef)})
    out = out[out["coef"] != 0]
    return out.reindex(out["coef"].abs().sort_values(ascending=False).index).reset_index(drop=True)
