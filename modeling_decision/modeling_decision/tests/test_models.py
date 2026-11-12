import numpy as np
import pytest

from src import data, models, split


def test_logistic_pipeline_fits_and_predicts(split_frames):
    tr, te = split_frames
    m = models.logistic_pipeline().set_params(clf__C=0.1).fit(tr[data.FEATURES], tr["y"])
    p = m.predict_proba(te[data.FEATURES])[:, 1]
    assert p.shape == (len(te),) and ((p >= 0) & (p <= 1)).all()


def test_gbm_pipeline_fits_and_predicts(split_frames):
    tr, te = split_frames
    m = models.gbm_pipeline().fit(tr[data.FEATURES], tr["y"])
    p = m.predict_proba(te[data.FEATURES])[:, 1]
    assert p.shape == (len(te),) and np.isfinite(p).all()


def test_unseen_category_at_predict_time(split_frames):
    tr, te = split_frames
    te2 = te.copy()
    te2.loc[te2.index[:5], "medical_specialty"] = "NeverSeenBefore"
    for pipe in (models.logistic_pipeline(), models.gbm_pipeline()):
        m = pipe.fit(tr[data.FEATURES], tr["y"])
        assert np.isfinite(m.predict_proba(te2[data.FEATURES])).all()


def test_models_learn_the_planted_signal(split_frames):
    tr, te = split_frames
    m = models.logistic_pipeline().set_params(clf__C=0.3).fit(tr[data.FEATURES], tr["y"])
    coef = models.logistic_coefficients(m)
    top = coef.set_index("feature")["coef"]
    # the planted effects: more prior stays -> higher risk, and they rank among the strongest
    assert top["num__number_inpatient"] > 0
    assert "num__number_inpatient" in list(coef["feature"].head(5))
    assert (coef["coef"] != 0).all()
    assert np.allclose(coef["odds_ratio"], np.exp(coef["coef"]))


def test_tuning_uses_the_locked_metric_and_grouped_folds(split_frames, monkeypatch):
    tr, _ = split_frames
    from src import config
    monkeypatch.setattr(config, "LOGIT_C_GRID", (0.01, 0.1))
    folds = split.cv_folds(tr)
    best, params, cv = models.fit_logistic(tr, folds, data.FEATURES)
    assert params["clf__C"] in (0.01, 0.1)
    assert len(cv) == 2 and cv["mean_test_score"].between(0, 1).all()


def test_scorer_matches_metric(split_frames):
    from src.metric import weekly_precision
    tr, te = split_frames
    m = models.logistic_pipeline().fit(tr[data.FEATURES], tr["y"])
    s = models.precision_at_budget_scorer(m, te[data.FEATURES], te["y"])
    assert s == pytest.approx(weekly_precision(te["y"], m.predict_proba(te[data.FEATURES])[:, 1], n_shuffles=5))
