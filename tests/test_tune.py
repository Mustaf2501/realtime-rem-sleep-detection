"""Smoke test for the nested-CV engine in modeling/tune.py.

Runs nested_loso_f1 on tiny synthetic data with a minimal budget, just to confirm
the outer/inner loop wiring and the returned shapes. Uses logistic regression (cheap)
so the test stays fast; the XGBoost path shares the same machinery.
"""
import numpy as np

from remdetect.features import FEATURE_NAMES
from remdetect.modeling.tune import logreg_search, nested_loso_f1


def test_nested_loso_f1_runs_and_aligns():
    rng = np.random.default_rng(0)
    n_subjects, per = 5, 40
    X = rng.normal(size=(n_subjects * per, len(FEATURE_NAMES)))
    groups = np.repeat(np.arange(n_subjects), per)
    y = (X[:, 0] + rng.normal(scale=0.5, size=n_subjects * per) > 0).astype(int)  # learnable, ~half REM

    estimator, param_space = logreg_search()
    out = nested_loso_f1(estimator, param_space, X, y, groups, n_iter=2, inner_splits=2)

    assert out["f1"].ndim == 1
    assert len(out["subjects"]) == out["f1"].size == len(out["chosen_params"])
    assert (out["f1"] >= 0).all() and (out["f1"] <= 1).all()
