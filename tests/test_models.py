"""Tests for the deployment model builder in modeling/model.py.

build_xgb must return a thresholded binary classifier that predicts 0/1 and whose
decision threshold is set at fit. Uses tiny synthetic data (6 feature columns,
matching FEATURE_NAMES) so the test is fast and needs no raw recordings.
"""
import numpy as np

from remdetect.features import FEATURE_NAMES
from remdetect.modeling.model import build_xgb


def test_build_xgb_returns_thresholded_binary_classifier():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, len(FEATURE_NAMES)))
    y = (rng.random(120) < 0.3).astype(int)

    model = build_xgb(beta=0.5).fit(X, y)

    assert 0.0 < model.threshold < 1.0                 # a real threshold was chosen
    pred = model.predict(X)
    assert set(np.unique(pred)) <= {0, 1}              # binary labels only
    assert model.predict_proba(X).shape == (120, 2)    # two-column probabilities
