"""Tests for the causal windowed features in features.py: the HR / activity
variability columns (rolling std) and how featurize wires them.

Causality (removing later epochs never changes an earlier row) is covered by the
truncation-invariance tests in test_features.py, which run the whole pipeline
including these columns. Here we check the helper against a brute-force reference.

    uv run --extra test python -m pytest tests/test_temporal_features.py -v
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features import (FEATURE_NAMES, ROLL_WIN_LONG, ROLL_WIN_SHORT, _rolling_std,
                      featurize)
from test_features import make_record


EXPECTED_NAMES = ["hr_mean", "hr_std_w10", "hr_std_w30", "activity", "act_std_w30", "time_h"]


# --------------------------------------------------------------------------- #
# the feature contract: names, order, count
# --------------------------------------------------------------------------- #
def test_feature_names_are_the_documented_contract():
    assert FEATURE_NAMES == EXPECTED_NAMES
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))       # unique


def test_featurize_has_one_column_per_named_feature():
    X = featurize(make_record(n_epochs=20, burst_epoch=6))
    assert X.shape == (20, len(FEATURE_NAMES))


# --------------------------------------------------------------------------- #
# rolling std (used for hr_std_w30 and act_std_w30)
# --------------------------------------------------------------------------- #
def test_rolling_std_matches_population_std_over_the_window():
    rng = np.random.default_rng(0)
    col = rng.normal(size=50)
    w = 10
    got = _rolling_std(col, w)
    for t in range(col.size):
        window = col[max(0, t - w + 1): t + 1]
        assert math.isclose(got[t], window.std(), rel_tol=1e-9, abs_tol=1e-9)


def test_rolling_std_expands_before_full_and_is_zero_at_the_start():
    assert _rolling_std(np.array([5.0, 5.0, 9.0]), 30)[0] == 0.0   # single value -> 0
    assert np.allclose(_rolling_std(np.full(20, 7.0), 10), 0.0)     # constant -> 0


# --------------------------------------------------------------------------- #
# featurize wires the helper to the right columns
# --------------------------------------------------------------------------- #
def test_derived_columns_are_the_helpers_applied():
    r = make_record(n_epochs=40, burst_epoch=6)
    X = featurize(r)
    f = FEATURE_NAMES.index
    hr_mean = X[:, f("hr_mean")]
    assert np.allclose(X[:, f("hr_std_w10")], _rolling_std(hr_mean, ROLL_WIN_SHORT))
    assert np.allclose(X[:, f("hr_std_w30")], _rolling_std(hr_mean, ROLL_WIN_LONG))
    assert np.allclose(X[:, f("act_std_w30")], _rolling_std(X[:, f("activity")], ROLL_WIN_LONG))
