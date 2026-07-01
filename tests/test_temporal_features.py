"""Tests for the causal temporal features added to features.py, and for the
deployment state contract that lets them be reproduced one epoch at a time in
another language (Dart on the watch).

Two things are asserted here:

  1. Correctness -- each helper (delta, causal rolling mean/std) matches an
     independent brute-force computation, and `featurize` lays the columns out in
     the order documented by FEATURE_NAMES.

  2. The contract -- the batch (cumsum) features equal a strictly streamed
     ring-buffer implementation. `online_derived` below IS the reference
     algorithm to port: a deque of the last BUFFER_LEN base values per signal,
     updated one epoch at a time, plus one scalar of previous-HR for the delta.
     If this test passes, the Dart port that mirrors `online_derived` reproduces
     the committed feature matrix exactly.

    uv run --extra test python -m pytest tests/test_temporal_features.py -v
"""
import math
import os
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features import (BUFFER_LEN, FEATURE_NAMES, ROLL_WIN_LONG, ROLL_WIN_SHORT,
                      _delta, _rolling_mean, _rolling_std, featurize)
from test_features import make_record, truncate


# --------------------------------------------------------------------------- #
# the feature contract: names, order, count
# --------------------------------------------------------------------------- #
EXPECTED_NAMES = [
    "hr", "activity", "time_h",          # base (unchanged)
    "hr_delta",                          # HR rate of change
    "hr_mean_w30",                       # HR ~15 min trend
    "hr_std_w10", "hr_std_w30",          # HR variability (~5 min, ~15 min)
    "act_mean_w30", "act_std_w30",       # activity stillness + texture
]


def test_feature_names_are_the_documented_contract():
    assert FEATURE_NAMES == EXPECTED_NAMES
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))   # unique


def test_featurize_has_one_column_per_named_feature():
    X = featurize(make_record(n_epochs=20, burst_epoch=6))
    assert X.shape == (20, len(FEATURE_NAMES))


def test_max_window_matches_buffer_len():
    # the on-device ring buffer must be the largest window we read from it
    assert BUFFER_LEN == max(ROLL_WIN_LONG, ROLL_WIN_SHORT) == 30


# --------------------------------------------------------------------------- #
# helper correctness vs brute force
# --------------------------------------------------------------------------- #
def test_delta_is_first_difference_with_zero_at_the_start():
    col = np.array([3.0, 5.0, 4.0, 10.0])
    assert np.allclose(_delta(col), [0.0, 2.0, -1.0, 6.0])


def test_rolling_mean_expands_then_slides():
    col = np.arange(1.0, 6.0)                      # [1,2,3,4,5]
    # window 3: [1], [1,2], [1,2,3], [2,3,4], [3,4,5]
    assert np.allclose(_rolling_mean(col, 3), [1.0, 1.5, 2.0, 3.0, 4.0])


def test_rolling_std_matches_population_std_over_the_window():
    rng = np.random.default_rng(0)
    col = rng.normal(size=50)
    w = 10
    got = _rolling_std(col, w)
    for t in range(col.size):
        window = col[max(0, t - w + 1): t + 1]
        assert math.isclose(got[t], window.std(), rel_tol=1e-9, abs_tol=1e-9)


def test_rolling_std_is_zero_on_a_constant_signal():
    assert np.allclose(_rolling_std(np.full(20, 7.0), 10), 0.0)


# --------------------------------------------------------------------------- #
# featurize wires the helpers to the right base columns
# --------------------------------------------------------------------------- #
def test_derived_columns_are_the_helpers_applied_to_the_base_columns():
    X = featurize(make_record(n_epochs=40, burst_epoch=6))
    hr, act = X[:, 0], X[:, 1]
    assert np.allclose(X[:, 3], _delta(hr))
    assert np.allclose(X[:, 4], _rolling_mean(hr, 30))
    assert np.allclose(X[:, 5], _rolling_std(hr, 10))
    assert np.allclose(X[:, 6], _rolling_std(hr, 30))
    assert np.allclose(X[:, 7], _rolling_mean(act, 30))
    assert np.allclose(X[:, 8], _rolling_std(act, 30))


# --------------------------------------------------------------------------- #
# causality: derived features never look ahead
# --------------------------------------------------------------------------- #
def test_temporal_features_are_causal():
    r = make_record(n_epochs=40, burst_epoch=6)
    k = 25
    assert np.allclose(featurize(r)[:k], featurize(truncate(r, k))[:k])


# --------------------------------------------------------------------------- #
# THE DEPLOYMENT CONTRACT: streamed ring-buffer reproduction == batch features.
# This function is the reference to port to Dart.
# --------------------------------------------------------------------------- #
def online_derived(hr_col, act_col):
    """Reproduce the 6 derived columns one epoch at a time, holding only:
        - hr_buf, act_buf : ring buffers of the last BUFFER_LEN base values
        - prev_hr         : one scalar (newest HR), for the delta
    No future sample is ever read. Returns (n, 6) in FEATURE_NAMES[3:] order."""
    hr_buf = deque(maxlen=BUFFER_LEN)
    act_buf = deque(maxlen=BUFFER_LEN)
    prev_hr = None
    rows = []

    def mean(buf, w):
        win = list(buf)[-w:]
        return sum(win) / len(win)

    def std(buf, w):
        win = list(buf)[-w:]
        m = sum(win) / len(win)                      # two-pass: subtract the mean first
        return math.sqrt(max(0.0, sum((v - m) ** 2 for v in win) / len(win)))

    for hr, act in zip(hr_col, act_col):
        delta = 0.0 if prev_hr is None else hr - prev_hr
        prev_hr = hr
        hr_buf.append(hr)
        act_buf.append(act)
        rows.append([
            delta,
            mean(hr_buf, ROLL_WIN_LONG),
            std(hr_buf, ROLL_WIN_SHORT),
            std(hr_buf, ROLL_WIN_LONG),
            mean(act_buf, ROLL_WIN_LONG),
            std(act_buf, ROLL_WIN_LONG),
        ])
    return np.array(rows)


def test_streamed_ring_buffer_reproduces_the_batch_features():
    """The whole point: a strictly one-epoch-at-a-time implementation (what runs on
    the watch) gives the same numbers as the vectorized cumsum build."""
    X = featurize(make_record(n_epochs=60, burst_epoch=20))
    batch_derived = X[:, 3:]
    streamed = online_derived(X[:, 0], X[:, 1])
    assert np.allclose(batch_derived, streamed, rtol=1e-9, atol=1e-9)


def test_contract_holds_on_real_data():
    """On real magnitudes the two-pass std keeps batch and streamed within tight
    float tolerance (the old mean(x^2)-mean(x)^2 form drifted by ~1e-3 here)."""
    from dataset import load_records
    X = featurize(load_records()[0])
    assert np.allclose(X[:, 3:], online_derived(X[:, 0], X[:, 1]), rtol=1e-9, atol=1e-9)
