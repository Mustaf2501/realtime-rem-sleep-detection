"""Tests for the feature extraction in features.py.

Each test builds a small, controlled input so the expected output is obvious by
inspection. Run with:

    uv run --extra test python -m pytest tests/ -v

They cover:
  - the feature matrix has the right shape and never contains NaN
  - heart rate is bucketed into (t-30, t] windows and gaps are forward-filled
  - the EMA smoothing matches a hand-computed result
  - the activity count for epoch i comes from the time block [30(i-1), 30i)
  - features are causal: the first k epochs don't change when the rest is removed
"""
import numpy as np
import pytest

from conftest import make_record
from remdetect.dataset import EPOCH_SEC, Record, load_records
from remdetect.features import (ACT_EMA_ALPHA, FEATURE_NAMES, HR_BASELINE_ALPHA,
                                HRV_WINDOW_SEC, _activity_counts, _causal_fill,
                                _ema, _epoch_hr_mean, _windowed_hr_std, featurize)

NAN = np.nan


# --------------------------------------------------------------------------- #
# relative HR and raw-series HRV (the added features)
# --------------------------------------------------------------------------- #
def test_hr_rel_is_hr_mean_minus_ema_baseline():
    r = make_record(n_epochs=20, hr_bpm=60.0)
    hr_mean = _epoch_hr_mean(r)
    expected = hr_mean - _ema(hr_mean, HR_BASELINE_ALPHA)
    got = featurize(r)[:, FEATURE_NAMES.index("hr_rel")]
    assert np.allclose(got, expected)


def test_windowed_hr_std_matches_trailing_window():
    # 5 epochs ending at 0,30,60,90,120 s; a raw HR reading every 5 s with varied values.
    epochs = np.arange(5) * EPOCH_SEC
    ht = np.arange(0.0, 121.0, 5.0)
    rng = np.random.default_rng(0)
    hr = 60.0 + rng.normal(0.0, 5.0, size=ht.size)
    r = Record("v", epochs, np.full(5, 2), ht, hr, np.zeros(1), np.zeros((1, 3)))

    got = _windowed_hr_std(r, HRV_WINDOW_SEC)
    for t, tau in enumerate(epochs):
        window = hr[(ht > tau - HRV_WINDOW_SEC) & (ht <= tau)]   # causal trailing window
        if window.size >= 2:
            assert np.isclose(got[t], window.std())             # population std, matches


def test_windowed_hr_std_is_causal():
    # Earlier epochs' HRV must not change when later epochs (and their HR) are dropped.
    r = make_record(n_epochs=20, hr_bpm=60.0)
    full = _windowed_hr_std(r, HRV_WINDOW_SEC)
    k = 8
    trunc = _windowed_hr_std(truncate(r, k), HRV_WINDOW_SEC)
    assert np.allclose(full[:k], trunc[:k])


def test_sample_entropy_zero_on_flat_and_higher_when_irregular():
    from remdetect.features import _sample_entropy
    assert _sample_entropy(np.full(30, 60.0)) == 0.0     # flat -> perfectly predictable
    rng = np.random.default_rng(0)
    smooth = np.cumsum(rng.normal(0, 0.3, 40))           # correlated random walk
    jerky = rng.normal(60, 5, 40)                        # independent noise
    assert _sample_entropy(jerky) > _sample_entropy(smooth)


# --------------------------------------------------------------------------- #
# helpers to build controlled synthetic recordings
# --------------------------------------------------------------------------- #
def truncate(r: Record, k: int) -> Record:
    """The same record observed only through the end of epoch k-1."""
    t_end = r.epoch_time[k - 1]
    hm, mm = r.hr_time <= t_end, r.motion_time <= t_end
    return Record(r.subject_id, r.epoch_time[:k], r.stage[:k],
                  r.hr_time[hm], r.hr[hm], r.motion_time[mm], r.motion[mm])


# --------------------------------------------------------------------------- #
# shape / sanity
# --------------------------------------------------------------------------- #
def test_featurize_shape_one_row_per_epoch():
    from remdetect.features import FEATURE_NAMES
    r = make_record(n_epochs=20)
    X = featurize(r)
    assert X.shape == (20, len(FEATURE_NAMES))   # one row per epoch, one col per named feature


def test_featurize_has_no_nans():
    X = featurize(make_record(n_epochs=20))
    assert np.isfinite(X).all()          # forward-fill removes every NaN


def test_real_data_shape_and_finite():
    """Sanity check on a real subject-night from the dataset."""
    from remdetect.features import FEATURE_NAMES
    r = load_records()[0]
    X = featurize(r)
    assert X.shape == (r.epoch_time.size, len(FEATURE_NAMES))
    assert np.isfinite(X).all()


# --------------------------------------------------------------------------- #
# time-of-night feature (column 2)
# --------------------------------------------------------------------------- #
def test_time_of_night_is_epoch_time_in_hours():
    from remdetect.features import FEATURE_NAMES
    r = make_record(n_epochs=20)
    X = featurize(r)
    assert np.allclose(X[:, FEATURE_NAMES.index("time_h")], r.epoch_time / 3600.0)


# --------------------------------------------------------------------------- #
# heart-rate bucketing + forward-fill
# --------------------------------------------------------------------------- #
def test_hr_mean_buckets_into_window_and_forward_fills():
    # 4 epochs ending at 0, 30, 60, 90 s. Place samples in known windows:
    #   epoch 0  (-30, 0] : sample at -5s  -> 50
    #   epoch 1  (0, 30]  : samples 10s,20s -> mean(60, 80) = 70
    #   epoch 2  (30, 60] : no samples       -> forward-fill -> 70
    #   epoch 3  (60, 90] : no samples       -> forward-fill -> 70
    #   sample at 100s is beyond the last epoch -> ignored
    r = Record(
        "hr", epoch_time=np.array([0.0, 30, 60, 90]), stage=np.array([2, 2, 2, 2]),
        hr_time=np.array([-5.0, 10, 20, 100]), hr=np.array([50.0, 60, 80, 999]),
        motion_time=np.zeros(1), motion=np.zeros((1, 3)))
    assert np.array_equal(_epoch_hr_mean(r), np.array([50.0, 70.0, 70.0, 70.0]))


# --------------------------------------------------------------------------- #
# EMA smoothing
# --------------------------------------------------------------------------- #
def test_ema_matches_hand_computation():
    # causal EMA, alpha=0.5, adjust=False:
    #   y0=0, y1=0, y2=0.5*8=4, y3=0.5*4=2, y4=0.5*2=1
    out = _ema(np.array([0.0, 0, 8, 0, 0]), alpha=0.5)
    assert np.allclose(out, [0.0, 0.0, 4.0, 2.0, 1.0])


def test_constant_hr_gives_constant_hr_feature():
    # HR steady at 60 bpm -> hr_mean is just the raw per-epoch mean = 60
    from remdetect.features import FEATURE_NAMES
    X = featurize(make_record(n_epochs=20, hr_bpm=60.0))
    assert np.allclose(X[:, FEATURE_NAMES.index("hr_mean")], 60.0)


# --------------------------------------------------------------------------- #
# activity counts: which time block, and zero when still
# --------------------------------------------------------------------------- #
def test_activity_is_zero_with_no_movement():
    counts = _activity_counts(make_record(n_epochs=20))
    assert np.all(counts == 0.0)


def test_activity_count_localizes_to_its_epoch_block():
    # Movement injected only into block [150, 180) s = epoch 6's window.
    counts = _activity_counts(make_record(n_epochs=20, burst_epoch=6))
    assert counts[6] == counts.max()        # the spike lands on epoch 6
    assert counts[6] > 0
    assert np.all(counts[:4] == 0.0)         # quiet epochs before stay zero
    assert counts[0] == 0.0                  # epoch 0 has no preceding block


def test_featurize_activity_column_is_smoothed_squared_counts():
    # paper: the summed count magnitude is squared, then EMA-smoothed
    from remdetect.features import FEATURE_NAMES
    r = make_record(n_epochs=20, burst_epoch=6)
    X = featurize(r)
    assert np.allclose(X[:, FEATURE_NAMES.index("activity")],
                       _ema(_activity_counts(r) ** 2, ACT_EMA_ALPHA))


# --------------------------------------------------------------------------- #
# forward-fill  (applied to the per-epoch HR mean during feature extraction,
# NOT to the raw record; only missing heart-rate epochs are filled)
# --------------------------------------------------------------------------- #
def test_forward_fill_carries_last_value_over_a_gap():
    raw =      np.array([60, NAN, NAN, 72, NAN, 68], float)
    expected = np.array([60, 60,  60,  72, 72,  68], float)
    assert np.array_equal(_causal_fill(raw), expected)


def test_forward_fill_leading_gap_uses_first_real_value():
    # nothing exists before the first sample, so back-fill with the first real one
    raw =      np.array([NAN, NAN, 50, 60], float)
    expected = np.array([50,  50,  50, 60], float)
    assert np.array_equal(_causal_fill(raw), expected)


def test_forward_fill_trailing_gap_holds_last_value():
    raw =      np.array([60, 70, NAN, NAN], float)
    expected = np.array([60, 70, 70,  70], float)
    assert np.array_equal(_causal_fill(raw), expected)


def test_forward_fill_leaves_complete_data_unchanged():
    raw = np.array([60, 61, 62, 63], float)
    assert np.array_equal(_causal_fill(raw), raw)


def test_forward_fill_all_missing_defaults_to_zero():
    # documented fallback: if there is no value to carry, fill with 0.0
    assert np.array_equal(_causal_fill(np.array([NAN, NAN], float)), np.array([0.0, 0.0]))


def test_forward_fill_is_only_used_on_hr_not_the_raw_record():
    # The raw record is untouched; only the per-epoch HR *mean* gets filled.
    # epoch 1's window (0, 30] has a sample; epoch 2's window (30, 60] has none,
    # so the epoch-2 mean is forward-filled from epoch 1 -- the record stays as-is.
    r = Record(
        "ff", epoch_time=np.array([0.0, 30, 60]), stage=np.array([2, 2, 2]),
        hr_time=np.array([15.0]), hr=np.array([66.0]),
        motion_time=np.zeros(1), motion=np.zeros((1, 3)))
    before = r.hr.copy()
    means = _epoch_hr_mean(r)
    assert np.array_equal(means, np.array([66.0, 66.0, 66.0]))  # filled both ways
    assert np.array_equal(r.hr, before)                          # record unchanged


# --------------------------------------------------------------------------- #
# causality: the past must not depend on the future
# --------------------------------------------------------------------------- #
def test_features_are_causal():
    # Movement at epoch 6; truncate at epoch 15 (a quiet region). The features for
    # the first 15 epochs must be identical whether or not the rest of the night
    # exists -- i.e. nothing looks ahead.
    r = make_record(n_epochs=20, burst_epoch=6)
    k = 15
    assert np.allclose(featurize(r)[:k], featurize(truncate(r, k))[:k])


# Truncation invariance is the operational definition of "no look-ahead": removing
# every epoch after k must not change any of the first k feature rows, in ANY
# column. This is a black-box check on the whole pipeline (base features, the
# agcounts activity filter, and the temporal features), so it catches leakage the
# per-column reasoning might miss. Bit-exact: a causal pipeline changes nothing.
@pytest.mark.parametrize("k", [30, 100, 300, 600, 900])
def test_featurize_never_changes_the_past_on_real_data(k):
    r = load_records()[0]
    if k >= r.epoch_time.size:
        pytest.skip("cut beyond this night")
    assert np.array_equal(featurize(r)[:k], featurize(truncate(r, k))[:k])


@pytest.mark.parametrize("k", [7, 9, 12, 16])
def test_activity_filter_does_not_leak_across_the_cut(k):
    # Put the movement burst in the epoch just before the cut, so any look-ahead in
    # the agcounts band-pass filter (future accel samples bleeding backward over the
    # epoch boundary) would show up as a changed count in an epoch we keep.
    r = make_record(n_epochs=20, burst_epoch=k - 1)
    assert np.array_equal(featurize(r)[:k], featurize(truncate(r, k))[:k])
