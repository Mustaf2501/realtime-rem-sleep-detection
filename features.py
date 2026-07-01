"""Feature extraction, following the paper (Mallela & Mallett, 2024), plus causal
temporal features over the two wearable signals.

Turns one subject-night into a causal per-epoch feature matrix: one row per 30 s
epoch, each value using only samples at or before the epoch end. The column order
is fixed by FEATURE_NAMES:

  0  hr            heart rate, mean bpm in the epoch, EMA-smoothed, cubed, /1000
  1  activity      ActiGraph activity counts (agcounts; Neishabouri 2022): accel
                   resampled to 30 Hz -> per-second count magnitudes, summed in the
                   epoch, squared, then EMA-smoothed
  2  time_h        time-of-night in hours
  3  hr_delta      hr[t] - hr[t-1]              (rate of change at REM transitions)
  4  hr_mean_w30   mean of hr over last 30 epochs (~15 min trend)
  5  hr_std_w10    std of hr over last 10 epochs (~5 min HR variability)
  6  hr_std_w30    std of hr over last 30 epochs (~15 min HR variability)
  7  act_mean_w30  mean of activity over last 30 epochs (sustained stillness)
  8  act_std_w30   std of activity over last 30 epochs (twitch-vs-still texture)

The counts come from agcounts; the EMA and gap-fill from pandas. The temporal
features (3-8) are computed in batch with cumulative sums, but each is a function of
the current and earlier epochs only -- never the future.

DEPLOYMENT STATE CONTRACT (how to reproduce columns 3-8 live, one epoch at a time,
in another language -- e.g. Dart on the watch). State is per *signal*, not per
feature: a single ring buffer sized to the largest window feeds every windowed
feature on that signal. Clear all state at the start of each night (sleep session).

  hr signal:
    ring buffer  : last BUFFER_LEN (=30) values of column `hr`  (circular float[])
    prev_hr      : one float = the value of `hr` at the previous epoch (for hr_delta)
  activity signal:
    ring buffer  : last BUFFER_LEN (=30) values of column `activity`  (circular float[])
  time_h:
    none -- it is the wall clock (seconds since lights-off / 3600)

  Total live state: 2 ring buffers (30 floats each) + 1 scalar  ~= 61 floats, fixed.

  Per-epoch update at time t, given the freshly computed base values hr_t, act_t:
    hr_delta     = 0 if first epoch else hr_t - prev_hr ; then prev_hr = hr_t
    push hr_t into the hr buffer (drop the oldest once full)
    push act_t into the activity buffer (drop the oldest once full)
    mean_w(buf)  = sum(last w in buf) / count(last w in buf)        # count < w until warm
    std_w(buf)   = let m = mean_w(buf); sqrt(max(0, mean((x - m)^2 over last w)))
                   # two-pass (subtract the mean first): stable on large-magnitude
                   # activity, where mean(x^2)-mean(x)^2 would lose all precision
    hr_mean_w30  = mean_w(hr_buf, 30)    hr_std_w10 = std_w(hr_buf, 10)
    hr_std_w30   = std_w(hr_buf, 30)
    act_mean_w30 = mean_w(act_buf, 30)   act_std_w30 = std_w(act_buf, 30)

  Windows expand before they are full (first epoch's std is 0). tests/
  test_temporal_features.py::online_derived is the executable reference for this
  contract and asserts it matches the batch build, epoch for epoch.

Two things the paper does not pin down. The EMA smoothing constants aren't stated
(0.30 here). The paper's final "normalize the EMA" is left out: a whole-recording
normalization would use the future, and scaling is the model's job anyway (trees
ignore it; a scale-sensitive model can fit a scaler on the training split).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from agcounts.extract import get_counts

from dataset import EPOCH_SEC, Record

HR_EMA_ALPHA = 0.30       # smoothing for heart-rate feature (not stated in the paper)
ACT_EMA_ALPHA = 0.30      # smoothing for activity feature (not stated in the paper)
ACCEL_FS = 30             # Hz; resampling grid for activity counts (the paper's rate)

ROLL_WIN_SHORT = 10       # epochs (~5 min): short-scale HR variability
ROLL_WIN_LONG = 30        # epochs (~15 min): REM-episode-scale trend / variability
BUFFER_LEN = max(ROLL_WIN_SHORT, ROLL_WIN_LONG)   # on-device ring-buffer length per signal

FEATURE_NAMES = [
    "hr", "activity", "time_h",
    "hr_delta",
    "hr_mean_w30",
    "hr_std_w10", "hr_std_w30",
    "act_mean_w30", "act_std_w30",
]


def featurize(record: Record) -> np.ndarray:
    """Causal (n_epochs, len(FEATURE_NAMES)) feature matrix for one night. Columns
    follow FEATURE_NAMES; temporal columns reset per night (this is called per
    record, so windows never cross a subject boundary)."""
    hr = (_ema(_epoch_hr_mean(record), HR_EMA_ALPHA) ** 3) / 1000.0
    activity = _ema(_activity_counts(record) ** 2, ACT_EMA_ALPHA)        # summed, then squared
    time_of_night = record.epoch_time / 3600.0
    return np.column_stack([
        hr,
        activity,
        time_of_night,
        _delta(hr),
        _rolling_mean(hr, ROLL_WIN_LONG),
        _rolling_std(hr, ROLL_WIN_SHORT),
        _rolling_std(hr, ROLL_WIN_LONG),
        _rolling_mean(activity, ROLL_WIN_LONG),
        _rolling_std(activity, ROLL_WIN_LONG),
    ])


def _delta(col: np.ndarray) -> np.ndarray:
    """First difference col[t]-col[t-1], with 0 at the first epoch (no prior value).
    Live: keep one scalar (the previous value)."""
    out = np.zeros_like(col, dtype=float)
    out[1:] = col[1:] - col[:-1]
    return out


def _rolling_mean(col: np.ndarray, w: int) -> np.ndarray:
    """Causal mean over the last w epochs (expanding until full).
    Live: mean of the last w values held in the signal's ring buffer."""
    return _causal_window_reduce(col, w, np.mean)


def _rolling_std(col: np.ndarray, w: int) -> np.ndarray:
    """Causal population std over the last w epochs (expanding until full). Uses a
    two-pass reduction (np.std subtracts the window mean before squaring), so it
    stays exact on large-magnitude signals where mean(x^2)-mean(x)^2 -- and pandas'
    rolling std -- lose precision (a still night's activity std must be exactly 0).
    0 on the first epoch. Live: keep the last w values (ring buffer) and apply the
    same two-pass formula."""
    return _causal_window_reduce(col, w, np.std)


def _causal_window_reduce(col: np.ndarray, w: int, reduce) -> np.ndarray:
    """Apply `reduce` (np.mean / np.std, ddof=0) to each causal window
    [max(0, t-w+1), t]. Windows expand until they reach w, then slide. Each window
    is reduced directly (two-pass, no cumulative sums) so the result matches a live
    ring-buffer computation epoch-for-epoch; see the deployment contract above."""
    col = np.asarray(col, dtype=float)
    n = col.size
    out = np.empty(n)
    warm = min(w - 1, n)
    for t in range(warm):                                  # expanding windows, t < w-1
        out[t] = reduce(col[: t + 1])
    if n >= w:                                             # full windows, vectorized
        windows = np.lib.stride_tricks.sliding_window_view(col, w)
        out[w - 1:] = reduce(windows, axis=1)
    return out


def _ema(x: np.ndarray, alpha: float) -> np.ndarray:
    """Causal exponential moving average (pandas; past-and-present only)."""
    return pd.Series(x).ewm(alpha=alpha, adjust=False).mean().to_numpy()


def _epoch_hr_mean(record: Record) -> np.ndarray:
    """Mean heart rate per epoch, using only samples in (t - EPOCH_SEC, t]. Epochs
    with no sample are forward-filled from the past so callers never see NaN."""
    epochs = record.epoch_time
    n = epochs.size
    epoch_of = np.searchsorted(epochs, record.hr_time, side="left")
    keep = (epoch_of < n) & (record.hr_time > epochs[np.clip(epoch_of, 0, n - 1)] - EPOCH_SEC)
    idx = epoch_of[keep]
    total = np.bincount(idx, record.hr[keep], minlength=n)
    count = np.bincount(idx, minlength=n)
    mean = np.divide(total, count, out=np.full(n, np.nan), where=count > 0)
    return _causal_fill(mean)


def _activity_counts(record: Record) -> np.ndarray:
    """Per-epoch summed activity-count magnitude (paper / Neishabouri 2022).

    Resample the accelerometer to a fixed 30 Hz grid, compute activity counts per
    SECOND with agcounts, take the vector magnitude per second, and sum the
    magnitudes within each 30 s epoch. Causal: epoch i sums only seconds
    [30(i-1), 30 i). Epoch 0 (window before t=0) has no counts.
    """
    n = record.epoch_time.size
    grid = np.arange(0.0, record.epoch_time[-1], 1.0 / ACCEL_FS)
    if grid.size < ACCEL_FS:                                  # under a second of data
        return np.zeros(n)
    accel = np.column_stack(
        [np.interp(grid, record.motion_time, record.motion[:, k]) for k in range(3)])
    per_second = get_counts(accel, freq=ACCEL_FS, epoch=1)    # (seconds, 3)
    magnitude = np.linalg.norm(per_second, axis=1)            # vector addition

    epoch_of_second = np.arange(magnitude.size) // int(EPOCH_SEC) + 1
    return np.bincount(epoch_of_second, weights=magnitude, minlength=n + 1)[:n]


def _causal_fill(x: np.ndarray) -> np.ndarray:
    """Forward-fill gaps from the past (pandas); leading gaps take the first finite
    value, all-missing -> 0. Defines the missing-heart-rate case out of existence."""
    return pd.Series(x).ffill().bfill().fillna(0.0).to_numpy()
