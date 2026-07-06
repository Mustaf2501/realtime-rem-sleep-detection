"""Feature extraction: a small causal per-epoch feature set from heart rate, motion,
and time. One row per 30 s epoch, each value using only samples at or before the
epoch end. Column order is fixed by FEATURE_NAMES:

  0  hr_mean       mean heart rate (bpm) in the epoch (t-30, t]
  1  hr_rel        hr_mean minus a slow causal EMA baseline: HR above the subject's
                   own recent level, so REM's HR rise shows without the large
                   between-subject offset that makes absolute hr_mean nearly useless
  2  hr_std_w10    std of hr_mean over the last 10 epochs (~5 min HR variability)
  3  hr_std_w30    std of hr_mean over the last 30 epochs (~15 min HR variability)
  4  hr_raw_std    std of the RAW ~0.2 Hz HR samples over the last ~2 min: short-scale
                   HR variability that the epoch-mean std above blurs out
  5  hr_entropy    sample entropy of the raw HR over the last ~4 min: the *character* of
                   the variability (irregular/jerky in REM vs smooth in NREM), which is
                   orthogonal to the std features above (magnitude of variability)
  6  activity      ActiGraph activity counts (agcounts; Neishabouri 2022): accel
                   resampled to 30 Hz -> per-second count magnitudes summed in the
                   epoch, squared, EMA-smoothed
  7  act_std_w30   std of activity over the last 30 epochs (still-vs-restless texture)
  8  time_h        time-of-night in hours (seconds since lights-off / 3600)

These cover the independent signal axes that carry REM information from a wrist
wearable: HR level (absolute and relative to a causal baseline), HR variability at
several timescales -- minutes (epoch-mean std) and short-scale (raw-series std),
since REM has more irregular heart rate than deep NREM -- motion level (atonia
separates REM/NREM from Wake), motion texture, and circadian timing (REM propensity
rises across the night). Variability is measured on the RAW HR, not a smoothed
transform, so it is preserved.

Everything is causal: each column at epoch t is a function of samples with time <= t
only, verified by the truncation-invariance tests in tests/test_features.py.

DEPLOYMENT STATE CONTRACT (to reproduce a row live, one epoch at a time, e.g. in Dart;
clear all state at the start of each night):
  hr_mean          accumulate HR readings within the epoch -> mean; hold last if none
  hr_rel           hr_mean minus a slow EMA of hr_mean (1 scalar)
  hr_std_w10/w30   ring buffer of the last 30 hr_mean values -> two-pass std over last 10 / 30
  hr_raw_std       ring buffer of the last ~2 min of raw HR readings -> std
  hr_entropy       ring buffer of the last ~4 min of raw HR readings -> sample entropy
  activity         accel ring buffer -> agcounts per second -> summed, squared, EMA (1 scalar)
  act_std_w30      ring buffer of the last 30 activity values -> two-pass std
  time_h           the clock
Total live state: 2 epoch ring buffers (30 floats) + a raw-HR ring buffer + 2 EMA scalars.

The paper's final "normalize the EMA" is left out: a whole-recording normalization
would use the future, and scaling is the model's job anyway (trees ignore it; a
scale-sensitive model can fit a scaler on the training split).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from agcounts.extract import get_counts

from remdetect.dataset import EPOCH_SEC, Record

ACT_EMA_ALPHA = 0.30      # smoothing for the activity level feature
ACCEL_FS = 30             # Hz; resampling grid for activity counts (the paper's rate)
ROLL_WIN_SHORT = 10       # epochs (~5 min): short-scale HR variability
ROLL_WIN_LONG = 30        # epochs (~15 min): long-scale HR / activity variability
HR_BASELINE_ALPHA = 0.02  # slow EMA -> per-subject HR baseline (~25 min); hr_rel = hr - baseline
HRV_WINDOW_SEC = 120.0    # trailing window for short-scale HR variability on the raw series
HRV_ENT_WINDOW_SEC = 240.0  # longer trailing window for HR sample entropy (needs more samples)

FEATURE_NAMES = ["hr_mean", "hr_rel", "hr_std_w10", "hr_std_w30", "hr_raw_std",
                 "hr_entropy", "activity", "act_std_w30", "time_h"]


def featurize(record: Record) -> np.ndarray:
    """Causal (n_epochs, len(FEATURE_NAMES)) feature matrix for one night. Columns
    follow FEATURE_NAMES; windowed columns reset per night (called per record, so
    windows never cross a subject boundary)."""
    hr_mean = _epoch_hr_mean(record)                                # raw mean HR per epoch
    hr_rel = hr_mean - _ema(hr_mean, HR_BASELINE_ALPHA)             # HR above recent baseline
    hr_raw_std = _windowed_hr_std(record, HRV_WINDOW_SEC)           # short-scale HRV on raw HR
    hr_entropy = _windowed_hr_stat(record, HRV_ENT_WINDOW_SEC,      # HR irregularity (character)
                                   _sample_entropy, min_samples=8)
    activity = _ema(_activity_counts(record) ** 2, ACT_EMA_ALPHA)   # summed, squared, smoothed
    time_h = record.epoch_time / 3600.0
    return np.column_stack([
        hr_mean,
        hr_rel,
        _rolling_std(hr_mean, ROLL_WIN_SHORT),     # HR variability (~5 min) on epoch-mean hr
        _rolling_std(hr_mean, ROLL_WIN_LONG),      # HR variability (~15 min) on epoch-mean hr
        hr_raw_std,
        hr_entropy,
        activity,
        _rolling_std(activity, ROLL_WIN_LONG),     # still-vs-restless texture
        time_h,
    ])


def _rolling_std(col: np.ndarray, w: int) -> np.ndarray:
    """Causal population std over the last w epochs (expanding until full). Two-pass
    (np.std subtracts the window mean first), so it stays exact on large-magnitude
    signals where mean(x^2)-mean(x)^2 would lose all precision. 0 on the first epoch.
    Live: keep the last w values (ring buffer) and apply the same two-pass formula."""
    col = np.asarray(col, dtype=float)
    n = col.size
    out = np.empty(n)
    warm = min(w - 1, n)
    for t in range(warm):                                  # expanding windows, t < w-1
        out[t] = col[: t + 1].std()
    if n >= w:                                             # full windows, vectorized
        windows = np.lib.stride_tricks.sliding_window_view(col, w)
        out[w - 1:] = windows.std(axis=1)
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


def _windowed_hr_stat(record: Record, window_sec: float, fn, min_samples: int = 2) -> np.ndarray:
    """Apply fn to the RAW heart-rate samples in the causal window (t - window_sec, t]
    at each epoch end t. fn takes a time-ordered 1-D array and returns a scalar.

    Causal: uses only samples with time <= t (searchsorted bounds at t and t-window).
    The raw HR series may be out of time order, so sort it first. Windows with fewer
    than min_samples are forward-filled from the past (0 before the first) -- no
    backfill, so it stays causal. Live: keep a ring buffer of the last window_sec of HR.
    """
    order = np.argsort(record.hr_time, kind="stable")     # raw HR may be out of time order
    ht, hr = record.hr_time[order], record.hr[order]
    epochs = record.epoch_time
    hi = np.searchsorted(ht, epochs, side="right")                 # samples with time <= t
    lo = np.searchsorted(ht, epochs - window_sec, side="right")    # samples with time <= t-w
    out = np.full(epochs.size, np.nan)
    for t in range(epochs.size):
        if hi[t] - lo[t] >= min_samples:
            out[t] = fn(hr[lo[t]:hi[t]])
    return pd.Series(out).ffill().fillna(0.0).to_numpy()


def _windowed_hr_std(record: Record, window_sec: float) -> np.ndarray:
    """Std of the raw HR in the causal trailing window -- short-scale HRV."""
    return _windowed_hr_stat(record, window_sec, np.std)


def _sample_entropy(hr_window: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """Sample entropy of a time-ordered HR window: higher = more irregular / less
    predictable (REM heart rate is jerkier than smooth NREM). Counts m- and
    (m+1)-length template matches within tolerance r = r_factor * std; returns 0 for a
    too-short or flat window. Character of variability, not magnitude -- so it adds
    signal orthogonal to the std features."""
    x = np.asarray(hr_window, dtype=float)
    n = x.size
    r = r_factor * x.std()
    if n < m + 2 or r == 0:
        return 0.0

    def _matches(length: int) -> int:
        templates = np.array([x[i:i + length] for i in range(n - length + 1)])
        dist = np.max(np.abs(templates[:, None, :] - templates[None, :, :]), axis=2)
        return int((dist <= r).sum()) - len(templates)   # drop the K self-matches

    b, a = _matches(m), _matches(m + 1)
    return float(-np.log(a / b)) if a > 0 and b > 0 else 0.0


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
