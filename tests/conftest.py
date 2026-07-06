"""Shared test helpers.

make_record builds a controlled synthetic subject-night so several test modules can
assert on features with an obvious expected output. Kept here (rather than in one
test file) so no test module imports another.
"""
import numpy as np

from remdetect.dataset import EPOCH_SEC, Record


def make_record(n_epochs=20, hr_bpm=60.0, burst_epoch=None) -> Record:
    """A flat, still night (constant HR, no motion). If burst_epoch is given, a
    1 Hz movement is injected into that epoch's 30 s block [30(e-1), 30e)."""
    epoch_time = np.arange(n_epochs) * EPOCH_SEC
    stage = np.full(n_epochs, 2)                       # all N2 (scored)

    hr_time = np.arange(0.0, n_epochs * EPOCH_SEC, 5.0)
    hr = np.full(hr_time.shape, hr_bpm)

    motion_time = np.arange(0.0, n_epochs * EPOCH_SEC, 1.0 / 30)
    motion = np.zeros((motion_time.size, 3))
    motion[:, 2] = 1.0                                 # gravity on z, no movement
    if burst_epoch is not None:
        lo, hi = EPOCH_SEC * (burst_epoch - 1), EPOCH_SEC * burst_epoch
        in_block = (motion_time >= lo) & (motion_time < hi)
        motion[in_block, 0] += 0.3 * np.sin(2 * np.pi * 1.0 * motion_time[in_block])

    return Record("synthetic", epoch_time, stage, hr_time, hr, motion_time, motion)
