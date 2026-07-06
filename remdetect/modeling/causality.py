"""Real-time causality guard.

The detector runs live on a watch, so the prediction for the epoch ending at t may
use only data up to t. This checks a fitted model's predict() for look-ahead, and is
called on every fold by the evaluation in tune.py.
"""
from __future__ import annotations

import numpy as np

_CUT_FRACTIONS = (0.25, 0.5, 0.75)   # points in the night where look-ahead is checked


class CausalityError(RuntimeError):
    """Raised when a model's earlier predictions change as later epochs change: it
    looks ahead and is not real-time. Carries the offending subject group."""

    def __init__(self, group: int):
        super().__init__(
            f"CAUSALITY CHECK FAILED on subject group {group}: predictions for earlier "
            "epochs change when later epochs are removed or altered, so the model looks "
            "ahead and is not real-time.")
        self.group = group


def _predictions_are_causal(model, X_test: np.ndarray, full_pred: np.ndarray) -> bool:
    """True if the model scores each epoch using only that epoch and earlier ones.

    At several cut points k, the first-k predictions must be identical (a) when epochs
    after k are removed, and (b) when their content is zeroed. A model that peeks at
    the future fails at least one of these.
    """
    n = len(X_test)
    for frac in _CUT_FRACTIONS:
        k = max(1, int(n * frac))
        if not np.array_equal(full_pred[:k], model.predict(X_test[:k])[:k]):
            return False
        altered_future = X_test.copy()
        altered_future[k:] = 0.0
        if not np.array_equal(full_pred[:k], model.predict(altered_future)[:k]):
            return False
    return True
