"""The REM-detection model (Weco rewrites this file).

Each 30 s epoch is classified REM / not-REM from the feature matrix built in
features.py. build_model returns a scikit-learn-compatible estimator:

    fit(X, y)            X = per-epoch feature rows, y = 1 for REM
    predict(X)           1 for predicted-REM epochs, one label per row

Training rows are many subjects' nights concatenated. A model that needs per-night
boundaries (a sequence model that resets state between nights) can instead declare
fit(X, y, groups), where groups[i] is the subject index of row i; the harness
passes it when the signature asks for it. predict always gets one subject's night
in chronological order.

Real-time constraint. The detector runs live on a watch, so the prediction for the
epoch ending at t may use only data up to t. predict must score each epoch from
that row and earlier ones. That rules out reading later epochs (bidirectional
nets, attention over the whole night), post-hoc smoothing with future predictions,
whole-sequence normalization, and tuning on the held-out subject. evaluate.py
checks this each fold and scores 0 on a violation; the features are already causal
and the threshold is set at train time.

Deployment. The chosen model runs on a phone (Flutter), one epoch at a time, so a
small model that exports to TFLite or ONNX is preferable. evaluate.py measures only
F1 and causality, not size or latency, so treat this as a guideline.

Anything about the model is open: the estimator, its hyperparameters and tuning,
the REM threshold, class weighting, calibration.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

REM_THRESHOLD = 0.24      # P(REM) >= threshold -> REM  (paper used 0.24)
RF_KWARGS = dict(n_estimators=200, min_samples_leaf=48, n_jobs=-1, random_state=0)


class RemModel:
    """Predicts REM by thresholding its own probabilities. Apply the threshold
    here, in predict — do NOT wrap a model in sklearn's FixedThresholdClassifier:
    that breaks on custom or groups-aware estimators. Keep this shape (fit /
    predict / predict_proba, classes_ set, fit accepts groups) for any model,
    including LSTMs and other custom estimators, and they compose cleanly."""

    def __init__(self, threshold: float = REM_THRESHOLD):
        self.threshold = threshold

    def fit(self, X, y, groups=None):     # groups is optional; ignore it if unused
        self.classes_ = np.array([0, 1])
        self.model_ = RandomForestClassifier(**RF_KWARGS).fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return (self.model_.predict_proba(X)[:, 1] >= self.threshold).astype(int)


def build_model():
    return RemModel()
