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

This model is a gradient-boosted tree (XGBoost) with a motor-atonia prior baked in
as monotone constraints: the activity features (activity, act_std_w30) can only
*lower* P(REM), since REM sleep has near-still motion -- a near-deterministic
exclusion that is the strongest REM-precision lever for wrist HR + motion + time.
The decision threshold is tuned at fit time for the precision-weighted F-beta(0.3)
that evaluate.py reports.

Deployment. The chosen model runs on a phone (Flutter), one epoch at a time, so a
small model that exports to TFLite or ONNX is preferable. XGBoost trees export
cleanly, and the monotone constraint is baked into the trees (nothing to carry at
inference). evaluate.py measures only REM F-beta and causality, not size or latency.

Anything about the model is open: the estimator, its hyperparameters and tuning,
the REM threshold, class weighting, calibration.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import fbeta_score
from xgboost import XGBClassifier

from features import FEATURE_NAMES

BETA = 0.3                                      # F-beta the decision threshold is tuned for
ATONIA_FEATURES = ("activity", "act_std_w30")   # motor atonia: motion can only lower P(REM)
XGB_KWARGS = dict(                              # tuned by leave-one-subject-out search
    n_estimators=299, max_depth=6, learning_rate=0.147, min_child_weight=3,
    subsample=0.8888, colsample_bytree=0.6232, reg_lambda=0.5921,
    objective="binary:logistic", eval_metric="logloss", tree_method="hist",
    n_jobs=-1, random_state=42)


def _monotone_constraints() -> str:
    """XGBoost monotone_constraints string: -1 on the activity features (atonia),
    0 elsewhere. Built from FEATURE_NAMES so it tracks the feature order."""
    return "(" + ",".join("-1" if n in ATONIA_FEATURES else "0" for n in FEATURE_NAMES) + ")"


class RemModel:
    """XGBoost REM detector with a motor-atonia monotone prior. Predicts REM by
    thresholding its own probability; the threshold is chosen at fit time for
    F-beta(0.3). Apply the threshold here, in predict -- do NOT wrap in sklearn's
    FixedThresholdClassifier (it breaks on custom / groups-aware estimators). Keep
    this shape (fit / predict / predict_proba, classes_ set, fit accepts groups) for
    any estimator and they compose cleanly."""

    def __init__(self, beta: float = BETA):
        self.beta = beta

    def fit(self, X, y, groups=None):     # groups is optional; ignore it if unused
        self.classes_ = np.array([0, 1])
        pos, neg = int((y == 1).sum()), int((y == 0).sum())
        self.model_ = XGBClassifier(
            scale_pos_weight=neg / max(pos, 1),
            monotone_constraints=_monotone_constraints(), **XGB_KWARGS).fit(X, y)
        p = self.model_.predict_proba(X)[:, 1]                     # tune threshold on train
        grid = np.linspace(0.05, 0.95, 181)
        self.threshold = float(max(grid, key=lambda t: fbeta_score(
            y, (p >= t).astype(int), beta=self.beta, pos_label=1, zero_division=0)))
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return (self.model_.predict_proba(X)[:, 1] >= self.threshold).astype(int)


def build_model():
    return RemModel()
