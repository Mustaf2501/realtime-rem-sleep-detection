"""The REM-detection models.

Each 30 s epoch is classified REM / not-REM from the feature matrix built in
features.py. A model is a scikit-learn-style estimator:

    fit(X, y)            X = per-epoch feature rows, y = 1 for REM
    predict(X)           1 for predicted-REM epochs, one label per row

Training rows are many subjects' nights concatenated. fit accepts an optional
`groups` (subject index per row) for models that reset state between nights; the
harness passes it when the signature asks. predict always gets one subject's night
in chronological order.

Real-time constraint. The detector runs live on a watch, so the prediction for the
epoch ending at t may use only data up to t. That rules out reading later epochs,
post-hoc smoothing, whole-sequence normalization, and tuning on the held-out
subject. evaluate.py checks this each fold and scores 0 on a violation; the features
are already causal and the threshold is set at train time.

build_xgb is the deployment model: XGBoost with a motor-atonia monotone prior (the
activity features can only *lower* P(REM), since REM sleep has near-still motion --
the strongest REM-precision lever), wrapped in ThresholdedModel, which tunes its
decision threshold for a given F-beta. beta is where the precision/recall trade-off
lives: beta<1 favors precision, beta>1 favors recall. (The model comparison uses the
sklearn-native estimators in tune.py instead.)
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import fbeta_score
from xgboost import XGBClassifier

from remdetect.features import FEATURE_NAMES

BETA = 0.3                                      # default F-beta the threshold is tuned for
ATONIA_FEATURES = ("activity", "act_std_w30")   # motor atonia: motion can only lower P(REM)
XGB_KWARGS = dict(                              # tuned by leave-one-subject-out search
    n_estimators=299, max_depth=6, learning_rate=0.147, min_child_weight=3,
    subsample=0.8888, colsample_bytree=0.6232, reg_lambda=0.5921,
    objective="binary:logistic", eval_metric="logloss", tree_method="hist",
    n_jobs=-1, random_state=42)


def monotone_constraints() -> str:
    """XGBoost monotone_constraints string: -1 on the activity features (atonia),
    0 elsewhere. Built from FEATURE_NAMES so it tracks the feature order."""
    return "(" + ",".join("-1" if n in ATONIA_FEATURES else "0" for n in FEATURE_NAMES) + ")"


class AtoniaXGB(XGBClassifier):
    """XGBoost that sets scale_pos_weight from the training class balance at fit, so
    the REM/not-REM imbalance is handled without knowing the counts up front."""

    def fit(self, X, y, **kw):
        pos, neg = int((y == 1).sum()), int((y == 0).sum())
        self.set_params(scale_pos_weight=neg / max(pos, 1))
        return super().fit(X, y, **kw)


class ThresholdedModel:
    """Wrap a probabilistic estimator and pick its decision threshold at fit time.

    fit trains the estimator, then chooses the threshold that maximizes F-beta on
    the training predictions; predict applies it (1 = REM). Keeping the threshold
    here -- not in sklearn's FixedThresholdClassifier, which breaks on custom /
    groups-aware estimators -- lets every model stay causal and comparable. fit
    accepts an optional `groups` (ignored) to match the harness contract.
    """

    def __init__(self, estimator, beta: float = BETA):
        self.estimator = estimator
        self.beta = beta

    def fit(self, X, y, groups=None):
        self.classes_ = np.array([0, 1])
        self.estimator.fit(X, y)
        p = self.estimator.predict_proba(X)[:, 1]                  # tune threshold on train
        grid = np.linspace(0.05, 0.95, 181)
        self.threshold = float(max(grid, key=lambda t: fbeta_score(
            y, (p >= t).astype(int), beta=self.beta, pos_label=1, zero_division=0)))
        return self

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def predict(self, X):
        return (self.estimator.predict_proba(X)[:, 1] >= self.threshold).astype(int)


def build_xgb(beta: float = BETA) -> ThresholdedModel:
    """XGBoost with the motor-atonia monotone prior, threshold tuned for beta."""
    est = AtoniaXGB(monotone_constraints=monotone_constraints(), **XGB_KWARGS)
    return ThresholdedModel(est, beta)


build_model = build_xgb   # the default deployment model (used by train.py / predict.py)
