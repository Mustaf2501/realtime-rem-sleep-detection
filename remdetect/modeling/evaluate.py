"""Leave-one-subject-out scoring for the models in model.py.

score(make_model, beta) runs the protocol and returns the stats for any model, so a
notebook can sweep models and beta values through one code path; main() prints the
default XGBoost breakdown. Reports the mean per-subject REM F-beta, computed per
held-out subject and averaged across folds.

Scoring follows the paper's Figure 1. Accuracy, precision, recall, and F-beta are
computed per held-out subject and averaged across folds (mean +/- SEM); a subject
with no scored REM is skipped, since REM precision and recall are undefined there.
The confusion matrix is pooled over all epochs and row-normalized (the paper
row-normalizes the matrix but averages the metrics across folds), so its
[REM, REM] cell is close to, but not exactly, the averaged recall.

Before scoring, each fold is checked for look-ahead (the real-time constraint in
model.py): if the first-k predictions change when later epochs are removed or
altered, the model is reading the future and score raises CausalityError.

Metrics and the figure are written to reports/ and reports/figures/ by plots.py;
this module only scores.
"""
from __future__ import annotations

import inspect

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, fbeta_score,
                             precision_score, recall_score)

from remdetect import plots, splits
from remdetect.modeling.model import build_xgb

STAGES = [0, 1]                      # not-REM, REM (binary)
REM_LABEL = 1                        # the class the metric is about
BETA = 0.3                           # F-beta < 1 weights precision over recall
_CUT_FRACTIONS = (0.25, 0.5, 0.75)   # points in the night where look-ahead is checked
_LABELS = ["not-REM", "REM"]


class CausalityError(RuntimeError):
    """Raised when a model's earlier predictions change as later epochs change --
    it looks ahead and is not real-time. Carries the offending subject group."""

    def __init__(self, group: int):
        super().__init__(
            f"CAUSALITY CHECK FAILED on subject group {group}: predictions for earlier "
            "epochs change when later epochs are removed or altered -- the model looks "
            "ahead and is not real-time.")
        self.group = group


def _fit(model, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
    """Fit the model, handing per-night subject boundaries to models that ask for
    them. A model whose fit signature declares `groups` (e.g. a sequence model
    that resets state between nights) receives groups[i] = the subject index of
    training row i; a plain tabular estimator just gets (X, y)."""
    if "groups" in inspect.signature(model.fit).parameters:
        return model.fit(X, y, groups=groups)
    return model.fit(X, y)


def _predictions_are_causal(model, X_test: np.ndarray, full_pred: np.ndarray) -> bool:
    """True if the model scores each epoch using only that epoch and earlier ones.

    At several cut points k we require the first-k predictions to be identical
    (a) when epochs after k are removed, and (b) when their content is zeroed.
    A model that peeks at the future fails at least one of these.
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


def _mean_sem(values: list[float]) -> tuple[float, float]:
    a = np.asarray(values, dtype=float)
    sem = float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0
    return float(a.mean()), sem


def score(make_model, beta: float = BETA) -> dict:
    """Leave-one-subject-out scoring for a model, returning its stats.

    make_model() returns a fresh estimator per fold (fit/predict, 1 = REM) with its
    threshold tuned for `beta`; the per-subject F-beta uses the same beta. Returns a
    dict: per-subject (mean, sem) for accuracy/precision/recall/fbeta, the pooled
    row-normalized confusion, the pooled per-class report, and fold counts. Raises
    CausalityError if any fold's predictions look ahead.
    """
    # features (X): (n_epochs, n_features) | labels (y): (n_epochs,), 1 == REM
    # subjects (groups): (n_epochs,)   -- from the committed matrix when present
    X, y, groups = splits.load_dataset()

    per_fold = {"accuracy": [], "precision": [], "recall": [], "fbeta": []}
    pooled_true, pooled_pred = [], []   # every epoch, for the pooled confusion (A)
    skipped = 0                         # subjects with no scored REM
    for train_idx, test_idx in splits.cross_validator().split(X, y, groups=groups):
        model = _fit(make_model(), X[train_idx], y[train_idx], groups[train_idx])
        X_test, y_test = X[test_idx], y[test_idx]
        y_pred = model.predict(X_test)

        if not _predictions_are_causal(model, X_test, y_pred):
            raise CausalityError(int(groups[test_idx][0]))

        pooled_true.append(y_test)             # the pooled confusion uses every subject
        pooled_pred.append(y_pred)

        if (y_test == REM_LABEL).sum() == 0:   # no REM -> REM metrics undefined, skip (B)
            skipped += 1
            continue

        per_fold["accuracy"].append(accuracy_score(y_test, y_pred))
        per_fold["precision"].append(precision_score(
            y_test, y_pred, labels=[REM_LABEL], average="macro", zero_division=0))
        per_fold["recall"].append(recall_score(
            y_test, y_pred, labels=[REM_LABEL], average="macro", zero_division=0))
        per_fold["fbeta"].append(fbeta_score(
            y_test, y_pred, beta=beta, labels=[REM_LABEL], average="macro", zero_division=0))

    pooled_true_all = np.concatenate(pooled_true)   # every epoch, every subject
    pooled_pred_all = np.concatenate(pooled_pred)
    return {
        "beta": beta,
        "n_subjects": len(per_fold["fbeta"]),
        "skipped": skipped,
        "stats": {name: _mean_sem(vals) for name, vals in per_fold.items()},
        "confusion": plots.row_normalized_confusion(   # pooled over all epochs (paper's A)
            pooled_true_all, pooled_pred_all, STAGES),
        "per_class": classification_report(            # per-class, pooled
            pooled_true_all, pooled_pred_all, labels=STAGES, target_names=_LABELS,
            output_dict=True, zero_division=0),
    }


def main(beta: float = BETA) -> float:
    result = score(lambda: build_xgb(beta), beta)
    stats, n_subjects, skipped = result["stats"], result["n_subjects"], result["skipped"]

    fbeta_mean = stats["fbeta"][0]
    print(f"REM F-beta (beta={beta}): {fbeta_mean:.4f}  [per-subject mean]")
    print("REM, per-subject mean +/- SEM:")
    for name in ("fbeta", "precision", "recall"):
        mean, sem = stats[name]
        print(f"  {name}: {mean:.4f} +/- {sem:.4f}")
    acc_mean, acc_sem = stats["accuracy"]
    note = f" ({skipped} skipped: no scored REM)" if skipped else ""
    print(f"overall accuracy: {acc_mean:.4f} +/- {acc_sem:.4f} SEM  "
          f"(per-subject mean over {n_subjects} folds{note}; beta={beta})")
    print("per class, pooled over all epochs:")
    for lbl in _LABELS:
        r = result["per_class"][lbl]
        print(f"  {lbl:8s} precision {r['precision']:.3f}  recall {r['recall']:.3f}  "
              f"f1 {r['f1-score']:.3f}  (n={int(r['support'])})")

    plots.save("xgboost", stats, result["confusion"], result["per_class"],
               n_subjects, _LABELS, beta)
    return fbeta_mean


if __name__ == "__main__":
    main()
