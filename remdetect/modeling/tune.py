"""Fair, publication-ready model comparison by nested leave-one-subject-out CV.

The comparison must not leak the test subject into hyperparameter selection, and it
must give each model an equal tuning budget. Nested cross-validation does both:

    Outer loop  : leave-one-subject-out. Estimates generalization to a new subject.
    Inner loop  : GroupKFold over the *training* subjects only, used to pick
                  hyperparameters. The outer test subject is never touched, so
                  tuning cannot leak.

For every outer fold we run a randomized hyperparameter search (the same n_iter for
every model) that scores REM F1 on the inner folds, refit the best config on all
outer-training data, and record the held-out subject's F1. The scaler lives inside
each pipeline, so it is refit within every fold. predict scores each epoch from its
own row, so the real-time constraint holds; we assert it per fold.

nested_loso_f1 returns per-subject F1, precision, and recall (aligned across models by
subject order), a pooled confusion matrix, and the hyperparameters chosen on each
fold, so results are reproducible and comparable across models.
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, fbeta_score, make_scorer,
                             precision_score, recall_score)
from sklearn.model_selection import (GroupKFold, LeaveOneGroupOut,
                                     RandomizedSearchCV, TunedThresholdClassifierCV)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from remdetect.config import REPORTS_DIR
from remdetect.modeling.causality import CausalityError, _predictions_are_causal
from remdetect.modeling.model import AtoniaXGB, monotone_constraints

SEED = 42
N_ITER = 25          # random-search draws per outer fold — the shared tuning budget
INNER_SPLITS = 5     # grouped inner folds for hyperparameter selection
REM_LABEL = 1


def logreg_search() -> tuple[Pipeline, dict]:
    """Standardized, class-weighted logistic regression; tune the regularization C."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=2000)),
    ])
    space = {"clf__C": loguniform(1e-3, 1e2)}
    return pipe, space


def xgb_search() -> tuple[AtoniaXGB, dict]:
    """XGBoost with the fixed motor-atonia monotone prior; tune the tree/regularization
    knobs. n_jobs=1 so the outer RandomizedSearchCV can parallelize across candidates."""
    est = AtoniaXGB(
        monotone_constraints=monotone_constraints(), objective="binary:logistic",
        eval_metric="logloss", tree_method="hist", n_jobs=1, random_state=SEED)
    space = {
        "n_estimators": randint(100, 400),
        "max_depth": randint(3, 8),
        "learning_rate": loguniform(1e-2, 3e-1),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_weight": randint(1, 6),
        "reg_lambda": uniform(0.0, 2.0),
    }
    return est, space


SEARCHES = {"logreg": logreg_search, "xgboost": xgb_search}


def nested_loso_f1(estimator, param_space, X, y, groups, *, n_iter=N_ITER,
                   inner_splits=INNER_SPLITS, seed=SEED, beta=None) -> dict:
    """Nested-LOSO scoring for a given estimator + hyperparameter space.

    The estimator and param_space are passed in (a notebook builds them in view), so
    this function is just the leakage-free protocol: outer LOSO, inner GroupKFold
    hyperparameter search (scored by F1), causality check, and per-subject REM F1 /
    precision / recall plus a confusion matrix pooled over the held-out epochs.
    Subjects with no scored REM are skipped (the REM metrics are undefined); the same
    subjects are skipped for every model, so the returned arrays stay aligned.

    If beta is set, each fold's tuned estimator has its decision threshold chosen to
    maximize F-beta on the training subjects (sklearn's TunedThresholdClassifierCV),
    and the result gains per-subject F-beta and the mean chosen threshold. With
    beta=None the estimator's own 0.5 cutoff is used.
    """
    fbeta_scorer = (make_scorer(fbeta_score, beta=beta, pos_label=REM_LABEL,
                                zero_division=0) if beta is not None else None)
    outer = LeaveOneGroupOut()
    subjects, f1s, precisions, recalls, chosen = [], [], [], [], []
    fbetas, thresholds, pooled_true, pooled_pred = [], [], [], []
    for train_idx, test_idx in outer.split(X, y, groups):
        subject = int(groups[test_idx][0])
        if (y[test_idx] == REM_LABEL).sum() == 0:      # REM metrics undefined -> skip
            continue
        search = RandomizedSearchCV(
            estimator, param_space, n_iter=n_iter, scoring="f1",
            cv=GroupKFold(n_splits=inner_splits), random_state=seed, n_jobs=-1)
        search.fit(X[train_idx], y[train_idx], groups=groups[train_idx])

        best = search.best_estimator_
        if beta is not None:                            # tune the threshold for F-beta
            best = TunedThresholdClassifierCV(best, scoring=fbeta_scorer, cv=inner_splits)
            best.fit(X[train_idx], y[train_idx])
            thresholds.append(float(best.best_threshold_))

        y_true, y_pred = y[test_idx], best.predict(X[test_idx])
        if not _predictions_are_causal(best, X[test_idx], y_pred):
            raise CausalityError(subject)

        subjects.append(subject)
        f1s.append(f1_score(y_true, y_pred, pos_label=REM_LABEL, zero_division=0))
        precisions.append(precision_score(y_true, y_pred, pos_label=REM_LABEL, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, pos_label=REM_LABEL, zero_division=0))
        if beta is not None:
            fbetas.append(fbeta_score(y_true, y_pred, beta=beta, pos_label=REM_LABEL, zero_division=0))
        pooled_true.append(y_true)
        pooled_pred.append(y_pred)
        chosen.append({k: _plain(v) for k, v in search.best_params_.items()})

    confusion = confusion_matrix(np.concatenate(pooled_true), np.concatenate(pooled_pred),
                                 labels=[0, REM_LABEL])   # [[TN, FP], [FN, TP]]
    res = {"subjects": subjects, "f1": np.array(f1s), "precision": np.array(precisions),
           "recall": np.array(recalls), "confusion": confusion, "chosen_params": chosen,
           "config": {"n_iter": n_iter, "inner_splits": inner_splits, "seed": seed, "beta": beta}}
    if beta is not None:
        res["fbeta"] = np.array(fbetas)
        res["beta"] = beta
        res["threshold_mean"] = float(np.mean(thresholds))
    return res


def _metric_block(values: np.ndarray) -> dict:
    """A per-subject metric summarized: mean, SEM, normal-approx 95% CI half-width,
    and the per-subject values themselves."""
    v = np.asarray(values, dtype=float)
    sem = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0
    return {"mean": float(v.mean()), "sem": sem, "ci95": 1.96 * sem,
            "per_subject": v.tolist()}


def to_report(name: str, res: dict) -> dict:
    """Package a nested_loso_f1 result into a JSON-ready per-model report.

    Each metric (f1, precision, recall) is its own self-describing block with mean,
    SEM, 95% CI, and per-subject values; confusion is the pooled 2x2 count matrix with
    its labels. All metrics share the same n_subjects.
    """
    report = {
        "model": name,
        "metric": "REM (per-subject, nested LOSO)",
        "config": res["config"],
        "n_subjects": int(res["f1"].size),
        "subjects": res["subjects"],
        "f1": _metric_block(res["f1"]),
        "precision": _metric_block(res["precision"]),
        "recall": _metric_block(res["recall"]),
        "confusion": {"matrix": res["confusion"].tolist(), "labels": ["not-REM", "REM"]},
        "chosen_params": res["chosen_params"],
    }
    if "fbeta" in res:      # threshold tuned for this F-beta
        report["fbeta"] = {**_metric_block(res["fbeta"]), "beta": res["beta"],
                           "threshold_mean": res["threshold_mean"]}
    return report


def save_report(report: dict) -> str:
    """Write one model's report to reports/<model>_nested.json."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, f"{report['model']}_nested.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path


def run_model(name: str, X, y, groups, *, n_iter=N_ITER,
              inner_splits=INNER_SPLITS, seed=SEED) -> dict:
    """Headless convenience: nested-CV a named model (from SEARCHES) and package its
    report. The training notebooks build the estimator + space inline instead, then
    call nested_loso_f1 + to_report directly, so the model is visible where you work."""
    estimator, param_space = SEARCHES[name]()
    res = nested_loso_f1(estimator, param_space, X, y, groups,
                         n_iter=n_iter, inner_splits=inner_splits, seed=seed)
    return to_report(name, res)


def _plain(v):
    """JSON-friendly scalar (numpy -> Python)."""
    return v.item() if isinstance(v, np.generic) else v


if __name__ == "__main__":     # headless mirror of the training notebooks (for CI)
    import sys

    from remdetect import splits

    model_name = sys.argv[1] if len(sys.argv) > 1 else "xgboost"
    X_all, y_all, groups_all = splits.load_dataset()
    report = run_model(model_name, X_all, y_all, groups_all)
    print(f"{model_name}: REM F1 {report['f1']['mean']:.4f} +/- "
          f"{report['f1']['ci95']:.4f} (95% CI, n={report['n_subjects']})")
    print("saved", save_report(report))
