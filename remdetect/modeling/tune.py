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

nested_loso_f1 returns the per-subject F1 array (aligned across models by subject
order) plus the hyperparameters chosen on each fold, so results are reproducible and
comparable with a paired test.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
                   inner_splits=INNER_SPLITS, seed=SEED) -> dict:
    """Nested-LOSO REM F1 for a given estimator + hyperparameter space.

    The estimator and param_space are passed in (a notebook builds them in view), so
    this function is just the leakage-free protocol: outer LOSO, inner GroupKFold
    hyperparameter search, causality check, per-subject F1. Subjects with no scored
    REM are skipped (F1 undefined); the same subjects are skipped for every model, so
    the returned arrays stay aligned by subject.
    """
    outer = LeaveOneGroupOut()
    subjects, f1s, chosen = [], [], []
    for train_idx, test_idx in outer.split(X, y, groups):
        subject = int(groups[test_idx][0])
        if (y[test_idx] == REM_LABEL).sum() == 0:      # REM F1 undefined -> skip
            continue
        search = RandomizedSearchCV(
            estimator, param_space, n_iter=n_iter, scoring="f1",
            cv=GroupKFold(n_splits=inner_splits), random_state=seed, n_jobs=-1)
        search.fit(X[train_idx], y[train_idx], groups=groups[train_idx])

        best = search.best_estimator_
        y_pred = best.predict(X[test_idx])
        if not _predictions_are_causal(best, X[test_idx], y_pred):
            raise CausalityError(subject)

        subjects.append(subject)
        f1s.append(f1_score(y[test_idx], y_pred, pos_label=REM_LABEL, zero_division=0))
        chosen.append({k: _plain(v) for k, v in search.best_params_.items()})

    return {"subjects": subjects, "f1": np.array(f1s), "chosen_params": chosen,
            "config": {"n_iter": n_iter, "inner_splits": inner_splits, "seed": seed}}


def summarize(f1: np.ndarray) -> dict:
    """Per-subject F1 -> mean, SEM, and a normal-approx 95% CI half-width."""
    mean = float(f1.mean())
    sem = float(f1.std(ddof=1) / np.sqrt(f1.size)) if f1.size > 1 else 0.0
    return {"mean": mean, "sem": sem, "ci95": 1.96 * sem, "n": int(f1.size)}


def to_report(name: str, res: dict) -> dict:
    """Package a nested_loso_f1 result into a JSON-ready per-model report."""
    return {
        "model": name,
        "metric": "REM F1 (per-subject, LOSO)",
        "config": res["config"],
        "subjects": res["subjects"],
        "per_subject_f1": res["f1"].tolist(),
        "chosen_params": res["chosen_params"],
        **summarize(res["f1"]),
    }


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
    from remdetect.modeling.compare import save_report

    model_name = sys.argv[1] if len(sys.argv) > 1 else "xgboost"
    X_all, y_all, groups_all = splits.load_dataset()
    report = run_model(model_name, X_all, y_all, groups_all)
    print(f"{model_name}: REM F1 {report['mean']:.4f} +/- {report['ci95']:.4f} "
          f"(95% CI, n={report['n']})")
    print("saved", save_report(report))
