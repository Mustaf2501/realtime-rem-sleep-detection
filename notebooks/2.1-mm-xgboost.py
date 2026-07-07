import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # XGBoost

    Nested leave-one-subject-out CV: outer LOSO for the held-out score, inner
    GroupKFold(5) to tune the tree and regularization params (randomized search,
    25 draws). The estimator and search space are defined below; `nested_loso_f1`
    runs the protocol and writes `reports/xgboost_nested.json`.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.stats import loguniform, randint, uniform
    from sklearn.base import clone

    from remdetect import splits
    from remdetect.features import FEATURE_NAMES
    from remdetect.modeling.compare import save_report
    from remdetect.modeling.model import AtoniaXGB, monotone_constraints
    from remdetect.modeling.tune import nested_loso_f1, to_report

    return (
        AtoniaXGB,
        FEATURE_NAMES,
        clone,
        loguniform,
        mo,
        monotone_constraints,
        nested_loso_f1,
        np,
        pd,
        plt,
        randint,
        save_report,
        splits,
        to_report,
        uniform,
    )


@app.cell
def _(splits):
    X, y, groups = splits.load_dataset()
    return X, groups, y


@app.cell
def _(AtoniaXGB, monotone_constraints):
    # Motor-atonia monotone prior: the activity features can only lower P(REM).
    # scale_pos_weight is set from the class balance at fit.
    estimator = AtoniaXGB(
        monotone_constraints=monotone_constraints(),
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=1,
        random_state=42,
    )
    estimator
    return (estimator,)


@app.cell
def _(loguniform, randint, uniform):
    param_space = {
        "n_estimators": randint(100, 400),
        "max_depth": randint(3, 8),
        "learning_rate": loguniform(1e-2, 3e-1),
        "subsample": uniform(0.6, 0.4),          # U(0.6, 1.0)
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_weight": randint(1, 6),
        "reg_lambda": uniform(0.0, 2.0),
    }
    param_space
    return (param_space,)


@app.cell
def _(
    X,
    estimator,
    groups,
    nested_loso_f1,
    param_space,
    save_report,
    to_report,
    y,
):
    res = nested_loso_f1(estimator, param_space, X, y, groups, n_iter=25, inner_splits=5)
    report = to_report("xgboost", res)
    save_report(report)
    return (report,)


@app.cell
def _(mo, report):
    mo.md(
        "| metric | mean ± SEM |\n"
        "|---|---|\n"
        f"| REM F1 | {report['f1']['mean']:.3f} ± {report['f1']['sem']:.3f} |\n"
        f"| precision | {report['precision']['mean']:.3f} ± {report['precision']['sem']:.3f} |\n"
        f"| recall | {report['recall']['mean']:.3f} ± {report['recall']['sem']:.3f} |\n\n"
        f"Per-subject means over n = {report['n_subjects']} subjects. "
        "Saved to `reports/xgboost_nested.json`."
    )
    return


@app.cell
def _(np, plt, report):
    _f1 = np.sort(np.array(report["f1"]["per_subject"]))
    _mean = report["f1"]["mean"]
    _fig, _ax = plt.subplots(figsize=(9, 3))
    _ax.bar(range(len(_f1)), _f1, color="0.6", edgecolor="black")
    _ax.axhline(_mean, color="crimson", lw=1, label=f"mean {_mean:.3f}")
    _ax.set_xlabel("subject (sorted by F1)")
    _ax.set_ylabel("held-out REM F1")
    _ax.set_title("Per-subject generalization")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Feature importances and hyperparameter stability
    """)
    return


@app.cell
def _(FEATURE_NAMES, X, clone, estimator, np, plt, y):
    _fitted = clone(estimator).fit(X, y)
    _imp = _fitted.feature_importances_
    _order = np.argsort(_imp)
    _fig, _ax = plt.subplots(figsize=(7, 3))
    _ax.barh([FEATURE_NAMES[i] for i in _order], _imp[_order], color="steelblue")
    _ax.set_xlabel("XGBoost feature importance")
    _ax.set_title("Feature importances")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(pd, report):
    # Spread of the tuned config across the 30 outer folds.
    chosen = pd.DataFrame(report["chosen_params"])
    chosen.describe().loc[["mean", "std", "min", "max"]].round(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Export for deployment

    Fit the XGBoost deployment model on every subject (no held-out split) and serialize it
    for the Flutter/Dart app: the booster as native JSON plus a metadata sidecar holding the
    feature order and the decision threshold.
    """)
    return


@app.cell
def _(X, mo, y):
    from remdetect.modeling.model import build_model
    from remdetect.modeling.train import save_model

    _deploy = build_model().fit(X, y)          # fit on all subjects, no held-out split
    _booster, _meta = save_model(_deploy)
    mo.md(f"""
    **Deployment model saved** (load these in Flutter/Dart):

    - `{_booster}` — XGBoost booster, native JSON
    - `{_meta}` — feature order and the decision threshold

    Rule: predict REM when P(REM) >= {_deploy.threshold:.3f}.
    """)
    return


if __name__ == "__main__":
    app.run()
