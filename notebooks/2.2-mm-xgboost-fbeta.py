import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # XGBoost operating points: F0.5 and F0.3

    Takes the F1-tuned XGBoost (same estimator and search space as `2.1-mm-xgboost`)
    and re-tunes only the decision threshold to maximize F0.5 and F0.3, which weight
    precision over recall. The threshold is chosen on the training subjects with
    `TunedThresholdClassifierCV`, so nothing leaks from the held-out subject. Saves
    `reports/xgboost_f05_nested.json` and `reports/xgboost_f03_nested.json`.

    Runs two nested-CV passes, so it takes a while.
    """)
    return


@app.cell
def _():
    import json

    import marimo as mo
    import numpy as np
    import pandas as pd
    from scipy.stats import loguniform, randint, uniform

    from remdetect import splits
    from remdetect.config import REPORTS_DIR
    from remdetect.modeling.model import AtoniaXGB, monotone_constraints
    from remdetect.modeling.tune import nested_loso_f1, save_report, to_report

    return (
        AtoniaXGB,
        REPORTS_DIR,
        json,
        loguniform,
        mo,
        monotone_constraints,
        nested_loso_f1,
        pd,
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
def _(AtoniaXGB, loguniform, monotone_constraints, randint, uniform):
    estimator = AtoniaXGB(
        monotone_constraints=monotone_constraints(),
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=1,
        random_state=42,
    )
    param_space = {
        "n_estimators": randint(100, 400),
        "max_depth": randint(3, 8),
        "learning_rate": loguniform(1e-2, 3e-1),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_weight": randint(1, 6),
        "reg_lambda": uniform(0.0, 2.0),
    }
    return estimator, param_space


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
    reports = {}
    for _beta, _name in [(0.5, "xgboost_f05"), (0.3, "xgboost_f03")]:
        _res = nested_loso_f1(estimator, param_space, X, y, groups,
                              n_iter=25, inner_splits=5, beta=_beta)
        reports[_name] = to_report(_name, _res)
        save_report(reports[_name])
    return (reports,)


@app.cell
def _(REPORTS_DIR, json, pd, reports):
    with open(REPORTS_DIR / "xgboost_nested.json") as _f:
        _base = json.load(_f)                       # the F1 baseline from 2.1

    rows = [{
        "operating point": "F1 (0.5 cutoff)",
        "Fbeta (own beta)": _base["f1"]["mean"],
        "precision": _base["precision"]["mean"],
        "recall": _base["recall"]["mean"],
        "threshold": 0.5,
    }]
    for _name, _r in reports.items():
        rows.append({
            "operating point": f"F{_r['fbeta']['beta']} (tuned)",
            "Fbeta (own beta)": _r["fbeta"]["mean"],
            "precision": _r["precision"]["mean"],
            "recall": _r["recall"]["mean"],
            "threshold": _r["fbeta"]["threshold_mean"],
        })
    pd.DataFrame(rows).round(3)
    return


if __name__ == "__main__":
    app.run()
