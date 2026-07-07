import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Logistic regression

    Nested leave-one-subject-out CV: outer LOSO for the held-out score, inner
    GroupKFold(5) to tune `C` (randomized search, 25 draws). The estimator and
    search space are defined below; `nested_loso_f1` runs the protocol and writes
    `reports/logreg_nested.json`.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import loguniform
    from sklearn.base import clone
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from remdetect import splits
    from remdetect.features import FEATURE_NAMES
    from remdetect.modeling.tune import nested_loso_f1, save_report, to_report

    return (
        FEATURE_NAMES,
        LogisticRegression,
        Pipeline,
        StandardScaler,
        clone,
        loguniform,
        mo,
        nested_loso_f1,
        np,
        plt,
        save_report,
        splits,
        to_report,
    )


@app.cell
def _(splits):
    X, y, groups = splits.load_dataset()
    return X, groups, y


@app.cell
def _(LogisticRegression, Pipeline, StandardScaler):
    # Standardized, class-weighted logistic regression. Scaler in the pipeline refits
    # per CV split.
    estimator = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=2000)),
    ])
    estimator
    return (estimator,)


@app.cell
def _(loguniform):
    param_space = {"clf__C": loguniform(1e-3, 1e2)}
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
    report = to_report("logreg", res)
    save_report(report)
    return (report,)


@app.cell
def _(mo):
    mo.md("""
    | metric | mean ± SEM |
    "
        "|---|---|
    "
        f"| REM F1 | {report['f1']['mean']:.3f} ± {report['f1']['sem']:.3f} |
    "
        f"| precision | {report['precision']['mean']:.3f} ± {report['precision']['sem']:.3f} |
    "
        f"| recall | {report['recall']['mean']:.3f} ± {report['recall']['sem']:.3f} |

    "
        f"Per-subject means over n = {report['n_subjects']} subjects. "
        "Saved to `reports/logreg_nested.json`.
    """)
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
    ### Coefficients (fit on all data)
    """)
    return


@app.cell
def _(FEATURE_NAMES, X, clone, estimator, np, plt, y):
    _fitted = clone(estimator).fit(X, y)
    _coef = _fitted.named_steps["clf"].coef_[0]
    _order = np.argsort(_coef)
    _fig, _ax = plt.subplots(figsize=(7, 3))
    _ax.barh([FEATURE_NAMES[i] for i in _order], _coef[_order], color="steelblue")
    _ax.axvline(0, color="0.5", lw=0.8)
    _ax.set_xlabel("standardized logistic coefficient")
    _ax.set_title("Logistic coefficients")
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
