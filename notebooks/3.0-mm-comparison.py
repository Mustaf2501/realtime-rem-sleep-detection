import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Comparison: logistic regression vs XGBoost

        Reads the per-model nested-CV reports (from `2.0-mm-logreg` and
        `2.1-mm-xgboost`). Both are scored by per-subject REM F1 on the same held-out
        subjects, so the difference is tested with a paired Wilcoxon signed-rank test.

        Run the training notebooks first (or `make train-logreg` / `train-xgboost`).
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from itertools import combinations

    from remdetect.modeling.compare import load_report, paired_test

    return combinations, load_report, mo, np, paired_test, pd, plt


@app.cell
def _(load_report):
    models = {name: load_report(name) for name in ("logreg", "xgboost")}
    return (models,)


@app.cell
def _(models, pd):
    summary = pd.DataFrame([
        {
            "model": m["model"],
            "REM F1": m["mean"],
            "precision": m["precision"]["mean"],
            "recall": m["recall"]["mean"],
            "95% CI ± (F1)": m["ci95"],
            "subjects": m["n"],
        }
        for m in models.values()
    ]).round(4)
    summary
    return


@app.cell
def _(combinations, models, mo, paired_test):
    _lines = ["**Paired Wilcoxon signed-rank (per-subject F1):**", ""]
    for _a, _b in combinations(models, 2):
        _t = paired_test(models[_a], models[_b])
        _diff = _t[f"mean_diff_{_a}_minus_{_b}"]
        _sig = "significant" if _t["pvalue"] < 0.05 else "n.s."
        _lines.append(
            f"- **{_a} vs {_b}:** {_diff:+.4f} F1, p = {_t['pvalue']:.4f} "
            f"→ {_t['winner']} ({_sig})"
        )
    mo.md("\n".join(_lines))
    return


@app.cell
def _(models, plt):
    _names = list(models)
    _means = [models[n]["mean"] for n in _names]
    _cis = [models[n]["ci95"] for n in _names]
    _fig, _ax = plt.subplots(figsize=(6, 4))
    _ax.bar(_names, _means, yerr=_cis, capsize=6, color="steelblue", edgecolor="black")
    _ax.set_ylabel("REM F1 (per-subject mean)")
    _ax.set_title("REM F1 with 95% CI")
    for _i, _m in enumerate(_means):
        _ax.text(_i, _m + 0.01, f"{_m:.3f}", ha="center", fontsize=9)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(models, np, plt):
    # Pooled confusion matrices (row-normalized), one per model.
    _names = list(models)
    _fig, _axes = plt.subplots(1, len(_names), figsize=(4 * len(_names), 3.4))
    _axes = np.atleast_1d(_axes)
    for _ax, _n in zip(_axes, _names):
        _cm = np.array(models[_n]["confusion"], dtype=float)
        _norm = _cm / _cm.sum(axis=1, keepdims=True)
        _ax.imshow(_norm, cmap="Blues", vmin=0, vmax=1)
        for _i in range(2):
            for _j in range(2):
                _ax.text(_j, _i, f"{int(_cm[_i, _j])}\n{_norm[_i, _j]:.2f}",
                         ha="center", va="center", fontsize=9)
        _ax.set_xticks([0, 1], models[_n]["confusion_labels"])
        _ax.set_yticks([0, 1], models[_n]["confusion_labels"])
        _ax.set_xlabel("predicted")
        _ax.set_ylabel("true")
        _ax.set_title(_n)
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
