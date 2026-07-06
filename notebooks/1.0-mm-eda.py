import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # EDA

    The feature matrix (`data/processed/featurematrix.npz`): shapes, REM prevalence,
    per-feature distributions, and per-subject structure. New feature ideas can be
    prototyped here before moving into `remdetect/features.py`.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    from remdetect import splits
    from remdetect.dataset import load_records, WAKE, N1, N2, N3, REM
    from remdetect.features import FEATURE_NAMES, featurize

    return (
        FEATURE_NAMES,
        N1,
        N2,
        N3,
        REM,
        WAKE,
        featurize,
        load_records,
        mo,
        np,
        plt,
        splits,
    )


@app.cell(hide_code=True)
def participant_header(mo):
    mo.md(r"""
    ## Per-participant view

    One subject's night: the PSG hypnogram and each feature traced across it, with
    REM epochs shaded.
    """)
    return


@app.cell
def participant_select(load_records, mo):
    records = load_records()
    subject = mo.ui.dropdown(
        options=[r.subject_id for r in records],
        value=records[0].subject_id,
        label="subject",
    )
    subject
    return records, subject


@app.cell
def participant_record(REM, mo, records, subject):
    record = next(r for r in records if r.subject_id == subject.value)
    mo.md(
        f"Subject {record.subject_id}: {record.epoch_time.size} epochs, "
        f"{int((record.stage == REM).sum())} REM "
        f"({100 * (record.stage == REM).mean():.1f}%)."
    )
    return (record,)


@app.cell
def hypnogram(N1, N2, N3, REM, WAKE, np, plt, record):
    # Wake on top, deep sleep at the bottom, REM just below Wake.
    _order = {WAKE: 4, REM: 3, N1: 2, N2: 1, N3: 0}
    _ypos = np.array([_order.get(int(s), np.nan) for s in record.stage])
    _t = record.epoch_time / 3600.0

    _fig, _ax = plt.subplots(figsize=(11, 2.8))
    _ax.step(_t, _ypos, where="post", color="0.4", lw=1.0)
    _rem = record.stage == REM
    _ax.scatter(_t[_rem], _ypos[_rem], s=10, color="crimson", zorder=3, label="REM")
    _ax.set_yticks([0, 1, 2, 3, 4])
    _ax.set_yticklabels(["N3", "N2", "N1", "REM", "Wake"])
    _ax.set_xlabel("time (h)")
    _ax.set_title(f"Hypnogram, subject {record.subject_id}")
    _ax.legend(loc="upper right", fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell
def participant_features(FEATURE_NAMES, REM, featurize, plt, record):
    _F = featurize(record)
    _t = record.epoch_time / 3600.0
    _rem = record.stage == REM

    _fig, _axes = plt.subplots(len(FEATURE_NAMES), 1, figsize=(11, 9), sharex=True)
    for _i, _ax in enumerate(_axes):
        _ax.plot(_t, _F[:, _i], color="steelblue", lw=0.9)
        _ax.fill_between(_t, 0, 1, where=_rem, transform=_ax.get_xaxis_transform(),
                         color="crimson", alpha=0.12, step="post")
        _ax.set_ylabel(FEATURE_NAMES[_i], fontsize=8, rotation=0, ha="right", va="center")
    _axes[-1].set_xlabel("time (h)")
    _axes[0].set_title(f"Features, subject {record.subject_id} (REM shaded)")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def general_header(mo):
    mo.md(r"""
    ## General analysis (all participants)

    The pooled feature matrix across all 31 subjects: class balance, per-feature
    distributions, and how REM is spread across nights.
    """)
    return


@app.cell
def _(splits):
    X, y, groups = splits.load_dataset()
    return X, groups, y


@app.cell
def _(X, groups, mo, np, y):
    mo.md(f"""
    - **{X.shape[0]:,}** epochs × **{X.shape[1]}** features
    - **{len(np.unique(groups))}** subjects
    - REM prevalence: **{100 * y.mean():.1f}%**
    """)
    return


@app.cell
def _(FEATURE_NAMES, X, np, plt, y):
    # Per-feature distributions, REM vs not-REM. Grid sizes to the feature count.
    _n = len(FEATURE_NAMES)
    _ncols = 4
    _nrows = int(np.ceil(_n / _ncols))
    _fig, _axes = plt.subplots(_nrows, _ncols, figsize=(3 * _ncols, 2.4 * _nrows))
    _flat = _axes.ravel()
    for _i, _ax in enumerate(_flat):
        if _i >= _n:
            _ax.axis("off")                          # blank any extra cells
            continue
        _ax.hist(X[y == 0, _i], bins=40, alpha=0.6, density=True, label="not-REM")
        _ax.hist(X[y == 1, _i], bins=40, alpha=0.6, density=True, label="REM")
        _ax.set_title(FEATURE_NAMES[_i], fontsize=9)
    _flat[0].legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(groups, np, plt, y):
    # REM epochs per subject.
    _subjects = np.unique(groups)
    _rem = [int(y[groups == g].sum()) for g in _subjects]
    _fig, _ax = plt.subplots(figsize=(11, 3))
    _ax.bar(range(len(_subjects)), _rem, color="0.6", edgecolor="black")
    _ax.set_xlabel("subject index")
    _ax.set_ylabel("REM epochs")
    _ax.set_title("REM epochs per subject")
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
