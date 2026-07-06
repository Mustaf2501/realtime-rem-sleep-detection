import marimo

__generated_with = "0.23.13"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Real-time REM detection on the Walch (2019) data

    A walk-through of the pipeline on the dataset, using the `remdetect` package
    (`dataset`, `features`, `splits`, `modeling.model`, `modeling.evaluate`):

    1. the raw wearable signals and PSG labels,
    2. the causal features,
    3. the real-time (look-ahead) check,
    4. leave-one-subject-out evaluation,
    5. the baseline REM detector,
    6. a tuned XGBoost.

    Feature extraction lives in `features.py`; `modeling/model.py` holds the model.
    """)
    return


@app.cell
def _():
    # '%matplotlib inline' command supported automatically in marimo
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report, f1_score, ConfusionMatrixDisplay
    from remdetect import features, splits
    from remdetect.dataset import load_records, REM
    from remdetect.modeling.model import build_model
    records = load_records()
    n_scored = sum((int(r.scored_mask.sum()) for r in records))
    n_rem = sum((int((r.stage == REM).sum()) for r in records))
    print(f'{len(records)} subjects | {n_scored} scored 30s epochs | REM prevalence {100 * n_rem / n_scored:.1f}%')

    def rem_spans(r):
        """Contiguous REM intervals (in hours) for shading plots."""
        flag = (r.stage == REM).astype(int)
        d = np.diff(flag, prepend=0, append=0)
        starts, ends = (np.where(d == 1)[0], np.where(d == -1)[0])
        h = r.epoch_time / 3600
        return [(h[_s], h[min(_e, len(h) - 1)]) for _s, _e in zip(starts, ends)]

    return (
        ConfusionMatrixDisplay,
        build_model,
        classification_report,
        features,
        np,
        plt,
        records,
        rem_spans,
        splits,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The data

    Each subject is one night: heart rate (~0.2 Hz) and triaxial accelerometer
    (~30 Hz) from an Apple Watch, with a polysomnography hypnogram at 30 s resolution.
    One subject's raw streams, REM periods shaded.
    """)
    return


@app.cell
def _(np, plt, records, rem_spans):
    r = records[0]
    mag = np.linalg.norm(r.motion, axis=1)
    _fig, _ax = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
    _ax[0].plot(r.hr_time / 3600, r.hr, lw=0.5)
    _ax[0].set_ylabel('HR (bpm)')
    _ax[0].set_xlim(0, r.epoch_time.max() / 3600)
    _ax[1].plot(r.motion_time[::50] / 3600, mag[::50], lw=0.3)
    _ax[1].set_ylabel('|accel| (g)')
    _ax[2].step(r.epoch_time / 3600, r.stage, where='post', lw=0.9)
    _ax[2].set_yticks([0, 1, 2, 3, 4])
    _ax[2].set_yticklabels(['W', 'N1', 'N2', 'N3', 'REM'])
    _ax[2].set_ylabel('stage')
    _ax[2].set_xlabel('hours')
    for a in _ax:
        for _s, _e in rem_spans(r):
            a.axvspan(_s, _e, color='tab:red', alpha=0.12)
    _fig.suptitle(f'Subject {r.subject_id}: raw wearable streams + PSG hypnogram (REM shaded)')
    plt.tight_layout()
    plt.show()
    return (r,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Features

    `features.featurize` turns the raw streams into one row per epoch. The first three
    columns are the paper's base features -- smoothed heart rate, ActiGraph activity
    counts (via `agcounts`), and time-of-night -- and the rest are causal *temporal*
    features over HR and activity: the rate of change, and rolling means / standard
    deviations over ~5 and ~15 min windows (REM shows elevated, variable HR with
    near-still motion). Every column uses only samples at or before the end of its
    epoch. The column order and each feature's on-device state are documented in
    `features.FEATURE_NAMES` and the deployment-state contract at the top of
    `features.py`.
    """)
    return


@app.cell
def _(features, plt, r, rem_spans):
    X_one = features.featurize(r)
    names = features.FEATURE_NAMES
    _fig, _ax = plt.subplots(len(names), 1, figsize=(9, 1.1 * len(names)), sharex=True)
    for k, name in enumerate(names):
        _ax[k].plot(r.epoch_time / 3600, X_one[:, k], lw=0.8)
        _ax[k].set_ylabel(name, fontsize=7, rotation=0, ha='right', va='center')
        for _s, _e in rem_spans(r):
            _ax[k].axvspan(_s, _e, color='tab:red', alpha=0.12)
    _ax[-1].set_xlabel('hours')
    _fig.suptitle('Causal features fed to the model (REM shaded)')
    plt.tight_layout()
    plt.show()
    print(f'feature matrix shape (epochs x features): {X_one.shape}')
    print('columns:', names)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Real-time (look-ahead) check

    A live watch scores each epoch using only data up to that moment. `evaluate.py`
    checks this every fold with `_predictions_are_causal`: at several cut points the
    first-k predictions must be unchanged when later epochs are removed and when their
    content is altered. Below, a causal model passes and a model that reads one epoch
    ahead does not.
    """)
    return


@app.cell
def _(build_model, np, records, splits):
    from remdetect.modeling.causality import _predictions_are_causal

    X, y, groups = splits.make_dataset(records)     # fixed causal features for every epoch
    train_idx, test_idx = next(splits.cross_validator().split(X, y, groups=groups))
    model = build_model().fit(X[train_idx], y[train_idx])
    X_test = X[test_idx]
    print("real-time (causal) model passes the guard :",
          _predictions_are_causal(model, X_test, model.predict(X_test)))

    class LookAheadModel:               # cheats: labels epoch i using epoch i+1
        def predict(self, Xa):
            nxt = np.r_[Xa[1:, 0], Xa[-1, 0]]
            return (nxt > 0.5).astype(int)

    cheat = LookAheadModel()
    print("look-ahead (cheating) model passes the guard:",
          _predictions_are_causal(cheat, X_test, cheat.predict(X_test)))
    return X, groups, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Leave-one-subject-out

    Epochs from one sleeper are correlated, so testing on a subject you trained on
    inflates the score. `splits` groups epochs by subject and uses scikit-learn's
    `LeaveOneGroupOut`: each subject is the test set once and never appears in its own
    training fold.
    """)
    return


@app.cell
def _(X, groups, splits, y):
    cv = splits.cross_validator()
    n_folds = cv.get_n_splits(groups=groups)
    disjoint = all((set(groups[_tr]).isdisjoint(set(groups[_te])) for _tr, _te in cv.split(X, y, groups=groups)))
    print(f'{n_folds} folds | every subject held out once | train/test subject-disjoint: {disjoint}')
    print(f'feature matrix: {X.shape[0]} epochs x {X.shape[1]} fixed features')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Baseline: binary REM detection

    The task is REM vs not-REM. We report the REM F-beta(0.3), plus precision and recall
    and the 2-class confusion, under leave-one-subject-out. This uses `build_model()`
    from `modeling/model.py` -- exactly the estimator `evaluate.py` scores.
    """)
    return


@app.cell
def _(
    ConfusionMatrixDisplay,
    X,
    build_model,
    classification_report,
    groups,
    np,
    plt,
    splits,
    y,
):
    from sklearn.metrics import fbeta_score, precision_score, recall_score
    BETA = 0.3
    _fb, _pr, _rc, yt_all, yp_all = ([], [], [], [], [])
    for _tr, _te in splits.cross_validator().split(X, y, groups=groups):
        _m = build_model().fit(X[_tr], y[_tr])
        _yp, _yt = (_m.predict(X[_te]), y[_te])
        yt_all.append(_yt)
        yp_all.append(_yp)
        if (_yt == 1).sum() == 0:
            continue
        _fb.append(fbeta_score(_yt, _yp, beta=BETA, pos_label=1, zero_division=0))
        _pr.append(precision_score(_yt, _yp, pos_label=1, zero_division=0))
        _rc.append(recall_score(_yt, _yp, pos_label=1, zero_division=0))
    _fb = np.array(_fb)
    print(f'REM: F-beta(0.3) = {_fb.mean():.4f} +/- {_fb.std(ddof=1) / np.sqrt(_fb.size):.4f} SEM | precision {np.mean(_pr):.3f}  recall {np.mean(_rc):.3f}  (per-subject mean over {_fb.size} folds)')
    yt_all, yp_all = (np.concatenate(yt_all), np.concatenate(yp_all))
    print(classification_report(yt_all, yp_all, labels=[0, 1], target_names=['not-REM', 'REM'], zero_division=0))
    ConfusionMatrixDisplay.from_predictions(yt_all, yp_all, display_labels=['not REM', 'REM'], normalize='true', cmap='Blues', values_format='.2f')
    plt.title('Leave-one-subject-out confusion (row-normalized)')
    plt.show()
    return BETA, fbeta_score, precision_score, recall_score


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Tuned XGBoost with a motor-atonia prior

    An XGBoost REM detector with two things baked in: **monotone constraints** encoding
    motor atonia -- the activity columns (`activity`, `act_std_w30`) can only *lower*
    P(REM), since REM has near-still motion -- and **hyperparameters tuned by successive
    halving** (`HalvingRandomSearchCV`), scored by best-threshold REM F-beta(0.3). Scored
    under leave-one-subject-out with the decision threshold tuned per training split. The
    atonia prior + threshold together are the strongest REM-precision lever we found.
    """)
    return


@app.cell
def _(
    BETA,
    X,
    fbeta_score,
    features,
    groups,
    np,
    precision_score,
    recall_score,
    splits,
    y,
):
    from sklearn.experimental import enable_halving_search_cv  # noqa: F401
    from sklearn.model_selection import HalvingRandomSearchCV, GroupKFold
    from sklearn.metrics import make_scorer
    from scipy.stats import randint, uniform, loguniform
    from xgboost import XGBClassifier
    ATONIA = ('activity', 'act_std_w30')
    # motor-atonia prior: activity columns can only DECREASE P(REM)
    MONO = '(' + ','.join(('-1' if n in ATONIA else '0' for n in features.FEATURE_NAMES)) + ')'

    def _best_fbeta(y_true, proba):
        p = proba[:, 1] if getattr(proba, 'ndim', 1) == 2 else proba  # threshold-free HPO score
        return max((fbeta_score(y_true, (p >= t).astype(int), beta=BETA, pos_label=1, zero_division=0) for t in np.linspace(0.05, 0.95, 37)))
    scorer = make_scorer(_best_fbeta, response_method='predict_proba')
    neg, pos = (int((y == 0).sum()), int((y == 1).sum()))
    fixed = dict(tree_method='hist', n_jobs=-1, random_state=42, monotone_constraints=MONO)
    space = dict(n_estimators=randint(150, 400), max_depth=randint(3, 7), learning_rate=loguniform(0.02, 0.2), min_child_weight=randint(1, 8), subsample=uniform(0.6, 0.4), colsample_bytree=uniform(0.6, 0.4), reg_lambda=loguniform(0.5, 10.0))
    base = XGBClassifier(scale_pos_weight=neg / pos, **fixed)
    best = HalvingRandomSearchCV(base, space, n_candidates=20, factor=3, scoring=scorer, cv=GroupKFold(3), random_state=42, n_jobs=1).fit(X, y, groups=groups).best_params_
    print('tuned params:', {k: round(v, 4) if isinstance(v, float) else int(v) for k, v in best.items()})
    _fb, _pr, _rc = ([], [], [])
    for _tr, _te in splits.cross_validator().split(X, y, groups=groups):
        spw = (y[_tr] == 0).sum() / max((y[_tr] == 1).sum(), 1)
        _m = XGBClassifier(**best, scale_pos_weight=spw, **fixed).fit(X[_tr], y[_tr])
        p_tr = _m.predict_proba(X[_tr])[:, 1]
        tau = float(max(np.linspace(0.05, 0.95, 181), key=lambda t: fbeta_score(y[_tr], (p_tr >= t).astype(int), beta=BETA, pos_label=1, zero_division=0)))
        _yp, _yt = ((_m.predict_proba(X[_te])[:, 1] >= tau).astype(int), y[_te])
        if (_yt == 1).sum() == 0:
            continue
        _fb.append(fbeta_score(_yt, _yp, beta=BETA, pos_label=1, zero_division=0))
        _pr.append(precision_score(_yt, _yp, pos_label=1, zero_division=0))
        _rc.append(recall_score(_yt, _yp, pos_label=1, zero_division=0))
    _fb = np.array(_fb)  # tune the decision threshold on TRAIN
    print(f'atonia-monotone tuned XGBoost: REM F-beta(0.3) = {_fb.mean():.4f} +/- {_fb.std(ddof=1) / np.sqrt(_fb.size):.4f} SEM | precision {np.mean(_pr):.3f}  recall {np.mean(_rc):.3f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    The pipeline runs end to end: wearable and PSG signals, the fixed causal features, a
    look-ahead check, and a per-subject leave-one-subject-out score. The task is binary
    REM vs not-REM, scored by precision-weighted REM F-beta(0.3). To improve the score,
    try a different model in `modeling/model.py` (the features in `features.py` stay
    fixed); the look-ahead check keeps every candidate deployable in real time.
    """)
    return


if __name__ == "__main__":
    app.run()
