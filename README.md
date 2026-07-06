# Real-time wearable REM detection

Detect REM sleep from a wrist wearable's heart rate, motion, and time of night, one
30-second epoch at a time. The data is the Walch et al. (2019) Apple Watch recordings
(31 subjects, one night each); the task is REM versus not-REM. Every model is scored
by leave-one-subject-out cross-validation and must be causal: it may use only data up
to the current epoch, so it can run live on the watch.

Feature extraction is fixed and lives in the `remdetect` package, so every model
trains on the same matrix. Models are compared both in marimo notebooks and from the
command line.

## Results

Nested leave-one-subject-out cross-validation, per-subject means over 30 subjects:

| model | REM F1 | precision | recall |
|-------|--------|-----------|--------|
| logistic regression | 0.51 | 0.42 | 0.70 |
| XGBoost | 0.59 | 0.52 | 0.73 |

XGBoost wins the paired Wilcoxon signed-rank test on F1 (p < 0.001), and it beats
logistic on both precision and recall. Each model's 95% CIs and pooled confusion
matrix are saved in `reports/` and shown in the `3.0-mm-comparison` notebook.
Regenerate with `make compare`.

## Layout

```
remdetect/                 importable library
├── config.py              paths
├── dataset.py             load the Walch recordings (parsed once, cached)
├── features.py            the fixed causal feature set
├── splits.py              build the feature matrix and the LOSO splitter
└── modeling/
    ├── model.py           XGBoost with a motor-atonia prior
    ├── tune.py            nested-CV engine and search spaces
    ├── compare.py         combine per-model reports, paired test
    ├── causality.py       the real-time causality guard
    ├── train.py           fit on the full set, serialize to models/
    └── predict.py         load the trained model and predict

notebooks/                 EDA, per-model training, and the comparison (marimo)
data/{raw,interim,processed}   recordings / parse cache / committed featurematrix.npz
models/                    serialized models
reports/                   metrics and figures
```

## Setup

Dependencies and the `remdetect` package (editable) are managed with
[uv](https://docs.astral.sh/uv/):

```bash
make sync
```

## Compare models

```bash
make train-logreg     # nested CV, writes reports/logreg_nested.json
make train-xgboost    # nested CV, writes reports/xgboost_nested.json
make compare          # paired test, writes reports/comparison_nested.json
```

Each `train-*` target runs nested leave-one-subject-out cross-validation: an outer
LOSO loop for the held-out score, an inner grouped loop to tune hyperparameters. The
notebooks `2.0-mm-logreg`, `2.1-mm-xgboost`, and `3.0-mm-comparison` do the same
work interactively.

## Deploy a model

```bash
make train            # fit on all data, write models/rem_xgb.json + model_meta.json
make predict          # reload and predict
```

## Add a model

The comparison takes any scikit-learn estimator and a search space. Define both in a
training notebook (see `2.1-mm-xgboost`) and call `nested_loso_f1`. Predictions must
stay causal: the guard in `causality.py` checks every fold and raises if a model
reads ahead.

## Add a feature

Edit `remdetect/features.py`. A feature must be causal, using only samples at or
before the current epoch; the truncation tests in `tests/test_features.py` enforce
this. Changing `features.py` rebuilds `data/processed/featurematrix.npz` on the next
run.

## Data

The feature matrix ships with the repo, so you can train and compare without the raw
recordings. To rebuild from the raw data, install the dataset (see `data/README.md`).

## Test

```bash
make test
```

## Pairing with an agent (marimo pair)

Open a notebook and an agent can join the running kernel to work alongside you:

```bash
make notebook         # uv run marimo edit notebooks/1.0-mm-eda.py --no-token
```
