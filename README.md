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

XGBoost beats logistic regression on all three metrics. Each model's 95% CIs and
pooled confusion matrix are saved in `reports/`, written by the training notebooks
(`2.0-mm-logreg`, `2.1-mm-xgboost`) or `make train-logreg` / `make train-xgboost`.

## Tuning Decision Threshold 
A second experiment keeps the F1-tuned XGBoost and re-tunes only its decision threshold to
favor precision by varying beta. 

| threshold tuned for | threshold | precision | recall |
|---------------------|-----------|-----------|--------|
| F1 (default)        | 0.50      | 0.52      | 0.73   |
| F0.5                | 0.75      | 0.64      | 0.46   |
| F0.3                | 0.82      | 0.70      | 0.35   |

Raising the threshold trades recall for precision. Tuning for F0.3 reaches 0.70
precision at 0.35 recall. Run it in the `2.2-mm-xgboost-fbeta` notebook; the reports save to
`reports/xgboost_f05_nested.json` and `reports/xgboost_f03_nested.json`.

## Layout

```
remdetect/                 importable library
├── config.py              paths
├── dataset.py             load the Walch recordings (parsed once, cached)
├── features.py            the fixed causal feature set
├── splits.py              build the feature matrix and the LOSO splitter
└── modeling/
    ├── model.py           XGBoost with a motor-atonia prior
    ├── tune.py            nested-CV engine, search spaces, report writing
    ├── causality.py       the real-time causality guard
    ├── train.py           fit on the full set, serialize to models/
    └── predict.py         load the trained model and predict

notebooks/                 EDA and per-model training (marimo)
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
```

Each `train-*` target runs nested leave-one-subject-out cross-validation: an outer
LOSO loop for the held-out score, an inner grouped loop to tune hyperparameters. The
notebooks `2.0-mm-logreg` and `2.1-mm-xgboost` do the same work interactively, and
each report holds the per-subject scores for comparing the two.

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
make notebooks        # uv run marimo edit notebooks/ --no-token
```
