"""Load the trained REM model from models/ and predict REM per epoch.

Mirrors train.py's serialization: reads the XGBoost booster and metadata (feature
order + decision threshold) from models/, and applies the threshold so a 1 means
predicted-REM. Reproduces predictions outside the training code — e.g. to check the
serialized model against golden vectors for the Flutter port.
"""
from __future__ import annotations

import json
import os

import numpy as np
from xgboost import XGBClassifier

from remdetect.config import MODELS_DIR
from remdetect.modeling.train import BOOSTER_FILE, META_FILE


def load() -> tuple[XGBClassifier, dict]:
    """Return (booster, meta) loaded from models/."""
    booster = XGBClassifier()
    booster.load_model(os.path.join(MODELS_DIR, BOOSTER_FILE))
    with open(os.path.join(MODELS_DIR, META_FILE)) as f:
        meta = json.load(f)
    return booster, meta


def predict_proba(X) -> np.ndarray:
    """P(REM) for each epoch row."""
    booster, _ = load()
    return booster.predict_proba(np.asarray(X, dtype=float))[:, 1]


def predict(X) -> np.ndarray:
    """1 for predicted-REM epochs, applying the trained decision threshold."""
    booster, meta = load()
    p = booster.predict_proba(np.asarray(X, dtype=float))[:, 1]
    return (p >= meta["threshold"]).astype(int)


if __name__ == "__main__":
    from remdetect import splits

    X, _, _ = splits.load_dataset()
    pred = predict(X)
    print(f"predicted REM on {int(pred.sum())} / {len(pred)} epochs")
