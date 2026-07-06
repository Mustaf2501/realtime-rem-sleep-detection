"""Train the REM model on the full feature set and serialize it to models/.

Fits build_model() on every labeled epoch — no held-out subject, since this is the
deployment model, not an evaluation (use evaluate.py for leave-one-subject-out
scoring). Writes the trained XGBoost booster and the metadata needed to reproduce a
prediction: the feature order and the decision threshold.
"""
from __future__ import annotations

import json
import os

from remdetect import splits
from remdetect.config import MODELS_DIR
from remdetect.features import FEATURE_NAMES
from remdetect.modeling.model import BETA, XGB_KWARGS, build_model

BOOSTER_FILE = "rem_xgb.json"
META_FILE = "model_meta.json"


def main() -> None:
    X, y, _ = splits.load_dataset()
    model = build_model().fit(X, y)

    os.makedirs(MODELS_DIR, exist_ok=True)
    booster_path = os.path.join(MODELS_DIR, BOOSTER_FILE)
    model.estimator.save_model(booster_path)   # XGBoost native JSON

    meta_path = os.path.join(MODELS_DIR, META_FILE)
    with open(meta_path, "w") as f:
        json.dump({
            "features": FEATURE_NAMES,          # column order predict() must match
            "threshold": model.threshold,       # P(REM) >= threshold -> REM
            "beta": BETA,
            "objective": XGB_KWARGS["objective"],
            "positive_class": "REM",
        }, f, indent=2)

    print(f"trained on {X.shape[0]} epochs x {X.shape[1]} features "
          f"(threshold={model.threshold:.3f})")
    print(f"saved {booster_path}")
    print(f"saved {meta_path}")


if __name__ == "__main__":
    main()
