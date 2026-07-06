"""Project paths, in one place.

Every module imports its directories from here rather than recomputing them from
__file__, so the layout is defined once. Paths follow the cookiecutter-data-science
convention: raw/interim/processed under data/, plus models/ and reports/.
"""
from __future__ import annotations

from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # original, immutable recordings
INTERIM_DIR = DATA_DIR / "interim"   # transformed intermediates (parse caches)
PROCESSED_DIR = DATA_DIR / "processed"   # final modeling datasets
EXTERNAL_DIR = DATA_DIR / "external"     # third-party data

MODELS_DIR = PROJ_ROOT / "models"    # serialized trained models
REPORTS_DIR = PROJ_ROOT / "reports"  # generated metrics
FIGURES_DIR = REPORTS_DIR / "figures"    # generated figures

# Walch et al. (2019) recordings and their per-subject parse cache.
WALCH_DIR = RAW_DIR / "walch2019"
CACHE_DIR = INTERIM_DIR / "walch2019_cache"
# The small, committed feature matrix — the fixed feature set every model trains on.
DATASET_FILE = PROCESSED_DIR / "featurematrix.npz"
