"""Combine per-model nested-CV reports and test whether they differ.

Training happens in the per-model notebooks (or `python -m remdetect.modeling.tune
<model>`), each of which writes reports/<model>_nested.json. This module only
*combines*: it loads two per-model reports and compares them with a paired Wilcoxon
signed-rank test over the per-subject F1 scores (the subjects are shared, so the
comparison is paired), then writes reports/comparison_nested.json.

    save_report(report)   # persist one model's nested-CV report
    load_report(name)     # read it back
    main()                # load logreg + xgboost, paired test, write the comparison
"""
from __future__ import annotations

import json
import os
from itertools import combinations

import numpy as np
from scipy.stats import wilcoxon

from remdetect.config import REPORTS_DIR

COMPARISON_FILE = "comparison_nested.json"
MODELS = ["logreg", "xgboost"]


def report_path(name: str) -> str:
    return os.path.join(REPORTS_DIR, f"{name}_nested.json")


def save_report(report: dict) -> str:
    """Write one model's nested-CV report to reports/<model>_nested.json."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = report_path(report["model"])
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path


def load_report(name: str) -> dict:
    path = report_path(name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Train it first: run the {name} notebook, or "
            f"`python -m remdetect.modeling.tune {name}`.")
    with open(path) as f:
        return json.load(f)


def paired_test(report_a: dict, report_b: dict) -> dict:
    """Paired Wilcoxon of report_a vs report_b on per-subject F1 (a - b)."""
    assert report_a["subjects"] == report_b["subjects"], "subject folds misaligned"
    a, b = np.array(report_a["per_subject_f1"]), np.array(report_b["per_subject_f1"])
    stat, pval = wilcoxon(a, b)
    diff = float(a.mean() - b.mean())
    return {"test": "wilcoxon_signed_rank", "statistic": float(stat), "pvalue": float(pval),
            f"mean_diff_{report_a['model']}_minus_{report_b['model']}": diff,
            "winner": report_a["model"] if diff > 0 else report_b["model"]}


def main() -> dict:
    reports = {}
    for name in MODELS:
        try:
            reports[name] = load_report(name)
        except FileNotFoundError as e:
            print(f"[compare] skipping {name}: {e}")
    if len(reports) < 2:
        raise RuntimeError("need at least two trained models to compare")

    names = list(reports)
    tests = {}
    for a, b in combinations(names, 2):
        t = paired_test(reports[a], reports[b])
        tests[f"{a}_vs_{b}"] = t
        print(f"[compare] {a} {reports[a]['mean']:.4f} vs {b} {reports[b]['mean']:.4f} "
              f"REM F1; Wilcoxon p = {t['pvalue']:.4f} -> {t['winner']} "
              f"({'significant' if t['pvalue'] < 0.05 else 'n.s.'})")

    report = {"metric": reports[names[0]]["metric"], "models": reports,
              "pairwise_tests": tests}
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(os.path.join(REPORTS_DIR, COMPARISON_FILE), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[compare] wrote {os.path.join(REPORTS_DIR, COMPARISON_FILE)}")
    return report


if __name__ == "__main__":
    main()
