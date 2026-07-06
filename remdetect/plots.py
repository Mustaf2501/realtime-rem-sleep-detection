"""Recording one model's leave-one-subject-out results to disk.

evaluate.py computes the numbers; this module records them. Given a model name and
its scored stats, it writes reports/<name>.json (the metrics) and
reports/figures/<name>.png (the confusion matrix + per-fold bars), keeping file IO,
JSON, and matplotlib out of the scoring code.

Public surface:
    row_normalized_confusion(y_true, y_pred, stages) -> (k, k) array
    save(name, stats, confusion, per_class, n_subjects, labels, beta) -> writes the files
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.metrics import confusion_matrix

from remdetect.config import FIGURES_DIR, REPORTS_DIR


def row_normalized_confusion(y_true: np.ndarray, y_pred: np.ndarray,
                             stages: list[int]) -> np.ndarray:
    """Confusion matrix over `stages`, each true-class row normalized to sum 1. A
    stage that never truly occurs leaves its row NaN rather than dividing by zero."""
    cm = confusion_matrix(y_true, y_pred, labels=stages).astype(float)
    row_totals = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, row_totals, out=np.full_like(cm, np.nan), where=row_totals > 0)


def save(name: str, stats: dict, confusion: np.ndarray, per_class: dict,
         n_subjects: int, labels: list[str], beta: float) -> None:
    """Write reports/<name>.json (metrics) and reports/figures/<name>.png (figure).
    Named by the model, so re-running the same model overwrites its own files."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    with open(os.path.join(REPORTS_DIR, name + ".json"), "w") as f:
        json.dump({
            "model": name,
            "metric_mean_rem_fbeta": stats["fbeta"][0],
            "beta": beta,
            "n_subjects": n_subjects,
            # REM one-vs-rest, per-subject mean/sem; accuracy is overall (all 5 stages)
            "rem_per_subject": {k: {"mean": m, "sem": s} for k, (m, s) in stats.items()},
            # overall multiclass breakdown, pooled over every epoch
            "per_class_pooled": {lbl: {"precision": per_class[lbl]["precision"],
                                       "recall": per_class[lbl]["recall"],
                                       "f1": per_class[lbl]["f1-score"],
                                       "support": int(per_class[lbl]["support"])}
                                 for lbl in labels},
            "overall_accuracy_pooled": per_class["accuracy"],
            "confusion_rownorm": confusion.tolist(),
            "confusion_labels": labels,
            "confusion_method": "pooled, row-normalized over all epochs",
        }, f, indent=2)

    _save_figure(os.path.join(FIGURES_DIR, name + ".png"), stats, confusion,
                 labels, beta, name)


def _save_figure(path: str, stats: dict, confusion: np.ndarray,
                 labels: list[str], beta: float, name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")   # headless: no display needed
    import matplotlib.pyplot as plt

    fig, (ax_cm, ax_bar) = plt.subplots(1, 2, figsize=(10, 4))

    n = len(labels)
    ax_cm.imshow(confusion, cmap="Blues", vmin=0, vmax=1)
    ax_cm.set_xticks(range(n), labels)
    ax_cm.set_yticks(range(n), labels)
    for i in range(n):
        for j in range(n):
            val = confusion[i, j]
            ax_cm.text(j, i, "" if np.isnan(val) else f"{val:.2f}",
                       ha="center", va="center", fontsize=8)
    ax_cm.set_xlabel("Predicted sleep stage")
    ax_cm.set_ylabel("True sleep stage")
    ax_cm.set_title("Confusion (row-normalized, pooled)")

    names = ["Accuracy\n(overall)", "REM\nprecision", "REM\nrecall"]
    means = [stats["accuracy"][0], stats["precision"][0], stats["recall"][0]]
    sems = [stats["accuracy"][1], stats["precision"][1], stats["recall"][1]]
    ax_bar.bar(names, means, yerr=sems, capsize=4, color="0.8", edgecolor="black")
    ax_bar.set_ylim(0, 1)
    ax_bar.set_ylabel("Ratio")
    ax_bar.set_title("Per-fold mean +/- SEM")

    fig.suptitle(f"REM detection - {name}  (REM F{beta} = {stats['fbeta'][0]:.3f})")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
