"""Shared evaluation: every experiment logs results through evaluate().

Results are appended to results/<experiment>.csv so report tables can be
generated from files rather than typed by hand.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

from src.config import RESULTS_DIR

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def compute_metrics(y_true, y_pred, y_score=None) -> dict:
    """Binary metrics with 'Parasitized' (label 1) as the positive class."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),  # sensitivity
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score) if y_score is not None else np.nan,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def evaluate(y_true, y_pred, y_score, system: str, fold: int, experiment: str = "main",
             results_dir: Path = RESULTS_DIR, **extra) -> dict:
    """Compute metrics for one system on one fold and append a row to the CSV.

    extra: any additional columns, e.g. train_frac=0.1 or corruption="blur".
    """
    row = {"system": system, "fold": fold, **extra, **compute_metrics(y_true, y_pred, y_score)}
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{experiment}.csv"
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)
    return row


def summarise(experiment: str = "main", by=("system",), results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """Mean and std of each metric across folds, one row per group."""
    df = pd.read_csv(results_dir / f"{experiment}.csv")
    return df.groupby(list(by))[METRICS].agg(["mean", "std"])


def to_latex(summary: pd.DataFrame) -> str:
    """Format a summarise() table as 'mean ± std' percentages for the report."""
    out = pd.DataFrame(index=summary.index)
    for m in METRICS:
        out[m] = [f"{100*a:.1f} $\\pm$ {100*b:.1f}" for a, b in
                  zip(summary[(m, "mean")], summary[(m, "std")].fillna(0))]
    return out.to_latex(escape=False)
