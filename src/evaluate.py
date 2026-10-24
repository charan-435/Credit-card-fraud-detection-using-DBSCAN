"""Stratified splits and imbalance-aware metrics.

Accuracy is included but should be treated as misleading at a 1% fraud rate
(a constant non-fraud predictor already scores ~0.99).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.config import POSITIVE_LABEL, RANDOM_STATE, TEST_SIZE


def stratified_split(X, y, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    """Train/test split that preserves class rates. Works on arrays or frames."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


def _as_numpy(a) -> np.ndarray:
    if isinstance(a, (pd.Series, pd.DataFrame)):
        return np.asarray(a)
    return np.asarray(a)


def classification_metrics(
    y_true,
    y_pred,
    y_score=None,
    pos_label: int = POSITIVE_LABEL,
) -> dict:
    """Return Recall, Precision, F1, PR-AUC, ROC-AUC, MCC, CM, and Accuracy.

    ``y_score`` should be P(positive) or a continuous decision value.
    PR-AUC / ROC-AUC are set to NaN if scores are not provided.
    """
    y_true = _as_numpy(y_true).ravel()
    y_pred = _as_numpy(y_pred).ravel()

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "accuracy_is_misleading": True,
        "precision": float(precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "pr_auc": float("nan"),
        "roc_auc": float("nan"),
    }

    if y_score is not None:
        scores = _as_numpy(y_score).ravel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metrics["pr_auc"] = float(average_precision_score(y_true, scores, pos_label=pos_label))
            metrics["roc_auc"] = float(roc_auc_score(y_true, scores, labels=[0, 1]))
    return metrics


def metrics_to_row(name: str, metrics: dict) -> dict:
    """Flat row for a results table."""
    return {"model": name, **metrics}
