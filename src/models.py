"""Paper classifiers, voting rules, boosting wrappers, and a small MLP."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC

from src.config import (
    KNN_N_NEIGHBORS,
    POSITIVE_LABEL,
    RANDOM_STATE,
    RF_MAX_DEPTH,
    RF_N_ESTIMATORS,
    SVM_C,
    SVM_KERNEL,
)
from src.evaluate import classification_metrics


def paper_rf(**kwargs) -> RandomForestClassifier:
    params = dict(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight=None,
    )
    params.update(kwargs)
    return RandomForestClassifier(**params)


def paper_knn(**kwargs) -> KNeighborsClassifier:
    params = dict(n_neighbors=KNN_N_NEIGHBORS, n_jobs=-1)
    params.update(kwargs)
    return KNeighborsClassifier(**params)


def paper_svm(linear_fallback: bool = False, **kwargs):
    if linear_fallback:
        params = dict(C=SVM_C, dual="auto", random_state=RANDOM_STATE)
        params.update(kwargs)
        return LinearSVC(**params)
    params = dict(kernel=SVM_KERNEL, C=SVM_C, probability=False, random_state=RANDOM_STATE, cache_size=1000)
    params.update(kwargs)
    return SVC(**params)


def majority_vote(pred_list: list[np.ndarray]) -> np.ndarray:
    stacked = np.vstack([np.asarray(p).ravel() for p in pred_list])
    return (stacked.sum(axis=0) >= 2).astype(int)


def disjunctive_vote(pred_list: list[np.ndarray]) -> np.ndarray:
    stacked = np.vstack([np.asarray(p).ravel() for p in pred_list])
    return (stacked.max(axis=0) >= 1).astype(int)


def _score_from_estimator(est, X) -> np.ndarray | None:
    if hasattr(est, "predict_proba"):
        proba = est.predict_proba(X)
        classes = list(est.classes_)
        if POSITIVE_LABEL in classes:
            return proba[:, classes.index(POSITIVE_LABEL)]
        return proba[:, -1]
    if hasattr(est, "decision_function"):
        return np.asarray(est.decision_function(X)).ravel()
    return None


@dataclass
class FitResult:
    name: str
    estimator: object
    metrics: dict
    seconds: float
    y_pred: np.ndarray
    y_score: np.ndarray | None


def fit_eval(name: str, estimator, X_train, y_train, X_test, y_test) -> FitResult:
    t0 = time.perf_counter()
    estimator.fit(X_train, y_train)
    y_pred = np.asarray(estimator.predict(X_test)).ravel()
    y_score = _score_from_estimator(estimator, X_test)
    seconds = time.perf_counter() - t0
    metrics = classification_metrics(y_test, y_pred, y_score)
    metrics["seconds"] = seconds
    return FitResult(name, estimator, metrics, seconds, y_pred, y_score)


def eval_votes(pred_map: dict[str, np.ndarray], y_test, score_map: dict[str, np.ndarray] | None = None) -> dict[str, dict]:
    names = list(pred_map)
    preds = [pred_map[n] for n in names]
    out = {}
    maj = majority_vote(preds)
    dve = disjunctive_vote(preds)
    maj_score = dve_score = None
    if score_map:
        scores = np.vstack([score_map[n] for n in names if score_map.get(n) is not None])
        if scores.size:
            maj_score = scores.mean(axis=0)
            dve_score = scores.max(axis=0)
    out["majority"] = classification_metrics(y_test, maj, maj_score)
    out["disjunctive"] = classification_metrics(y_test, dve, dve_score)
    return out
