"""DBSCAN cluster-label augmentation: paper-style (full-data) and leakage-safe.

Paper: fit DBSCAN, append labels as a feature, refit, repeat.
sklearn DBSCAN has no predict(). Leakage-safe test labels are the label of the
nearest training point (1-NN in the current feature space).
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from src.config import DBSCAN_EPS, DBSCAN_MIN_SAMPLES, RANDOM_STATE


def _fit_labels(X: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    return DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(X)


def _assign_by_nearest(X_train: np.ndarray, labels_train: np.ndarray, X_query: np.ndarray) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=1, n_jobs=-1)
    nn.fit(X_train)
    idx = nn.kneighbors(X_query, return_distance=False)[:, 0]
    return labels_train[idx]


def augment_full(
    X: np.ndarray,
    n_iterations: int = 2,
    eps: float = DBSCAN_EPS,
    min_samples: int = DBSCAN_MIN_SAMPLES,
) -> tuple[np.ndarray, list[dict]]:
    """Paper-style: cluster the whole matrix, append labels, repeat. Can leak."""
    X_aug = np.asarray(X, dtype=float)
    history = []
    for i in range(n_iterations):
        labels = _fit_labels(X_aug, eps, min_samples)
        n_clusters = int(len(set(labels) - {-1}))
        n_noise = int((labels == -1).sum())
        history.append(
            {
                "iteration": i + 1,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "noise_frac": n_noise / len(labels),
            }
        )
        X_aug = np.column_stack([X_aug, labels.astype(float)])
    return X_aug, history


def augment_leakage_safe(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_iterations: int = 2,
    eps: float = DBSCAN_EPS,
    min_samples: int = DBSCAN_MIN_SAMPLES,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Fit DBSCAN on train only; map test points via 1-NN to a train label."""
    X_tr = np.asarray(X_train, dtype=float)
    X_te = np.asarray(X_test, dtype=float)
    history = []
    for i in range(n_iterations):
        labels_tr = _fit_labels(X_tr, eps, min_samples)
        labels_te = _assign_by_nearest(X_tr, labels_tr, X_te)
        n_clusters = int(len(set(labels_tr) - {-1}))
        n_noise = int((labels_tr == -1).sum())
        history.append(
            {
                "iteration": i + 1,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "noise_frac": n_noise / len(labels_tr),
            }
        )
        X_tr = np.column_stack([X_tr, labels_tr.astype(float)])
        X_te = np.column_stack([X_te, labels_te.astype(float)])
    return X_tr, X_te, history


def k_distance(X: np.ndarray, k: int = 4) -> np.ndarray:
    """Sorted distance to the k-th neighbour (used to judge DBSCAN eps)."""
    nn = NearestNeighbors(n_neighbors=k, n_jobs=-1)
    nn.fit(X)
    dist = nn.kneighbors(X)[0][:, -1]
    return np.sort(dist)


# RANDOM_STATE imported for a stable public surface; DBSCAN itself is deterministic
# given the data order, which we freeze via the subset id files.
_ = RANDOM_STATE
