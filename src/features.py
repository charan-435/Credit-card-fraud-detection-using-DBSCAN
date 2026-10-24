"""Leakage-safe extra features justified (or rejected) by EDA."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


def scale_pos_weight(y) -> float:
    y = np.asarray(y).ravel()
    n_pos = max(int((y == 1).sum()), 1)
    n_neg = int((y == 0).sum())
    return n_neg / n_pos


def kmeans_features(X_train, X_test, n_clusters: int = 8, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    km.fit(X_train)
    d_tr = km.transform(X_train)
    d_te = km.transform(X_test)
    id_tr = km.predict(X_train).reshape(-1, 1).astype(float)
    id_te = km.predict(X_test).reshape(-1, 1).astype(float)
    return np.hstack([X_train, id_tr, d_tr]), np.hstack([X_test, id_te, d_te])


def isolation_forest_score(X_train, X_test, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=random_state, n_jobs=-1)
    iso.fit(X_train)
    s_tr = -iso.score_samples(X_train).reshape(-1, 1)
    s_te = -iso.score_samples(X_test).reshape(-1, 1)
    return np.hstack([X_train, s_tr]), np.hstack([X_test, s_te])


def lof_score(X_train, X_test, n_neighbors: int = 20) -> tuple[np.ndarray, np.ndarray]:
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, n_jobs=-1)
    lof.fit(X_train)
    s_tr = -lof.score_samples(X_train).reshape(-1, 1)
    s_te = -lof.score_samples(X_test).reshape(-1, 1)
    return np.hstack([X_train, s_tr]), np.hstack([X_test, s_te])


def log_amount(X, amount_col: int = -1) -> np.ndarray:
    """log1p of Amount (last column in FEATURE_COLS). Fit-free, no leakage."""
    X = np.asarray(X, dtype=float).copy()
    X[:, amount_col] = np.log1p(np.clip(X[:, amount_col], a_min=0, a_max=None))
    return X


def top_k_columns(X, idx: list[int]) -> np.ndarray:
    return np.asarray(X)[:, idx]


def maybe_scale(X_train, X_test) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test), scaler
