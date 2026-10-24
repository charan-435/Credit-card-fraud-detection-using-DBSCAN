"""Leakage-safe scaler + optional PCA. Fit on train, transform test."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class ScalePca:
    """StandardScaler, then PCA. PCA is optional (paper applies it anyway)."""

    def __init__(self, use_pca: bool = True, pca_variance: float = 0.95, random_state: int = 42):
        self.use_pca = use_pca
        self.pca_variance = pca_variance
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca: PCA | None = None

    def fit(self, X):
        Z = self.scaler.fit_transform(X)
        if self.use_pca:
            n_features = Z.shape[1]
            self.pca = PCA(n_components=min(self.pca_variance, n_features), random_state=self.random_state)
            self.pca.fit(Z)
        return self

    def transform(self, X) -> np.ndarray:
        Z = self.scaler.transform(X)
        if self.pca is not None:
            return self.pca.transform(Z)
        return Z

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)

    @property
    def n_components_(self) -> int | None:
        if self.pca is None:
            return None
        return int(self.pca.n_components_)
