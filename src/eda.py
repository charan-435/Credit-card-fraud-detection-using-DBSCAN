"""Compute EDA stats and figures for D1 (and class counts for all subsets)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp, skew
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    FEATURE_COLS,
    PLOTS_DIR,
    POSITIVE_LABEL,
    RANDOM_STATE,
    TABLES_DIR,
    TARGET_COL,
)
from src.data import features_and_target, load_cleaned, load_named_subsets, load_subset, subset_summary
from src.dbscan_augment import k_distance
from src.evaluate import classification_metrics
from src.preprocess import ScalePca
from sklearn.cluster import DBSCAN

sns.set_theme(style="whitegrid", context="notebook")


def _save(fig: plt.Figure, name: str) -> str:
    folder = PLOTS_DIR / "eda"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def univariate_separability(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rows = []
    yv = y.to_numpy()
    for col in X.columns:
        x = X[col].to_numpy()
        auc = roc_auc_score(yv, x)
        auc = max(auc, 1.0 - auc)
        ks = ks_2samp(x[yv == 1], x[yv == 0]).statistic
        rows.append({"feature": col, "auc": float(auc), "ks": float(ks)})
    mi = mutual_info_classif(X, y, random_state=RANDOM_STATE)
    out = pd.DataFrame(rows)
    out["mi"] = mi
    return out.sort_values("auc", ascending=False).reset_index(drop=True)


def run_eda() -> dict:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (PLOTS_DIR / "eda").mkdir(parents=True, exist_ok=True)

    raw = load_cleaned()
    subsets = load_named_subsets(raw)
    d1 = subsets["D1"]
    X, y = features_and_target(d1)

    summary: dict = {
        "raw_n": int(len(raw)),
        "raw_class_counts": {int(k): int(v) for k, v in raw[TARGET_COL].value_counts().items()},
        "nulls": int(raw.isnull().sum().sum()),
        "duplicates": int(raw.duplicated().sum()),
        "dtypes": {c: str(t) for c, t in raw.dtypes.items()},
        "subsets": {name: subset_summary(frame) for name, frame in subsets.items()},
    }

    # 2.1 class counts
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    frames = [("raw 50/50", raw)] + [(n, subsets[n]) for n in ("D1", "D2", "D3")]
    for ax, (title, frame) in zip(axes, frames):
        vc = frame[TARGET_COL].value_counts().sort_index()
        ax.bar(["non-fraud (0)", "fraud (1)"], vc.values, color=["#4c78a8", "#e45756"])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
        for i, v in enumerate(vc.values):
            ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Class counts: raw file vs paper subsets")
    plots = {"class_counts": _save(fig, "01_class_counts.png")}

    # 2.2 describe
    desc = X.describe().T
    desc["skew"] = X.apply(lambda s: float(skew(s.dropna())))
    desc.to_json(TABLES_DIR / "eda_describe.json", indent=2)
    summary["amount_mean"] = float(X["Amount"].mean())
    summary["amount_std"] = float(X["Amount"].std())
    summary["v_mean_abs"] = float(X[[c for c in FEATURE_COLS if c != "Amount"]].mean().abs().mean())
    summary["amount_skew"] = float(skew(X["Amount"]))
    v_skew = X[[c for c in FEATURE_COLS if c != "Amount"]].apply(lambda s: float(skew(s)))
    summary["v_skew_max"] = {"feature": v_skew.abs().idxmax(), "skew": float(v_skew.abs().max())}

    # 2.3 Amount vs V scale
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    sns.kdeplot(X["Amount"], ax=axes[0], fill=True)
    axes[0].set_title("Amount (raw)")
    sns.kdeplot(np.log1p(X["Amount"]), ax=axes[1], fill=True)
    axes[1].set_title("log1p(Amount)")
    plots["amount_dist"] = _save(fig, "02_amount_dist.png")

    # 2.4 univariate ranking
    rank = univariate_separability(X, y)
    rank.to_json(TABLES_DIR / "eda_univariate.json", orient="records", indent=2)
    summary["top_auc"] = rank.head(8).to_dict(orient="records")
    summary["amount_auc"] = float(rank.loc[rank.feature == "Amount", "auc"].iloc[0])
    fig, ax = plt.subplots(figsize=(7, 6))
    top = rank.head(15).iloc[::-1]
    ax.barh(top["feature"], top["auc"], color="#4c78a8")
    ax.axvline(0.5, color="grey", ls="--", lw=1)
    ax.set_xlabel("univariate ROC-AUC (direction-invariant)")
    ax.set_title("Feature-vs-class separability on D1")
    plots["univariate_auc"] = _save(fig, "03_univariate_auc.png")

    fig, ax = plt.subplots(figsize=(7, 6))
    top_mi = rank.sort_values("mi").tail(15)
    ax.barh(top_mi["feature"], top_mi["mi"], color="#72b7b2")
    ax.set_xlabel("mutual information")
    ax.set_title("Mutual information with Class (D1)")
    plots["univariate_mi"] = _save(fig, "04_univariate_mi.png")

    # 2.5 distributions of top features
    top6 = rank.head(6)["feature"].tolist()
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, col in zip(axes.ravel(), top6):
        sns.kdeplot(data=d1, x=col, hue=TARGET_COL, common_norm=False, ax=ax, fill=True, alpha=0.4)
        ax.set_title(col)
    fig.suptitle("Top-6 univariate features by class (D1)")
    fig.tight_layout()
    plots["top_kde"] = _save(fig, "05_top_kde.png")

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, col in zip(axes.ravel(), top6):
        sns.boxplot(data=d1, x=TARGET_COL, y=col, ax=ax)
        ax.set_title(col)
    fig.suptitle("Top-6 features: boxplots by class")
    fig.tight_layout()
    plots["top_box"] = _save(fig, "06_top_box.png")

    # 2.6 correlation
    corr = X.corr()
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(corr, ax=ax, cmap="vlag", center=0, xticklabels=False, yticklabels=False)
    ax.set_title("Feature correlation (D1)")
    plots["corr"] = _save(fig, "07_corr.png")
    # target correlation via point-biserial == pearson with 0/1
    target_corr = X.apply(lambda s: float(np.corrcoef(s, y)[0, 1])).abs().sort_values(ascending=False)
    summary["top_target_corr"] = target_corr.head(8).to_dict()
    off_diag = corr.abs().where(~np.eye(len(corr), dtype=bool))
    summary["max_feature_corr"] = float(off_diag.max().max())
    summary["max_feature_corr_pair"] = [
        str(off_diag.max().idxmax()),
        str(off_diag[off_diag.max().idxmax()].idxmax()),
    ]

    # 2.7 outliers: IQR flags
    q1 = X.quantile(0.25)
    q3 = X.quantile(0.75)
    iqr = q3 - q1
    outlier = (X < (q1 - 1.5 * iqr)) | (X > (q3 + 1.5 * iqr))
    n_flagged = outlier.any(axis=1)
    fraud = y == POSITIVE_LABEL
    summary["outlier_frac_all"] = float(n_flagged.mean())
    summary["outlier_frac_fraud"] = float(n_flagged[fraud].mean())
    summary["outlier_frac_legit"] = float(n_flagged[~fraud].mean())
    summary["fraud_among_outliers"] = float(fraud[n_flagged].mean()) if n_flagged.any() else 0.0
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(
        ["legit flagged", "fraud flagged"],
        [summary["outlier_frac_legit"], summary["outlier_frac_fraud"]],
        color=["#4c78a8", "#e45756"],
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of class with any IQR outlier")
    ax.set_title("IQR outliers by class (D1)")
    plots["outliers"] = _save(fig, "08_outliers.png")

    # 2.8 2D projections on stratified 8k subsample
    rng = np.random.RandomState(RANDOM_STATE)
    n_sub = min(8000, len(d1))
    # keep ~1% fraud in subsample: take all fraud + fill with legit
    fraud_idx = np.flatnonzero(y == 1)
    legit_idx = np.flatnonzero(y == 0)
    n_f = min(len(fraud_idx), max(80, int(0.05 * n_sub)))  # slightly oversample fraud for visibility
    n_l = n_sub - n_f
    take_f = rng.choice(fraud_idx, size=n_f, replace=False)
    take_l = rng.choice(legit_idx, size=n_l, replace=False)
    take = np.concatenate([take_f, take_l])
    Xs = StandardScaler().fit_transform(X.iloc[take])
    ys = y.iloc[take].to_numpy()

    pca2 = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(pca2[ys == 0, 0], pca2[ys == 0, 1], s=6, alpha=0.25, label="legit", c="#4c78a8")
    ax.scatter(pca2[ys == 1, 0], pca2[ys == 1, 1], s=14, alpha=0.8, label="fraud", c="#e45756")
    ax.set_title("PCA 2D (8k subsample, fraud oversampled for visibility)")
    ax.legend(markerscale=2)
    plots["pca2"] = _save(fig, "09_pca2.png")
    summary["pca2_var"] = PCA(n_components=2, random_state=RANDOM_STATE).fit(Xs).explained_variance_ratio_.tolist()

    tsne = TSNE(n_components=2, perplexity=30, random_state=RANDOM_STATE, init="pca")
    tsne2 = tsne.fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(tsne2[ys == 0, 0], tsne2[ys == 0, 1], s=6, alpha=0.25, label="legit", c="#4c78a8")
    ax.scatter(tsne2[ys == 1, 0], tsne2[ys == 1, 1], s=14, alpha=0.8, label="fraud", c="#e45756")
    ax.set_title("t-SNE 2D (same 8k subsample)")
    ax.legend(markerscale=2)
    plots["tsne2"] = _save(fig, "10_tsne2.png")

    umap_ok = False
    try:
        import umap

        um = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=RANDOM_STATE)
        um2 = um.fit_transform(Xs)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.scatter(um2[ys == 0, 0], um2[ys == 0, 1], s=6, alpha=0.25, label="legit", c="#4c78a8")
        ax.scatter(um2[ys == 1, 0], um2[ys == 1, 1], s=14, alpha=0.8, label="fraud", c="#e45756")
        ax.set_title("UMAP 2D (same 8k subsample)")
        ax.legend(markerscale=2)
        plots["umap2"] = _save(fig, "11_umap2.png")
        umap_ok = True
    except Exception as exc:
        summary["umap_error"] = str(exc)

    summary["umap_ok"] = umap_ok

    # 2.9 DBSCAN diagnostics on D1 TRAIN only
    train = load_subset("D1", df=raw, split="train")
    Xtr, ytr = features_and_target(train)
    scaler = StandardScaler()
    Ztr = scaler.fit_transform(Xtr)
    kd = k_distance(Ztr, k=4)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(kd, lw=1)
    ax.axhline(2.0, color="#e45756", ls="--", label="paper eps=2")
    ax.set_title("4-distance plot (D1 train, scaled)")
    ax.set_xlabel("points sorted by 4-NN distance")
    ax.set_ylabel("distance")
    ax.legend()
    plots["kdist"] = _save(fig, "12_kdistance.png")
    summary["kdist_median"] = float(np.median(kd))
    summary["kdist_p90"] = float(np.quantile(kd, 0.90))
    summary["kdist_p99"] = float(np.quantile(kd, 0.99))
    summary["frac_kdist_gt_eps2"] = float((kd > 2.0).mean())

    db = DBSCAN(eps=2.0, min_samples=4, n_jobs=-1)
    labels = db.fit_predict(Ztr)
    ytr_np = ytr.to_numpy()
    n_clusters = int(len(set(labels) - {-1}))
    noise = labels == -1
    summary["dbscan"] = {
        "n_clusters": n_clusters,
        "noise_frac": float(noise.mean()),
        "fraud_n": int((ytr_np == 1).sum()),
        "fraud_in_noise": int(((ytr_np == 1) & noise).sum()),
        "fraud_in_cluster": int(((ytr_np == 1) & ~noise).sum()),
        "fraud_noise_rate": float(((ytr_np == 1) & noise).sum() / max((ytr_np == 1).sum(), 1)),
        "noise_that_is_fraud": float((ytr_np[noise] == 1).mean()) if noise.any() else 0.0,
    }

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    cats = ["fraud in noise", "fraud in a cluster", "legit in noise"]
    vals = [
        summary["dbscan"]["fraud_in_noise"],
        summary["dbscan"]["fraud_in_cluster"],
        int(((ytr_np == 0) & noise).sum()),
    ]
    ax.bar(cats, vals, color=["#e45756", "#f2a07b", "#4c78a8"])
    ax.set_title("Where DBSCAN (eps=2, minPts=4) puts D1-train points")
    ax.tick_params(axis="x", rotation=15)
    plots["dbscan_where"] = _save(fig, "13_dbscan_where.png")

    # 2.10 too-easy probe, leakage-safe
    test = load_subset("D1", df=raw, split="test")
    Xte, yte = features_and_target(test)
    pipe = ScalePca(use_pca=False)
    Ztr2 = pipe.fit_transform(Xtr)
    Zte2 = pipe.transform(Xte)
    probe = {}
    for name, clf in [
        ("logreg", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
        ("tree_d3", DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=RANDOM_STATE)),
    ]:
        clf.fit(Ztr2, ytr)
        pred = clf.predict(Zte2)
        score = clf.predict_proba(Zte2)[:, 1]
        met = classification_metrics(yte, pred, score)
        probe[name] = {k: met[k] for k in ("recall", "precision", "f1", "pr_auc", "roc_auc", "mcc")}
    summary["probe"] = probe

    # extra PCA on already-PCA features: variance curve
    pca_full = PCA(random_state=RANDOM_STATE).fit(Ztr)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(np.cumsum(pca_full.explained_variance_ratio_), marker="o", ms=3)
    ax.axhline(0.95, color="grey", ls="--")
    ax.set_title("Cumulative PCA variance on scaled D1 train")
    ax.set_xlabel("components")
    ax.set_ylabel("cumulative explained variance")
    plots["pca_var"] = _save(fig, "14_pca_variance.png")
    summary["pca_n_for_95"] = int(np.searchsorted(np.cumsum(pca_full.explained_variance_ratio_), 0.95) + 1)

    summary["plots"] = plots
    path = TABLES_DIR / "eda_summary.json"
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("raw_n", "subsets", "probe", "dbscan", "top_auc", "amount_auc")}, indent=2, default=str))
    print("wrote", path)
    return summary


if __name__ == "__main__":
    run_eda()
