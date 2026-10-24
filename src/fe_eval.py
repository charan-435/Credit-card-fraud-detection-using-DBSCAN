"""Step 4: leakage-safe feature variants, same LightGBM workhorse."""

from __future__ import annotations

import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_recall_curve

from src.config import FEATURE_COLS, RANDOM_STATE, TABLES_DIR
from src.data import features_and_target, load_cleaned, load_subset
from src.dbscan_augment import augment_leakage_safe
from src.evaluate import classification_metrics
from src.features import isolation_forest_score, kmeans_features, lof_score, scale_pos_weight
from src.preprocess import ScalePca

TOP_AUC = ["V14", "V12", "V4", "V11", "V10", "V3", "V9", "V2"]


def _lgbm(y_train, extra=None):
    params = dict(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight(y_train),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    if extra:
        params.update(extra)
    return lgb.LGBMClassifier(**params)


def _recall_at_precision(y_true, scores, min_precision=0.95) -> float:
    p, r, _ = precision_recall_curve(y_true, scores)
    ok = np.where(p[:-1] >= min_precision)[0]
    if len(ok) == 0:
        return 0.0
    return float(r[ok].max())


def _eval_xy(name, X_tr, y_tr, X_te, y_te) -> dict:
    t0 = time.perf_counter()
    clf = _lgbm(y_tr)
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    score = clf.predict_proba(X_te)[:, 1]
    met = classification_metrics(y_te, pred, score)
    met["seconds"] = time.perf_counter() - t0
    met["recall_at_p95"] = _recall_at_precision(y_te, score, 0.95)
    met["variant"] = name
    met["n_features"] = int(X_tr.shape[1])
    return met


def variants_d1() -> list[dict]:
    raw = load_cleaned()
    train = load_subset("D1", df=raw, split="train")
    test = load_subset("D1", df=raw, split="test")
    X_tr, y_tr = features_and_target(train)
    X_te, y_te = features_and_target(test)
    y_tr_np, y_te_np = y_tr.to_numpy(), y_te.to_numpy()

    pipe = ScalePca(use_pca=False, random_state=RANDOM_STATE)
    Z_tr = pipe.fit_transform(X_tr)
    Z_te = pipe.transform(X_te)

    rows = []
    rows.append(_eval_xy("scaled_raw", Z_tr, y_tr_np, Z_te, y_te_np))

    idx = [FEATURE_COLS.index(c) for c in TOP_AUC]
    rows.append(_eval_xy("top8_auc", Z_tr[:, idx], y_tr_np, Z_te[:, idx], y_te_np))

    km_tr, km_te = kmeans_features(Z_tr, Z_te, n_clusters=8, random_state=RANDOM_STATE)
    rows.append(_eval_xy("kmeans_id_dist", km_tr, y_tr_np, km_te, y_te_np))

    if_tr, if_te = isolation_forest_score(Z_tr, Z_te, random_state=RANDOM_STATE)
    rows.append(_eval_xy("isolation_forest", if_tr, y_tr_np, if_te, y_te_np))

    lof_tr, lof_te = lof_score(Z_tr, Z_te)
    rows.append(_eval_xy("lof", lof_tr, y_tr_np, lof_te, y_te_np))

    db_tr, db_te, _ = augment_leakage_safe(Z_tr, Z_te, n_iterations=1)
    rows.append(_eval_xy("dbscan_labels_x1", db_tr, y_tr_np, db_te, y_te_np))

    sm = SMOTE(random_state=RANDOM_STATE)
    Z_s, y_s = sm.fit_resample(Z_tr, y_tr_np)
    rows.append(_eval_xy("smote_on_scaled", Z_s, y_s, Z_te, y_te_np))

    return rows


def eval_winner_on(name: str, variant: str) -> dict:
    raw = load_cleaned()
    train = load_subset(name, df=raw, split="train")
    test = load_subset(name, df=raw, split="test")
    X_tr, y_tr = features_and_target(train)
    X_te, y_te = features_and_target(test)
    pipe = ScalePca(use_pca=False, random_state=RANDOM_STATE)
    Z_tr = pipe.fit_transform(X_tr)
    Z_te = pipe.transform(X_te)
    y_tr_np, y_te_np = y_tr.to_numpy(), y_te.to_numpy()
    if variant == "scaled_raw":
        return _eval_xy(variant, Z_tr, y_tr_np, Z_te, y_te_np)
    if variant == "isolation_forest":
        a, b = isolation_forest_score(Z_tr, Z_te, random_state=RANDOM_STATE)
        return _eval_xy(variant, a, y_tr_np, b, y_te_np)
    if variant == "lof":
        a, b = lof_score(Z_tr, Z_te)
        return _eval_xy(variant, a, y_tr_np, b, y_te_np)
    if variant == "kmeans_id_dist":
        a, b = kmeans_features(Z_tr, Z_te, n_clusters=8, random_state=RANDOM_STATE)
        return _eval_xy(variant, a, y_tr_np, b, y_te_np)
    if variant == "top8_auc":
        idx = [FEATURE_COLS.index(c) for c in TOP_AUC]
        return _eval_xy(variant, Z_tr[:, idx], y_tr_np, Z_te[:, idx], y_te_np)
    if variant == "dbscan_labels_x1":
        a, b, _ = augment_leakage_safe(Z_tr, Z_te, n_iterations=1)
        return _eval_xy(variant, a, y_tr_np, b, y_te_np)
    if variant == "smote_on_scaled":
        Z_s, y_s = SMOTE(random_state=RANDOM_STATE).fit_resample(Z_tr, y_tr_np)
        return _eval_xy(variant, Z_s, y_s, Z_te, y_te_np)
    raise ValueError(variant)


def run_fe() -> pd.DataFrame:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    rows = variants_d1()
    for r in rows:
        r["dataset"] = "D1"
    winner = max(rows, key=lambda d: d["pr_auc"])["variant"]
    print("D1 winner by PR-AUC:", winner)
    for ds in ("D2", "D3"):
        met = eval_winner_on(ds, winner)
        met["dataset"] = ds
        rows.append(met)
    df = pd.DataFrame(rows)
    payload = {"winner": winner, "rows": rows}
    (TABLES_DIR / "fe_results.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(df[["dataset", "variant", "pr_auc", "recall", "precision", "f1", "recall_at_p95", "seconds"]].to_string(index=False))
    return df


if __name__ == "__main__":
    run_fe()
