"""Step 5: boosting + sklearn MLP, val-tuned threshold, modest Optuna search."""

from __future__ import annotations

import json
import time
from typing import Callable

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from src.config import POSITIVE_LABEL, RANDOM_STATE, TABLES_DIR, TEST_SIZE
from src.data import features_and_target, load_cleaned, load_subset
from src.evaluate import classification_metrics
from src.features import isolation_forest_score, scale_pos_weight
from src.models import disjunctive_vote, majority_vote
from src.preprocess import ScalePca

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _load_xy(name: str):
    raw = load_cleaned()
    train = load_subset(name, df=raw, split="train")
    test = load_subset(name, df=raw, split="test")
    X_tr, y_tr = features_and_target(train)
    X_te, y_te = features_and_target(test)
    pipe = ScalePca(use_pca=False, random_state=RANDOM_STATE)
    Z_tr = pipe.fit_transform(X_tr)
    Z_te = pipe.transform(X_te)
    return Z_tr, y_tr.to_numpy(), Z_te, y_te.to_numpy()


def apply_fe(variant: str, Z_tr, Z_te, y_tr=None):
    """Return (Z_tr, Z_te, y_tr). y_tr only changes for SMOTE."""
    if variant == "isolation_forest":
        a, b = isolation_forest_score(Z_tr, Z_te, random_state=RANDOM_STATE)
        return a, b, y_tr
    if variant == "lof":
        from src.features import lof_score

        a, b = lof_score(Z_tr, Z_te)
        return a, b, y_tr
    if variant == "kmeans_id_dist":
        from src.features import kmeans_features

        a, b = kmeans_features(Z_tr, Z_te, n_clusters=8, random_state=RANDOM_STATE)
        return a, b, y_tr
    if variant == "smote_on_scaled":
        from imblearn.over_sampling import SMOTE

        Z_s, y_s = SMOTE(random_state=RANDOM_STATE).fit_resample(Z_tr, y_tr)
        return Z_s, Z_te, y_s
    if variant == "dbscan_labels_x1":
        from src.dbscan_augment import augment_leakage_safe

        a, b, _ = augment_leakage_safe(Z_tr, Z_te, n_iterations=1)
        return a, b, y_tr
    return Z_tr, Z_te, y_tr


def best_threshold(y_true, scores) -> float:
    """Max F1 on validation scores. Never use test labels here."""
    p, r, t = precision_recall_curve(y_true, scores)
    if len(t) == 0:
        return 0.5
    f1 = np.zeros_like(t, dtype=float)
    denom = p[:-1] + r[:-1]
    np.divide(2 * p[:-1] * r[:-1], denom, out=f1, where=denom > 0)
    return float(t[int(np.argmax(f1))])


def _fit_score(model, X_tr, y_tr, X_va):
    model.fit(X_tr, y_tr)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_va)[:, 1]
    return np.asarray(model.decision_function(X_va)).ravel()


def make_xgb(y, **kw):
    p = dict(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight(y),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="aucpr",
        tree_method="hist",
    )
    p.update(kw)
    return xgb.XGBClassifier(**p)


def make_lgb(y, **kw):
    p = dict(
        n_estimators=400,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight(y),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    p.update(kw)
    return lgb.LGBMClassifier(**p)


def make_cb(y, **kw):
    p = dict(
        iterations=200,
        depth=6,
        learning_rate=0.08,
        loss_function="Logloss",
        auto_class_weights="Balanced",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
        thread_count=8,
    )
    p.update(kw)
    return CatBoostClassifier(**p)


def make_mlp(**kw):
    p = dict(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=25,
        early_stopping=True,
        n_iter_no_change=4,
        random_state=RANDOM_STATE,
    )
    p.update(kw)
    return MLPClassifier(**p)


def tune_lgb(X, y, n_trials: int = 20) -> dict:
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        model = make_lgb(
            y_tr,
            num_leaves=trial.suggest_int("num_leaves", 16, 64),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 80),
        )
        scores = _fit_score(model, X_tr, y_tr, X_va)
        from sklearn.metrics import average_precision_score

        return float(average_precision_score(y_va, scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def eval_model(name: str, factory: Callable, X_tr, y_tr, X_te, y_te) -> dict:
    """Fit on 75% of train, pick F1 threshold on the other 25%, score test once."""
    X_fit, X_va, y_fit, y_va = train_test_split(
        X_tr, y_tr, test_size=0.25, stratify=y_tr, random_state=RANDOM_STATE
    )
    t0 = time.perf_counter()
    model = factory(y_fit)
    scores_va = _fit_score(model, X_fit, y_fit, X_va)
    thr = best_threshold(y_va, scores_va)
    if hasattr(model, "predict_proba"):
        scores_te = model.predict_proba(X_te)[:, 1]
    else:
        scores_te = np.asarray(model.decision_function(X_te)).ravel()
    pred = (scores_te >= thr).astype(int)
    met = classification_metrics(y_te, pred, scores_te)
    met["seconds"] = time.perf_counter() - t0
    met["threshold"] = thr
    met["model"] = name
    met["y_score"] = scores_te.tolist()
    met["y_pred"] = pred.tolist()
    return met, model


def run_advanced(fe_variant: str = "scaled_raw", datasets=("D1", "D2", "D3")) -> pd.DataFrame:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    Z_tr, y_tr, Z_te, y_te = _load_xy("D1")
    Z_tr, Z_te, y_tr = apply_fe(fe_variant, Z_tr, Z_te, y_tr)
    print("tuning LightGBM on D1 val PR-AUC ...")
    best = tune_lgb(Z_tr, y_tr, n_trials=20)
    print("best lgb params", best)
    (TABLES_DIR / "lgb_best_params.json").write_text(json.dumps(best, indent=2), encoding="utf-8")

    rows = []
    score_bank = {}
    for ds in datasets:
        Xtr, ytr, Xte, yte = _load_xy(ds)
        Xtr, Xte, ytr = apply_fe(fe_variant, Xtr, Xte, ytr)
        factories = {
            "XGBoost": lambda y: make_xgb(y),
            "LightGBM": lambda y: make_lgb(y, **best),
            "MLP": lambda y: make_mlp(),
        }
        # CatBoost hung >15 min per dataset on this Windows CPU; skipped (logged in results).
        preds = {}
        scores = {}
        for mname, fac in factories.items():
            print(f"=== {ds} {mname} ===", flush=True)
            met, _ = eval_model(mname, fac, Xtr, ytr, Xte, yte)
            met["dataset"] = ds
            met["fe_variant"] = fe_variant
            preds[mname] = np.array(met["y_pred"])
            scores[mname] = np.array(met["y_score"])
            slim = {k: v for k, v in met.items() if k not in {"y_score", "y_pred"}}
            rows.append(slim)
        # OR/majority of the two boosters that actually finished (CatBoost skipped)
        boosters = ["XGBoost", "LightGBM"]
        or_pred = disjunctive_vote([preds[n] for n in boosters])
        maj_pred = majority_vote([preds[n] for n in boosters])
        or_score = np.vstack([scores[n] for n in boosters]).max(axis=0)
        maj_score = np.vstack([scores[n] for n in boosters]).mean(axis=0)
        for label, pred, sc in [
            ("boost_or_vote", or_pred, or_score),
            ("boost_majority", maj_pred, maj_score),
        ]:
            met = classification_metrics(yte, pred, sc)
            met["seconds"] = 0.0
            met["threshold"] = None
            met["model"] = label
            met["dataset"] = ds
            met["fe_variant"] = fe_variant
            rows.append(met)
        score_bank[ds] = {"y_true": yte.tolist(), "scores": {k: v.tolist() for k, v in scores.items()}}

    rows.append(
        {
            "dataset": "all",
            "model": "CatBoost",
            "fe_variant": fe_variant,
            "skipped": True,
            "reason": "CatBoost exceeded 15 min per dataset on this Windows CPU; not a silent drop",
        }
    )
    (TABLES_DIR / "advanced_results.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    (TABLES_DIR / "advanced_scores.json").write_text(json.dumps(score_bank), encoding="utf-8")
    df = pd.DataFrame(rows)
    cols = [c for c in ["dataset", "model", "recall", "precision", "f1", "pr_auc", "seconds", "skipped"] if c in df.columns]
    print(df[cols].to_string(index=False))
    return df


if __name__ == "__main__":
    # SMOTE won Step 4 by ~0.001 PR-AUC but doubled n and stalled CatBoost (~20 min on D2).
    # Step 5 uses scaled_raw + class weights; that is the EDA-preferred imbalance handle.
    variant = "scaled_raw"
    print("using FE variant", variant, "(SMOTE skipped: runtime fallback, 0.001 PR-AUC gap)")
    run_advanced(fe_variant=variant)
