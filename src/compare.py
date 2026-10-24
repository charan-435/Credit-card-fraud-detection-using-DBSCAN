"""Step 6: 5-fold CV leaderboard, curves, importances."""

from __future__ import annotations

import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import StratifiedKFold

from src.advanced import make_lgb, make_mlp, make_xgb
from src.config import PLOTS_DIR, RANDOM_STATE, TABLES_DIR
from src.data import features_and_target, load_cleaned, load_named_subsets
from src.evaluate import classification_metrics
from src.preprocess import ScalePca

sns.set_theme(style="whitegrid")


def _cv_model(factory, X, y, n_splits=5) -> dict:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_rows = []
    t0 = time.perf_counter()
    for i, (tr, te) in enumerate(skf.split(X, y), start=1):
        pipe = ScalePca(use_pca=False, random_state=RANDOM_STATE)
        X_tr = pipe.fit_transform(X.iloc[tr] if hasattr(X, "iloc") else X[tr])
        X_te = pipe.transform(X.iloc[te] if hasattr(X, "iloc") else X[te])
        y_tr = y.iloc[tr] if hasattr(y, "iloc") else y[tr]
        y_te = y.iloc[te] if hasattr(y, "iloc") else y[te]
        model = factory(np.asarray(y_tr))
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        if hasattr(model, "predict_proba"):
            score = model.predict_proba(X_te)[:, 1]
        else:
            score = np.asarray(model.decision_function(X_te)).ravel()
        met = classification_metrics(y_te, pred, score)
        met["fold"] = i
        fold_rows.append(met)
    seconds = time.perf_counter() - t0
    df = pd.DataFrame(fold_rows)
    keys = ["recall", "precision", "f1", "pr_auc", "roc_auc", "mcc"]
    out = {f"{k}_mean": float(df[k].mean()) for k in keys}
    out.update({f"{k}_std": float(df[k].std(ddof=1)) for k in keys})
    out["seconds"] = seconds
    out["folds"] = fold_rows
    return out


def run_compare(datasets=("D1", "D2", "D3")) -> pd.DataFrame:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (PLOTS_DIR / "compare").mkdir(parents=True, exist_ok=True)
    raw = load_cleaned()
    subsets = load_named_subsets(raw)
    lgb_params = {}
    p = TABLES_DIR / "lgb_best_params.json"
    if p.exists():
        lgb_params = json.loads(p.read_text(encoding="utf-8"))

    rows = []
    curve_payload = {}
    importances = None
    for ds in datasets:
        frame = subsets[ds]
        X, y = features_and_target(frame)
        factories = {
            "XGBoost": lambda yy: make_xgb(yy),
            "LightGBM": lambda yy, lp=lgb_params: make_lgb(yy, **lp),
            "MLP": lambda yy: make_mlp(),
        }
        for mname, fac in factories.items():
            print(f"CV {ds} {mname}", flush=True)
            stats = _cv_model(fac, X, y)
            stats["dataset"] = ds
            stats["model"] = mname
            slim = {k: v for k, v in stats.items() if k != "folds"}
            rows.append(slim)

        # holdout curves from frozen split using LightGBM
        from src.data import load_subset

        train = load_subset(ds, df=raw, split="train")
        test = load_subset(ds, df=raw, split="test")
        Xtr, ytr = features_and_target(train)
        Xte, yte = features_and_target(test)
        pipe = ScalePca(use_pca=False, random_state=RANDOM_STATE)
        Ztr, Zte = pipe.fit_transform(Xtr), pipe.transform(Xte)
        model = make_lgb(ytr, **lgb_params)
        model.fit(Ztr, ytr)
        score = model.predict_proba(Zte)[:, 1]
        pred = model.predict(Zte)
        curve_payload[ds] = {
            "y_true": yte.to_numpy().tolist(),
            "y_score": score.tolist(),
            "y_pred": pred.tolist(),
        }
        if ds == "D1":
            importances = pd.Series(model.feature_importances_, index=list(X.columns)).sort_values(ascending=False)

        p_c, r_c, _ = precision_recall_curve(yte, score)
        fpr, tpr, _ = roc_curve(yte, score)
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
        axes[0].plot(r_c, p_c)
        axes[0].set_xlabel("recall")
        axes[0].set_ylabel("precision")
        axes[0].set_title(f"{ds} LightGBM PR curve")
        axes[1].plot(fpr, tpr)
        axes[1].set_xlabel("FPR")
        axes[1].set_ylabel("TPR")
        axes[1].set_title(f"{ds} LightGBM ROC")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "compare" / f"{ds.lower()}_pr_roc.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

        cm = confusion_matrix(yte, pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(3.8, 3.4))
        ConfusionMatrixDisplay(cm, display_labels=["legit", "fraud"]).plot(ax=ax, colorbar=False)
        ax.set_title(f"{ds} LightGBM confusion (holdout)")
        fig.savefig(PLOTS_DIR / "compare" / f"{ds.lower()}_cm.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    if importances is not None:
        fig, ax = plt.subplots(figsize=(6, 5))
        importances.head(15).iloc[::-1].plot.barh(ax=ax, color="#4c78a8")
        ax.set_title("LightGBM gain importance (D1)")
        fig.savefig(PLOTS_DIR / "compare" / "lgb_importance_d1.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        importances.to_json(TABLES_DIR / "lgb_importance_d1.json")

    (TABLES_DIR / "cv_results.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    (TABLES_DIR / "holdout_scores.json").write_text(json.dumps(curve_payload), encoding="utf-8")
    df = pd.DataFrame(rows)
    print(df[["dataset", "model", "recall_mean", "recall_std", "pr_auc_mean", "pr_auc_std", "seconds"]].to_string(index=False))
    return df


if __name__ == "__main__":
    run_compare()
