"""Paper Case study 1+2: individual models, majority vote, disjunctive vote.

Two modes
---------
paper_style : scale+PCA (+ optional DBSCAN loop) on the FULL subset, then split.
leakage_safe: split first; fit transforms on train only.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RANDOM_STATE, TABLES_DIR, TEST_SIZE
from src.data import features_and_target, load_cleaned, load_subset
from src.dbscan_augment import augment_full, augment_leakage_safe
from src.evaluate import classification_metrics
from src.models import eval_votes, fit_eval, paper_knn, paper_rf, paper_svm
from src.preprocess import ScalePca


PAPER_TABLES = {
    "case1": {
        "D1": {
            "RandomForest": {"recall": 0.78, "f1": 0.87, "precision": 0.98},
            "KNN": {"recall": 0.68, "f1": 0.81, "precision": 1.00},
            "SVM": {"recall": 0.75, "f1": 0.86, "precision": 1.00},
            "majority": {"recall": 0.79, "f1": 0.88, "precision": 1.00},
        },
        "D2": {
            "RandomForest": {"recall": 0.91, "f1": 0.95, "precision": 1.00},
            "KNN": {"recall": 0.81, "f1": 0.90, "precision": 1.00},
            "SVM": {"recall": 0.90, "f1": 0.95, "precision": 1.00},
            "majority": {"recall": 0.92, "f1": 0.96, "precision": 1.00},
        },
        "D3": {
            "RandomForest": {"recall": 0.92, "f1": 0.96, "precision": 0.99},
            "KNN": {"recall": 0.87, "f1": 0.93, "precision": 1.00},
            "SVM": {"recall": 0.91, "f1": 0.95, "precision": 1.00},
            "majority": {"recall": 0.93, "f1": 0.97, "precision": 1.00},
        },
    },
    "dve": {
        "D1": {"recall": 0.988, "f1": 0.994, "precision": 1.00},
        "D2": {"recall": 0.994, "f1": 0.997, "precision": 1.00},
        "D3": {"recall": 0.995, "f1": 0.998, "precision": 1.00},
    },
}


def _row(dataset, mode, augment, model, metrics: dict, paper: dict | None = None) -> dict:
    row = {
        "dataset": dataset,
        "mode": mode,
        "augment": augment,
        "model": model,
        **{k: metrics.get(k) for k in ("precision", "recall", "f1", "pr_auc", "roc_auc", "mcc", "tn", "fp", "fn", "tp", "accuracy", "seconds")},
    }
    if paper:
        row["paper_recall"] = paper.get("recall")
        row["paper_f1"] = paper.get("f1")
        row["paper_precision"] = paper.get("precision")
        if metrics.get("recall") is not None and paper.get("recall") is not None:
            row["recall_gap"] = float(metrics["recall"]) - float(paper["recall"])
    return row


def _fit_three(X_tr, y_tr, X_te, y_te, svm_linear: bool) -> tuple[dict, dict, dict]:
    results = {}
    preds = {}
    scores = {}
    for name, est in [
        ("RandomForest", paper_rf()),
        ("KNN", paper_knn()),
        ("SVM", paper_svm(linear_fallback=svm_linear)),
    ]:
        fit = fit_eval(name, est, X_tr, y_tr, X_te, y_te)
        results[name] = fit
        preds[name] = fit.y_pred
        if fit.y_score is not None:
            scores[name] = fit.y_score
    votes = eval_votes(preds, y_te, scores)
    return results, votes, preds


def run_one(
    name: str,
    mode: str,
    n_dbscan_iter: int,
    svm_linear: bool = False,
    use_pca: bool = True,
) -> list[dict]:
    """Run one subset × mode. ``n_dbscan_iter=0`` is Case study 1 (no augmentation)."""
    raw = load_cleaned()
    rows = []
    augment_tag = f"dbscan_x{n_dbscan_iter}" if n_dbscan_iter else "none"
    t0 = time.perf_counter()

    if mode == "paper_style":
        frame = load_subset(name, df=raw)
        X, y = features_and_target(frame)
        pipe = ScalePca(use_pca=use_pca, random_state=RANDOM_STATE)
        Z = pipe.fit_transform(X)
        if n_dbscan_iter:
            Z, hist = augment_full(Z, n_iterations=n_dbscan_iter)
        else:
            hist = []
        from sklearn.model_selection import train_test_split

        Z_tr, Z_te, y_tr, y_te = train_test_split(
            Z, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
        )
    elif mode == "leakage_safe":
        train = load_subset(name, df=raw, split="train")
        test = load_subset(name, df=raw, split="test")
        X_tr, y_tr = features_and_target(train)
        X_te, y_te = features_and_target(test)
        pipe = ScalePca(use_pca=use_pca, random_state=RANDOM_STATE)
        Z_tr = pipe.fit_transform(X_tr)
        Z_te = pipe.transform(X_te)
        if n_dbscan_iter:
            Z_tr, Z_te, hist = augment_leakage_safe(Z_tr, Z_te, n_iterations=n_dbscan_iter)
        else:
            hist = []
    else:
        raise ValueError(mode)

    prep_s = time.perf_counter() - t0
    models, votes, _ = _fit_three(Z_tr, y_tr, Z_te, y_te, svm_linear=svm_linear)

    paper_case = PAPER_TABLES["case1"][name] if n_dbscan_iter == 0 else {}
    for mname, fit in models.items():
        fit.metrics["seconds"] = fit.seconds
        paper = paper_case.get(mname) if n_dbscan_iter == 0 else None
        rows.append(_row(name, mode, augment_tag, mname, fit.metrics, paper))

    for vote_name, met in votes.items():
        met["seconds"] = prep_s
        paper = None
        if n_dbscan_iter == 0 and vote_name == "majority":
            paper = paper_case.get("majority")
        if n_dbscan_iter > 0 and vote_name == "disjunctive":
            paper = PAPER_TABLES["dve"][name]
        rows.append(_row(name, mode, augment_tag, vote_name, met, paper))

    for row in rows:
        row["prep_seconds"] = prep_s
        row["dbscan_history"] = hist
        row["svm_linear"] = svm_linear
        row["use_pca"] = use_pca
        row["n_train"] = int(len(y_tr))
        row["n_test"] = int(len(y_te))
    return rows


def choose_iterations(max_iter: int = 3) -> dict:
    """PLAN A2: try 1/2/3 DBSCAN iterations on paper-style D1 DVE, pick closest recall to 0.988."""
    target = PAPER_TABLES["dve"]["D1"]["recall"]
    tried = []
    for n in range(1, max_iter + 1):
        rows = run_one("D1", "paper_style", n_dbscan_iter=n, svm_linear=False)
        dve = next(r for r in rows if r["model"] == "disjunctive")
        tried.append(
            {
                "n_iterations": n,
                "recall": dve["recall"],
                "f1": dve["f1"],
                "abs_gap": abs(dve["recall"] - target),
            }
        )
    best = min(tried, key=lambda d: d["abs_gap"])
    return {"target_recall": target, "tried": tried, "chosen": best["n_iterations"]}


def _save_rows(all_rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(all_rows)
    keep = [c for c in df.columns if c != "dbscan_history"]
    df[keep].to_json(TABLES_DIR / "baseline_results.json", orient="records", indent=2)
    df.to_pickle(TABLES_DIR / "baseline_results.pkl")
    return df


def run_baseline(
    datasets: tuple[str, ...] = ("D1", "D2", "D3"),
    n_dbscan_iter: int | None = None,
    skip_d3_rbf: bool | None = None,
) -> pd.DataFrame:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    pick_path = TABLES_DIR / "dbscan_iteration_choice.json"
    if n_dbscan_iter is None:
        choice = choose_iterations(3)
        pick_path.write_text(json.dumps(choice, indent=2), encoding="utf-8")
        n_dbscan_iter = int(choice["chosen"])
        print("A2 iteration choice:", choice)
    else:
        pick_path.write_text(json.dumps({"chosen": n_dbscan_iter, "note": "forced"}, indent=2), encoding="utf-8")

    all_rows: list[dict] = []
    d3_linear = bool(skip_d3_rbf)
    for name in datasets:
        svm_linear = bool(d3_linear and name == "D3")
        for mode in ("paper_style", "leakage_safe"):
            print(f"=== {name} {mode} no-aug svm_linear={svm_linear} ===", flush=True)
            all_rows.extend(run_one(name, mode, n_dbscan_iter=0, svm_linear=svm_linear))
            print(f"=== {name} {mode} dbscan x{n_dbscan_iter} ===", flush=True)
            all_rows.extend(run_one(name, mode, n_dbscan_iter=n_dbscan_iter, svm_linear=svm_linear))
            _save_rows(all_rows)
            # PLAN A6: if D1/D2 RBF SVM is already slow, do not start D3 RBF.
            if skip_d3_rbf is None and name in {"D1", "D2"}:
                svm_times = [r["seconds"] for r in all_rows if r["model"] == "SVM" and r.get("seconds")]
                if svm_times and max(svm_times) > 180:
                    d3_linear = True
                    print("A6: SVM RBF already >180s; D3 will use LinearSVC", flush=True)

    df = _save_rows(all_rows)
    print(df[["dataset", "mode", "augment", "model", "recall", "f1", "precision", "seconds"]].to_string(index=False))
    return df


if __name__ == "__main__":
    run_baseline()
