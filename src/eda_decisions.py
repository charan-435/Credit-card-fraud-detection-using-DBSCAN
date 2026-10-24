"""Build the findings → decision table from eda_summary.json."""

from __future__ import annotations

import json

import pandas as pd

from src.config import TABLES_DIR


def build_decisions(summary: dict) -> pd.DataFrame:
    pr = summary["probe"]["logreg"]["pr_auc"]
    tree_pr = summary["probe"]["tree_d3"]["pr_auc"]
    db = summary["dbscan"]
    max_corr = summary["max_feature_corr"]
    amt_auc = summary["amount_auc"]
    kdist_med = summary["kdist_median"]

    if summary["pca_n_for_95"] >= 24:
        pca_decision = (
            "skip extra PCA in our methods (keep it only in the paper replica); "
            "95% variance still needs almost all components"
        )
    elif max_corr < 0.5:
        pca_decision = "skip extra PCA in our methods (keep it only in the paper replica)"
    else:
        pca_decision = "keep a 95% PCA option; leftover correlation is real"

    if abs(summary["amount_skew"]) < 0.2 and amt_auc < 0.6:
        amt_decision = "standardise Amount; skip log1p (skew ~0) and treat Amount as a weak feature"
    elif amt_auc < 0.65:
        amt_decision = "keep Amount standardised; also try log1p as an FE variant"
    else:
        amt_decision = "Amount is informative — keep raw and log forms"

    if db["fraud_noise_rate"] >= 0.4:
        db_decision = (
            "do not rely on DBSCAN labels as the main signal; "
            "try IsolationForest/LOF scores (fraud often looks like noise)"
        )
    elif db["n_clusters"] <= 2:
        db_decision = "DBSCAN with eps=2 is too coarse. Keep it as a paper replica only; try KMeans distances"
    else:
        db_decision = "keep DBSCAN labels as one FE candidate; also try KMeans distances"

    if pr >= 0.9:
        model_decision = (
            "trees/boosting as the fair FE workhorse; skip autoencoders "
            "(linear model already saturates); still run XGBoost/LightGBM/CatBoost "
            "and sklearn MLP for the paper's missing-baseline gap"
        )
        imb_decision = "class weights / scale_pos_weight first; SMOTE is optional because features already separate"
        ae_decision = "skip autoencoder reconstruction error"
    else:
        model_decision = "boosting + MLP + anomaly scores; autoencoder is in scope"
        imb_decision = "compare class weights vs SMOTE/ADASYN"
        ae_decision = "try autoencoder reconstruction error"

    rows = [
        {
            "finding": "Raw file is 50/50; D1-D3 are 1% by construction",
            "decision": "treat paper imbalance-handling claims as methods on a synthetic 1% sample",
            "reason": "cannot claim real-world rarity from this file",
        },
        {
            "finding": (
                f"Amount scale >> V features (mean {summary['amount_mean']:.0f} vs |V| "
                f"{summary['v_mean_abs']:.2f}); skew={summary['amount_skew']:.3f}"
            ),
            "decision": amt_decision,
            "reason": "KNN/DBSCAN/SVM are distance-based",
        },
        {
            "finding": f"Max |feature corr| = {max_corr:.3f}; PCA-95 needs {summary['pca_n_for_95']} components",
            "decision": pca_decision,
            "reason": "V1-V28 are already PCA features; a second PCA is only for paper fidelity",
        },
        {
            "finding": (
                f"Top univariate AUC is {summary['top_auc'][0]['feature']}="
                f"{summary['top_auc'][0]['auc']:.3f}; Amount AUC={amt_auc:.3f}"
            ),
            "decision": "use top-MI / top-AUC selection as one FE variant; do not drop Amount until measured",
            "reason": "a few V columns carry most of the signal",
        },
        {
            "finding": (
                f"IQR outlier share: fraud {summary['outlier_frac_fraud']:.2f} vs "
                f"legit {summary['outlier_frac_legit']:.2f}"
            ),
            "decision": "include IsolationForest and LOF scores as FE candidates",
            "reason": "if fraud lives in tails, anomaly scores beat cluster ids",
        },
        {
            "finding": (
                f"DBSCAN eps=2 on D1 train: {db['n_clusters']} clusters, noise={db['noise_frac']:.3f}, "
                f"fraud-in-noise={db['fraud_noise_rate']:.3f}; k-dist median={kdist_med:.3f}"
            ),
            "decision": db_decision,
            "reason": "paper hyperparameters may not match the k-distance knee",
        },
        {
            "finding": f"Leakage-safe logreg PR-AUC={pr:.3f}, depth-3 tree PR-AUC={tree_pr:.3f}",
            "decision": model_decision,
            "reason": "a shallow probe is the honesty check the paper skipped",
        },
        {
            "finding": "1% class rate after sampling",
            "decision": imb_decision,
            "reason": "compare rebalancing to DBSCAN augmentation with the same model",
        },
        {
            "finding": f"Near-linear separability probe PR-AUC={pr:.3f}",
            "decision": ae_decision,
            "reason": "PLAN: skip AE if 2.10 already shows near-linear separation",
        },
        {
            "finding": "Paper uses RF(n=3), KNN(k=3), RBF-SVM",
            "decision": "replicate those as baseline only; Step 5 uses XGBoost, LightGBM, CatBoost, MLP",
            "reason": "n_estimators=3 is an unusually weak forest; boosting is the missing strong baseline",
        },
        {
            "finding": "PyTorch DLL failed to load on this Windows Anaconda env",
            "decision": "use sklearn MLPClassifier with early stopping instead of PyTorch",
            "reason": "do not silently drop DL; document the fallback",
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    summary = json.loads((TABLES_DIR / "eda_summary.json").read_text(encoding="utf-8"))
    decisions = build_decisions(summary)
    out = TABLES_DIR / "eda_decisions.json"
    decisions.to_json(out, orient="records", indent=2)
    print(decisions.to_string(index=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
