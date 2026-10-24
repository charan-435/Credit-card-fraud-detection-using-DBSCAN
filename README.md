# Credit-card fraud detection (DBSCAN-augmented DVE)

Reproduction and extension of Ghalwash et al., Scientific Reports (2025) 15:39754. Design is in [PLAN.md](PLAN.md). Numbers below are **computed in this repo**, not copied from the paper.

## Setup

Place Kaggle **Credit Card Fraud Detection Dataset 2023** at `creditcard_2023.csv` in the repo root (~310 MB, gitignored).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = "."
```

## How to run

```powershell
python -m src.data          # D1/D2/D3 ids (already committed)
python -m src.eda           # plots + eda_summary.json
python -m src.eda_decisions
python -m src.baseline      # paper vs leakage-safe
python -m src.fe_eval
python -m src.advanced
python -m src.compare
python -m src.runtime_plot
```

Open the single notebook `notebooks/pipeline.ipynb` and run all cells. It only displays saved JSON/plots; it does not retrain. First CSV load is cached as parquet under `results/cache/` (gitignored).

## Branch map

| Branch | Status |
|---|---|
| `main` | PLAN.md + .gitignore only |
| `dev` | all finished steps merged |
| `step-01-setup` | skeleton, frozen D1/D2/D3 |
| `step-02-eda` | visual EDA + decisions |
| `step-03-baseline-paper` | paper pipeline vs leakage-safe |
| `step-04-feature-engineering` | LightGBM FE bake-off |
| `step-05-models` | XGBoost / LightGBM / MLP |
| `step-06-final-comparison` | 5-fold CV + plots |
| `step-07-realistic-data` | skipped (no 2013 file) |

## Sampling

Raw file is **50/50** (284,315 / 284,315). Paper 1% rate is **constructed**. Nested seed `42`: D1 ⊂ D2 ⊂ D3 (40,400 / 80,800 / 121,200). Stratified 80/20. `Class == 1` is fraud.

## Results (short)

**The paper's Table 11 was not reproduced.** Closest paper-style DVE recall is 0.90 / 0.89 / 0.93 on D1/D2/D3 vs claimed 0.988 / 0.994 / 0.995. Leakage-safe DVE is similar, so fitting DBSCAN before the split does **not** explain the gap. Iterations 1–3 were tried (PLAN A2); 1 was closest.

The 2023 set is nearly separable: leakage-safe logistic PR-AUC **0.95**, ROC-AUC **0.998** on D1. V14 univariate AUC is 0.977. Amount does nothing (AUC 0.52). DBSCAN `eps=2` put **78% of D1-train fraud in noise**.

Holdout DVE (dbscan ×1):

| Set | Mode | Recall | Precision | F1 |
|---|---|---|---|---|
| D1 | paper-style | 0.900 | 0.923 | 0.911 |
| D1 | leakage-safe | 0.900 | 0.973 | 0.935 |
| D3 | paper-style | 0.925 | 0.945 | 0.935 |
| D3 | leakage-safe | 0.933 | 0.907 | 0.920 |

5-fold CV mean ± std (scaled features, class weights; default 0.5 threshold):

| Set | Model | Recall | PR-AUC |
|---|---|---|---|
| D1 | XGBoost | 0.925 ± 0.023 | 0.956 ± 0.016 |
| D3 | XGBoost | 0.943 ± 0.015 | 0.973 ± 0.011 |
| D3 | LightGBM | 0.935 ± 0.014 | 0.975 ± 0.010 |
| D3 | MLP | 0.898 ± 0.042 | 0.952 ± 0.027 |

Holdout with val-tuned F1 threshold: D1 XGBoost recall 0.913 / precision 0.973 / PR-AUC 0.976. Booster majority vote reached precision 1.0 at recall 0.913 (F1 0.954). OR-vote of boosters does **not** recover Table 11.

FE bake-off (same LightGBM): SMOTE won D1 PR-AUC by **0.001** over IsolationForest/DBSCAN labels (all ~0.97). Extra cluster features are not doing the work. SMOTE was not reused in Step 5 because it doubled `n` and stalled CatBoost.

**Fallbacks (not silent):** PyTorch DLL failed → sklearn MLP. SHAP imports torch → LightGBM gain importances. CatBoost >15 min/dataset on this CPU → skipped. 2013 ULB file absent → Step 7 skipped.

Accuracy is ~0.99 for a constant non-fraud predictor at 1% fraud; do not rank on it.

## Conclusion

Table 11's 0.995 recall does not hold under a faithful reimplementation on the same public file. This dataset is an easy, artificially imbalanced slice of a balanced 2023 dump. Boosting with class weights beats the paper's 3-tree RF + OR-vote without DBSCAN labels. That is not evidence of robustness on naturally rare, drifting fraud.
