"""Write notebooks/pipeline.ipynb — one simple notebook for the whole study."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "notebooks" / "pipeline.ipynb"

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}

cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip() + "\n"))


md(
    """# Credit-card fraud detection

Reproduce Ghalwash et al. (Scientific Reports 2025) — DBSCAN labels + RF/KNN/SVM + OR-vote — then try simpler, stronger models.

This notebook **displays saved results**. Heavy fitting already ran via `python -m src.eda` / `baseline` / `fe_eval` / `advanced` / `compare`. Run those first if a JSON file is missing.

1. Data and subsets
2. EDA
3. Paper baseline (with and without leakage)
4. Feature-engineering bake-off
5. XGBoost / LightGBM / MLP
6. Final comparison
7. 2013 data (skipped)
"""
)

md("## 0. Setup")

code(
    """
import json
import sys
from pathlib import Path

import pandas as pd
from IPython.display import Image, Markdown, display

ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATA_PATH, PLOTS_DIR, POSITIVE_LABEL, RANDOM_STATE, TABLES_DIR
from src.data import load_cleaned, load_named_subsets, subset_summary

pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 120)


def show_img(path, takeaway=None):
    path = Path(path)
    if not path.exists():
        print("missing", path)
        return
    display(Image(filename=str(path)))
    if takeaway:
        display(Markdown("**Takeaway.** " + takeaway))


def load_json(name):
    path = TABLES_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run the matching `python -m src.*` command first.")
    return json.loads(path.read_text(encoding="utf-8"))


print("root", ROOT)
print("csv exists", DATA_PATH.exists(), "| seed", RANDOM_STATE, "| fraud label", POSITIVE_LABEL)
"""
)

md(
    """## 1. Data and subsets

Paper uses three 1% fraud slices: D1=40,400, D2=80,800, D3=121,200. The Kaggle 2023 file is actually **50/50**. We build the 1% rate by nested sampling (D1 ⊂ D2 ⊂ D3), seed 42.
"""
)

code(
    """
raw = load_cleaned()
print("raw shape", raw.shape)
print(raw["Class"].value_counts())
print("nulls", int(raw.isnull().sum().sum()), "dups", int(raw.duplicated().sum()))

subsets = load_named_subsets(raw)
rows = []
for name, frame in subsets.items():
    info = subset_summary(frame)
    info["name"] = name
    rows.append(info)
display(pd.DataFrame(rows).set_index("name"))

ids = {n: set(subsets[n]["id"]) for n in ("D1", "D2", "D3")}
print("D1 inside D2", ids["D1"].issubset(ids["D2"]))
print("D2 inside D3", ids["D2"].issubset(ids["D3"]))
"""
)

md(
    """**Takeaway.** `Class == 1` is fraud. Drop-na/drop-duplicates does nothing here. The paper's "imbalanced European 2023 data" is a **constructed** 1% sample of a balanced file.
"""
)

md("## 2. EDA")

code(
    """
summary = load_json("eda_summary.json")
print("Amount mean", round(summary["amount_mean"], 1), "Amount AUC", round(summary["amount_auc"], 3))
print("top features", [(r["feature"], round(r["auc"], 3)) for r in summary["top_auc"][:5]])
print("logreg probe", summary["probe"]["logreg"])
print("DBSCAN", summary["dbscan"])
"""
)

md("### Class counts")
code(
    'show_img(PLOTS_DIR / "eda" / "01_class_counts.png", "Raw file is 50/50. D1–D3 are 1% only because we undersampled.")'
)

md("### Amount vs V-features")
code(
    'show_img(PLOTS_DIR / "eda" / "02_amount_dist.png", "Amount is a different scale (needs StandardScaler). Skew is ~0, so skip log1p.")'
)

md("### Which columns separate fraud")
code(
    """
rank = pd.read_json(TABLES_DIR / "eda_univariate.json")
display(rank.head(8))
show_img(PLOTS_DIR / "eda" / "03_univariate_auc.png", "V14 is already almost a detector (AUC 0.977). Amount is useless (AUC 0.52).")
show_img(PLOTS_DIR / "eda" / "04_univariate_mi.png", "MI agrees on the same handful of V columns.")
"""
)

md("### Top features by class")
code(
    """
show_img(PLOTS_DIR / "eda" / "05_top_kde.png", "Red/blue densities barely overlap. A linear model can do well — this set may be too easy.")
show_img(PLOTS_DIR / "eda" / "06_top_box.png", "Same story in boxplots: location shift, not just a few outliers.")
"""
)

md("### Correlation and a second PCA")
code(
    """
show_img(PLOTS_DIR / "eda" / "07_corr.png", "Some leftover correlation (max |r| ~0.70).")
show_img(PLOTS_DIR / "eda" / "14_pca_variance.png", f"95% variance still needs {summary['pca_n_for_95']} of 29 components. Extra PCA is cosmetic; keep it only in the paper replica.")
"""
)

md("### Outliers")
code(
    'show_img(PLOTS_DIR / "eda" / "08_outliers.png", "95% of fraud vs 49% of legit trip an IQR flag. Anomaly scores are more natural than cluster IDs.")'
)

md("### 2D views (8k subsample)")
code(
    """
show_img(PLOTS_DIR / "eda" / "09_pca2.png", "Even linear 2D already separates many fraud points.")
show_img(PLOTS_DIR / "eda" / "10_tsne2.png", "t-SNE also shows structure. Use as a picture, not as a feature.")
show_img(PLOTS_DIR / "eda" / "11_umap2.png", "UMAP agrees. Compact groups justify trying KMeans distances later.")
"""
)

md("### Is DBSCAN appropriate?")
code(
    """
show_img(PLOTS_DIR / "eda" / "12_kdistance.png", "Paper eps=2 sits near the median 4-distance, not a sharp knee. About 46% of points are farther than 2.")
show_img(PLOTS_DIR / "eda" / "13_dbscan_where.png", "78% of D1-train fraud is labelled noise. Noise is still ~98% legitimate. Cluster labels are a weak fraud signal.")
"""
)

md("### Too easy? Shallow probe (leakage-safe D1)")
code(
    """
display(pd.DataFrame(summary["probe"]).T)
display(Markdown(
    "**Takeaway.** Logistic PR-AUC is 0.95 and ROC-AUC is 0.998. Table 11's 0.995 recall is not surprising on this file, "
    "and it is not evidence the method works on messy real fraud. Skip autoencoders."
))
"""
)

md("### Findings → decisions")
code(
    """
decisions = pd.read_json(TABLES_DIR / "eda_decisions.json")
display(decisions)
"""
)

md(
    """## 3. Paper baseline

Same models as the paper: RF (`n_estimators=3`), KNN (`k=3`), SVM (RBF, `C=1`).

- **paper_style:** scale + PCA + DBSCAN on the **full** subset, then split (can leak).
- **leakage_safe:** split first; DBSCAN on train; test labels = nearest train point.

OR-vote (DVE) = fraud if **any** model says fraud. Majority = at least two. DBSCAN iterations tried: 1, 2, 3 on D1; **1** was closest to Table 11.
"""
)

code(
    """
choice = load_json("dbscan_iteration_choice.json")
print("iteration search", choice)

base = pd.DataFrame(load_json("baseline_results.json"))
keep = [c for c in ["dataset", "mode", "augment", "model", "precision", "recall", "f1", "pr_auc", "fp", "fn", "seconds", "paper_recall", "recall_gap"] if c in base.columns]
display(base[keep].sort_values(["dataset", "mode", "augment", "model"]))
"""
)

md("### Versus paper Table 11 (DVE)")
code(
    """
dve = base[(base.model == "disjunctive") & (base.augment.astype(str).str.contains("dbscan"))]
display(dve[[c for c in ["dataset", "mode", "recall", "precision", "f1", "pr_auc", "paper_recall", "recall_gap", "fp", "fn"] if c in dve.columns]])

gap = dve["recall_gap"].abs().min() if "recall_gap" in dve.columns else None
paper_mean = dve.loc[dve.mode == "paper_style", "recall"].mean()
safe_mean = dve.loc[dve.mode == "leakage_safe", "recall"].mean()
display(Markdown(
    f"**Takeaway.** Closest |recall gap| vs Table 11 is {gap:.3f}. "
    f"Mean DVE recall paper-style={paper_mean:.3f}, leakage-safe={safe_mean:.3f}. "
    "We did **not** reproduce 0.988–0.995. Leakage does not explain the gap — both modes look similar."
))
"""
)

md(
    """## 4. Feature engineering

Fair test: **same LightGBM**, leakage-safe, D1 first.

| EDA finding | Variant |
|---|---|
| Linear model already strong | skip autoencoder |
| Amount skew ~0 | skip log(Amount) |
| Extra PCA needs 26/29 components | no second PCA here |
| Fraud often looks like noise | IsolationForest, LOF |
| Some 2D blobs | KMeans id + distances |
| V14/V12/V4 dominate | top-8 AUC features |
| 1% rate is synthetic | SMOTE vs class weights |
| Paper idea | DBSCAN labels (train-only) |
"""
)

code(
    """
fe = load_json("fe_results.json")
display(Markdown(f"**Winner by D1 PR-AUC:** `{fe['winner']}`"))
display(pd.DataFrame(fe["rows"])[["dataset", "variant", "pr_auc", "recall", "precision", "f1", "recall_at_p95", "seconds"]])
display(Markdown(
    "**Takeaway.** SMOTE wins by about 0.001 over IsolationForest and DBSCAN labels. "
    "Everything sits in a 0.96–0.98 PR-AUC band. Cluster features are not doing heavy lifting. "
    "Step 5 therefore uses **scaled features + class weights** (SMOTE doubled n and stalled CatBoost)."
))
"""
)

md(
    """## 5. Stronger models

XGBoost, LightGBM (20 Optuna trials on D1 **validation** PR-AUC), sklearn MLP (PyTorch failed to load). Threshold: max F1 on val, freeze, apply to test once.

CatBoost hung >15 min per dataset on this CPU — skipped, not dropped quietly. OR/majority vote of XGBoost+LightGBM tests whether the paper's voting idea still helps.
"""
)

code(
    """
params = load_json("lgb_best_params.json")
adv = pd.DataFrame(load_json("advanced_results.json"))
display(Markdown(f"**Tuned LightGBM params:** `{params}`"))
cols = [c for c in ["dataset", "model", "fe_variant", "threshold", "precision", "recall", "f1", "pr_auc", "roc_auc", "mcc", "seconds", "skipped", "reason"] if c in adv.columns]
display(adv[cols])

d1 = adv[adv.dataset == "D1"].copy()
if "skipped" in d1.columns:
    d1 = d1[~d1["skipped"].fillna(False).astype(bool)]
d1 = d1.dropna(subset=["pr_auc"])
if len(d1):
    best = d1.sort_values("pr_auc", ascending=False).iloc[0]
    display(Markdown(f"**Takeaway.** Best D1 PR-AUC is **{best['model']}** ({best['pr_auc']:.3f}). Voting helps only if OR/majority beats that single model."))
"""
)

md(
    """## 6. Final comparison

5-fold CV mean ± std for XGBoost / LightGBM / MLP. Accuracy is misleading at 1% fraud (~0.99 by predicting all legit). SHAP needs torch here, so we show LightGBM gain importances.
"""
)

code(
    """
cv = pd.DataFrame(load_json("cv_results.json"))
show = cv.copy()
for m in ["recall", "precision", "f1", "pr_auc", "roc_auc", "mcc"]:
    show[m] = show[f"{m}_mean"].map(lambda x: f"{x:.3f}") + " ± " + show[f"{m}_std"].map(lambda x: f"{x:.3f}")
display(Markdown("**5-fold CV**"))
display(show[["dataset", "model", "recall", "precision", "f1", "pr_auc", "roc_auc", "mcc", "seconds"]])

display(Markdown("**Paper DVE holdout**"))
display(dve[[c for c in ["dataset", "mode", "recall", "precision", "f1", "pr_auc", "fp", "fn"] if c in dve.columns]])
"""
)

code(
    """
plot_dir = PLOTS_DIR / "compare"
for name, note in [
    ("d1_pr_roc.png", "D1 LightGBM PR / ROC"),
    ("d2_pr_roc.png", "D2"),
    ("d3_pr_roc.png", "D3"),
    ("d1_cm.png", "D1 confusion (holdout)"),
    ("d2_cm.png", "D2"),
    ("d3_cm.png", "D3"),
    ("lgb_importance_d1.png", "V14 / V12 / V4 dominate, same as EDA"),
    ("runtime.png", "Boosting is cheaper than the paper SVM on D3"),
]:
    p = plot_dir / name
    if p.exists():
        display(Markdown(f"**{note}**"))
        show_img(p)
"""
)

md(
    """## 7. Realistic 2013 data — skipped

PLAN said: only run on the ULB `creditcard.csv` (~0.17% fraud) if the file is present. It is not. I did not download it. Drop `data/creditcard.csv` in if you want this check later.
"""
)

code(
    """
candidates = [ROOT / "data" / "creditcard.csv", ROOT / "creditcard.csv"]
print({str(p): p.exists() for p in candidates})
print("Step 7 skipped." if not any(p.exists() for p in candidates) else "File found.")
"""
)

md(
    """## Conclusion

- **Table 11 does not reproduce.** Our DVE recall is about 0.90–0.93, not 0.988–0.995. Paper-style and leakage-safe are close, so leakage is not the main story.
- **This 2023 Kaggle file is too easy.** It is balanced until we force 1% fraud. A logistic probe already has PR-AUC 0.95.
- **DBSCAN labels are a weak extra feature.** Most fraud is noise; most noise is still legit.
- **XGBoost / LightGBM with class weights beat the paper's 3-tree RF + OR-vote** without needing cluster labels.
- **Limits:** synthetic 1% sample, no drift, no 2013 file, sklearn MLP instead of PyTorch, CatBoost skipped on this CPU.

Do not copy 0.995 into a report as *our* number. Use the tables above.
"""
)

nb.cells = cells
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print("wrote", out, "cells", len(cells))
