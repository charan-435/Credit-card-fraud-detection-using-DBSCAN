# Plan: Reproduce and improve DBSCAN-augmented DVE fraud detection

Paper: Ghalwash et al., *Enhancing credit card fraud detection using DBSCAN-augmented disjunctive voting ensemble*, Scientific Reports (2025) 15:39754.

This plan is written **before any code**. No notebooks, `src/`, or training until you approve it.

---

## What is already on disk

| File | Size | Notes |
|---|---|---|
| `creditcard_2023.csv` | 310 MB | Kaggle 2023 set. **Do not commit** (GitHub 100 MB limit). |
| `s41598-025-22960-w.pdf` | 2 MB | The paper (CC BY 4.0). Keep local; optional in git. |

Quick local check of the CSV (not a full EDA):

- Shape: **568,630 × 31** (`id`, `V1`–`V28`, `Amount`, `Class`)
- Class counts: **284,315 / 284,315** (exactly 50/50)
- Nulls: 0, duplicate rows: 0

This already disagrees with the paper in two places:

1. Paper says “~550,000” records; the file has **568,630**.
2. Paper treats the source as naturally imbalanced at 1% fraud. This Kaggle 2023 file is **balanced**. The 1% rate in D1/D2/D3 is **constructed by undersampling fraud**, not found in the raw file.

Git state: `main` exists, **no commits yet**. Dataset and paper are untracked.

---

## Paper method (what we will treat as the baseline)

### Data construction

From a 242,400-row extraction, three **1% fraud** subsets:

| Subset | Non-fraud | Fraud | Total | Test size (80/20) per paper |
|---|---|---|---|---|
| D1 | 40,000 | 400 | 40,400 | 8,000 non-fraud + 80 fraud |
| D2 | 80,000 | 800 | 80,800 | 16,000 + 160 |
| D3 | 120,000 | 1,200 | 121,200 | 24,000 + 240 (paper writes 24,001) |

### Pipeline (as described)

1. Drop missing values / duplicates.
2. `StandardScaler`.
3. PCA (again, even though V1–V28 are already PCA features).
4. **DBSCAN augmentation loop** (`eps=2`, `min_samples=4`): fit DBSCAN → append cluster labels as a feature → refit DBSCAN on the augmented matrix → “repeat multiple times”.
5. **Then** 80/20 split. Models: RF (`n_estimators=3`, `max_depth=None`), KNN (`k=3`), SVM (RBF, `C=1.0`).
6. Majority vote (Case study 1, original data) and **disjunctive / OR vote** (Case study 2, augmented data): fraud if **any** of RF/KNN/SVM predicts fraud.

### Numbers we must try to match

| Table | Setting | Recall (fraud) | F1 | Precision |
|---|---|---|---|---|
| 3 | D1, original, RF / KNN / SVM / majority | 0.78 / 0.68 / 0.75 / 0.79 | 0.87 / 0.81 / 0.86 / 0.88 | ~1.0 |
| 5 | D2, original, RF / KNN / SVM / majority | 0.91 / 0.81 / 0.90 / 0.92 | 0.95 / 0.90 / 0.95 / 0.96 | 1.0 |
| 7 | D3, original, RF / KNN / SVM / majority | 0.92 / 0.87 / 0.91 / 0.93 | 0.96 / 0.93 / 0.95 / 0.97 | ~1.0 |
| 9 | Majority across D1–D3 | 0.79 / 0.92 / 0.93 | 0.88 / 0.96 / 0.97 | 1.0 |
| 11 | DVE on augmented D1–D3 | **0.988 / 0.994 / 0.995** | **0.994 / 0.997 / 0.998** | 1.0 |

Table 2: DBSCAN `(eps=2, min_samples=4)` is their chosen setting (runtime 12s / 51s / 185s on D1/D2/D3).

Paper tables label fraud as class `(0)`. In this CSV, **`Class == 1` is fraud**. We will keep `Class == 1` as the positive class and note the paper’s inverted label in the baseline notebook.

---

## Ambiguities (do not silently “fix”)

These are under-specified in the paper. Each has a **default** and a **fallback**. If a default fails or looks wrong, I will stop and propose the fallback instead of changing the plan quietly.

| # | Ambiguity | Default | Fallback |
|---|---|---|---|
| A1 | How were D1/D2/D3 sampled from the balanced 568k rows? | Nested stratified sample, seed `42`: draw D3 first (120k non-fraud + 1.2k fraud), then D2 ⊂ D3, then D1 ⊂ D2. Matches “expanding dataset size”. | Independent samples of each size. Compare both if nested results are far from Tables 3/5/7. |
| A2 | How many DBSCAN iterations? Paper says “multiple times”. | Try **1, 2, 3** iterations on D1 (cheap). Use the count whose DVE recall is closest to Table 11, and freeze that count for D2/D3. | If none is close, report all three and do not chase the paper number. |
| A3 | DBSCAN before or after the split? Figure 1: augment **then** split. | Paper-style = fit on the **full** subset (leakage). Also run leakage-safe = fit on **train only**, assign test points via nearest train core-point / `kneighbors`. | If sklearn has no clean “predict” for DBSCAN, document the exact assignment rule we use. |
| A4 | Is PCA applied to `Amount` too? Drop `id`? | Always drop `id`. Scale `V1`–`V28` + `Amount`, then PCA. Keep enough components to explain ≥95% variance **or** `n_components = n_features` if the paper’s intent is “run PCA” rather than reduce. | If EDA shows Amount is already on a different scale and PCA hurts, keep a no-PCA ablation. |
| A5 | Positive class in metrics | `Class == 1` (fraud). | None; paper’s `(0)` is treated as a writing inconsistency. |
| A6 | SVM on D3 + 5-fold CV | Paper-style: sklearn `SVC(kernel="rbf", C=1.0)` even if slow. For our later CV comparison, cap SVM at D1/D2 or use a 50k train subsample on D3 and say so. | Linear SVM / `NuSVC` only if RBF cannot finish in ~30 min; flag it. |
| A7 | 2013 ULB dataset for Step 7 | Use it **only if present** under `data/` (or you add it). | Skip Step 7 and say so. |

---

## Hard rules (every later step)

1. **No leakage in our methods.** Scaler, PCA, clusterers, anomaly models, SMOTE, and threshold tuning fit on **train** (or inner CV train) only. Paper-style leakage is isolated to Step 3a and labelled as such.
2. **Seeds.** `RANDOM_STATE = 42` everywhere (numpy, sklearn, torch, sampling).
3. **Splits.** Stratified 80/20. Final comparison uses stratified **5-fold CV**, mean ± std.
4. **Metrics.** Recall, Precision, F1, PR-AUC, ROC-AUC, MCC, confusion matrix. Accuracy is reported but marked **misleading** at 1% fraud (~99% by predicting all non-fraud).
5. **EDA → decisions.** Every modelling choice after Step 2 is written as `EDA finding → decision → reason`.
6. **Do not hardcode paper claims as our results.** Always compute.

---

## Git workflow

`main` stays empty of experiment code. Work lands on `dev`.

```
main          # .gitignore + this PLAN.md only (after you approve)
  └─ dev      # integration branch
       ├─ step-01-setup
       ├─ step-02-eda
       ├─ step-03-baseline-paper
       ├─ step-04-feature-engineering
       ├─ step-05-models
       ├─ step-06-final-comparison
       └─ step-07-realistic-data   # only if 2013 data exists
```

Per step:

1. Branch from current `dev`: `git checkout -b step-0N-...`
2. Implement, commit with a why-focused message.
3. Merge into `dev` (`git checkout dev && git merge step-0N-...`).
4. Do **not** merge to `main` unless you ask.

`.gitignore` will exclude `creditcard_2023.csv`, `data/*.csv`, notebooks checkpoints, `__pycache__/`, `.venv/`, model dumps, Optuna db files.

First commit on `main` (after approval): `.gitignore` + `PLAN.md` only. Then create `dev` from `main`.

---

## Repo layout (created from Step 1 onward)

```
PLAN.md
README.md
requirements.txt
.gitignore
creditcard_2023.csv          # local only, gitignored
s41598-025-22960-w.pdf      # local, optional
notebooks/
  pipeline.ipynb             # single notebook for all steps
src/
  config.py                  # seeds, paths, DBSCAN/model defaults
  data.py                    # load, subset D1/D2/D3, stratified split
  preprocess.py              # scaler / PCA fit on train
  dbscan_augment.py          # paper-style + leakage-safe
  features.py                # cluster / anomaly / selection helpers
  models.py                  # RF/KNN/SVM, boosting, MLP, voting
  evaluate.py                # metrics, CV, plots
  viz.py                     # EDA plotting helpers
results/                     # tables/plots written by notebooks (gitkeep)
```

Environment: Python 3.11+, `requirements.txt` with pinned versions (pandas, numpy, scikit-learn, matplotlib, seaborn, scipy, umap-learn, xgboost, lightgbm, catboost, imbalanced-learn, optuna, shap, torch, jupyter, ipykernel). If a package fails to install on Windows, I will say so and propose a fallback.

---

## Steps

### Step 1 — Setup (`step-01-setup`)

**Goal.** Reproducible project skeleton and exact D1/D2/D3 construction.

**Work**

- `requirements.txt`, `.gitignore`, `src/config.py`, `src/data.py`, `src/evaluate.py` (split + metric helpers only).
- Loader that reads `creditcard_2023.csv`, drops `id`, documents class mapping.
- Build D1/D2/D3 with the nested sampling rule (A1). Persist subset **index files** (row ids) under `results/subsets/` so later steps reuse the same rows.
- Stratified 80/20 split utility.
- Notebook `01_setup_and_subsets.ipynb`: print counts, fraud rate, overlap D1⊂D2⊂D3, and the sampling description.
- Short README stub: how to create the venv and launch Jupyter.

**Decision criteria (must pass before merge)**

- [ ] D1/D2/D3 counts match the paper exactly.
- [ ] Fraud rate is 1.0% in each.
- [ ] Nested inclusion holds (or independent sampling is documented if we switched).
- [ ] Same seed reproduces the same indices.
- [ ] CSV is gitignored; `python -c "from src.data import load_raw"` works.

**Stop / ask you if:** the CSV path is wrong, or pandas cannot load the 310 MB file.

---

### Step 2 — EDA (`step-02-eda`)

**Goal.** Understand this dataset well enough that later methods are justified, not defaulted.

**Notebook `02_eda.ipynb`** — plots inline, 1–2 line takeaway under each figure.

| Block | What | Why |
|---|---|---|
| 2.1 | Class counts: full 50/50 vs D1/D2/D3 at 1% | Show the paper’s “imbalance” is artificial |
| 2.2 | Missing, dups, dtypes, describe() | Confirm paper preprocessing is a no-op on quality |
| 2.3 | Amount scale vs V1–V28; skewness; need for log(Amount)? | Scaling / transform decision |
| 2.4 | Feature-vs-Class AUC, KS, mutual information; rank features | Which features actually separate fraud |
| 2.5 | KDE / violin / box for top-k discriminative features | Visual confirmation of 2.4 |
| 2.6 | Correlation heatmap; collinearity after extra PCA | Whether paper’s second PCA is useful or harmful |
| 2.7 | Outliers: IQR / z-score by class | Whether DBSCAN noise ≠ fraud |
| 2.8 | 2D PCA and t-SNE (or UMAP if it installs) on a **stratified subsample** (~8k) colored by class | Cluster structure / separability |
| 2.9 | k-distance plot (k=4); DBSCAN on D1 train; #clusters, noise %, fraud-in-noise vs fraud-in-cluster | Is DBSCAN even appropriate? |
| 2.10 | “Too easy?” probe: LogisticRegression and DecisionTree(max_depth=3) on leakage-safe D1 split; PR-AUC / recall | Flag inflated scores if a shallow model already looks near-perfect |
| 2.11 | **Findings and decisions** markdown table | Feature engineering + model shortlist |

**Decision criteria**

- [ ] Every required plot is in the notebook with a takeaway.
- [ ] Closing table has rows of the form `finding → decision → reason` covering: extra PCA yes/no, Amount transform, DBSCAN keep/replace, imbalance method, which models to try in Steps 4–5.
- [ ] If 2.10 PR-AUC is already very high, the table must say the 2023 set is nearly separable and that paper-level 0.995 recall is not surprising.

**Stop / ask you if:** UMAP/t-SNE is too slow even on 8k (fall back to PCA-only 2D).

---

### Step 3 — Baseline replication (`step-03-baseline-paper`)

**Goal.** Faithful paper pipeline, plus a leakage-safe twin. Compare to Tables 3, 5, 7, 9, 11.

**Two versions, both on D1/D2/D3**

**(a) Paper-style (may leak)**  
Preprocess + DBSCAN loop on the **full** subset → then 80/20 split → RF/KNN/SVM → majority and OR vote. Hyperparameters frozen to the paper. Iteration count from A2.

**(b) Leakage-safe**  
Split first. Fit scaler/PCA/DBSCAN on train. Map test cluster labels with a documented rule (nearest train neighbor’s label). Same models and votes.

Also run Case study 1: individual models + majority vote **without** DBSCAN, paper-style vs leakage-safe.

**Outputs**

- Results tables: Accuracy (flagged), Precision, Recall, F1, PR-AUC, ROC-AUC, MCC, TN/FP/FN/TP.
- Side-by-side vs paper Tables 3/5/7/9/11.
- Short discussion: gap size; whether (a) is closer than (b); whether leakage is a plausible explanation.

**Decision criteria**

- [ ] Both (a) and (b) are implemented and labelled.
- [ ] We do not claim “reproduced” unless recall is within ~0.03 of Table 11 on at least D1. Otherwise we report the gap.
- [ ] SVM runtime is logged; if D3 RBF > 30 min, apply A6 and say so.

**Stop / ask you if:** paper-style DVE recall is far below ~0.99 **and** leakage-safe is similar — then the issue is sampling/iterations/class label, not only leakage. I will propose A1/A2 switches rather than invent a new method.

---

### Step 4 — Feature engineering alternatives (`step-04-feature-engineering`)

**Goal.** Replace or supplement DBSCAN using **only** methods the EDA table supports. Fair comparison: **one fixed model** (default LightGBM with `scale_pos_weight = n_neg/n_pos`, unless Step 2 says trees fail).

**Candidate pool (run a method only if EDA justifies it)**

| Candidate | Likely EDA trigger |
|---|---|
| KMeans / GMM cluster id + distance-to-centroid | 2D projection shows compact blobs |
| Isolation Forest / LOF scores | Fraud sits in the tail / DBSCAN noise |
| One-Class SVM score | Dense legitimate mass, sparse fraud |
| Autoencoder reconstruction error | Nonlinear structure; shallow models not enough |
| Amount log/quantile transform; selected interactions of top MI features | Skew / a few features dominate |
| SMOTE / ADASYN vs class weights | To test “augmentation” vs rebalancing |
| Feature selection (MI / extra-trees importances) | Many near-zero univariate AUCs |

Each variant: leakage-safe, same D1 (all methods) then winner on D2/D3. Same split indices as Step 1.

**Decision criteria**

- [ ] At least 3 EDA-justified variants besides “no extra features”.
- [ ] Winner = highest **PR-AUC**, with recall at a precision-constrained operating point (e.g. precision ≥ 0.95) reported too.
- [ ] Markdown cell: `EDA finding → this method → metric vs DBSCAN labels`.
- [ ] Autoencoder skipped if 2.10 already shows near-linear separability (not worth the complexity).

---

### Step 5 — Models beyond the paper (`step-05-models`)

**Goal.** Stronger learners on the **winning feature set** from Step 4 (plus a “raw scaled features” control).

**Models**

- XGBoost, LightGBM, CatBoost with `scale_pos_weight` / class weights.
- MLP (PyTorch): class-weighted BCE **or** focal loss if recall at high precision is still weak. Early stopping on val PR-AUC, seed, deterministic flags as far as Windows/CPU allows.
- Optional TabNet: **skip** unless CatBoost/MLP clearly stall; it is heavy.
- Optional OR-vote / stacking of the top 3 heterogeneous models, tuned on val, not test.

**Tuning**

- Modest Optuna or RandomizedSearch (≈40 trials, inner 3-fold or a val split).
- Decision threshold chosen on validation to maximize F1 **or** to hit a recall target; never on test.

**Decision criteria**

- [ ] Each model has CV or a held-out val protocol; test is touched once per model.
- [ ] Threshold rule is written down.
- [ ] Slow models: document D3 subsample / fewer trials rather than drop silently.

---

### Step 6 — Final comparison (`step-06-final-comparison`)

**Goal.** One honest leaderboard and a written conclusion.

**Table (D1, D2, D3):** paper-style baseline, leakage-safe baseline, best FE + fixed model, XGBoost, LightGBM, CatBoost, MLP, optional ensemble. **5-fold CV mean ± std** for Recall, Precision, F1, PR-AUC, ROC-AUC, MCC. Runtime column.

**Plots:** PR curves, ROC curves, confusion matrices, recall–precision trade-off, runtime bars.

**Stats:** paired fold differences vs leakage-safe DVE (mean Δ recall, rough interval). Skip a formal test if folds are too correlated; say so.

**Interpretability:** SHAP (or tree importances if SHAP is too slow on D3) for the best model.

**Conclusion (short):** what held up, what did not, whether Table 11 is credible on this set, limitations (artificial 1% sample, near-separability, no concept drift).

**Decision criteria**

- [ ] Every number in the table is computed in this repo.
- [ ] Accuracy is visually secondary.
- [ ] README updated with the result summary.

---

### Step 7 — Optional sanity check (`step-07-realistic-data`)

**Goal.** See if the paper method and our winner survive a **naturally** rare-fraud set.

- Dataset: Kaggle/ULB `creditcard.csv` (2013), fraud rate ~0.17%, if you add it.
- Run leakage-safe paper pipeline + Step 6 winner. Same metrics.

**Decision criteria**

- [ ] If the file is missing: **skip**, write one markdown cell, do not download without asking.
- [ ] If present: report the drop vs D1–D3; that drop is the generalization check.

---

## Deliverables (end state)

1. `PLAN.md` (this file)
2. One notebook (`notebooks/pipeline.ipynb`): short markdown, inline plots, reasoning cells, computed tables
3. `src/` as above
4. `README.md`: setup, how to run, branch map, result summary
5. Git history: one branch per step, merged to `dev`, `main` limited to plan/ignore/README

---

## What I will do immediately after you approve

1. Commit `.gitignore` + `PLAN.md` on `main`.
2. Create `dev` from `main`.
3. Branch `step-01-setup` and start Step 1 only.

I will not write notebooks or `src/` until you say the plan is approved. If you want changes (nested vs independent sampling, skip DL, skip CatBoost, etc.), say so before I start.
