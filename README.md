# Credit‑Card Fraud Detection with DBSCAN

Implementation and extension of **"Credit Card Fraud Detection Using DBSCAN and Ensemble Learning"** (Scientific Reports, 2025) on the [Kaggle Credit Card Fraud Detection Dataset 2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023).

---

##  Overview

The paper proposes using DBSCAN cluster labels as extra features fed into RF, KNN, and SVM classifiers with disjunctive (OR) voting. This notebook reproduces that idea, fixes a data-leakage issue, and benchmarks against modern gradient-boosted baselines (XGBoost, CatBoost).

**Key questions answered:**
- Does DBSCAN augmentation actually improve fraud detection?
- Does cluster-ID encoding cause issues for distance-based models?
- How do modern boosting methods compare against the paper's paper-style ensemble?

---

##  Dataset

| | |
|---|---|
| **Source** | [Kaggle – Credit Card Fraud Detection Dataset 2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023) |
| **Records** | 568,630 transactions |
| **Features** | 28 PCA-anonymised components (V1–V28) + Amount + Class |
| **Fraud rate** | ~50% in source; downsampled to ~1% for realistic evaluation |
| **File** | `creditcard_2023.csv` (~310 MB, not tracked in Git) |

---

## 🛠️ Methodology

### 1. Data preparation
- Stratified train/test split (80/20) → 64,640 train | 16,160 test
- Standard scaling fitted **only** on training data to avoid leakage

### 2. Exploratory Data Analysis
- Class distribution, amount histograms by class
- Top univariate AUC features: **V14 (0.967), V4 (0.949), V12 (0.948)**
- Correlation analysis and outlier profiling

### 3. DBSCAN clustering (leakage-safe)
- `eps=2.0`, `min_samples=4`, fitted on **training set only**
- 380 clusters found; ~25% of train points marked as noise
- Test points assigned to nearest training cluster via 1-NN (distance > eps → noise)
- Two feature representations tested:
  - **Numeric IDs** – paper-style: iterative integer cluster labels appended as features
  - **Noise flag** – binary `is_noise` indicator (cleaner encoding)

### 4. Models evaluated

| Group | Models |
|---|---|
| Paper-style (Section 6.1) | RF (n=3), KNN (k=3), SVM (RBF), majority vote, disjunctive OR |
| Broader baselines (Section 6.2) | Logistic Regression, RF (n=200), XGBoost, CatBoost |

Each model tested on: original features / + numeric IDs / + noise flag

### 5. Evaluation metrics
Precision, Recall, F1, PR-AUC, MCC, and raw FP/FN counts — accuracy alone is misleading at 1% fraud prevalence.

---

##  Results

### Paper-style classifiers (original features vs. DBSCAN numeric IDs)

| Model | Augmentation | Precision | Recall | F1 | PR-AUC | FP | FN |
|---|---|---|---|---|---|---|---|
| RF (n=3) | Original | 0.971 | 0.850 | 0.907 | 0.858 | 4 | 24 |
| KNN (k=3) | Original | 0.935 | 0.894 | 0.914 | 0.898 | 10 | 17 |
| SVM RBF | Original | 0.966 | 0.875 | 0.918 | 0.958 | 5 | 20 |
| Majority vote | Original | 0.972 | 0.881 | 0.925 | — | 4 | 19 |
| **Disjunctive OR** | **Original** | **0.924** | **0.906** | **0.915** | — | **12** | **15** |
| Disjunctive OR | Numeric IDs | 0.918 | 0.912 | **0.915** | — | 13 | 14 |
| Disjunctive OR | + Noise flag | 0.924 | 0.912 | **0.918** | — | 12 | 14 |

### Stronger baselines vs. paper reference (Disjunctive OR | Numeric IDs: F1=0.915, Recall=0.912)

| Model | Augmentation | Precision | Recall | F1 | PR-AUC | FP | FN |
|---|---|---|---|---|---|---|---|
| XGBoost | + Noise flag | 0.901 | 0.906 | 0.903 | 0.955 | 16 | 15 |
| RF (n=200) | + Noise flag | 0.972 | 0.863 | 0.914 | 0.959 | 4 | 22 |
| CatBoost | + Noise flag | 0.902 | 0.925 | 0.914 | 0.956 | 16 | 12 |
| **CatBoost** | **Original** | **0.915** | **0.938** | **0.926** | **0.958** | **14** | **10** |

> **Takeaway:** CatBoost on raw features (F1=0.926, Recall=0.938) outperforms the paper-style disjunctive OR ensemble (F1=0.915) — with fewer missed fraud cases. The DBSCAN noise flag provides a modest lift for KNN/SVM but hurts SVM on numeric IDs due to false distance ordering.

---

##  Key Findings

1. **DBSCAN noise flag ≈ marginal improvement** — the binary noise indicator is a cleaner encoding than numeric cluster IDs and adds a small consistent lift for some models.
2. **Numeric IDs can hurt** — distance-based models (KNN, RBF SVM) are sensitive to the arbitrary ordering of cluster IDs; SVM recall dropped from 0.875 → 0.781 with numeric IDs.
3. **CatBoost dominates** — modern gradient boosting with 200 iterations vs. the paper's 3-tree RF explains the gap; it's not a surprising result, just a useful benchmark.
4. **Leakage matters** — the paper fits DBSCAN before splitting; the leakage-safe version shows similar trends, so leakage alone doesn't explain any reported improvement.

---

##  Usage

```bash
# 1. Download dataset from Kaggle and place it in the repo root
#    https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023

# 2. Install dependencies
pip install numpy pandas scikit-learn xgboost catboost matplotlib jupyter

# 3. Open the final notebook
jupyter notebook credit_card_fraud_detection.ipynb
```

---

##  Repository Structure

```
.
├── credit_card_fraud_detection.ipynb   # Main notebook (all steps self-contained)
├── Research_Paper.pdf                  # Original paper (Scientific Reports, 2025)
├── credit_card_fraud_detection_presentation.pdf
├── creditcard_2023.csv                 # Dataset (not tracked in Git)
├── .gitignore
└── README.md
```

---

##  Limitations

- Only **one** stratified split evaluated — run-to-run variation is unknown.
- 1% fraud prevalence is **constructed by sampling** from a balanced source dataset; real deployment rates differ.
- Decision threshold is fixed at 0.5 — not tuned to any business cost function.
- Nearest-neighbour DBSCAN assignment is an approximation for unseen test points.

---

