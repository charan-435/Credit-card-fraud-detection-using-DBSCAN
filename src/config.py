"""Shared paths, seeds, and paper hyperparameters."""

from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.20
POSITIVE_LABEL = 1  # fraud in this CSV; paper tables write fraud as class (0)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "creditcard_2023.csv"
SUBSET_DIR = PROJECT_ROOT / "results" / "subsets"
CACHE_DIR = PROJECT_ROOT / "results" / "cache"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

ID_COL = "id"
TARGET_COL = "Class"
FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]

# Paper Section "Data collection phase"
SUBSET_SPECS = {
    "D1": {"n_nonfraud": 40_000, "n_fraud": 400},
    "D2": {"n_nonfraud": 80_000, "n_fraud": 800},
    "D3": {"n_nonfraud": 120_000, "n_fraud": 1_200},
}

# Paper "Hyperparameter configuration"
DBSCAN_EPS = 2.0
DBSCAN_MIN_SAMPLES = 4
RF_N_ESTIMATORS = 3
RF_MAX_DEPTH = None
KNN_N_NEIGHBORS = 3
SVM_KERNEL = "rbf"
SVM_C = 1.0
