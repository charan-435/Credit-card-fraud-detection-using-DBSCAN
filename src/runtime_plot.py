"""Runtime bars from already-computed JSON (no refitting)."""

import json

import matplotlib.pyplot as plt
import pandas as pd

from src.config import PLOTS_DIR, TABLES_DIR


def main():
    (PLOTS_DIR / "compare").mkdir(parents=True, exist_ok=True)
    cv = pd.DataFrame(json.loads((TABLES_DIR / "cv_results.json").read_text(encoding="utf-8")))
    base = pd.DataFrame(json.loads((TABLES_DIR / "baseline_results.json").read_text(encoding="utf-8")))
    adv = pd.DataFrame(json.loads((TABLES_DIR / "advanced_results.json").read_text(encoding="utf-8")))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    pivot = cv.pivot(index="dataset", columns="model", values="seconds")
    pivot.plot.bar(ax=axes[0], rot=0)
    axes[0].set_ylabel("seconds (5-fold total)")
    axes[0].set_title("CV runtime")

    svm = base[(base.model == "SVM") & (base.mode == "leakage_safe") & (base.augment == "none")]
    axes[1].bar(svm["dataset"], svm["seconds"], color="#e45756", label="SVM RBF holdout")
    xgb = adv[adv.model == "XGBoost"]
    axes[1].bar([d + " " for d in xgb["dataset"]], xgb["seconds"], color="#4c78a8", label="XGBoost holdout")
    axes[1].set_title("Holdout fit time")
    axes[1].legend()
    fig.tight_layout()
    path = PLOTS_DIR / "compare" / "runtime.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
