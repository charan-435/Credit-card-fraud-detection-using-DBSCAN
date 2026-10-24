"""Small plotting helpers for EDA and comparison notebooks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import PLOTS_DIR, POSITIVE_LABEL, TARGET_COL

sns.set_theme(style="whitegrid", context="notebook")


def save_fig(fig: plt.Figure, name: str, subdir: str = "eda") -> Path:
    folder = PLOTS_DIR / subdir
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    return path


def class_count_bar(counts: dict, title: str, ax=None):
    ax = ax or plt.gca()
    labels = list(counts.keys())
    values = list(counts.values())
    ax.bar(labels, values, color=["#4c78a8", "#e45756"])
    ax.set_title(title)
    ax.set_ylabel("count")
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    return ax


def ranked_bar(series: pd.Series, title: str, xlabel: str, ax=None, color="#4c78a8"):
    ax = ax or plt.gca()
    s = series.sort_values()
    ax.barh(s.index.astype(str), s.values, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    return ax
