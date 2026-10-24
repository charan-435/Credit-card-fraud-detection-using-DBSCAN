"""Load the 2023 Kaggle file and build the paper's D1/D2/D3 subsets.

Sampling rule (PLAN A1, default)
--------------------------------
The raw file is balanced 50/50. The paper's 1% fraud rate is constructed.

Nested draw with ``RANDOM_STATE``:
1. Sample D3 = 120,000 non-fraud + 1,200 fraud.
2. Sample D2 from D3 (80,000 + 800).
3. Sample D1 from D2 (40,000 + 400).

So D1 ⊂ D2 ⊂ D3. Row identity is the original ``id`` column.

``id`` is kept in subset tables for reproducibility and dropped when building
model matrices via :func:`features_and_target`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CACHE_DIR,
    DATA_PATH,
    FEATURE_COLS,
    ID_COL,
    POSITIVE_LABEL,
    RANDOM_STATE,
    SUBSET_DIR,
    SUBSET_SPECS,
    TARGET_COL,
    TEST_SIZE,
)


def load_raw(path: Path | None = None, use_cache: bool = True) -> pd.DataFrame:
    """Load the Kaggle CSV. Raises a clear error if the file is missing."""
    path = Path(path) if path is not None else DATA_PATH
    cache_path = CACHE_DIR / "raw.parquet"
    if use_cache and path == DATA_PATH and cache_path.exists():
        return pd.read_parquet(cache_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place creditcard_2023.csv in the "
            "repo root (Kaggle: nelgiriyewithana/credit-card-fraud-detection-dataset-2023)."
        )
    df = pd.read_csv(path)
    expected = {ID_COL, *FEATURE_COLS, TARGET_COL}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")
    if use_cache and path == DATA_PATH:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            df.to_parquet(cache_path, index=False)
        except Exception:
            pass
    return df


def load_cleaned() -> pd.DataFrame:
    """Load + drop nulls/duplicates (no-op on this file, kept for the paper pipeline)."""
    return clean(load_raw())


def load_named_subsets(df: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Reload frozen D1/D2/D3 from saved ids."""
    if df is None:
        df = load_cleaned()
    return {name: load_subset(name, df=df) for name in ("D1", "D2", "D3")}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Paper preprocessing that applies here: drop nulls and duplicate rows."""
    out = df.dropna()
    out = out.drop_duplicates()
    return out.reset_index(drop=True)


def _sample_class(frame: pd.DataFrame, n: int, rng: int) -> pd.DataFrame:
    if n > len(frame):
        raise ValueError(f"Need {n} rows but only {len(frame)} available.")
    return frame.sample(n=n, random_state=rng)


def build_paper_subsets(
    df: pd.DataFrame,
    nested: bool = True,
    random_state: int = RANDOM_STATE,
) -> dict[str, pd.DataFrame]:
    """Return D1/D2/D3 as shuffled dataframes with original ``id`` values.

    ``nested=True`` is the PLAN default (D1 ⊂ D2 ⊂ D3).
    ``nested=False`` draws each size independently from the full file.
    """
    fraud = df[df[TARGET_COL] == POSITIVE_LABEL]
    nonfraud = df[df[TARGET_COL] != POSITIVE_LABEL]
    spec_d1, spec_d2, spec_d3 = (SUBSET_SPECS[k] for k in ("D1", "D2", "D3"))

    if nested:
        fraud_d3 = _sample_class(fraud, spec_d3["n_fraud"], random_state)
        nonfraud_d3 = _sample_class(nonfraud, spec_d3["n_nonfraud"], random_state)
        fraud_d2 = _sample_class(fraud_d3, spec_d2["n_fraud"], random_state)
        nonfraud_d2 = _sample_class(nonfraud_d3, spec_d2["n_nonfraud"], random_state)
        fraud_d1 = _sample_class(fraud_d2, spec_d1["n_fraud"], random_state)
        nonfraud_d1 = _sample_class(nonfraud_d2, spec_d1["n_nonfraud"], random_state)
        parts = {
            "D3": (nonfraud_d3, fraud_d3),
            "D2": (nonfraud_d2, fraud_d2),
            "D1": (nonfraud_d1, fraud_d1),
        }
    else:
        parts = {}
        for name, spec in SUBSET_SPECS.items():
            parts[name] = (
                _sample_class(nonfraud, spec["n_nonfraud"], random_state),
                _sample_class(fraud, spec["n_fraud"], random_state),
            )

    subsets: dict[str, pd.DataFrame] = {}
    for name, (neg, pos) in parts.items():
        combined = pd.concat([neg, pos], axis=0)
        subsets[name] = combined.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return subsets


def subset_summary(frame: pd.DataFrame) -> dict:
    n_pos = int((frame[TARGET_COL] == POSITIVE_LABEL).sum())
    n_neg = int((frame[TARGET_COL] != POSITIVE_LABEL).sum())
    n = len(frame)
    return {
        "n": n,
        "n_nonfraud": n_neg,
        "n_fraud": n_pos,
        "fraud_rate": n_pos / n if n else 0.0,
    }


def nested_inclusion(subsets: dict[str, pd.DataFrame]) -> dict[str, bool]:
    """Check D1 ⊂ D2 ⊂ D3 using original ids."""
    ids = {name: set(frame[ID_COL]) for name, frame in subsets.items()}
    return {
        "D1_subset_of_D2": ids["D1"].issubset(ids["D2"]),
        "D2_subset_of_D3": ids["D2"].issubset(ids["D3"]),
    }


def stratified_split_frame(
    frame: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """80/20 stratified split. Returns (train, test) dataframes."""
    train, test = train_test_split(
        frame,
        test_size=test_size,
        stratify=frame[TARGET_COL],
        random_state=random_state,
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def features_and_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Drop ``id``; return X (V1–V28 + Amount) and y (Class)."""
    return frame[FEATURE_COLS].copy(), frame[TARGET_COL].copy()


def persist_subsets(
    subsets: dict[str, pd.DataFrame],
    out_dir: Path | None = None,
    random_state: int = RANDOM_STATE,
) -> Path:
    """Write id lists, train/test ids, and a JSON manifest. Returns the directory."""
    out_dir = Path(out_dir) if out_dir is not None else SUBSET_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "random_state": random_state,
        "test_size": TEST_SIZE,
        "positive_label": POSITIVE_LABEL,
        "sampling": "nested D1 subset of D2 subset of D3",
        "subsets": {},
    }

    for name in ("D1", "D2", "D3"):
        frame = subsets[name]
        key = name.lower()
        frame[[ID_COL]].to_csv(out_dir / f"{key}_ids.csv", index=False)

        train, test = stratified_split_frame(frame, random_state=random_state)
        train[[ID_COL]].to_csv(out_dir / f"{key}_train_ids.csv", index=False)
        test[[ID_COL]].to_csv(out_dir / f"{key}_test_ids.csv", index=False)

        info = subset_summary(frame)
        info["train_n"] = len(train)
        info["test_n"] = len(test)
        info["train_fraud"] = int((train[TARGET_COL] == POSITIVE_LABEL).sum())
        info["test_fraud"] = int((test[TARGET_COL] == POSITIVE_LABEL).sum())
        manifest["subsets"][name] = info

    manifest["inclusion"] = nested_inclusion(subsets)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_dir


def load_ids(name: str, split: str | None = None, out_dir: Path | None = None) -> pd.Index:
    """Load saved ids. ``split`` is None (full subset), 'train', or 'test'."""
    out_dir = Path(out_dir) if out_dir is not None else SUBSET_DIR
    key = name.lower()
    if split is None:
        path = out_dir / f"{key}_ids.csv"
    elif split in {"train", "test"}:
        path = out_dir / f"{key}_{split}_ids.csv"
    else:
        raise ValueError("split must be None, 'train', or 'test'")
    if not path.exists():
        raise FileNotFoundError(f"No saved ids at {path}. Run persist_subsets first.")
    return pd.read_csv(path)[ID_COL]


def subset_from_ids(df: pd.DataFrame, ids: pd.Index) -> pd.DataFrame:
    """Select rows by original id, preserving the saved order."""
    indexed = df.set_index(ID_COL)
    return indexed.loc[ids].reset_index()


def load_subset(
    name: str,
    df: pd.DataFrame | None = None,
    split: str | None = None,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    """Reload a persisted subset (and optional train/test split) from ids."""
    if df is None:
        df = load_raw()
    ids = load_ids(name, split=split, out_dir=out_dir)
    return subset_from_ids(df, ids)


def build_and_persist(
    path: Path | None = None,
    nested: bool = True,
    random_state: int = RANDOM_STATE,
) -> dict[str, pd.DataFrame]:
    """End-to-end: load → clean → sample D1/D2/D3 → write id files."""
    raw = clean(load_raw(path))
    subsets = build_paper_subsets(raw, nested=nested, random_state=random_state)
    persist_subsets(subsets, random_state=random_state)
    return subsets


if __name__ == "__main__":
    built = build_and_persist()
    for name, frame in built.items():
        print(name, subset_summary(frame))
    print("inclusion", nested_inclusion(built))
    print(f"wrote ids to {SUBSET_DIR}")
