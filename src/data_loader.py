"""Load ApacheJIT data and filter to Apache Camel."""

from pathlib import Path

import pandas as pd

from config import (
    COMMIT_COL, DATA_DIR, DATE_COL, LABEL_COL, METRIC_COLS, PROJECT, TRAIN_RATIO, VAL_RATIO,
)


def find_dataset_csv() -> Path:
    candidates = [
        DATA_DIR / "dataset" / "apachejit_total.csv",
        DATA_DIR / "apachejit" / "dataset" / "apachejit_total.csv",
        DATA_DIR / "apachejit_total.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "apachejit_total.csv not found. Download from "
        "https://zenodo.org/records/5907847 and extract to data/"
    )


def load_camel_commits() -> pd.DataFrame:
    csv_path = find_dataset_csv()
    df = pd.read_csv(csv_path)
    df = df[df["project"].str.lower() == PROJECT.lower()].copy()
    df[LABEL_COL] = df[LABEL_COL].map(
        lambda x: 1 if str(x).lower() in ("true", "1", "1.0") else 0
    )
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce", unit="s")
    df = df.dropna(subset=[DATE_COL, COMMIT_COL, LABEL_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    for col in METRIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def sample_commits(df: pd.DataFrame, n: int, random_state: int = 42) -> pd.DataFrame:
    """Stratified sample so both classes are represented when subsampling."""
    if len(df) <= n:
        return df.copy()

    pos_df = df[df[LABEL_COL] == 1]
    neg_df = df[df[LABEL_COL] == 0]
    if pos_df.empty or neg_df.empty:
        return df.sample(n=n, random_state=random_state).sort_values(DATE_COL).reset_index(drop=True)

    n_pos = max(1, round(n * len(pos_df) / len(df)))
    n_pos = min(n_pos, len(pos_df), n - 1)
    n_neg = n - n_pos

    parts = [
        pos_df.sample(n=n_pos, random_state=random_state),
        neg_df.sample(n=n_neg, random_state=random_state),
    ]
    return pd.concat(parts).sort_values(DATE_COL).reset_index(drop=True)


def chronological_split(df: pd.DataFrame, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    """Split a date-sorted frame in sequence: train | validation | test."""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    return train, val, test


def describe_split(name: str, part: pd.DataFrame) -> str:
    return (
        f"{name:5s} {len(part):6d} commits, {int(part[LABEL_COL].sum()):5d} buggy "
        f"({part[LABEL_COL].mean():.1%}), "
        f"{part[DATE_COL].min().date()} -> {part[DATE_COL].max().date()}"
    )
