"""Dataset preparation utilities for the credit-card fraud project.

The functions in this module are intentionally leakage-safe: the train/test
split is created before any learned preprocessing is fitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COLUMN = "Class"
REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


@dataclass(frozen=True)
class DataSummary:
    """Store core dataset quality and class-imbalance facts."""

    samples: int
    model_features: int
    total_columns: int
    missing_values: int
    infinite_feature_values: int
    duplicate_rows: int
    legitimate_count: int
    fraud_count: int
    fraud_ratio: float

    def as_dict(self) -> dict[str, Any]:
        """Return the summary as a regular dictionary."""

        return {
            "samples": self.samples,
            "model_features": self.model_features,
            "total_columns": self.total_columns,
            "missing_values": self.missing_values,
            "infinite_feature_values": self.infinite_feature_values,
            "duplicate_rows": self.duplicate_rows,
            "legitimate_count": self.legitimate_count,
            "fraud_count": self.fraud_count,
            "fraud_ratio": self.fraud_ratio,
        }


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load the real dataset and enforce the project schema.

    Raises a clear error when the included header-only placeholder has not yet
    been replaced by the real Kaggle dataset.
    """

    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. "
            "Place the real creditcard.csv file in data/creditcard.csv."
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(
            "data/creditcard.csv currently contains only the required header. "
            "Replace it with the real dataset before training."
        )

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Keep only the project-defined columns and enforce canonical order.
    df = df.loc[:, REQUIRED_COLUMNS].copy()

    missing_value_count = int(df.isna().sum().sum())
    if missing_value_count:
        raise ValueError(
            f"Dataset contains {missing_value_count} missing values. "
            "The assignment does not define an imputation policy, so the "
            "pipeline stops rather than silently inventing one."
        )

    feature_array = df[FEATURE_COLUMNS].to_numpy(dtype=float, copy=False)
    infinite_count = int(np.isinf(feature_array).sum())
    if infinite_count:
        raise ValueError(
            f"Dataset contains {infinite_count} infinite feature values."
        )

    target_values = set(df[TARGET_COLUMN].astype(int).unique().tolist())
    if target_values != {0, 1}:
        raise ValueError(
            f"Expected target labels {{0, 1}}, found: {sorted(target_values)}"
        )

    return df


def dataset_summary(df: pd.DataFrame) -> DataSummary:
    """Compute class distribution and data-quality summary statistics."""

    counts = df[TARGET_COLUMN].value_counts().to_dict()
    legitimate = int(counts.get(0, 0))
    fraud = int(counts.get(1, 0))
    total = len(df)

    feature_array = df[FEATURE_COLUMNS].to_numpy(dtype=float, copy=False)

    return DataSummary(
        samples=total,
        model_features=len(FEATURE_COLUMNS),
        total_columns=len(REQUIRED_COLUMNS),
        missing_values=int(df.isna().sum().sum()),
        infinite_feature_values=int(np.isinf(feature_array).sum()),
        duplicate_rows=int(df.duplicated().sum()),
        legitimate_count=legitimate,
        fraud_count=fraud,
        fraud_ratio=(fraud / total) if total else 0.0,
    )


def dataset_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Return names, dtypes, non-null counts, and unique counts."""

    rows = []
    for column in df.columns:
        rows.append(
            {
                "Column": column,
                "Dtype": str(df[column].dtype),
                "Non-Null": int(df[column].notna().sum()),
                "Unique": int(df[column].nunique(dropna=True)),
            }
        )

    return pd.DataFrame(rows)


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Generate descriptive statistics for all numeric project columns."""

    return (
        df[REQUIRED_COLUMNS]
        .describe()
        .T
        .reset_index()
        .rename(columns={"index": "Feature"})
    )


def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the 30 model features from the binary target."""

    X = df.loc[:, FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int).copy()
    return X, y


def stratified_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create the required stratified train/test split.

    The returned test set must remain untouched during preprocessing fitting,
    cross-validation, model selection, hyperparameter selection, and threshold
    tuning.
    """

    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
