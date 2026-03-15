"""
Build model-ready datasets from cleaned data.

Merges cleaned tables, applies feature engineering, creates
train/validation/test splits at the SKU-store level, and
saves the final datasets to the processed directory.
"""

import pandas as pd
from pathlib import Path


def merge_tables(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge related tables (sales, calendar, prices, stores) into a single frame.

    Parameters
    ----------
    tables : dict[str, pd.DataFrame]
        Dictionary of cleaned dataframes.

    Returns
    -------
    pd.DataFrame
        Merged dataframe.
    """
    ...


def create_train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Time-based split into train, validation, and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset sorted by date.
    train_ratio, val_ratio, test_ratio : float
        Proportions for each split.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train, val, test) DataFrames.
    """
    ...


def build_and_save(tables: dict[str, pd.DataFrame], output_dir: str) -> None:
    """End-to-end pipeline: merge → split → save to disk.

    Parameters
    ----------
    tables : dict[str, pd.DataFrame]
    output_dir : str
        Path to ``data/processed``.
    """
    ...
