"""
Clean and preprocess raw data.

Handles missing values, dtype corrections, outlier detection,
and synthetic field generation for inventory-specific columns
(stock-on-hand, lead time, reorder cost) that may be absent
from public datasets.
"""

import pandas as pd
import numpy as np


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute or drop missing values based on column type and missingness rate.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with missing values handled.
    """
    ...


def remove_outliers(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Flag and remove statistical outliers from specified columns.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list[str]
        Columns to check for outliers.

    Returns
    -------
    pd.DataFrame
    """
    ...


def generate_synthetic_inventory_fields(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic stock-on-hand, lead time, and reorder cost fields.

    Used when public datasets lack inventory-specific columns needed for
    the replenishment logic.

    Parameters
    ----------
    df : pd.DataFrame
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional synthetic columns.
    """
    ...


def clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline: missing values → outliers → dtype fixes.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Fully cleaned dataframe.
    """
    ...
