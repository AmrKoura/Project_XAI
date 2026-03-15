"""
Lag and rolling-window feature engineering.

Creates lagged sales values and rolling statistics (mean, std, min, max)
to capture recent demand history for each SKU-store combination.
"""

import pandas as pd
import numpy as np


def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "sales",
    group_cols: list[str] | None = None,
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """Add lagged values of the target column per group.

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
    group_cols : list[str] | None
        Columns defining the SKU-store group (e.g. ``['store_id', 'item_id']``).
    lags : list[int] | None
        Lag periods (default: ``[7, 14, 28]``).

    Returns
    -------
    pd.DataFrame
    """
    ...


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str = "sales",
    group_cols: list[str] | None = None,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Add rolling mean/std/min/max for specified window sizes.

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
    group_cols : list[str] | None
    windows : list[int] | None
        Window sizes (default: ``[7, 14, 28]``).

    Returns
    -------
    pd.DataFrame
    """
    ...


def build_lag_features(
    df: pd.DataFrame,
    target_col: str = "sales",
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Run the full lag-feature pipeline (lags + rolling stats).

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
    group_cols : list[str] | None

    Returns
    -------
    pd.DataFrame
    """
    ...
