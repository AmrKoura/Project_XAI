"""
Lag and rolling-window feature engineering.

Creates lagged sales values and rolling statistics (mean, std, min, max)
to capture recent demand history for each SKU-store combination.
"""

import pandas as pd
import numpy as np


def _pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _resolve_group_cols(df: pd.DataFrame, group_cols: list[str] | None) -> list[str]:
    if group_cols:
        return [col for col in group_cols if col in df.columns]

    candidates = [
        ["store_id", "item_id"],
        ["Store", "item_id"],
        ["Store", "id"],
        ["id"],
    ]
    for cols in candidates:
        if all(col in df.columns for col in cols):
            return cols
    return []


def _sort_by_time(df: pd.DataFrame, group_cols: list[str]) -> tuple[pd.DataFrame, str | None]:
    out = df.copy()
    date_col = _pick_first_existing(out, ["date", "Date"])
    if date_col is not None:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        sort_cols = [*group_cols, date_col] if group_cols else [date_col]
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out, date_col


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
    out = df.copy()
    if target_col not in out.columns:
        return out

    lags = lags or [7, 14, 28]
    group_cols = _resolve_group_cols(out, group_cols)
    out, _ = _sort_by_time(out, group_cols)

    out[target_col] = pd.to_numeric(out[target_col], errors="coerce")
    if group_cols:
        grouped = out.groupby(group_cols, sort=False)[target_col]
        for lag in lags:
            out[f"{target_col}_lag_{lag}"] = grouped.shift(lag)
    else:
        for lag in lags:
            out[f"{target_col}_lag_{lag}"] = out[target_col].shift(lag)

    return out


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
    out = df.copy()
    if target_col not in out.columns:
        return out

    windows = windows or [7, 14, 28]
    group_cols = _resolve_group_cols(out, group_cols)
    out, _ = _sort_by_time(out, group_cols)

    out[target_col] = pd.to_numeric(out[target_col], errors="coerce")
    if group_cols:
        shifted = out.groupby(group_cols, sort=False)[target_col].shift(1)
        temp = out[group_cols].copy()
        temp["shifted"] = shifted
        grouped_shifted = temp.groupby(group_cols, sort=False)["shifted"]
        for window in windows:
            out[f"{target_col}_roll_mean_{window}"] = grouped_shifted.transform(
                lambda s: s.rolling(window, min_periods=1).mean()
            )
            out[f"{target_col}_roll_std_{window}"] = grouped_shifted.transform(
                lambda s: s.rolling(window, min_periods=1).std()
            )
            out[f"{target_col}_roll_min_{window}"] = grouped_shifted.transform(
                lambda s: s.rolling(window, min_periods=1).min()
            )
            out[f"{target_col}_roll_max_{window}"] = grouped_shifted.transform(
                lambda s: s.rolling(window, min_periods=1).max()
            )
    else:
        shifted = out[target_col].shift(1)
        for window in windows:
            out[f"{target_col}_roll_mean_{window}"] = shifted.rolling(window, min_periods=1).mean()
            out[f"{target_col}_roll_std_{window}"] = shifted.rolling(window, min_periods=1).std()
            out[f"{target_col}_roll_min_{window}"] = shifted.rolling(window, min_periods=1).min()
            out[f"{target_col}_roll_max_{window}"] = shifted.rolling(window, min_periods=1).max()

    return out


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
    out = add_lag_features(df, target_col=target_col, group_cols=group_cols)
    out = add_rolling_features(out, target_col=target_col, group_cols=group_cols)
    return out
