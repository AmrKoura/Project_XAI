"""
Calendar and time-based feature engineering.

Extracts day-of-week, month, week-of-year, holiday proximity,
pay-day indicators, and other temporal patterns from date columns.
"""

import pandas as pd
import numpy as np


def _ensure_datetime(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    if date_col in out.columns:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    return out


def add_basic_date_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add day_of_week, month, week_of_year, day_of_month, quarter.

    Parameters
    ----------
    df : pd.DataFrame
    date_col : str

    Returns
    -------
    pd.DataFrame
        DataFrame with new time columns appended.
    """
    out = _ensure_datetime(df, date_col)
    if date_col not in out.columns:
        return out

    dt = out[date_col].dt
    out["day_of_week"] = dt.dayofweek.astype("Int8")
    out["day_of_month"] = dt.day.astype("Int8")
    out["week_of_year"] = dt.isocalendar().week.astype("Int16")
    out["month"] = dt.month.astype("Int8")
    out["quarter"] = dt.quarter.astype("Int8")
    out["is_month_start"] = dt.is_month_start.astype("Int8")
    out["is_month_end"] = dt.is_month_end.astype("Int8")
    out["is_year_start"] = dt.is_year_start.astype("Int8")
    out["is_year_end"] = dt.is_year_end.astype("Int8")

    # Cyclical encoding keeps weekly/yearly seasonality smooth for tree models.
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7.0)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)
    return out


def add_holiday_proximity(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add days-to-nearest-holiday and is_holiday flag.

    Parameters
    ----------
    df : pd.DataFrame
    date_col : str

    Returns
    -------
    pd.DataFrame
    """
    out = _ensure_datetime(df, date_col)
    if date_col not in out.columns:
        return out

    event_cols = [
        col for col in ["event_name_1", "event_name_2", "event_type_1", "event_type_2"] if col in out.columns
    ]
    if event_cols:
        has_event = pd.Series(False, index=out.index)
        for col in event_cols:
            has_event = has_event | out[col].notna() & (out[col].astype(str).str.strip() != "")
        out["is_holiday"] = has_event.astype("Int8")
    else:
        out["is_holiday"] = 0

    out["days_to_nearest_holiday"] = np.nan
    valid_dates = out[date_col].notna()
    holiday_dates = np.sort(out.loc[(out["is_holiday"] == 1) & valid_dates, date_col].unique())

    if len(holiday_dates) == 0:
        out.loc[valid_dates, "days_to_nearest_holiday"] = 999
        return out

    current_dates = out.loc[valid_dates, date_col].to_numpy(dtype="datetime64[ns]")
    holiday_dates = holiday_dates.astype("datetime64[ns]")

    idx = np.searchsorted(holiday_dates, current_dates)
    prev_idx = np.clip(idx - 1, 0, len(holiday_dates) - 1)
    next_idx = np.clip(idx, 0, len(holiday_dates) - 1)

    prev_dist = np.abs((current_dates - holiday_dates[prev_idx]).astype("timedelta64[D]").astype(float))
    next_dist = np.abs((holiday_dates[next_idx] - current_dates).astype("timedelta64[D]").astype(float))
    out.loc[valid_dates, "days_to_nearest_holiday"] = np.minimum(prev_dist, next_dist)
    return out


def add_weekend_flag(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add binary is_weekend flag.

    Parameters
    ----------
    df : pd.DataFrame
    date_col : str

    Returns
    -------
    pd.DataFrame
    """
    out = _ensure_datetime(df, date_col)
    if date_col not in out.columns:
        return out

    out["is_weekend"] = (out[date_col].dt.dayofweek >= 5).fillna(False).astype("Int8")
    return out


def build_time_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Run the full time-feature pipeline.

    Parameters
    ----------
    df : pd.DataFrame
    date_col : str

    Returns
    -------
    pd.DataFrame
        DataFrame with all time-based features added.
    """
    out = add_basic_date_features(df, date_col=date_col)
    out = add_holiday_proximity(out, date_col=date_col)
    out = add_weekend_flag(out, date_col=date_col)
    return out
