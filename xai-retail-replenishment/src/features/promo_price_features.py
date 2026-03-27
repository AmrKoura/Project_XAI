"""
Promotional and price-related feature engineering.

Captures the effect of active promotions, price changes,
competitor pricing, and discount depth on SKU-level demand.
"""

import pandas as pd
import numpy as np


def _pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _resolve_group_cols(df: pd.DataFrame) -> list[str]:
    candidates = [["store_id", "item_id"], ["Store", "item_id"], ["Store", "id"], ["id"]]
    for cols in candidates:
        if all(col in df.columns for col in cols):
            return cols
    return []


def add_promo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add promotion-related flags and duration features.

    Includes is_promo, promo_duration_days, days_since_last_promo, etc.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    out = df.copy()

    # Event-driven promo proxy.
    event_cols = [col for col in ["event_name_1", "event_type_1", "event_name_2", "event_type_2"] if col in out.columns]
    if event_cols:
        event_nonempty = pd.Series(False, index=out.index)
        for col in event_cols:
            event_nonempty = event_nonempty | (out[col].notna() & (out[col].astype(str).str.strip() != ""))
        out["is_event_day"] = event_nonempty.astype("int8")
    else:
        out["is_event_day"] = 0

    snap_cols = [col for col in ["snap_CA", "snap_TX", "snap_WI"] if col in out.columns]
    if snap_cols:
        out["snap_any"] = out[snap_cols].fillna(0).max(axis=1).astype("int8")
    else:
        out["snap_any"] = 0

    if {"state_id", "snap_CA", "snap_TX", "snap_WI"}.issubset(out.columns):
        state = out["state_id"].astype(str).str.upper()
        out["snap_relevant"] = np.select(
            [state.eq("CA"), state.eq("TX"), state.eq("WI")],
            [out["snap_CA"], out["snap_TX"], out["snap_WI"]],
            default=0,
        ).astype("int8")
    else:
        out["snap_relevant"] = out["snap_any"].astype("int8")

    promo_col = _pick_first_existing(out, ["Promo", "promo", "promo_flag"])
    if promo_col is not None:
        direct_promo = pd.to_numeric(out[promo_col], errors="coerce").fillna(0).clip(lower=0, upper=1)
    else:
        direct_promo = pd.Series(0, index=out.index, dtype=float)

    out["is_promo"] = np.maximum.reduce([
        np.asarray(direct_promo, dtype=float),
        np.asarray(out["is_event_day"], dtype=float),
        np.asarray(out["snap_relevant"], dtype=float),
    ]).astype("int8")

    date_col = _pick_first_existing(out, ["date", "Date"])
    group_cols = _resolve_group_cols(out)
    if date_col is None:
        out["promo_duration_days"] = out["is_promo"].astype("int16")
        out["days_since_last_promo"] = np.nan
        return out

    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    sort_cols = [*group_cols, date_col] if group_cols else [date_col]
    out = out.sort_values(sort_cols).reset_index(drop=True)

    if group_cols:
        grp = out.groupby(group_cols, sort=False)
        block_id = (out["is_promo"] != grp["is_promo"].shift(1).fillna(0)).groupby([out[c] for c in group_cols], sort=False).cumsum()
        run_len = out.groupby([*group_cols, block_id], sort=False).cumcount() + 1
    else:
        block_id = (out["is_promo"] != out["is_promo"].shift(1).fillna(0)).cumsum()
        run_len = out.groupby(block_id, sort=False).cumcount() + 1

    out["promo_duration_days"] = np.where(out["is_promo"] == 1, run_len, 0).astype("int16")

    if group_cols:
        last_promo_date = out[date_col].where(out["is_promo"] == 1).groupby([out[c] for c in group_cols], sort=False).ffill()
    else:
        last_promo_date = out[date_col].where(out["is_promo"] == 1).ffill()
    out["days_since_last_promo"] = (out[date_col] - last_promo_date).dt.days

    return out


def add_price_features(df: pd.DataFrame, price_col: str = "sell_price") -> pd.DataFrame:
    """Add price-change indicators and discount depth.

    Includes price_change_pct, is_price_drop, discount_depth, etc.

    Parameters
    ----------
    df : pd.DataFrame
    price_col : str

    Returns
    -------
    pd.DataFrame
    """
    out = df.copy()
    if price_col not in out.columns:
        alt = _pick_first_existing(out, ["sell_price", "price", "Price"])
        if alt is None:
            return out
        price_col = alt

    out[price_col] = pd.to_numeric(out[price_col], errors="coerce")

    date_col = _pick_first_existing(out, ["date", "Date"])
    group_cols = _resolve_group_cols(out)
    if date_col is not None:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        sort_cols = [*group_cols, date_col] if group_cols else [date_col]
        out = out.sort_values(sort_cols).reset_index(drop=True)

    if group_cols:
        grp = out.groupby(group_cols, sort=False)[price_col]
        out["price_change_pct"] = grp.pct_change()
        out["price_delta"] = grp.diff()
        rolling_ref = grp.transform(lambda s: s.shift(1).rolling(28, min_periods=1).median())
    else:
        out["price_change_pct"] = out[price_col].pct_change()
        out["price_delta"] = out[price_col].diff()
        rolling_ref = out[price_col].shift(1).rolling(28, min_periods=1).median()

    out["is_price_drop"] = (out["price_change_pct"] < 0).fillna(False).astype("int8")
    out["is_price_increase"] = (out["price_change_pct"] > 0).fillna(False).astype("int8")

    denom = rolling_ref.replace(0, np.nan)
    out["discount_depth"] = ((denom - out[price_col]) / denom).clip(lower=0).fillna(0.0)
    out["price_deviation_28"] = (out[price_col] - rolling_ref).fillna(0.0)
    return out


def add_competitor_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add competitor-relative pricing features (if available).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    out = df.copy()
    own_price = _pick_first_existing(out, ["sell_price", "price", "Price"])
    comp_price = _pick_first_existing(out, ["competitor_price", "CompetitionPrice", "comp_price"])

    if own_price is not None and comp_price is not None:
        own = pd.to_numeric(out[own_price], errors="coerce")
        comp = pd.to_numeric(out[comp_price], errors="coerce")
        out["price_to_competitor_ratio"] = own / comp.replace(0, np.nan)
        out["price_gap_vs_competitor"] = own - comp

    if "CompetitionDistance" in out.columns:
        dist = pd.to_numeric(out["CompetitionDistance"], errors="coerce")
        out["competition_pressure"] = 1.0 / (1.0 + dist)

    return out


def build_promo_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full promotional/price feature pipeline.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    out = add_promo_features(df)
    out = add_price_features(out)
    out = add_competitor_price_features(out)
    return out
