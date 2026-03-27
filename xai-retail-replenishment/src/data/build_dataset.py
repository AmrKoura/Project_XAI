"""
Build model-ready datasets from cleaned data.

Merges cleaned tables, applies feature engineering, creates
train/validation/test splits at the SKU-store level, and
saves the final datasets to the processed directory.
"""

import pandas as pd
from pathlib import Path

from .clean_data import generate_synthetic_inventory_fields
from features.time_features import build_time_features
from features.lag_features import build_lag_features
from features.promo_price_features import build_promo_price_features


INVENTORY_COLS = [
    "stock_on_hand",
    "lead_time_days",
    "reorder_cost",
    "supplier_min_order_qty",
    "supplier_order_multiple",
]


def _pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first matching column name from *candidates* found in DataFrame."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _coalesce_inventory_suffix_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Coalesce inventory columns from merge suffixes into canonical names.

    Handles patterns like ``lead_time_days_x``/``lead_time_days_y`` that can
    appear when upstream tables already contain synthetic inventory fields.
    """
    merged = df.copy()

    for base_col in INVENTORY_COLS:
        candidates = [
            col
            for col in [base_col, f"{base_col}_x", f"{base_col}_y"]
            if col in merged.columns
        ]
        if not candidates:
            continue

        merged[base_col] = merged[candidates[0]]
        for col in candidates[1:]:
            merged[base_col] = merged[base_col].combine_first(merged[col])

        drop_cols = [col for col in candidates if col != base_col]
        if drop_cols:
            merged = merged.drop(columns=drop_cols)

    return merged


def _impute_sparse_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Impute sparse fields needed by modeling logic."""
    out = df.copy()

    if "sell_price" in out.columns:
        out["sell_price"] = pd.to_numeric(out["sell_price"], errors="coerce")
        if {"store_id", "item_id"}.issubset(out.columns):
            out["sell_price"] = out["sell_price"].fillna(
                out.groupby(["store_id", "item_id"], sort=False)["sell_price"].transform("median")
            )
        if "item_id" in out.columns:
            out["sell_price"] = out["sell_price"].fillna(
                out.groupby("item_id", sort=False)["sell_price"].transform("median")
            )
        out["sell_price"] = out["sell_price"].fillna(out["sell_price"].median())

    if "is_outlier" in out.columns:
        out["is_outlier"] = pd.to_numeric(out["is_outlier"], errors="coerce").fillna(0).astype("Int8")

    return out


def _drop_rows_without_date(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing date since they cannot be placed on time axis."""
    out = df.copy()
    date_col = _pick_first_existing(out, ["date", "Date"])
    if date_col is None:
        return out

    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out[out[date_col].notna()].copy()
    return out


def _apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create model features for time, lag/rolling, and promo/price effects."""
    out = df.copy()
    date_col = _pick_first_existing(out, ["date", "Date"]) or "date"
    target_col = _pick_first_existing(out, ["sales", "Sales"]) or "sales"

    group_cols: list[str] | None = None
    if {"store_id", "item_id"}.issubset(out.columns):
        group_cols = ["store_id", "item_id"]
    elif {"Store", "id"}.issubset(out.columns):
        group_cols = ["Store", "id"]
    elif "id" in out.columns:
        group_cols = ["id"]

    out = build_time_features(out, date_col=date_col)
    out = build_lag_features(out, target_col=target_col, group_cols=group_cols)
    out = build_promo_price_features(out)

    # Fill expected NaNs from shifted lag/rolling windows and sparse promo history.
    fill_zero_prefixes = [
        f"{target_col}_lag_",
        f"{target_col}_roll_",
        "price_change_pct",
        "price_delta",
        "discount_depth",
        "price_deviation_28",
        "days_since_last_promo",
    ]
    for col in out.columns:
        if any(col.startswith(prefix) for prefix in fill_zero_prefixes):
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    flag_cols = [
        "is_event_day",
        "snap_any",
        "snap_relevant",
        "is_promo",
        "is_price_drop",
        "is_price_increase",
        "is_weekend",
        "is_holiday",
    ]
    for col in flag_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype("Int8")

    return out


def _build_rossmann_dataset(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge Rossmann train table with store metadata."""
    train = tables.get("train")
    store = tables.get("store")
    if train is None:
        raise ValueError("Rossmann merge requires a 'train' table.")

    merged = train.copy()
    if store is not None and "Store" in merged.columns and "Store" in store.columns:
        merged = merged.merge(store, on="Store", how="left")

    if "Date" in merged.columns:
        merged["Date"] = pd.to_datetime(merged["Date"], errors="coerce")

    merged["dataset_source"] = "rossmann"
    return merged


def _build_m5_dataset(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build long-format M5 training dataset by unpivoting day columns and joining context."""
    calendar = tables.get("calendar")
    prices = tables.get("prices")
    sales = tables.get("sales_train_validation")
    if sales is None:
        sales = tables.get("sales_train_evaluation")

    if calendar is None or prices is None or sales is None:
        raise ValueError(
            "M5 merge requires 'calendar', 'prices', and one sales table "
            "('sales_train_validation' or 'sales_train_evaluation')."
        )

    sales = sales.copy()
    day_cols = [col for col in sales.columns if col.startswith("d_")]
    if not day_cols:
        raise ValueError("M5 sales table does not include day columns (d_*).")

    # Keep all non-day columns as identifiers so optional synthetic fields survive the melt.
    present_id_cols = [col for col in sales.columns if col not in day_cols]

    long_sales = sales.melt(
        id_vars=present_id_cols,
        value_vars=day_cols,
        var_name="d",
        value_name="sales",
    )

    merged = long_sales.merge(calendar, on="d", how="left")
    if {"store_id", "item_id", "wm_yr_wk"}.issubset(merged.columns) and {"store_id", "item_id", "wm_yr_wk"}.issubset(prices.columns):
        merged = merged.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    if "date" in merged.columns:
        merged["date"] = pd.to_datetime(merged["date"], errors="coerce")

    merged["dataset_source"] = "m5"
    return merged


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
    if not tables:
        raise ValueError("Received empty tables dictionary.")

    keys = set(tables.keys())
    if {"train", "store"}.intersection(keys):
        return _build_rossmann_dataset(tables)
    if {"calendar", "prices"}.issubset(keys):
        return _build_m5_dataset(tables)

    raise ValueError(
        "Could not infer dataset type from table keys. "
        "Expected Rossmann-like keys ('train', 'store') or M5-like keys "
        "('calendar', 'prices', sales table)."
    )


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
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-9:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    if len(df) == 0:
        raise ValueError("Input DataFrame is empty.")

    date_col = _pick_first_existing(df, ["Date", "date"])

    working = df.copy()
    if date_col is not None:
        working[date_col] = pd.to_datetime(working[date_col], errors="coerce")
        working = working.sort_values(date_col).reset_index(drop=True)

        unique_dates = working[date_col].dropna().sort_values().unique()
        if len(unique_dates) >= 3:
            n_dates = len(unique_dates)
            train_end = max(1, int(n_dates * train_ratio))
            val_end = max(train_end + 1, int(n_dates * (train_ratio + val_ratio)))
            val_end = min(val_end, n_dates - 1)

            train_dates = set(unique_dates[:train_end])
            val_dates = set(unique_dates[train_end:val_end])
            test_dates = set(unique_dates[val_end:])

            train_df = working[working[date_col].isin(train_dates)].copy()
            val_df = working[working[date_col].isin(val_dates)].copy()
            test_df = working[working[date_col].isin(test_dates)].copy()
            return train_df, val_df, test_df

    # Fallback to row-order split if no usable date column exists.
    n = len(working)
    train_end = max(1, int(n * train_ratio))
    val_end = max(train_end + 1, int(n * (train_ratio + val_ratio)))
    val_end = min(val_end, n - 1)

    train_df = working.iloc[:train_end].copy()
    val_df = working.iloc[train_end:val_end].copy()
    test_df = working.iloc[val_end:].copy()
    return train_df, val_df, test_df


def build_and_save(tables: dict[str, pd.DataFrame], output_dir: str) -> None:
    """End-to-end pipeline: merge → split → save to disk.

    Parameters
    ----------
    tables : dict[str, pd.DataFrame]
    output_dir : str
        Path to ``data/processed``.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged = merge_tables(tables)
    merged = _coalesce_inventory_suffix_columns(merged)
    merged = _impute_sparse_columns(merged)
    merged = _drop_rows_without_date(merged)
    merged = generate_synthetic_inventory_fields(merged)
    merged = _apply_feature_engineering(merged)
    train_df, val_df, test_df = create_train_val_test_split(merged)

    # Base forecasting-ready splits.
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    # Full merged table for reproducibility.
    merged.to_csv(out_dir / "full_merged.csv", index=False)

    # Optional targeted views for downstream components.
    forecasting_cols = [
        col
        for col in merged.columns
        if col
        not in {
            "stock_on_hand",
            "lead_time_days",
            "reorder_cost",
            "supplier_min_order_qty",
            "supplier_order_multiple",
        }
    ]
    replenishment_cols = [
        col
        for col in merged.columns
        if col in {
            "Date",
            "date",
            "Store",
            "store_id",
            "item_id",
            "sku_id",
            "Sales",
            "sales",
            "stock_on_hand",
            "lead_time_days",
            "reorder_cost",
            "supplier_min_order_qty",
            "supplier_order_multiple",
        }
    ]

    merged[forecasting_cols].to_csv(out_dir / "forecasting_dataset.csv", index=False)
    if replenishment_cols:
        merged[replenishment_cols].to_csv(out_dir / "replenishment_dataset.csv", index=False)
