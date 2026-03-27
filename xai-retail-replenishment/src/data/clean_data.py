"""
Clean and preprocess raw data.

Handles missing values, dtype corrections, outlier detection,
and synthetic field generation for inventory-specific columns
(stock-on-hand, lead time, reorder cost) that may be absent
from public datasets.
"""

import pandas as pd
import numpy as np


def _pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first matching column name from *candidates* found in DataFrame."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


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
    cleaned = df.copy()

    # Drop columns that are effectively unusable.
    high_missing_cols = [
        col for col in cleaned.columns if cleaned[col].isna().mean() > 0.95
    ]
    if high_missing_cols:
        cleaned = cleaned.drop(columns=high_missing_cols)

    # Normalize mixed holiday codes early (Rossmann has int/string mix in StateHoliday).
    if "StateHoliday" in cleaned.columns:
        cleaned["StateHoliday"] = cleaned["StateHoliday"].astype(str)

    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns.tolist()
    datetime_cols = cleaned.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    categorical_cols = [
        col
        for col in cleaned.columns
        if col not in numeric_cols and col not in datetime_cols
    ]

    for col in numeric_cols:
        if cleaned[col].isna().any():
            if set(cleaned[col].dropna().unique()).issubset({0, 1}):
                cleaned[col] = cleaned[col].fillna(0)
            else:
                cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    for col in datetime_cols:
        if cleaned[col].isna().any():
            cleaned[col] = cleaned[col].ffill().bfill()

    for col in categorical_cols:
        if cleaned[col].isna().any():
            mode = cleaned[col].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "unknown"
            cleaned[col] = cleaned[col].fillna(fill_value)

    return cleaned


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
    filtered = df.copy()
    valid_cols = [
        col
        for col in columns
        if col in filtered.columns and pd.api.types.is_numeric_dtype(filtered[col])
    ]
    if not valid_cols:
        return filtered

    outlier_mask = pd.Series(False, index=filtered.index)

    # Use a conservative IQR threshold to avoid removing legitimate demand spikes.
    for col in valid_cols:
        q1 = filtered[col].quantile(0.25)
        q3 = filtered[col].quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            continue

        lower = q1 - 3.0 * iqr
        upper = q3 + 3.0 * iqr
        outlier_mask = outlier_mask | (filtered[col] < lower) | (filtered[col] > upper)

    filtered["is_outlier"] = outlier_mask.astype(int)
    filtered = filtered.loc[~outlier_mask].copy()
    return filtered


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
    enriched = df.copy()
    rng = np.random.default_rng(seed)

    date_col = _pick_first_existing(enriched, ["Date", "date"])
    sales_col = _pick_first_existing(enriched, ["Sales", "sales", "sales_units", "units_sold"])
    store_col = _pick_first_existing(enriched, ["Store", "store_id", "store"])
    sku_col = _pick_first_existing(enriched, ["item_id", "sku_id", "id"])
    assortment_col = _pick_first_existing(enriched, ["Assortment", "assortment"])
    store_type_col = _pick_first_existing(enriched, ["StoreType", "store_type"])

    if date_col is not None:
        enriched[date_col] = pd.to_datetime(enriched[date_col], errors="coerce")

    sort_cols = [col for col in [store_col, sku_col, date_col] if col is not None]
    if sort_cols:
        enriched = enriched.sort_values(sort_cols).reset_index(drop=True)

    # Simulated lead time by store type (or fallback distribution if missing).
    if "lead_time_days" not in enriched.columns:
        if store_type_col is not None:
            lt_ranges = {
                "a": (5, 9),
                "b": (6, 10),
                "c": (7, 12),
                "d": (8, 14),
            }
            st = enriched[store_type_col].astype(str).str.lower()
            lows = st.map(lambda x: lt_ranges.get(x, (5, 14))[0]).to_numpy()
            highs = st.map(lambda x: lt_ranges.get(x, (5, 14))[1]).to_numpy()
            enriched["lead_time_days"] = [
                int(rng.integers(low, high + 1)) for low, high in zip(lows, highs)
            ]
        else:
            enriched["lead_time_days"] = rng.integers(5, 15, size=len(enriched)).astype(int)

    # Simulated supplier constraint: minimum order quantity by assortment tier.
    if "supplier_min_order_qty" not in enriched.columns:
        if assortment_col is not None:
            base_moq = enriched[assortment_col].astype(str).str.lower().map(
                {"a": 12, "b": 18, "c": 24}
            ).fillna(12)
            multiplier = rng.choice([1, 2, 3], size=len(enriched), p=[0.6, 0.3, 0.1])
            enriched["supplier_min_order_qty"] = (base_moq.to_numpy() * multiplier).astype(int)
        else:
            enriched["supplier_min_order_qty"] = rng.choice([12, 24, 36], size=len(enriched)).astype(int)

    if "supplier_order_multiple" not in enriched.columns:
        enriched["supplier_order_multiple"] = rng.choice([6, 12], size=len(enriched), p=[0.7, 0.3]).astype(int)

    # Reorder cost based on store/product tiers + noise.
    if "reorder_cost" not in enriched.columns:
        store_tier = (
            pd.factorize(enriched[store_col])[0] % 4 if store_col is not None else np.zeros(len(enriched))
        )
        product_tier = (
            pd.factorize(enriched[sku_col])[0] % 5 if sku_col is not None else np.zeros(len(enriched))
        )
        noise = rng.uniform(0.0, 5.0, size=len(enriched))
        enriched["reorder_cost"] = np.round(20 + 2.0 * store_tier + 1.5 * product_tier + noise, 2)

    # Simulated stock-on-hand derived from recent sales demand + replenishment buffer.
    if "stock_on_hand" not in enriched.columns:
        if sales_col is None:
            base_demand = pd.Series(rng.integers(5, 40, size=len(enriched)), index=enriched.index)
        else:
            base_demand = pd.to_numeric(enriched[sales_col], errors="coerce").fillna(0)

        if store_col is not None and sku_col is not None and sales_col is not None:
            grp = enriched.groupby([store_col, sku_col], sort=False)
            rolling_mean = grp[sales_col].transform(lambda s: pd.to_numeric(s, errors="coerce").fillna(0).rolling(14, min_periods=1).mean())
            rolling_std = grp[sales_col].transform(lambda s: pd.to_numeric(s, errors="coerce").fillna(0).rolling(14, min_periods=1).std().fillna(0))
        else:
            rolling_mean = base_demand.rolling(14, min_periods=1).mean()
            rolling_std = base_demand.rolling(14, min_periods=1).std().fillna(0)

        safety_days = rng.integers(2, 7, size=len(enriched))
        target_cover = enriched["lead_time_days"].to_numpy() + safety_days
        simulated = (rolling_mean.to_numpy() * target_cover) + rng.normal(0, rolling_std.to_numpy() + 1.0)
        enriched["stock_on_hand"] = np.maximum(0, np.round(simulated)).astype(int)

    return enriched


def clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run the cleaning pipeline: missing values → outliers → dtype fixes.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Fully cleaned dataframe.
    """
    cleaned = handle_missing_values(df)

    numeric_candidates = [
        "Sales",
        "sales",
        "Customers",
        "customers",
        "sell_price",
        "CompetitionDistance",
    ]
    outlier_cols = [col for col in numeric_candidates if col in cleaned.columns]
    if outlier_cols:
        cleaned = remove_outliers(cleaned, outlier_cols)

    # Downcast numeric dtypes where safe to reduce memory footprint.
    int_cols = cleaned.select_dtypes(include=["int64", "Int64"]).columns
    for col in int_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], downcast="integer")

    float_cols = cleaned.select_dtypes(include=["float64"]).columns
    for col in float_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], downcast="float")

    # Convert low-cardinality object columns to categorical for efficiency.
    obj_cols = cleaned.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        nunique = cleaned[col].nunique(dropna=False)
        if nunique <= max(20, int(0.05 * len(cleaned))):
            cleaned[col] = cleaned[col].astype("category")

    return cleaned
