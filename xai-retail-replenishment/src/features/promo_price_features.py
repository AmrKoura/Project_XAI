"""
Promotional and price-related feature engineering.

Captures the effect of active promotions, price changes,
competitor pricing, and discount depth on SKU-level demand.
"""

import pandas as pd
import numpy as np


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
    ...


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
    ...


def add_competitor_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add competitor-relative pricing features (if available).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    ...


def build_promo_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full promotional/price feature pipeline.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    ...
