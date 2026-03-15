"""
Q7 — Stockout distortion analysis.

Answers: "Could past stockouts have distorted the demand signal for this SKU?"

Uses SHAP on a stockout flag feature and counterfactual reasoning to
assess whether historical zero-sales periods were true zero-demand
or supply-side censored data (stockouts masking real demand).
"""

import pandas as pd
import numpy as np
import shap


def flag_potential_stockouts(
    df: pd.DataFrame,
    sales_col: str = "sales",
    stock_col: str = "stock_on_hand",
) -> pd.DataFrame:
    """Add a binary stockout indicator based on zero sales + low stock.

    Parameters
    ----------
    df : pd.DataFrame
    sales_col, stock_col : str

    Returns
    -------
    pd.DataFrame
        With new ``'is_stockout'`` column.
    """
    ...


def shap_on_stockout_flag(
    model: object,
    X: pd.DataFrame,
    stockout_feature: str = "is_stockout",
) -> pd.DataFrame:
    """Measure the SHAP contribution of the stockout flag across instances.

    Parameters
    ----------
    model : object
    X : pd.DataFrame
    stockout_feature : str

    Returns
    -------
    pd.DataFrame
        Rows where stockout flag had a significant SHAP impact.
    """
    ...


def estimate_censored_demand(
    df: pd.DataFrame,
    sku_id: str,
) -> pd.DataFrame:
    """Estimate what demand would have been during stockout periods.

    Uses surrounding non-stockout periods to interpolate likely demand.

    Parameters
    ----------
    df : pd.DataFrame
    sku_id : str

    Returns
    -------
    pd.DataFrame
        With ``'estimated_demand'`` column for stockout periods.
    """
    ...


def generate_stockout_text(
    sku_id: str,
    n_stockout_periods: int,
    avg_shap_impact: float,
) -> str:
    """Generate a natural-language stockout distortion warning.

    Parameters
    ----------
    sku_id : str
    n_stockout_periods : int
    avg_shap_impact : float

    Returns
    -------
    str
    """
    ...
