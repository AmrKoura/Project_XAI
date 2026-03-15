"""
Q6 — Temporal SHAP analysis.

Answers: "Is this SKU experiencing a demand spike or stable growth?"

Groups SHAP values by time-related features to reveal whether recent
demand changes are driven by seasonal patterns, trends, or anomalies.
"""

import pandas as pd
import numpy as np
import shap


def compute_temporal_shap(
    model: object,
    X: pd.DataFrame,
    time_features: list[str] | None = None,
) -> pd.DataFrame:
    """Compute SHAP contributions from time-related features over periods.

    Parameters
    ----------
    model : object
    X : pd.DataFrame
    time_features : list[str] | None
        Time-related feature names (e.g. ``['day_of_week', 'month', 'week_of_year']``).

    Returns
    -------
    pd.DataFrame
        SHAP contributions grouped by time period.
    """
    ...


def classify_demand_pattern(
    temporal_shap_df: pd.DataFrame,
    sku_id: str,
) -> str:
    """Classify demand pattern as spike, stable growth, seasonal, or declining.

    Parameters
    ----------
    temporal_shap_df : pd.DataFrame
    sku_id : str

    Returns
    -------
    str
        Pattern label.
    """
    ...


def generate_temporal_explanation_text(
    sku_id: str,
    pattern: str,
    top_time_features: pd.DataFrame,
) -> str:
    """Generate a natural-language explanation of the temporal demand pattern.

    Parameters
    ----------
    sku_id : str
    pattern : str
    top_time_features : pd.DataFrame

    Returns
    -------
    str
    """
    ...
