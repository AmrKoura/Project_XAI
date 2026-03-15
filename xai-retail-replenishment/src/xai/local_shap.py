"""
Q1 — Local SHAP explanations.

Answers: "Why is the system recommending a reorder for this SKU right now?"

Computes per-instance SHAP values to produce waterfall charts and
natural-language explanations for individual SKU forecasts.
"""

import pandas as pd
import numpy as np
import shap


def compute_local_shap(
    model: object,
    X: pd.DataFrame,
    idx: int,
) -> shap.Explanation:
    """Compute SHAP values for a single prediction.

    Parameters
    ----------
    model : object
        Trained LightGBM / XGBoost model.
    X : pd.DataFrame
        Feature matrix.
    idx : int
        Row index of the SKU instance to explain.

    Returns
    -------
    shap.Explanation
    """
    ...


def get_top_contributors(
    shap_values: shap.Explanation,
    n: int = 5,
) -> pd.DataFrame:
    """Extract the top-N features contributing to the prediction.

    Parameters
    ----------
    shap_values : shap.Explanation
    n : int

    Returns
    -------
    pd.DataFrame
        Columns: ``['feature', 'shap_value', 'feature_value']``.
    """
    ...


def generate_local_explanation_text(
    shap_values: shap.Explanation,
    forecast_value: float,
    sku_id: str,
) -> str:
    """Generate a plain-English explanation for a single SKU forecast.

    Parameters
    ----------
    shap_values : shap.Explanation
    forecast_value : float
    sku_id : str

    Returns
    -------
    str
        Human-readable explanation string.
    """
    ...
