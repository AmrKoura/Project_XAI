"""
Q4 — Counterfactual analysis.

Answers: "What would happen to the forecast if a promotion were added or removed?"

Modifies individual features (promotion status, price, lead time) and
re-runs the model to show the causal-like effect on the forecast.
Partial Dependence Plots (PDP) are used as supporting evidence.
"""

import pandas as pd
import numpy as np


def generate_counterfactual(
    model: object,
    X_row: pd.Series,
    feature: str,
    new_value: float | int,
) -> dict[str, float]:
    """Generate a what-if counterfactual for a single feature change.

    Parameters
    ----------
    model : object
    X_row : pd.Series
        Original feature values for one instance.
    feature : str
        Feature to modify (e.g. ``'promo'``, ``'sell_price'``).
    new_value : float | int
        New value for the feature.

    Returns
    -------
    dict[str, float]
        ``{'original_pred': ..., 'counterfactual_pred': ..., 'delta': ...}``
    """
    ...


def batch_counterfactuals(
    model: object,
    X_row: pd.Series,
    feature: str,
    values: list[float | int],
) -> pd.DataFrame:
    """Sweep a feature across multiple values and record predictions.

    Parameters
    ----------
    model : object
    X_row : pd.Series
    feature : str
    values : list[float | int]

    Returns
    -------
    pd.DataFrame
        Columns: ``['feature_value', 'prediction', 'delta']``
    """
    ...


def partial_dependence(
    model: object,
    X: pd.DataFrame,
    feature: str,
    grid_resolution: int = 50,
) -> pd.DataFrame:
    """Compute Partial Dependence Plot data for a feature.

    Parameters
    ----------
    model : object
    X : pd.DataFrame
    feature : str
    grid_resolution : int

    Returns
    -------
    pd.DataFrame
        Columns: ``['feature_value', 'avg_prediction']``.
    """
    ...


def generate_counterfactual_text(
    sku_id: str,
    feature: str,
    original: float,
    new_val: float,
    delta: float,
) -> str:
    """Generate a natural-language what-if statement.

    Parameters
    ----------
    sku_id, feature : str
    original, new_val, delta : float

    Returns
    -------
    str
    """
    ...
