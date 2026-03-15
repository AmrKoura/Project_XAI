"""
Q3 + Q9 — Forecast uncertainty and cold-start analysis.

Answers:
  Q3: "How confident is the model in this forecast?"
  Q9: "Is the model less reliable for this SKU due to limited history?"

Uses quantile forecasts, prediction intervals, and subgroup evaluation
to quantify confidence and flag cold-start / low-data SKUs.
"""

import pandas as pd
import numpy as np


def compute_prediction_interval(
    q10: float,
    q90: float,
) -> dict[str, float]:
    """Return the prediction interval width and bounds.

    Parameters
    ----------
    q10, q90 : float
        10th and 90th percentile predictions.

    Returns
    -------
    dict[str, float]
        ``{'lower': q10, 'upper': q90, 'width': q90 - q10}``
    """
    ...


def confidence_label(interval_width: float, median: float) -> str:
    """Map forecast uncertainty to a human-readable confidence label.

    Parameters
    ----------
    interval_width : float
    median : float

    Returns
    -------
    str
        One of ``'high'``, ``'moderate'``, ``'low'``.
    """
    ...


def detect_cold_start_skus(
    df: pd.DataFrame,
    sku_col: str = "item_id",
    date_col: str = "date",
    min_history_days: int = 90,
) -> list[str]:
    """Identify SKUs with insufficient historical data.

    Parameters
    ----------
    df : pd.DataFrame
    sku_col, date_col : str
    min_history_days : int

    Returns
    -------
    list[str]
        List of SKU IDs flagged as cold-start.
    """
    ...


def subgroup_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: pd.Series,
) -> pd.DataFrame:
    """Evaluate forecast accuracy per subgroup (e.g. new vs established SKUs).

    Parameters
    ----------
    y_true, y_pred : np.ndarray
    groups : pd.Series
        Group labels for each observation.

    Returns
    -------
    pd.DataFrame
        Per-group metrics.
    """
    ...


def generate_confidence_text(
    sku_id: str,
    median: float,
    q10: float,
    q90: float,
    is_cold_start: bool,
) -> str:
    """Generate a natural-language confidence statement for a SKU.

    Parameters
    ----------
    sku_id : str
    median, q10, q90 : float
    is_cold_start : bool

    Returns
    -------
    str
    """
    ...
