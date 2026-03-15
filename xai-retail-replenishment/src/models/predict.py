"""
Generate point and quantile predictions from trained models.

Produces forecast DataFrames with columns for the median prediction
and the 10th/90th percentile bounds used in safety stock calculations.
"""

import pandas as pd
import numpy as np


def predict_point(model: object, X: pd.DataFrame) -> np.ndarray:
    """Generate point (median) forecasts.

    Parameters
    ----------
    model : object
        Trained LightGBM / XGBoost model.
    X : pd.DataFrame
        Feature matrix.

    Returns
    -------
    np.ndarray
        Predicted values.
    """
    ...


def predict_quantiles(
    models: dict[str, object],
    X: pd.DataFrame,
    quantiles: list[float] | None = None,
) -> pd.DataFrame:
    """Generate quantile forecasts (10th, 50th, 90th percentiles).

    Parameters
    ----------
    models : dict[str, object]
        Dict mapping quantile labels to trained models.
    X : pd.DataFrame
    quantiles : list[float] | None

    Returns
    -------
    pd.DataFrame
        Columns: ``['q10', 'q50', 'q90']``.
    """
    ...


def forecast_lead_time(
    models: dict[str, object],
    X_future: pd.DataFrame,
    lead_time_days: int = 7,
) -> pd.DataFrame:
    """Aggregate forecasts over the lead-time horizon.

    Parameters
    ----------
    models : dict[str, object]
    X_future : pd.DataFrame
    lead_time_days : int

    Returns
    -------
    pd.DataFrame
        Total demand forecast and uncertainty bounds for the lead-time period.
    """
    ...
