"""
Train baseline forecasting models (ARIMA, Exponential Smoothing).

These classical statistical models serve as benchmarks against the
primary LightGBM forecaster, evaluated on the same train/test splits.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def train_arima(
    train: pd.Series,
    order: tuple[int, int, int] = (1, 1, 1),
) -> object:
    """Fit an ARIMA model on a single SKU time series.

    Parameters
    ----------
    train : pd.Series
        Training sales series.
    order : tuple[int, int, int]
        (p, d, q) order.

    Returns
    -------
    object
        Fitted ARIMA result.
    """
    ...


def train_exponential_smoothing(
    train: pd.Series,
    seasonal_periods: int = 7,
) -> object:
    """Fit a Holt-Winters Exponential Smoothing model.

    Parameters
    ----------
    train : pd.Series
    seasonal_periods : int

    Returns
    -------
    object
        Fitted ExponentialSmoothing result.
    """
    ...


def train_all_baselines(
    train: pd.Series,
) -> dict[str, object]:
    """Train all baseline models and return a dict of fitted objects.

    Parameters
    ----------
    train : pd.Series

    Returns
    -------
    dict[str, object]
        ``{'arima': ..., 'exp_smoothing': ...}``
    """
    ...
