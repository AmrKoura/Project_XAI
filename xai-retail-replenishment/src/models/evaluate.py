"""
Evaluate forecast models using accuracy, bias, and business metrics.

Metrics:
  - Accuracy  : MAE, RMSE, SMAPE, Quantile Loss (Pinball)
  - Bias      : Forecast Bias, MAD, Tracking Signal
  - Business  : Value Add, Service Level, Stockout Rate, Inventory Turnover
"""

import pandas as pd
import numpy as np


# ── Accuracy metrics ─────────────────────────────────────────────────────────

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    ...


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    ...


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error."""
    ...


def quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """Pinball (quantile) loss at quantile *q*.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
    q : float
        Quantile level (e.g. 0.1, 0.5, 0.9).

    Returns
    -------
    float
    """
    ...


# ── Bias metrics ─────────────────────────────────────────────────────────────

def forecast_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean of (forecast − actual)."""
    ...


def mean_absolute_deviation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAD — used in safety stock calculation."""
    ...


def tracking_signal(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Cumulative bias / MAD — alerts when |TS| > 4."""
    ...


# ── Business / inventory KPIs ───────────────────────────────────────────────

def value_add(y_true: np.ndarray, y_pred: np.ndarray, y_naive: np.ndarray) -> float:
    """Percentage MAE improvement over a naïve baseline.

    Parameters
    ----------
    y_true, y_pred, y_naive : np.ndarray
    """
    ...


def service_level(demand: np.ndarray, available_stock: np.ndarray) -> float:
    """Fraction of periods where stock met demand."""
    ...


def stockout_rate(available_stock: np.ndarray) -> float:
    """Fraction of periods where stock reached zero."""
    ...


def inventory_turnover(total_sales: float, avg_inventory: float) -> float:
    """Sales / average inventory over the period."""
    ...


# ── Aggregation ──────────────────────────────────────────────────────────────

def evaluate_all(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_naive: np.ndarray | None = None,
    quantile_preds: dict[float, np.ndarray] | None = None,
) -> dict[str, float]:
    """Compute all configured metrics and return a summary dict.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
    y_naive : np.ndarray | None
    quantile_preds : dict[float, np.ndarray] | None

    Returns
    -------
    dict[str, float]
    """
    ...
