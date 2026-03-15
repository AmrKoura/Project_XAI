"""
Safety stock calculation.

Supports two methods:
  - Quantile-based: safety stock = q90 − q50 (from quantile forecasts)
  - MAD-based: safety stock = z × MAD × √lead_time
"""

import numpy as np
from scipy import stats


def safety_stock_quantile(
    q50: float,
    q90: float,
    lead_time_days: int = 7,
) -> float:
    """Compute safety stock from quantile forecast spread.

    Parameters
    ----------
    q50 : float
        Median forecast for the lead-time period.
    q90 : float
        90th percentile forecast.
    lead_time_days : int

    Returns
    -------
    float
    """
    ...


def safety_stock_mad(
    mad: float,
    lead_time_days: int = 7,
    service_level: float = 0.95,
) -> float:
    """Compute safety stock using MAD and a z-score for the target service level.

    Parameters
    ----------
    mad : float
        Mean Absolute Deviation of forecast errors.
    lead_time_days : int
    service_level : float
        Target service level (e.g. 0.95).

    Returns
    -------
    float
    """
    ...


def compute_safety_stock(
    method: str = "quantile",
    **kwargs,
) -> float:
    """Dispatch to the appropriate safety stock calculation method.

    Parameters
    ----------
    method : str
        ``'quantile'`` or ``'mad'``.
    **kwargs
        Arguments forwarded to the specific method.

    Returns
    -------
    float
    """
    ...
