"""
Safety stock calculation.

Supports two methods:
  - Quantile-based: safety stock = q90 − q50 (from quantile forecasts)
  - MAD-based: safety stock = z × 1.4826 × MAD × √lead_time
"""

import math
import numpy as np
from scipy import stats


def safety_stock_quantile(
    q50: float,
    q90: float,
    lead_time_days: int = 7,
    forecast_horizon: int = 7,
) -> float:
    """Compute safety stock from quantile forecast spread.

    Safety stock = (q90 − q50) × √(lead_time / forecast_horizon).

    Dividing by √forecast_horizon normalises the spread back to a per-day
    uncertainty, then multiplying by √lead_time scales it to the lead-time
    window. This keeps safety stock consistent across different forecast
    windows (7/14/28-day) since forecast errors scale with √horizon.

    Parameters
    ----------
    q50 : float
        Median forecast for the window period.
    q90 : float
        90th-percentile forecast for the window period.
    lead_time_days : int
        Replenishment lead time in days (default 7).
    forecast_horizon : int
        Number of days in the forecast window (7, 14, or 28).

    Returns
    -------
    float
        Safety stock units (≥ 0).
    """
    spread = max(0.0, q90 - q50)
    scale  = np.sqrt(lead_time_days / float(forecast_horizon))
    return float(math.ceil(spread * scale))


def safety_stock_mad(
    mad: float,
    lead_time_days: int = 7,
    service_level: float = 0.95,
) -> float:
    """Compute safety stock using MAD and a z-score for the target service level.

    Formula: SS = z × (1.4826 × MAD) × √lead_time_days
    The 1.4826 factor converts MAD to an equivalent standard deviation
    under the assumption of normally-distributed forecast errors.

    Parameters
    ----------
    mad : float
        Mean Absolute Deviation of forecast errors.
    lead_time_days : int
        Replenishment lead time in days.
    service_level : float
        Target fill-rate / cycle-service level (e.g. 0.95).

    Returns
    -------
    float
        Safety stock units (≥ 0).
    """
    if not (0 < service_level < 1):
        raise ValueError("service_level must be in (0, 1).")

    z     = float(stats.norm.ppf(service_level))
    sigma = 1.4826 * mad
    return float(math.ceil(z * sigma * np.sqrt(lead_time_days)))


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
        Arguments forwarded to the specific method:
        - quantile: ``q50``, ``q90``, ``lead_time_days``
        - mad:      ``mad``, ``lead_time_days``, ``service_level``

    Returns
    -------
    float
    """
    if method == "quantile":
        return safety_stock_quantile(**kwargs)
    elif method == "mad":
        return safety_stock_mad(**kwargs)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'quantile' or 'mad'.")
