"""
Replenishment rules engine.

Computes reorder quantities based on:
  - Median (point) forecast over the lead-time period
  - Safety stock buffer from quantile uncertainty
  - Variable lead time
  - Current stock-on-hand

Formula: Reorder Qty = max(0, forecast_demand + safety_stock − stock_on_hand)
"""

import math
import pandas as pd
import numpy as np


def compute_reorder_quantity(
    forecast_demand: float,
    safety_stock: float,
    stock_on_hand: float,
) -> float:
    """Calculate the recommended reorder quantity.

    Parameters
    ----------
    forecast_demand : float
        Median forecast demand over the lead-time period.
    safety_stock : float
        Safety stock buffer.
    stock_on_hand : float
        Current inventory level.

    Returns
    -------
    float
        Reorder quantity (floored at 0).
    """
    return float(math.ceil(max(0.0, forecast_demand + safety_stock - stock_on_hand)))


def should_reorder(
    stock_on_hand: float,
    reorder_point: float,
) -> bool:
    """Determine whether a reorder should be triggered.

    A reorder is triggered when current stock falls at or below the
    reorder point (ROP).

    Parameters
    ----------
    stock_on_hand : float
    reorder_point : float

    Returns
    -------
    bool
    """
    return stock_on_hand <= reorder_point


def compute_reorder_point(
    avg_daily_demand: float,
    lead_time_days: int,
    safety_stock: float,
) -> float:
    """Calculate the reorder point (ROP).

    ROP = (avg daily demand × lead time) + safety stock

    Parameters
    ----------
    avg_daily_demand : float
    lead_time_days : int
    safety_stock : float

    Returns
    -------
    float
    """
    return float(avg_daily_demand * lead_time_days + safety_stock)


def generate_replenishment_card(
    sku_id: str,
    forecast: dict,
    stock_on_hand: float,
    lead_time_days: int,
    safety_stock: float,
) -> dict:
    """Build a replenishment summary card for a single SKU.

    Parameters
    ----------
    sku_id : str
    forecast : dict
        Keys: ``'q10'``, ``'q50'``, ``'q90'``.
    stock_on_hand : float
    lead_time_days : int
    safety_stock : float

    Returns
    -------
    dict
        Keys:
        ``'sku_id'``, ``'forecast_q50'``, ``'forecast_q10'``, ``'forecast_q90'``,
        ``'safety_stock'``, ``'stock_on_hand'``, ``'lead_time_days'``,
        ``'reorder_point'``, ``'reorder_qty'``, ``'trigger_reorder'``,
        ``'urgency'``, ``'confidence_band'``.
    """
    q50 = float(forecast["q50"])
    q10 = float(forecast["q10"])
    q90 = float(forecast["q90"])

    avg_daily = q50 / 7.0
    rop        = compute_reorder_point(avg_daily, lead_time_days, safety_stock)
    reorder_qty = compute_reorder_quantity(q50, safety_stock, stock_on_hand)
    trigger     = should_reorder(stock_on_hand, rop)

    # Urgency: driven by how many units need to be ordered
    days_of_stock = (stock_on_hand / avg_daily) if avg_daily > 0 else float("inf")
    if reorder_qty == 0:
        urgency = "LOW"
    elif reorder_qty <= 10:
        urgency = "HIGH"
    else:
        urgency = "CRITICAL"

    # Confidence band width as % of median forecast
    band_width = q90 - q10
    confidence_band_pct = (band_width / q50 * 100) if q50 > 0 else float("nan")

    return {
        "sku_id":              sku_id,
        "forecast_q10":        int(round(q10)),
        "forecast_q50":        int(round(q50)),
        "forecast_q90":        int(round(q90)),
        "safety_stock":        int(safety_stock),
        "stock_on_hand":       round(stock_on_hand, 1),
        "lead_time_days":      lead_time_days,
        "days_of_stock":       round(days_of_stock, 1),
        "reorder_point":       round(rop, 1),
        "reorder_qty":         int(reorder_qty),
        "trigger_reorder":     trigger,
        "urgency":             urgency,
        "confidence_band_pct": round(confidence_band_pct, 1),
    }
