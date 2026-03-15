"""
Replenishment rules engine.

Computes reorder quantities based on:
  - Median (point) forecast over the lead-time period
  - Safety stock buffer from quantile uncertainty
  - Variable lead time
  - Current stock-on-hand

Formula: Reorder Qty = Total Demand Estimate − Current Stock
"""

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
    ...


def should_reorder(
    stock_on_hand: float,
    reorder_point: float,
) -> bool:
    """Determine whether a reorder should be triggered.

    Parameters
    ----------
    stock_on_hand : float
    reorder_point : float

    Returns
    -------
    bool
    """
    ...


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
    ...


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
        Card data with reorder qty, urgency flag, confidence, etc.
    """
    ...
