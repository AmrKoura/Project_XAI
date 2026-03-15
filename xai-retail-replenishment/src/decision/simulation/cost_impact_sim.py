"""
Cost-impact simulation.

Runs Monte Carlo simulations over the quantile forecast distribution
to quantify the financial impact of different order quantities.
Used in the What-If simulator dashboard page.
"""

import pandas as pd
import numpy as np


def simulate_order_cost(
    q10: float,
    q50: float,
    q90: float,
    order_qty: float,
    unit_margin: float,
    holding_cost_per_unit: float,
    n_simulations: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """Simulate expected total cost for a given order quantity.

    Parameters
    ----------
    q10, q50, q90 : float
    order_qty : float
    unit_margin : float
    holding_cost_per_unit : float
    n_simulations : int
    seed : int

    Returns
    -------
    dict[str, float]
        ``{'mean_stockout_cost', 'mean_overstock_cost', 'mean_total_cost',
           'p5_total_cost', 'p95_total_cost'}``
    """
    ...


def compare_order_quantities(
    q10: float,
    q50: float,
    q90: float,
    quantities: list[float],
    unit_margin: float,
    holding_cost_per_unit: float,
) -> pd.DataFrame:
    """Compare cost profiles across multiple candidate order quantities.

    Parameters
    ----------
    q10, q50, q90 : float
    quantities : list[float]
    unit_margin, holding_cost_per_unit : float

    Returns
    -------
    pd.DataFrame
        One row per quantity with cost breakdown.
    """
    ...
