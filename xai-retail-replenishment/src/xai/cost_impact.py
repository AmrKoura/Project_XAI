"""
Q10 — Business cost-impact modeling.

Answers: "What is the financial cost of under- or over-ordering
          based on this forecast?"

Uses probabilistic simulation across the quantile forecast distribution
to estimate expected stockout cost, overstock holding cost, and the
optimal order quantity for a given service-level target.
"""

import pandas as pd
import numpy as np


def compute_stockout_cost(
    demand: float,
    ordered: float,
    unit_margin: float,
) -> float:
    """Estimate lost-sales cost when order < actual demand.

    Parameters
    ----------
    demand : float
        Actual or forecast demand.
    ordered : float
        Quantity ordered.
    unit_margin : float
        Revenue margin per unit.

    Returns
    -------
    float
        Estimated stockout cost.
    """
    ...


def compute_overstock_cost(
    demand: float,
    ordered: float,
    holding_cost_per_unit: float,
) -> float:
    """Estimate holding cost when order > actual demand.

    Parameters
    ----------
    demand : float
    ordered : float
    holding_cost_per_unit : float

    Returns
    -------
    float
    """
    ...


def simulate_cost_distribution(
    q10: float,
    q50: float,
    q90: float,
    order_qty: float,
    unit_margin: float,
    holding_cost: float,
    n_simulations: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    """Monte Carlo simulation of total cost across demand scenarios.

    Parameters
    ----------
    q10, q50, q90 : float
        Quantile forecasts defining the demand distribution.
    order_qty : float
    unit_margin, holding_cost : float
    n_simulations : int
    seed : int

    Returns
    -------
    pd.DataFrame
        Columns: ``['demand', 'stockout_cost', 'overstock_cost', 'total_cost']``.
    """
    ...


def optimal_order_quantity(
    q10: float,
    q50: float,
    q90: float,
    unit_margin: float,
    holding_cost: float,
    service_level: float = 0.95,
) -> float:
    """Find the order quantity that minimises expected total cost.

    Parameters
    ----------
    q10, q50, q90 : float
    unit_margin, holding_cost : float
    service_level : float

    Returns
    -------
    float
    """
    ...


def generate_cost_impact_text(
    sku_id: str,
    order_qty: float,
    expected_stockout_cost: float,
    expected_overstock_cost: float,
) -> str:
    """Generate a natural-language cost-impact summary.

    Parameters
    ----------
    sku_id : str
    order_qty : float
    expected_stockout_cost, expected_overstock_cost : float

    Returns
    -------
    str
    """
    ...
