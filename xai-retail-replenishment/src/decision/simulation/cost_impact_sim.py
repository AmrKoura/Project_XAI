"""
Cost-impact simulation.

Thin wrapper around xai.cost_impact that provides a decision-layer
interface for the What-If simulator and dashboard callbacks.
"""

import pandas as pd
import numpy as np

from xai.cost_impact import simulate_cost_distribution, optimal_order_quantity


def simulate_order_cost(
    q10: float,
    q50: float,
    q90: float,
    order_qty: float,
    unit_margin: float,
    holding_cost_per_unit: float,
    n_simulations: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    """Simulate expected total cost for a given order quantity.

    Delegates to ``xai.cost_impact.simulate_cost_distribution`` and
    returns a compact summary dictionary suitable for dashboard display.

    Parameters
    ----------
    q10, q50, q90 : float
        Quantile forecasts from uncertainty.py.
    order_qty : float
        The order quantity to evaluate.
    unit_margin : float
        Margin per unit (cost of a lost sale).
    holding_cost_per_unit : float
        Cost to hold one excess unit for the period.
    n_simulations : int
    seed : int

    Returns
    -------
    dict[str, float]
        ``{'mean_stockout_cost', 'mean_overstock_cost', 'mean_total_cost',
           'p5_total_cost', 'p95_total_cost'}``
    """
    sim = simulate_cost_distribution(
        q10=q10, q50=q50, q90=q90,
        order_qty=order_qty,
        unit_margin=unit_margin,
        holding_cost=holding_cost_per_unit,
        n_simulations=n_simulations,
        seed=seed,
    )
    return {
        "mean_stockout_cost":  round(float(sim["stockout_cost"].mean()),         2),
        "mean_overstock_cost": round(float(sim["overstock_cost"].mean()),        2),
        "mean_total_cost":     round(float(sim["total_cost"].mean()),            2),
        "p5_total_cost":       round(float(sim["total_cost"].quantile(0.05)),    2),
        "p95_total_cost":      round(float(sim["total_cost"].quantile(0.95)),    2),
    }


def compare_order_quantities(
    q10: float,
    q50: float,
    q90: float,
    quantities: list[float],
    unit_margin: float,
    holding_cost_per_unit: float,
    n_simulations: int = 5_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare cost profiles across multiple candidate order quantities.

    Parameters
    ----------
    q10, q50, q90 : float
    quantities : list[float]
        Candidate order quantities to compare.
    unit_margin, holding_cost_per_unit : float
    n_simulations : int
    seed : int

    Returns
    -------
    pd.DataFrame
        Columns: ``['order_qty', 'mean_stockout_cost', 'mean_overstock_cost',
                    'mean_total_cost', 'p95_total_cost']``.
        Sorted by ``mean_total_cost`` ascending.
    """
    # Compute optimal for reference
    opt = optimal_order_quantity(
        q10=q10, q50=q50, q90=q90,
        unit_margin=unit_margin,
        holding_cost=holding_cost_per_unit,
    )
    optimal_qty = opt["optimal_qty"]

    rows = []
    for oq in quantities:
        costs = simulate_order_cost(
            q10=q10, q50=q50, q90=q90,
            order_qty=oq,
            unit_margin=unit_margin,
            holding_cost_per_unit=holding_cost_per_unit,
            n_simulations=n_simulations,
            seed=seed,
        )
        rows.append({
            "order_qty":           round(float(oq), 2),
            "is_optimal":          abs(oq - optimal_qty) < 0.5,
            **costs,
        })

    df = pd.DataFrame(rows).sort_values("mean_total_cost").reset_index(drop=True)
    return df
