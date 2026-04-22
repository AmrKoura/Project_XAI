"""
Q10 — Business cost-impact modeling.

Answers: "What is the financial cost of under- or over-ordering
          based on this forecast?"

Uses probabilistic simulation across the quantile forecast distribution
to estimate expected stockout cost, overstock holding cost, and the
optimal order quantity via the newsvendor critical ratio.

Demand is modelled as log-normal (always positive, right-skewed) fitted
from the q10/q50/q90 quantile forecasts produced by uncertainty.py.
"""

import pandas as pd
import numpy as np


# ── cost primitives ───────────────────────────────────────────────────────────

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
        Revenue margin per unit (opportunity cost of each lost sale).

    Returns
    -------
    float
        Stockout cost = max(0, demand - ordered) * unit_margin.
    """
    return float(max(0.0, demand - ordered) * unit_margin)


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
        Cost to hold one unsold unit for the forecast period.

    Returns
    -------
    float
        Overstock cost = max(0, ordered - demand) * holding_cost_per_unit.
    """
    return float(max(0.0, ordered - demand) * holding_cost_per_unit)


# ── distribution fitting ──────────────────────────────────────────────────────

def _fit_lognormal(q10: float, q50: float, q90: float) -> tuple[float, float]:
    """Fit log-normal (mu, sigma) from three quantile points.

    Uses q50 as the median (mu = log(q50)) and the q10/q90 spread to
    estimate sigma. Falls back to a small sigma if the spread is zero.

    Returns
    -------
    tuple[float, float]
        (mu, sigma) of the underlying normal distribution.
    """
    q50  = max(q50, 1e-6)
    mu   = float(np.log(q50))

    if q90 > q10 > 0:
        # z=1.28 corresponds to the 90th percentile of a standard normal.
        sigma = float((np.log(max(q90, 1e-6)) - np.log(max(q10, 1e-6))) / (2 * 1.28))
        sigma = max(sigma, 0.01)
    else:
        sigma = 0.3  # default moderate spread

    return mu, sigma


# ── simulation ────────────────────────────────────────────────────────────────

def simulate_cost_distribution(
    q10: float,
    q50: float,
    q90: float,
    order_qty: float,
    unit_margin: float,
    holding_cost: float,
    n_simulations: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Monte Carlo simulation of total cost across demand scenarios.

    Demand is sampled from a log-normal distribution fitted to the
    q10/q50/q90 quantile forecasts — ensuring all sampled values are
    positive and the distribution is right-skewed as retail demand tends
    to be.

    Parameters
    ----------
    q10, q50, q90 : float
        Quantile forecasts from uncertainty.py.
    order_qty : float
        The order quantity being evaluated.
    unit_margin : float
        Margin per unit (cost of a lost sale).
    holding_cost : float
        Cost to hold one excess unit for the period.
    n_simulations : int
    seed : int

    Returns
    -------
    pd.DataFrame
        Columns: ``['demand', 'stockout_cost', 'overstock_cost', 'total_cost']``.
    """
    mu, sigma = _fit_lognormal(q10, q50, q90)
    rng       = np.random.default_rng(seed)
    demands   = rng.lognormal(mean=mu, sigma=sigma, size=n_simulations)

    stockout  = np.maximum(0.0, demands - order_qty) * unit_margin
    overstock = np.maximum(0.0, order_qty - demands) * holding_cost
    total     = stockout + overstock

    return pd.DataFrame({
        "demand":        demands,
        "stockout_cost": stockout,
        "overstock_cost":overstock,
        "total_cost":    total,
    })


# ── optimal order ─────────────────────────────────────────────────────────────

def optimal_order_quantity(
    q10: float,
    q50: float,
    q90: float,
    unit_margin: float,
    holding_cost: float,
) -> dict:
    """Find the order quantity that minimises expected total cost.

    Uses the newsvendor critical ratio:
        critical_ratio = unit_margin / (unit_margin + holding_cost)
        Q* = F⁻¹(critical_ratio)

    where F is the log-normal CDF fitted from q10/q50/q90.
    This gives the theoretically optimal order quantity without needing
    a separate service_level parameter — the cost ratio implicitly
    defines the service level.

    Parameters
    ----------
    q10, q50, q90 : float
    unit_margin : float
    holding_cost : float

    Returns
    -------
    dict
        Keys: ``'optimal_qty'``, ``'critical_ratio'``, ``'implied_service_level'``,
        ``'mu'``, ``'sigma'``.
    """
    if unit_margin <= 0:
        raise ValueError("unit_margin must be > 0.")
    if holding_cost <= 0:
        raise ValueError("holding_cost must be > 0.")

    critical_ratio = unit_margin / (unit_margin + holding_cost)
    mu, sigma      = _fit_lognormal(q10, q50, q90)

    # Inverse CDF of log-normal at the critical ratio.
    from scipy.stats import lognorm
    optimal_qty = float(lognorm.ppf(critical_ratio, s=sigma, scale=np.exp(mu)))
    optimal_qty = max(0.0, optimal_qty)

    return {
        "optimal_qty":            round(optimal_qty,    2),
        "critical_ratio":         round(critical_ratio, 4),
        "implied_service_level":  round(critical_ratio * 100, 1),
        "mu":                     round(mu,    4),
        "sigma":                  round(sigma, 4),
    }


# ── NLG ──────────────────────────────────────────────────────────────────────

def generate_cost_impact_text(
    sku_id: str,
    q50: float,
    optimal_qty: float,
    expected_stockout_cost: float,
    expected_overstock_cost: float,
    unit_margin: float,
    holding_cost: float,
    critical_ratio: float,
) -> str:
    """Generate a natural-language cost-impact summary.

    Parameters
    ----------
    sku_id : str
    q50 : float
        Point forecast (median demand).
    optimal_qty : float
        Newsvendor-optimal order quantity.
    expected_stockout_cost : float
        Mean stockout cost from Monte Carlo simulation.
    expected_overstock_cost : float
        Mean overstock cost from Monte Carlo simulation.
    unit_margin, holding_cost : float
    critical_ratio : float
        unit_margin / (unit_margin + holding_cost).

    Returns
    -------
    str
    """
    total_cost  = expected_stockout_cost + expected_overstock_cost
    dominant    = "stockout" if expected_stockout_cost > expected_overstock_cost else "overstock"
    order_vs_forecast = optimal_qty - q50

    lines = [
        f"=== Cost-Impact Analysis: {sku_id} ===",
        f"Point forecast (q50):       {q50:.1f} units",
        f"Optimal order quantity:     {optimal_qty:.1f} units  "
        f"({'above' if order_vs_forecast >= 0 else 'below'} forecast by {abs(order_vs_forecast):.1f} units)",
        f"",
        f"Cost parameters:",
        f"  Unit margin (stockout cost):  ${unit_margin:.2f}/unit",
        f"  Holding cost (overstock):     ${holding_cost:.2f}/unit",
        f"  Critical ratio:               {critical_ratio:.2%}",
        f"",
        f"Expected costs at optimal order quantity:",
        f"  Stockout cost:   ${expected_stockout_cost:.2f}",
        f"  Overstock cost:  ${expected_overstock_cost:.2f}",
        f"  Total:           ${total_cost:.2f}",
        f"",
        f"Dominant risk: {dominant.upper()}",
    ]

    if dominant == "stockout":
        lines.append(
            f"The stockout risk outweighs overstock cost — the high unit margin "
            f"(${unit_margin:.2f}) means lost sales are more expensive than holding excess stock. "
            f"Consider ordering above the point forecast."
        )
    else:
        lines.append(
            f"The overstock risk outweighs stockout cost — holding excess inventory "
            f"(${holding_cost:.2f}/unit) is relatively expensive. "
            f"Stay close to or below the point forecast."
        )

    return "\n".join(lines)
