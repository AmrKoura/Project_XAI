"""
What-if simulator.

Interactive scenario tool allowing planners to modify promotion status,
lead time, current stock, and service-level target, then see how the
forecast, uncertainty, and replenishment recommendations change.
"""

import pandas as pd
import numpy as np

from xai.uncertainty import compute_prediction_interval
from decision.safety_stock import compute_safety_stock
from decision.replenishment_rules import (
    generate_replenishment_card,
    compute_reorder_quantity,
)
from decision.simulation.cost_impact_sim import simulate_order_cost


def simulate_scenario(
    model: object,
    X_row: pd.DataFrame,
    overrides: dict[str, float | int],
    stock_on_hand: float = 0.0,
    lead_time_days: int = 7,
    residual_std: float = 5.0,
    unit_margin: float = 3.50,
    holding_cost: float = 0.80,
) -> dict:
    """Run a single what-if scenario with user-specified parameter changes.

    Parameters
    ----------
    model : object
        Trained sklearn Pipeline.
    X_row : pd.DataFrame
        Single-row baseline feature values for the SKU.
    overrides : dict[str, float | int]
        Feature columns to override, e.g. ``{'snap_CA': 1, 'sell_price': 2.5}``.
        Only features present in X_row.columns are applied; unknown keys are ignored.
    stock_on_hand : float
        Current inventory level for replenishment calculation.
    lead_time_days : int
        Replenishment lead time in days.
    residual_std : float
        Per-SKU residual standard deviation (from uncertainty.compute_sku_residual_std).
    unit_margin : float
        Margin per unit (lost-sale cost).
    holding_cost : float
        Holding cost per excess unit.

    Returns
    -------
    dict
        ``{'baseline_pred', 'scenario_pred', 'delta', 'delta_pct',
           'interval_baseline', 'interval_scenario',
           'safety_stock', 'reorder_qty', 'cost_summary',
           'overrides_applied', 'overrides_ignored'}``
    """
    if not isinstance(X_row, pd.DataFrame) or len(X_row) != 1:
        raise TypeError("X_row must be a single-row pd.DataFrame.")

    # Baseline prediction
    baseline_pred = float(model.predict(X_row)[0])

    # Apply overrides — only known columns
    X_scenario = X_row.copy()
    applied, ignored = {}, {}
    for feat, val in overrides.items():
        if feat in X_scenario.columns:
            X_scenario[feat] = val
            applied[feat] = val
        else:
            ignored[feat] = val

    scenario_pred = float(model.predict(X_scenario)[0])
    delta     = round(scenario_pred - baseline_pred, 4)
    delta_pct = round((delta / baseline_pred * 100) if baseline_pred != 0 else 0.0, 2)

    # Prediction intervals
    interval_base = compute_prediction_interval(baseline_pred, residual_std)
    interval_scen = compute_prediction_interval(scenario_pred, residual_std)

    # Safety stock (quantile method, scenario forecast)
    safety_stock = compute_safety_stock(
        method="quantile",
        q50=interval_scen["q50"],
        q90=interval_scen["q90"],
        lead_time_days=lead_time_days,
    )

    # Reorder quantity
    reorder_qty = compute_reorder_quantity(
        forecast_demand=interval_scen["q50"],
        safety_stock=safety_stock,
        stock_on_hand=stock_on_hand,
    )

    # Cost simulation at scenario order qty
    cost_summary = simulate_order_cost(
        q10=interval_scen["q10"],
        q50=interval_scen["q50"],
        q90=interval_scen["q90"],
        order_qty=reorder_qty,
        unit_margin=unit_margin,
        holding_cost_per_unit=holding_cost,
    )

    return {
        "baseline_pred":     round(baseline_pred, 4),
        "scenario_pred":     round(scenario_pred, 4),
        "delta":             delta,
        "delta_pct":         delta_pct,
        "interval_baseline": interval_base,
        "interval_scenario": interval_scen,
        "safety_stock":      round(safety_stock, 2),
        "reorder_qty":       round(reorder_qty,  2),
        "cost_summary":      cost_summary,
        "overrides_applied": applied,
        "overrides_ignored": ignored,
    }


def simulate_batch_scenarios(
    model: object,
    X_row: pd.DataFrame,
    scenarios: list[dict],
    stock_on_hand: float = 0.0,
    lead_time_days: int = 7,
    residual_std: float = 5.0,
    unit_margin: float = 3.50,
    holding_cost: float = 0.80,
) -> pd.DataFrame:
    """Run multiple what-if scenarios and return a comparison table.

    Parameters
    ----------
    model : object
    X_row : pd.DataFrame
        Single-row baseline.
    scenarios : list[dict]
        Each dict is an ``overrides`` mapping passed to ``simulate_scenario``.
        Optionally each dict may include a ``'label'`` key for display.
    stock_on_hand, lead_time_days, residual_std, unit_margin, holding_cost
        Shared replenishment / cost parameters.

    Returns
    -------
    pd.DataFrame
        Columns: ``['label', 'scenario_pred', 'delta', 'delta_pct',
                    'safety_stock', 'reorder_qty', 'mean_total_cost']``.
    """
    rows = []
    for i, sc in enumerate(scenarios):
        overrides = {k: v for k, v in sc.items() if k != "label"}
        label     = sc.get("label", f"Scenario {i+1}")
        result    = simulate_scenario(
            model=model, X_row=X_row, overrides=overrides,
            stock_on_hand=stock_on_hand, lead_time_days=lead_time_days,
            residual_std=residual_std, unit_margin=unit_margin,
            holding_cost=holding_cost,
        )
        rows.append({
            "label":           label,
            "scenario_pred":   result["scenario_pred"],
            "delta":           result["delta"],
            "delta_pct":       result["delta_pct"],
            "safety_stock":    result["safety_stock"],
            "reorder_qty":     result["reorder_qty"],
            "mean_total_cost": result["cost_summary"]["mean_total_cost"],
        })

    return pd.DataFrame(rows)


def generate_scenario_summary_text(scenario_result: dict) -> str:
    """Generate a natural-language summary of a what-if scenario.

    Parameters
    ----------
    scenario_result : dict
        Output of ``simulate_scenario``.

    Returns
    -------
    str
    """
    r         = scenario_result
    direction = "increase" if r["delta"] >= 0 else "decrease"
    applied   = r["overrides_applied"]
    ignored   = r["overrides_ignored"]

    feat_str = ", ".join(f"{k}={v}" for k, v in applied.items()) if applied else "none"

    lines = [
        "=== What-If Scenario Summary ===",
        f"Changes applied: {feat_str}",
        f"",
        f"Baseline forecast:  {r['baseline_pred']:.1f} units",
        f"Scenario forecast:  {r['scenario_pred']:.1f} units",
        f"Impact: {'+' if r['delta'] >= 0 else ''}{r['delta']:.1f} units "
        f"({r['delta_pct']:+.1f}%)",
        f"",
        f"Prediction interval (scenario):",
        f"  q10: {r['interval_scenario']['q10']:.1f}  |  "
        f"q50: {r['interval_scenario']['q50']:.1f}  |  "
        f"q90: {r['interval_scenario']['q90']:.1f}",
        f"",
        f"Replenishment recommendation:",
        f"  Safety stock: {r['safety_stock']:.1f} units",
        f"  Recommended order qty: {r['reorder_qty']:.1f} units",
        f"",
        f"Expected cost at recommended order qty:",
        f"  Mean stockout cost:  ${r['cost_summary']['mean_stockout_cost']:.2f}",
        f"  Mean overstock cost: ${r['cost_summary']['mean_overstock_cost']:.2f}",
        f"  Mean total cost:     ${r['cost_summary']['mean_total_cost']:.2f}",
        f"  P95 total cost:      ${r['cost_summary']['p95_total_cost']:.2f}",
    ]

    if ignored:
        ign_str = ", ".join(ignored.keys())
        lines.append(f"\nNote: features not found in model and were ignored: {ign_str}")

    if abs(r["delta_pct"]) >= 10:
        lines.append(
            f"\nThis scenario causes a significant {direction} in forecast demand "
            f"({abs(r['delta_pct']):.1f}%). Review replenishment quantity carefully."
        )
    elif abs(r["delta_pct"]) < 2:
        lines.append(
            f"\nThis scenario has minimal impact on the forecast — "
            f"replenishment recommendations remain essentially unchanged."
        )

    return "\n".join(lines)
