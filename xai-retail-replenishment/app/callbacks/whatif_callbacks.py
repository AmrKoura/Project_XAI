"""
Callbacks for the What-If Simulator page.

Handles scenario parameter changes, re-forecasting, replenishment
recalculation, and cost-impact chart updates.
"""

from dash import Input, Output, State, callback
import pandas as pd


def register_whatif_callbacks(app) -> None:
    """Register all callbacks for the What-If Simulator page.

    Parameters
    ----------
    app : dash.Dash
    """
    ...


def _run_scenario(
    sku_id: str,
    promo: int,
    lead_time: int,
    stock_on_hand: float,
    service_level: float,
) -> dict:
    """Callback: run the what-if scenario and return updated predictions.

    Parameters
    ----------
    sku_id : str
    promo : int
    lead_time : int
    stock_on_hand : float
    service_level : float

    Returns
    -------
    dict
        Scenario results including forecast, reorder qty, and cost estimates.
    """
    ...


def _update_cost_chart(scenario_result: dict) -> dict:
    """Callback: update the cost-impact distribution chart.

    Parameters
    ----------
    scenario_result : dict

    Returns
    -------
    dict
        Plotly figure dict.
    """
    ...
