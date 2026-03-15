"""
Callbacks for the Dashboard page.

Handles data loading, SKU card rendering, alert flagging,
and periodic refresh of the overview table.
"""

from dash import Input, Output, callback
import pandas as pd


def register_dashboard_callbacks(app) -> None:
    """Register all callbacks for the Dashboard page.

    Parameters
    ----------
    app : dash.Dash
    """
    ...


def _update_sku_overview(filter_value: str) -> dict:
    """Callback: refresh the SKU overview table based on filters.

    Parameters
    ----------
    filter_value : str

    Returns
    -------
    dict
    """
    ...


def _update_replenishment_cards(selected_skus: list[str]) -> list[dict]:
    """Callback: rebuild replenishment cards for selected SKUs.

    Parameters
    ----------
    selected_skus : list[str]

    Returns
    -------
    list[dict]
    """
    ...
