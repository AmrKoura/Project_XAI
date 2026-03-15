"""
Callbacks for the SKU Explorer page.

Handles SKU selection, forecast chart updates, SHAP waterfall
rendering, and natural-language brief generation.
"""

from dash import Input, Output, callback
import pandas as pd


def register_sku_callbacks(app) -> None:
    """Register all callbacks for the SKU Explorer page.

    Parameters
    ----------
    app : dash.Dash
    """
    ...


def _update_forecast_chart(sku_id: str) -> dict:
    """Callback: update the time-series chart for the selected SKU.

    Parameters
    ----------
    sku_id : str

    Returns
    -------
    dict
        Plotly figure dict.
    """
    ...


def _update_shap_waterfall(sku_id: str) -> dict:
    """Callback: render local SHAP waterfall for the selected SKU.

    Parameters
    ----------
    sku_id : str

    Returns
    -------
    dict
    """
    ...


def _update_nlg_brief(sku_id: str) -> str:
    """Callback: generate the natural-language explanation brief.

    Parameters
    ----------
    sku_id : str

    Returns
    -------
    str
    """
    ...
