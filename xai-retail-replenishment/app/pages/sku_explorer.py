"""
SKU Explorer page — time series, local SHAP, and NLG brief.

Detailed breakdown for a selected SKU showing:
  - Forecast time series with quantile bands
  - Local SHAP waterfall chart
  - Natural-language explanation brief
Addresses Q1, Q3 & Q6.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc


def layout() -> html.Div:
    """Build the SKU Explorer page layout.

    Returns
    -------
    html.Div
    """
    ...


def build_sku_selector() -> dcc.Dropdown:
    """Create the SKU selection dropdown.

    Returns
    -------
    dcc.Dropdown
    """
    ...


def build_forecast_chart_container() -> dcc.Graph:
    """Placeholder container for the time-series + quantile bands chart.

    Returns
    -------
    dcc.Graph
    """
    ...


def build_shap_waterfall_container() -> dcc.Graph:
    """Placeholder container for the local SHAP waterfall.

    Returns
    -------
    dcc.Graph
    """
    ...


def build_nlg_brief_container() -> html.Div:
    """Placeholder container for the natural-language explanation.

    Returns
    -------
    html.Div
    """
    ...
