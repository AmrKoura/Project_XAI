"""
Dashboard page — SKU overview and replenishment cards.

Landing page displaying demand forecasts and replenishment summary
cards for all SKUs. Flags products with high uncertainty, imminent
reorder triggers, and data-quality concerns (Q2 & Q8).
"""

from dash import html, dcc
import dash_bootstrap_components as dbc


def layout() -> html.Div:
    """Build the dashboard page layout.

    Returns
    -------
    html.Div
    """
    ...


def build_sku_overview_table(data: dict) -> dbc.Table:
    """Render the SKU summary table with forecast and reorder status.

    Parameters
    ----------
    data : dict

    Returns
    -------
    dbc.Table
    """
    ...


def build_replenishment_cards(cards: list[dict]) -> html.Div:
    """Render a grid of replenishment cards for all active SKUs.

    Parameters
    ----------
    cards : list[dict]

    Returns
    -------
    html.Div
    """
    ...
