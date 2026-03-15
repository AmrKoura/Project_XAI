"""
Dashboard layout helpers.

Utility functions for building reusable Dash components:
replenishment cards, KPI tiles, alert badges, and formatted tables.
"""

import pandas as pd
from dash import html, dcc
import dash_bootstrap_components as dbc


def make_kpi_card(title: str, value: str, color: str = "primary") -> dbc.Card:
    """Build a styled KPI card component.

    Parameters
    ----------
    title : str
    value : str
    color : str

    Returns
    -------
    dbc.Card
    """
    ...


def make_replenishment_card(card_data: dict) -> dbc.Card:
    """Build a replenishment summary card for a single SKU.

    Parameters
    ----------
    card_data : dict
        Output from ``replenishment_rules.generate_replenishment_card``.

    Returns
    -------
    dbc.Card
    """
    ...


def make_alert_badge(label: str, severity: str = "warning") -> dbc.Badge:
    """Create a coloured alert badge (e.g. high uncertainty, low stock).

    Parameters
    ----------
    label : str
    severity : str

    Returns
    -------
    dbc.Badge
    """
    ...


def format_metrics_table(metrics: dict[str, float]) -> dbc.Table:
    """Render a metrics dictionary as a Bootstrap table.

    Parameters
    ----------
    metrics : dict[str, float]

    Returns
    -------
    dbc.Table
    """
    ...
