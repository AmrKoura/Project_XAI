"""
Overall Dash app layout.

Defines the top-level navigation bar and page container that holds
the four dashboard views (Dashboard, SKU Explorer, Explanations, What-If).
"""

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc


def create_navbar() -> dbc.NavbarSimple:
    """Build the top navigation bar with page links.

    Returns
    -------
    dbc.NavbarSimple
    """
    ...


def create_layout() -> html.Div:
    """Assemble the full app layout: navbar + page container.

    Returns
    -------
    html.Div
    """
    ...
