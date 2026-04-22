"""
Overall Dash app layout.

Defines the top-level navigation bar and page container that holds
the four dashboard views (Overview, SKU Explorer, Explanations, What-If).
"""

from dash import html, dcc
import dash_bootstrap_components as dbc


def _tab(label: str, href: str) -> html.Li:
    return html.Li([
        html.B(className="left-curve"),
        html.B(className="right-curve"),
        dbc.NavLink(label, href=href, active="exact"),
    ], className="nav-tab-item")


def create_navbar() -> html.Nav:
    """Build the folder-tab navigation bar with page links, model selector, and dark mode toggle."""
    return html.Nav([
        html.A("XAI Retail Replenishment", href="/", className="app-brand"),

        html.Ul([
            _tab("Overview",     "/"),
            _tab("SKU Explorer", "/sku-explorer"),
            _tab("Explanations", "/explanations"),
            _tab("What-If",      "/what-if"),
            _tab("Reports",      "/reports"),
        ], className="nav-tab-list"),

        html.Div([
            dbc.Select(
                id="model-dropdown",
                options=[
                    {"label": "7-day",  "value": "7d"},
                    {"label": "14-day", "value": "14d"},
                    {"label": "28-day", "value": "28d"},
                ],
                value="7d",
            ),
            dbc.Button(
                "",
                id="theme-toggle",
                color="link",
                style={"padding": "0", "border": "none", "background": "none"},
            ),
        ], className="navbar-right"),
    ], className="app-navbar")


def create_layout() -> html.Div:
    """Assemble the full app layout: theme link + navbar + stores + page container."""
    return html.Div([
        create_navbar(),
        dcc.Location(id="url", refresh=False),

        # Persistent stores
        dcc.Store(id="theme-store", data="light"),
        dcc.Store(id="model-store", data="7d"),

        # Page content wrapped in loading spinner for model switches
        dcc.Loading(
            id="page-loading",
            type="circle",
            children=html.Div(
                id="page-content",
                style={"padding": "24px 32px", "minHeight": "calc(100vh - 56px)"},
            ),
        ),
    ])
