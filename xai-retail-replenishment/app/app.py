"""
Dash application entry point.

Launches the XAI Retail Replenishment dashboard with four pages:
  1. Overview     — SKU overview + replenishment cards
  2. SKU Explorer — time series + local SHAP + NLG brief
  3. Explanations — global SHAP, PDP, comparative SHAP, feature audit
  4. What-If      — counterfactual scenario simulator
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, html, clientside_callback

from app.layout import create_layout
from app.pages import dashboard, sku_explorer, explanations, what_if, reports
from app.callbacks.dashboard_callbacks import register_dashboard_callbacks
from app.callbacks.sku_callbacks import register_sku_callbacks
from app.callbacks.whatif_callbacks import register_whatif_callbacks
from app.callbacks.explanations_callbacks import register_explanations_callbacks
from app.callbacks.report_callbacks import register_report_callbacks
import app.data_store as ds


def create_app() -> dash.Dash:
    """Initialise and configure the Dash application."""
    ds.load("7d")

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.FLATLY],
        suppress_callback_exceptions=True,
        title="XAI Retail Replenishment",
    )
    app.layout = create_layout()
    return app


def register_callbacks(app: dash.Dash) -> None:
    """Register all page and global callbacks."""

    # ── dark mode toggle (clientside — no server round-trip) ─────────────────
    clientside_callback(
        """
        function(n_clicks, current_theme) {
            if (!n_clicks) return [current_theme || 'light', ''];
            var next = (current_theme === 'dark') ? 'light' : 'dark';
            if (next === 'dark') {
                document.body.classList.add('dark-mode');
            } else {
                document.body.classList.remove('dark-mode');
            }
            return [next, ''];
        }
        """,
        Output("theme-store",  "data"),
        Output("theme-toggle", "children"),
        Input("theme-toggle",  "n_clicks"),
        State("theme-store",   "data"),
        prevent_initial_call=True,
    )

    # ── model switch ──────────────────────────────────────────────────────────
    @app.callback(
        Output("model-store", "data"),
        Input("model-dropdown", "value"),
        prevent_initial_call=True,
    )
    def switch_model(model_key: str) -> str:
        ds.reload(model_key)
        return model_key

    # ── page routing ──────────────────────────────────────────────────────────
    @app.callback(
        Output("page-content", "children"),
        Input("url", "pathname"),
        Input("model-store", "data"),
    )
    def display_page(pathname: str, _model_key: str):
        if pathname in ("/", "/overview"):
            return dashboard.layout()
        if pathname == "/sku-explorer":
            return sku_explorer.layout()
        if pathname == "/explanations":
            return explanations.layout()
        if pathname == "/what-if":
            return what_if.layout()
        if pathname == "/reports":
            return reports.layout()
        return html.Div([
            html.H2("404 — Page not found"),
            html.P(f"No page at '{pathname}'."),
        ], style={"padding": "40px"})

    # ── page-level callbacks ──────────────────────────────────────────────────
    register_dashboard_callbacks(app)
    register_sku_callbacks(app)
    register_whatif_callbacks(app)
    register_explanations_callbacks(app)
    register_report_callbacks(app)


if __name__ == "__main__":
    app = create_app()
    register_callbacks(app)
    app.run(debug=True, host="127.0.0.1", port=8050)
