"""
Dash application entry point.

Launches the XAI Retail Replenishment dashboard with four pages:
  1. Dashboard  — SKU overview + replenishment cards
  2. SKU Explorer — time series + local SHAP + NLG brief
  3. Explanations — global SHAP, PDP, comparative SHAP, feature audit
  4. What-If — counterfactual simulator
"""

import dash
import dash_bootstrap_components as dbc
from app.layout import create_layout


def create_app() -> dash.Dash:
    """Initialise and configure the Dash application.

    Returns
    -------
    dash.Dash
    """
    ...


def register_callbacks(app: dash.Dash) -> None:
    """Register all page callbacks on the app instance.

    Parameters
    ----------
    app : dash.Dash
    """
    ...


if __name__ == "__main__":
    app = create_app()
    register_callbacks(app)
    app.run(debug=True, host="127.0.0.1", port=8050)
