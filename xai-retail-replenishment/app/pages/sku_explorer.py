"""
SKU Explorer page — forecast time series, local SHAP waterfall, NLG brief.

Addresses Q1 (local SHAP), Q3 (uncertainty), Q9 (cold start).
"""

from dash import html, dcc
import dash_bootstrap_components as dbc


def layout() -> html.Div:
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H3([
                    "SKU Explorer",
                    html.Span("ⓘ", id="sku-page-info", style={
                        "fontSize": "14px", "color": "rgba(108,117,125,0.6)",
                        "cursor": "help", "marginLeft": "8px", "verticalAlign": "middle",
                    }),
                ], className="mb-0"),
                dbc.Tooltip(
                    "Dive into a single product. Pick any SKU to see its sales forecast, "
                    "the key factors driving demand, and how confident the model is in its prediction.",
                    target="sku-page-info", placement="right",
                ),
            ], width="auto"),
        ], align="end", className="mb-3"),

        dbc.Row([
            dbc.Col([
                dbc.Label("Select SKU", html_for="sku-selector"),
                dcc.Dropdown(
                    id="sku-selector",
                    clearable=False,
                    placeholder="Loading SKUs…",
                    style={"width": "280px"},
                ),
            ], width="auto"),
        ], className="mb-4"),

        html.Div(id="sku-summary-cards", className="mb-4"),

        dbc.Card([
            dbc.CardHeader(html.Strong("Forecast vs Actual — Test Period")),
            dbc.CardBody(
                dcc.Graph(
                    id="sku-forecast-chart",
                    config={"displayModeBar": False},
                    style={"height": "320px"},
                )
            ),
        ], className="mb-4 shadow-sm"),

        dbc.Card([
            dbc.CardHeader(
                html.Span([
                    html.Strong("Temporal Demand Pattern (Q6)"),
                    html.Span(
                        " ⓘ",
                        id="temporal-chart-info",
                        style={
                            "cursor": "pointer",
                            "fontSize": "14px",
                            "color": "#6c757d",
                            "marginLeft": "6px",
                            "userSelect": "none",
                        },
                    ),
                    dbc.Tooltip(
                        [
                            html.P(
                                "This section shows how feature importance has shifted over time "
                                "for this SKU, and classifies its overall demand pattern.",
                                className="mb-2",
                                style={"fontSize": "12px"},
                            ),
                            html.P(
                                "How to read it:",
                                className="mb-1 fw-bold",
                                style={"fontSize": "12px"},
                            ),
                            html.Ul([
                                html.Li("Heatmap rows = features, columns = dates — colour shows SHAP impact (blue = up, red = down)", style={"fontSize": "12px"}),
                                html.Li("Features with shifting colours over time are the main sources of demand volatility", style={"fontSize": "12px"}),
                                html.Li("The pattern badge (e.g. STABLE, SPIKE) summarises the overall demand trajectory", style={"fontSize": "12px"}),
                                html.Li("Top temporal drivers are ranked by how much their SHAP value varies across dates", style={"fontSize": "12px"}),
                            ], className="mb-0 ps-3"),
                        ],
                        target="temporal-chart-info",
                        placement="right",
                        style={"maxWidth": "380px", "textAlign": "left"},
                    ),
                ])
            ),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(
                        dcc.Graph(
                            id="sku-temporal-heatmap",
                            config={"displayModeBar": False},
                            style={"height": "340px"},
                        ),
                        md=8,
                    ),
                    dbc.Col(
                        html.Div(id="sku-temporal-summary"),
                        md=4,
                    ),
                ]),
            ]),
        ], className="mb-4 shadow-sm"),

        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader(
                        html.Span([
                            html.Strong("Top SHAP Contributors"),
                            html.Span(
                                " ⓘ",
                                id="shap-chart-info",
                                style={
                                    "cursor": "pointer",
                                    "fontSize": "14px",
                                    "color": "#6c757d",
                                    "marginLeft": "6px",
                                    "userSelect": "none",
                                },
                            ),
                            dbc.Tooltip(
                                [
                                    html.P(
                                        "Each bar shows how much a feature pushed the forecast "
                                        "UP (blue) or DOWN (red) compared to the model's baseline.",
                                        className="mb-2",
                                        style={"fontSize": "12px"},
                                    ),
                                    html.P(
                                        "How to read it:",
                                        className="mb-1 fw-bold",
                                        style={"fontSize": "12px"},
                                    ),
                                    html.Ul([
                                        html.Li("Longer bar = stronger influence on this prediction", style={"fontSize": "12px"}),
                                        html.Li("Blue bars increase the forecast (e.g. high recent sales)", style={"fontSize": "12px"}),
                                        html.Li("Red bars decrease the forecast (e.g. low lag values)", style={"fontSize": "12px"}),
                                        html.Li("Only the top 10 features by impact are shown", style={"fontSize": "12px"}),
                                    ], className="mb-0 ps-3"),
                                ],
                                target="shap-chart-info",
                                placement="right",
                                style={"maxWidth": "340px", "textAlign": "left"},
                            ),
                        ])
                    ),
                    dbc.CardBody(
                        dcc.Graph(
                            id="sku-shap-chart",
                            config={"displayModeBar": False},
                            style={"height": "380px"},
                        )
                    ),
                ], className="h-100 shadow-sm"),
                md=7,
            ),
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader(html.Strong("Explanation Brief")),
                    dbc.CardBody([
                        html.Div(id="sku-nlg-brief"),
                        html.Div(id="sku-temporal-nlg"),
                    ]),
                ], className="h-100 shadow-sm"),
                md=5,
            ),
        ], className="mb-4"),
    ])
