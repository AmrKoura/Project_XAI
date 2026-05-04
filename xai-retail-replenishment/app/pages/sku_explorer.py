"""
SKU Explorer page — forecast, local SHAP, temporal pattern,
comparative SHAP, stockout risk.
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

        html.Div(id="sku-date-context", className="mb-3"),

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

        # Forecast vs Actual
        dbc.Card([
            dbc.CardHeader(html.Span([
                html.Strong("Forecast vs Actual — Test Period"),
                html.Span(" ⓘ", id="forecast-chart-info", style={
                    "cursor": "pointer", "fontSize": "14px",
                    "color": "#6c757d", "marginLeft": "6px", "userSelect": "none",
                }),
                dbc.Tooltip(
                    "This chart shows how well the model predicted past sales on data it has never seen (the test set). "
                    "It is not a future forecast — it is a measure of model accuracy on historical data.",
                    target="forecast-chart-info", placement="right",
                    style={"maxWidth": "340px"},
                ),
            ])),
            dbc.CardBody(
                dcc.Graph(id="sku-forecast-chart",
                          config={"displayModeBar": False},
                          style={"height": "320px"}),
            ),
        ], className="mb-4 shadow-sm"),

        # Temporal Demand Pattern
        dbc.Card([
            dbc.CardHeader(
                html.Span([
                    html.Strong("Temporal Demand Pattern"),
                    html.Span(" ⓘ", id="temporal-chart-info", style={
                        "cursor": "pointer", "fontSize": "14px",
                        "color": "#6c757d", "marginLeft": "6px", "userSelect": "none",
                    }),
                    dbc.Tooltip(
                        [
                            html.P("Shows which features drove demand UP or DOWN each week for this SKU.",
                                   className="mb-2", style={"fontSize": "12px"}),
                            html.Ul([
                                html.Li("Each line = a feature; Y-axis = how many units it added or removed from the forecast", style={"fontSize": "12px"}),
                                html.Li("Hover a point to see plain-English: 'Christmas increased forecast by +5 units'", style={"fontSize": "12px"}),
                                html.Li("Top 5 features shown — ranked by how much their influence changes week to week", style={"fontSize": "12px"}),
                                html.Li("Pattern badge on the right summarises the overall demand trajectory", style={"fontSize": "12px"}),
                                html.Li("Neural Network: shows demand trend only (SHAP requires a tree-based model)", style={"fontSize": "12px"}),
                            ], className="mb-0 ps-3"),
                        ],
                        target="temporal-chart-info", placement="right",
                        style={"maxWidth": "420px", "textAlign": "left"},
                    ),
                ])
            ),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(
                        dcc.Graph(id="sku-temporal-heatmap",
                                  config={"displayModeBar": False},
                                  style={"height": "340px"}),
                        md=8,
                    ),
                    dbc.Col(html.Div(id="sku-temporal-summary"), md=4),
                ]),
            ]),
        ], className="mb-4 shadow-sm"),

        # SHAP + NLG
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader(
                        html.Span([
                            html.Strong("Top SHAP Contributors"),
                            html.Span(" ⓘ", id="shap-chart-info", style={
                                "cursor": "pointer", "fontSize": "14px",
                                "color": "#6c757d", "marginLeft": "6px", "userSelect": "none",
                            }),
                            dbc.Tooltip(
                                [
                                    html.P("Each bar shows how much a feature pushed the forecast UP (blue) or DOWN (red).",
                                           className="mb-2", style={"fontSize": "12px"}),
                                    html.Ul([
                                        html.Li("Longer bar = stronger influence", style={"fontSize": "12px"}),
                                        html.Li("Top 10 features by impact are shown", style={"fontSize": "12px"}),
                                    ], className="mb-0 ps-3"),
                                ],
                                target="shap-chart-info", placement="right",
                                style={"maxWidth": "340px", "textAlign": "left"},
                            ),
                        ])
                    ),
                    dbc.CardBody(
                        dcc.Graph(id="sku-shap-chart",
                                  config={"displayModeBar": False},
                                  style={"height": "380px"}),
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

        # Censored Demand + SHAP Distortion
        dbc.Card([
            dbc.CardHeader(
                html.Span([
                    html.Strong("Censored Demand Analysis"),
                    html.Span(" ⓘ", id="censored-info", style={
                        "cursor": "pointer", "fontSize": "14px",
                        "color": "#6c757d", "marginLeft": "6px", "userSelect": "none",
                    }),
                    dbc.Tooltip(
                        [
                            html.P("Identifies weeks where actual sales were zero — a likely sign of a stockout rather than truly zero demand.",
                                   className="mb-2", style={"fontSize": "12px"}),
                            html.Ul([
                                html.Li("Black line = actual recorded sales", style={"fontSize": "12px"}),
                                html.Li("Red dashed line = model prediction (what true demand likely was)", style={"fontSize": "12px"}),
                                html.Li("Red shading = suspected stockout week (actual sales = 0)", style={"fontSize": "12px"}),
                                html.Li("The gap between the lines during shaded weeks = estimated lost demand", style={"fontSize": "12px"}),
                            ], className="mb-0 ps-3"),
                        ],
                        target="censored-info", placement="right",
                        style={"maxWidth": "400px", "textAlign": "left"},
                    ),
                ])
            ),
            dbc.CardBody([
                dcc.Loading(type="circle", children=
                    dcc.Graph(id="sku-censored-chart",
                              config={"displayModeBar": False},
                              style={"height": "320px"}),
                ),
                html.Div(id="sku-stockout-nlg", className="mt-3"),
            ]),
        ], className="mb-4 shadow-sm"),

        # Comparative SHAP
        dbc.Card([
            dbc.CardHeader(
                html.Span([
                    html.Strong("Comparative SHAP"),
                    html.Span(" ⓘ", id="comp-shap-info", style={
                        "cursor": "pointer", "fontSize": "14px",
                        "color": "#6c757d", "marginLeft": "6px", "userSelect": "none",
                    }),
                    dbc.Tooltip(
                        [
                            html.P("Compares how two different models explain the forecast for the selected SKU.",
                                   className="mb-2", style={"fontSize": "12px"}),
                            html.Ul([
                                html.Li("Model A = currently selected model", style={"fontSize": "12px"}),
                                html.Li("Model B = any other model you pick", style={"fontSize": "12px"}),
                                html.Li("Left chart: SHAP difference (A − B) — which features each model weights differently", style={"fontSize": "12px"}),
                                html.Li("Right chart: side-by-side SHAP values for both models", style={"fontSize": "12px"}),
                                html.Li("Requires tree-based models — Neural Network does not support SHAP", style={"fontSize": "12px"}),
                            ], className="mb-0 ps-3"),
                        ],
                        target="comp-shap-info", placement="right",
                        style={"maxWidth": "420px", "textAlign": "left"},
                    ),
                ])
            ),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Model A (current)", html_for="comp-model-a-display"),
                        html.Div(id="comp-model-a-display", className="mt-1"),
                    ], md=5),
                    dbc.Col(
                        html.Div("vs", className="fw-bold text-muted text-center mt-4 pt-1"),
                        md=1,
                    ),
                    dbc.Col([
                        dbc.Label("Model B", html_for="comp-model-b"),
                        dcc.Dropdown(id="comp-model-b", placeholder="Select model B…", clearable=False),
                    ], md=5),
                ], className="mb-4 g-2"),
                dcc.Loading(type="circle", children=dbc.Row([
                    dbc.Col(
                        dcc.Graph(id="comp-diff-chart",
                                  config={"displayModeBar": False},
                                  style={"height": "380px"}),
                        md=6,
                    ),
                    dbc.Col(
                        dcc.Graph(id="comp-side-chart",
                                  config={"displayModeBar": False},
                                  style={"height": "380px"}),
                        md=6,
                    ),
                ])),
                html.Div(id="comp-nlg", className="mt-3"),
            ]),
        ], className="mb-4 shadow-sm"),
    ])
