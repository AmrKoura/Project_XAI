"""
Explanations page — Global SHAP (Q2) + Feature Quality Audit (Q8).
"""

from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc


def layout() -> html.Div:
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H3([
                    "Explanations",
                    html.Span("ⓘ", id="explanations-page-info", style={
                        "fontSize": "14px", "color": "rgba(108,117,125,0.6)",
                        "cursor": "help", "marginLeft": "8px", "verticalAlign": "middle",
                    }),
                ], className="mb-0"),
                dbc.Tooltip(
                    "Understand why the model makes its predictions. See which factors matter most, "
                    "how they influence demand, and whether the model is consistent across all products.",
                    target="explanations-page-info", placement="right",
                ),
            ], width="auto"),
        ], align="end", className="mb-4"),

        # Q2 — Global Feature Importance
        dbc.Card([
            dbc.CardHeader(
                html.Span([
                    html.Strong("Global Feature Importance"),
                    html.Span(
                        " ⓘ",
                        id="global-shap-info",
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
                            html.P("What is a SHAP value?", className="mb-1 fw-bold", style={"fontSize": "12px"}),
                            html.P(
                                "SHAP (SHapley Additive exPlanations) measures how much each feature "
                                "contributed to a single prediction compared to the model's average output. "
                                "A positive SHAP value means the feature pushed the forecast up; "
                                "a negative value means it pushed it down.",
                                className="mb-2",
                                style={"fontSize": "12px"},
                            ),
                            html.P("What this chart shows:", className="mb-1 fw-bold", style={"fontSize": "12px"}),
                            html.Ul([
                                html.Li("Each bar = one feature's average impact across all SKUs and time periods", style={"fontSize": "12px"}),
                                html.Li("The metric is Mean |SHAP| — the average of the absolute SHAP values, so direction is ignored and only magnitude counts", style={"fontSize": "12px"}),
                                html.Li("Longer bar = stronger overall influence on demand forecasts", style={"fontSize": "12px"}),
                                html.Li("Features at the top dominate predictions; features at the bottom barely matter", style={"fontSize": "12px"}),
                            ], className="mb-0 ps-3"),
                        ],
                        target="global-shap-info",
                        placement="right",
                        style={"maxWidth": "380px", "textAlign": "left"},
                    ),
                ])
            ),
            dbc.CardBody(
                dbc.Row([
                    dbc.Col(
                        dcc.Graph(
                            id="global-shap-chart",
                            config={"displayModeBar": False},
                            style={"height": "420px"},
                        ),
                        md=8,
                    ),
                    dbc.Col(html.Div(id="global-shap-nlg"), md=4),
                ]),
            ),
        ], className="mb-4 shadow-sm"),

        # Q8 — Feature Quality Audit
        dbc.Card([
            dbc.CardHeader(
                html.Span([
                    html.Strong("Feature Quality Audit"),
                    html.Span(
                        " ⓘ",
                        id="audit-info",
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
                                "Checks each feature for data quality issues that could "
                                "affect model reliability.",
                                className="mb-2",
                                style={"fontSize": "12px"},
                            ),
                            html.P("Flag types:", className="mb-1 fw-bold", style={"fontSize": "12px"}),
                            html.Ul([
                                html.Li("ok — feature looks healthy, no issues detected", style={"fontSize": "12px"}),
                                html.Li("zero_variance — feature has the same value for every row (adds no information)", style={"fontSize": "12px"}),
                                html.Li("mostly_zero — over 80% of values are zero (sparse signal)", style={"fontSize": "12px"}),
                                html.Li("high_corr — very similar to another feature (>0.95 correlation), may be redundant", style={"fontSize": "12px"}),
                                html.Li("high_missing — more than 10% of values are missing", style={"fontSize": "12px"}),
                            ], className="mb-0 ps-3"),
                        ],
                        target="audit-info",
                        placement="right",
                        style={"maxWidth": "380px", "textAlign": "left"},
                    ),
                ])
            ),
            dbc.CardBody([
                html.Div(id="audit-summary-cards", className="mb-4"),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Feature", className="small fw-bold mb-1"),
                        dcc.Dropdown(
                            id="audit-feature-filter",
                            placeholder="Search or select features…",
                            multi=True,
                            searchable=True,
                            optionHeight=30,
                            style={"fontSize": "13px"},
                        ),
                    ], md=6),
                    dbc.Col([
                        dbc.Label("Status", className="small fw-bold mb-1"),
                        dcc.Dropdown(
                            id="audit-flag-filter",
                            options=[
                                {"label": "ok",            "value": "ok"},
                                {"label": "zero_variance", "value": "zero_variance"},
                                {"label": "mostly_zero",   "value": "mostly_zero"},
                                {"label": "high_corr",     "value": "high_corr"},
                                {"label": "high_missing",  "value": "high_missing"},
                            ],
                            placeholder="All",
                            multi=True,
                            searchable=False,
                            style={"fontSize": "13px"},
                        ),
                    ], md=4),
                    dbc.Col(
                        dbc.Button("Clear", id="audit-clear-filters",
                                   color="secondary", outline=True, size="sm"),
                        md=2, className="d-flex align-items-end pb-1",
                    ),
                ], className="mb-3 g-2"),
                html.Div(id="feature-audit-table"),
            ]),
        ], className="mb-4 shadow-sm"),

        # Q5 — Comparative SHAP
        dbc.Card([
            dbc.CardHeader(
                html.Span([
                    html.Strong("Comparative SHAP"),
                    html.Span(
                        " ⓘ",
                        id="comp-shap-info",
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
                                "Compares why two SKUs receive different forecasts by "
                                "averaging their SHAP values across all time periods.",
                                className="mb-2",
                                style={"fontSize": "12px"},
                            ),
                            html.P("How to read it:", className="mb-1 fw-bold", style={"fontSize": "12px"}),
                            html.Ul([
                                html.Li("Left chart: SHAP difference (A − B) — how much more/less each feature pushes SKU A's forecast vs SKU B's", style={"fontSize": "12px"}),
                                html.Li("Right chart: side-by-side SHAP values so you can see both profiles at once", style={"fontSize": "12px"}),
                                html.Li("SKU B is auto-suggested based on feature similarity — you can override it", style={"fontSize": "12px"}),
                            ], className="mb-0 ps-3"),
                        ],
                        target="comp-shap-info",
                        placement="right",
                        style={"maxWidth": "380px", "textAlign": "left"},
                    ),
                ])
            ),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("SKU A", html_for="comp-sku-a"),
                        dcc.Dropdown(id="comp-sku-a", placeholder="Select SKU A…", clearable=False),
                    ], md=5),
                    dbc.Col(
                        html.Div("vs", className="fw-bold text-muted text-center mt-4 pt-1"),
                        md=1,
                    ),
                    dbc.Col([
                        dbc.Label("SKU B (auto-suggested)", html_for="comp-sku-b"),
                        dcc.Dropdown(id="comp-sku-b", placeholder="Select SKU B…", clearable=False),
                    ], md=5),
                ], className="mb-4 g-2"),

                dcc.Loading(type="circle", children=dbc.Row([
                    dbc.Col(
                        dcc.Graph(id="comp-diff-chart", config={"displayModeBar": False},
                                  style={"height": "380px"}),
                        md=6,
                    ),
                    dbc.Col(
                        dcc.Graph(id="comp-side-chart", config={"displayModeBar": False},
                                  style={"height": "380px"}),
                        md=6,
                    ),
                ])),

                html.Div(id="comp-nlg", className="mt-3"),
            ]),
        ], className="mb-4 shadow-sm"),

        # Q7 — Stockout Risk Analysis
        dbc.Card([
            dbc.CardHeader(
                html.Span([
                    html.Strong("Stockout Risk Analysis"),
                    html.Span(
                        " ⓘ",
                        id="stockout-info",
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
                                "Identifies SKUs where past stockouts may have distorted the demand signal — "
                                "causing the model to under-forecast because it learned from artificially low sales.",
                                className="mb-2",
                                style={"fontSize": "12px"},
                            ),
                            html.P("How stockouts are detected:", className="mb-1 fw-bold", style={"fontSize": "12px"}),
                            html.Ul([
                                html.Li("Periods where sales_lag_7 ≤ 0.5 are flagged as suspected stockouts (proxy for missing inventory data)", style={"fontSize": "12px"}),
                                html.Li("The SHAP distortion chart shows how much the lag feature suppresses the forecast during those periods", style={"fontSize": "12px"}),
                                html.Li("The censored demand chart estimates what sales would have been if stock was available", style={"fontSize": "12px"}),
                            ], className="mb-0 ps-3"),
                        ],
                        target="stockout-info",
                        placement="right",
                        style={"maxWidth": "400px", "textAlign": "left"},
                    ),
                ])
            ),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("SKU", className="small fw-bold mb-1"),
                        dcc.Dropdown(
                            id="stockout-sku-col-filter",
                            placeholder="Search or select SKUs…",
                            multi=True,
                            searchable=True,
                            optionHeight=30,
                            style={"fontSize": "13px"},
                        ),
                    ], md=6),
                    dbc.Col([
                        dbc.Label("Risk Level", className="small fw-bold mb-1"),
                        dcc.Dropdown(
                            id="stockout-risk-col-filter",
                            options=[
                                {"label": "HIGH",     "value": "HIGH"},
                                {"label": "MODERATE", "value": "MODERATE"},
                                {"label": "LOW",      "value": "LOW"},
                            ],
                            placeholder="All",
                            multi=True,
                            searchable=False,
                            style={"fontSize": "13px"},
                        ),
                    ], md=4),
                    dbc.Col(
                        dbc.Button("Clear", id="stockout-col-clear-filters",
                                   color="secondary", outline=True, size="sm"),
                        md=2, className="d-flex align-items-end pb-1",
                    ),
                ], className="mb-3 g-2"),

                html.Div(id="stockout-global-table", className="mb-4"),

                html.Hr(),

                dbc.Row([
                    dbc.Col([
                        dbc.Label("Select SKU for detailed analysis", html_for="stockout-sku-selector"),
                        dcc.Dropdown(
                            id="stockout-sku-selector",
                            placeholder="Select an at-risk SKU…",
                            clearable=False,
                        ),
                    ], md=4),
                ], className="mb-3"),

                dcc.Loading(type="circle", children=html.Div([
                    dbc.Row([
                        dbc.Col(
                            dcc.Graph(id="stockout-pred-chart", config={"displayModeBar": False},
                                      style={"height": "280px"}),
                            md=6,
                        ),
                        dbc.Col(
                            dcc.Graph(id="stockout-shap-chart", config={"displayModeBar": False},
                                      style={"height": "280px"}),
                            md=6,
                        ),
                    ], className="mb-3"),
                    dcc.Graph(id="stockout-censored-chart", config={"displayModeBar": False},
                              style={"height": "300px"}),
                    html.Div(id="stockout-nlg", className="mt-3"),
                ])),
            ]),
        ], className="mb-4 shadow-sm"),

        # Q9 — Model Reliability & Cold-Start
        dbc.Card([
            dbc.CardHeader(
                html.Span([
                    html.Strong("Model Reliability & Cold-Start Detection"),
                    html.Span(
                        " ⓘ",
                        id="reliability-info",
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
                                "Evaluates how reliable the model's forecasts are across different "
                                "SKU groups and checks whether any SKUs have too little history "
                                "to forecast accurately.",
                                className="mb-2",
                                style={"fontSize": "12px"},
                            ),
                            html.P("What each section shows:", className="mb-1 fw-bold", style={"fontSize": "12px"}),
                            html.Ul([
                                html.Li("Interval Coverage — what % of actual sales fell inside the model's 80% prediction interval (target: 80%)", style={"fontSize": "12px"}),
                                html.Li("Confidence Distribution — how many SKUs the model is HIGH / MODERATE / LOW confidence about", style={"fontSize": "12px"}),
                                html.Li("Cold-Start — SKUs with fewer than 13 weeks of history flagged as unreliable", style={"fontSize": "12px"}),
                                html.Li("Subgroup Performance — MAE/RMSE/SMAPE broken down by product category", style={"fontSize": "12px"}),
                            ], className="mb-0 ps-3"),
                        ],
                        target="reliability-info",
                        placement="right",
                        style={"maxWidth": "400px", "textAlign": "left"},
                    ),
                ])
            ),
            dbc.CardBody([
                html.Div(id="reliability-stat-cards", className="mb-4"),
                dbc.Row([
                    dbc.Col(
                        dcc.Graph(id="confidence-dist-chart",
                                  config={"displayModeBar": False},
                                  style={"height": "280px"}),
                        md=5,
                    ),
                    dbc.Col(html.Div(id="subgroup-eval-table", style={"paddingTop": "45px"}), md=7),
                ]),
                html.Div(id="reliability-nlg", className="mt-3"),
            ]),
        ], className="shadow-sm"),
    ])
