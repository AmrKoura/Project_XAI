"""
Model Analysis page — Global Feature Importance, Feature Quality Audit,
Model Reliability & Cold-Start Detection.
"""

from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc


def layout() -> html.Div:
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H3([
                    "Model Analysis",
                    html.Span("ⓘ", id="explanations-page-info", style={
                        "fontSize": "14px", "color": "rgba(108,117,125,0.6)",
                        "cursor": "help", "marginLeft": "8px", "verticalAlign": "middle",
                    }),
                ], className="mb-0"),
                dbc.Tooltip(
                    "Understand how the selected model works — which features drive its "
                    "forecasts, whether the data is healthy, and how reliable its predictions are.",
                    target="explanations-page-info", placement="right",
                ),
            ], width="auto"),
        ], align="end", className="mb-4"),

        # Feature Intelligence (merged: Global SHAP + Audit)
        dbc.Card([
            dbc.CardHeader(
                html.Span([
                    html.Strong("Feature Intelligence"),
                    html.Span(" ⓘ", id="global-shap-info", style={
                        "cursor": "pointer", "fontSize": "14px",
                        "color": "#6c757d", "marginLeft": "6px", "userSelect": "none",
                    }),
                    dbc.Tooltip(
                        [
                            html.P("What this section shows:", className="mb-1 fw-bold",
                                   style={"fontSize": "12px"}),
                            html.Ul([
                                html.Li("Top chart — how much each feature influences the model's forecasts on average (SHAP)", style={"fontSize": "12px"}),
                                html.Li("Table — data quality of each feature cross-referenced with model reliance", style={"fontSize": "12px"}),
                                html.Li("Risk column — the most important flag: features with quality issues that the model heavily relies on", style={"fontSize": "12px"}),
                            ], className="mb-3 ps-3"),
                            html.Hr(style={"margin": "6px 0"}),
                            html.P([
                                html.Strong("Note: ", style={"fontSize": "12px"}),
                                html.Span(
                                    "The audit is data-based and stays the same across model types for the same window. "
                                    "The SHAP chart updates per model type.",
                                    style={"fontSize": "12px"},
                                ),
                            ], className="mb-0"),
                        ],
                        target="global-shap-info", placement="right",
                        style={"maxWidth": "400px", "textAlign": "left"},
                    ),
                ])
            ),
            dbc.CardBody([
                # SHAP chart + combined NLG
                dbc.Row([
                    dbc.Col(
                        dcc.Graph(id="global-shap-chart",
                                  config={"displayModeBar": False},
                                  style={"height": "380px"}),
                        md=8,
                    ),
                    dbc.Col(html.Div(id="global-shap-nlg"), md=4),
                ], className="mb-4"),

                html.Hr(className="mb-4"),

                # Audit section
                html.Div(id="audit-summary-cards", className="mb-4"),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Filter by feature", className="small fw-bold mb-1"),
                        dcc.Dropdown(
                            id="audit-feature-filter",
                            placeholder="Search or select features…",
                            multi=True, searchable=True,
                            optionHeight=30, style={"fontSize": "13px"},
                        ),
                    ], md=6),
                    dbc.Col([
                        dbc.Label("Filter by status", className="small fw-bold mb-1"),
                        dcc.Dropdown(
                            id="audit-flag-filter",
                            options=[
                                {"label": "ok",            "value": "ok"},
                                {"label": "zero_variance", "value": "zero_variance"},
                                {"label": "mostly_zero",   "value": "mostly_zero"},
                                {"label": "high_corr",     "value": "high_corr"},
                                {"label": "high_missing",  "value": "high_missing"},
                            ],
                            placeholder="All statuses", multi=True, searchable=False,
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

        # Model Reliability & Cold-Start Detection
        dbc.Card([
            dbc.CardHeader(
                html.Span([
                    html.Strong("Model Reliability"),
                    html.Span(" ⓘ", id="reliability-info", style={
                        "cursor": "pointer", "fontSize": "14px",
                        "color": "#6c757d", "marginLeft": "6px", "userSelect": "none",
                    }),
                    dbc.Tooltip(
                        [
                            html.P(
                                "Evaluates how reliable the model's forecasts are across different "
                                "SKU groups and checks whether any SKUs have too little history "
                                "to forecast accurately.",
                                className="mb-2", style={"fontSize": "12px"},
                            ),
                            html.P("What each section shows:", className="mb-1 fw-bold", style={"fontSize": "12px"}),
                            html.Ul([
                                html.Li("Interval Coverage — % of actuals inside the 80% band (target: 80%)", style={"fontSize": "12px"}),
                                html.Li("Avg Error (MAE) — how many units off the model is per week on average", style={"fontSize": "12px"}),
                                html.Li("Systematic Bias — whether the model consistently over- or under-forecasts", style={"fontSize": "12px"}),
                                html.Li("Category Accuracy — SMAPE per product category, color-coded Good / OK / Poor", style={"fontSize": "12px"}),
                            ], className="mb-0 ps-3"),
                        ],
                        target="reliability-info", placement="right",
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
