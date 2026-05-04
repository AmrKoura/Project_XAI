"""
What-If Simulator page — Demand Scenario + Cost Impact (combined).
"""

from dash import html, dcc
import dash_bootstrap_components as dbc


def layout() -> html.Div:
    return html.Div([

        # ── page header ──────────────────────────────────────────────────────
        dbc.Row([
            dbc.Col([
                html.H3([
                    "What-If Simulator",
                    html.Span("ⓘ", id="whatif-page-info", style={
                        "fontSize": "14px", "color": "rgba(108,117,125,0.6)",
                        "cursor": "help", "marginLeft": "8px", "verticalAlign": "middle",
                    }),
                ], className="mb-0"),
                dbc.Tooltip(
                    "Adjust price and SNAP status to generate a new demand scenario, "
                    "then instantly see the cost impact and optimal order quantity.",
                    target="whatif-page-info", placement="right",
                ),
            ], width="auto"),
        ], align="end", className="mb-4"),

        # ── SKU selector ─────────────────────────────────────────────────────
        dbc.Card([
            dbc.CardBody(
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Select SKU", html_for="whatif-sku",
                                  className="small fw-bold mb-1"),
                        dcc.Dropdown(id="whatif-sku",
                                     placeholder="Select a SKU to analyse…",
                                     clearable=False, searchable=True),
                    ], md=5),
                    dbc.Col(
                        html.Small(
                            "Adjust the scenario inputs and hit Run to see the new forecast "
                            "and its cost implications together.",
                            className="text-muted align-self-center",
                            style={"fontSize": "13px"},
                        ), md=7,
                    ),
                ]),
                style={"padding": "14px 20px"},
            ),
        ], className="mb-4 shadow-sm"),

        dcc.Store(id="whatif-current-store"),
        dcc.Store(id="whatif-scenario-a-store"),

        # ── Combined Scenario + Cost Impact ──────────────────────────────────
        dbc.Card([
            dbc.CardHeader(
                html.Span([
                    html.Strong("Demand Scenario & Cost Impact"),
                    html.Span(" ⓘ", id="whatif-q4-info", style={
                        "cursor": "pointer", "fontSize": "14px",
                        "color": "#6c757d", "marginLeft": "6px", "userSelect": "none",
                    }),
                    dbc.Tooltip([
                        html.P("How it works:", className="mb-1 fw-bold",
                               style={"fontSize": "12px"}),
                        html.Ul([
                            html.Li("Starts from the actual next-period forecast for the selected SKU", style={"fontSize": "12px"}),
                            html.Li("Adjust price and/or SNAP, then hit Run", style={"fontSize": "12px"}),
                            html.Li("The model re-predicts with your modified inputs", style={"fontSize": "12px"}),
                            html.Li("Cost impact auto-updates from the new forecast", style={"fontSize": "12px"}),
                        ], className="mb-0 ps-3"),
                        html.Hr(style={"margin": "8px 0"}),
                        html.P([
                            html.Strong("Important: ", style={"fontSize": "12px"}),
                            html.Span(
                                "Results show what the model associates with these inputs based on "
                                "historical patterns — not causal predictions. The model may not "
                                "correctly capture price elasticity.",
                                style={"fontSize": "12px"},
                            ),
                        ], className="mb-0"),
                    ], target="whatif-q4-info", placement="right",
                       style={"maxWidth": "400px", "textAlign": "left"}),
                ])
            ),
            dbc.CardBody([
                dbc.Row([
                    # ── Controls ─────────────────────────────────────────────
                    dbc.Col([
                        html.P(html.Small(
                            "Pre-filled with next forecast period values (Apr 30, 2016).",
                            className="text-muted"), className="mb-3"),

                        html.P("Demand inputs", className="small fw-bold text-uppercase mb-2",
                               style={"letterSpacing": "0.5px", "color": "#6c757d"}),

                        dbc.Label("Price ($)", id="whatif-price-tip",
                                  className="small fw-bold mb-0",
                                  style={"textDecoration": "underline",
                                         "textDecorationStyle": "dashed",
                                         "textDecorationColor": "rgba(108,117,125,0.7)",
                                         "cursor": "help"}),
                        dbc.Tooltip("Average selling price. Lower price typically lifts demand.",
                                    target="whatif-price-tip", placement="right"),
                        dbc.InputGroup([
                            dbc.InputGroupText("$", style={"fontSize": "13px"}),
                            dbc.Input(id="whatif-price", type="number",
                                      min=0.01, step=0.01,
                                      style={"fontSize": "13px"}),
                        ], size="sm", className="mb-3"),

                        dbc.Label("SNAP CA Active", id="whatif-snap-tip",
                                  className="small fw-bold mb-0",
                                  style={"textDecoration": "underline",
                                         "textDecorationStyle": "dashed",
                                         "textDecorationColor": "rgba(108,117,125,0.7)",
                                         "cursor": "help"}),
                        dbc.Tooltip("Whether SNAP benefit redemption is active this week.",
                                    target="whatif-snap-tip", placement="right"),
                        dbc.Switch(id="whatif-snap", value=False,
                                   label="", className="mb-4"),

                        html.Hr(className="my-3"),
                        html.P("Cost parameters", className="small fw-bold text-uppercase mb-2",
                               style={"letterSpacing": "0.5px", "color": "#6c757d"}),

                        dbc.Label("Gross Margin %", id="whatif-margin-tip",
                                  className="small fw-bold mb-0",
                                  style={"textDecoration": "underline",
                                         "textDecorationStyle": "dashed",
                                         "textDecorationColor": "rgba(108,117,125,0.7)",
                                         "cursor": "help"}),
                        dbc.Tooltip(
                            "Profit margin as % of sell price lost per unit on a stockout. "
                            "E.g. 25% on a $2.24 item = $0.56 opportunity cost per missed sale.",
                            target="whatif-margin-tip", placement="right"),
                        dbc.InputGroup([
                            dbc.Input(id="whatif-unit-margin", type="number",
                                      value=25, min=1, max=80, step=1,
                                      style={"fontSize": "13px"}),
                            dbc.InputGroupText("%", style={"fontSize": "13px"}),
                        ], size="sm", className="mb-3"),

                        dbc.Label("Holding Cost %", id="whatif-holding-tip",
                                  className="small fw-bold mb-0",
                                  style={"textDecoration": "underline",
                                         "textDecorationStyle": "dashed",
                                         "textDecorationColor": "rgba(108,117,125,0.7)",
                                         "cursor": "help"}),
                        dbc.Tooltip(
                            "Cost to hold one unsold unit as % of sell price (storage, spoilage). "
                            "E.g. 8% on a $2.24 item = $0.18/unit.",
                            target="whatif-holding-tip", placement="right"),
                        dbc.InputGroup([
                            dbc.Input(id="whatif-holding-cost", type="number",
                                      value=8, min=1, max=50, step=1,
                                      style={"fontSize": "13px"}),
                            dbc.InputGroupText("%", style={"fontSize": "13px"}),
                        ], size="sm", className="mb-3"),

                        html.Hr(className="my-3"),
                        html.P("Order override", className="small fw-bold text-uppercase mb-2",
                               style={"letterSpacing": "0.5px", "color": "#6c757d"}),

                        dbc.Label("I plan to order (units)", id="whatif-custom-order-tip",
                                  className="small fw-bold mb-0",
                                  style={"textDecoration": "underline",
                                         "textDecorationStyle": "dashed",
                                         "textDecorationColor": "rgba(108,117,125,0.7)",
                                         "cursor": "help"}),
                        dbc.Tooltip(
                            "Enter your intended order quantity to see how its cost compares "
                            "to the model's optimal. Useful when supplier minimums or shelf "
                            "constraints prevent ordering the exact optimal amount.",
                            target="whatif-custom-order-tip", placement="right",
                        ),
                        dbc.Input(id="whatif-custom-order", type="number",
                                  min=0, step=1, placeholder="Leave blank → use optimal",
                                  style={"fontSize": "13px"},
                                  className="mb-4"),

                        html.Div([
                            dbc.Button("Run", id="whatif-run-q4", size="sm",
                                       className="me-2",
                                       style={"backgroundColor": "#7B61FF",
                                              "borderColor": "#7B61FF", "color": "#fff"}),
                            dbc.Button("Reset", id="whatif-reset-controls",
                                       color="secondary", outline=True, size="sm"),
                        ], className="d-flex"),

                    ], md=4),

                    # ── Results ───────────────────────────────────────────────
                    dbc.Col([
                        html.Div(id="whatif-causal-warning"),
                        html.Div(id="whatif-impact-cards", className="mb-3"),
                        dcc.Loading(type="circle", children=dcc.Graph(
                            id="whatif-compare-chart",
                            config={"displayModeBar": False},
                            style={"height": "240px"},
                        )),

                        html.Hr(className="my-3"),

                        html.Div(id="whatif-cost-cards", className="mb-2"),
                        html.Div(id="whatif-order-comparison", className="mb-3"),
                        dcc.Loading(type="circle", children=dcc.Graph(
                            id="whatif-cost-chart",
                            config={"displayModeBar": False},
                            style={"height": "280px"},
                        )),
                        dcc.Loading(type="circle", children=dcc.Graph(
                            id="whatif-risk-chart",
                            config={"displayModeBar": False},
                            style={"height": "260px"},
                        )),
                    ], md=8),
                ], className="mb-3"),

                html.Div(id="whatif-q4-nlg"),
                html.Div(id="whatif-q10-nlg", className="mt-2"),

                html.Hr(className="mt-4 mb-3"),
                dbc.Row([
                    dbc.Col(
                        html.Small("Run a scenario above, then save it to compare against a second scenario.",
                                   className="text-muted", style={"fontSize": "13px"}),
                        md=8,
                    ),
                    dbc.Col(
                        html.Div([
                            dbc.Button("📌 Save as Scenario A", id="whatif-lock-a",
                                       color="primary", outline=True, size="sm",
                                       className="me-2"),
                            dbc.Button("✕ Clear", id="whatif-clear-a",
                                       color="secondary", outline=True, size="sm"),
                        ], className="d-flex justify-content-end"),
                        md=4,
                    ),
                ], className="mb-3"),

                html.Div(id="whatif-scenario-comparison"),
            ]),
        ], className="shadow-sm"),
    ])
