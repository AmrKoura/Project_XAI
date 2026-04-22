"""
What-If Simulator page — Demand Counterfactual (Q4) + Cost Impact (Q10).
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
                    "Test out 'what if' scenarios before making decisions. Change the price, "
                    "discount, or SNAP status and instantly see how demand is expected to respond.",
                    target="whatif-page-info", placement="right",
                ),
            ], width="auto"),
        ], align="end", className="mb-4"),

        # ── shared SKU selector ──────────────────────────────────────────────
        dbc.Card([
            dbc.CardBody(
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Select SKU", html_for="whatif-sku",
                                  className="small fw-bold mb-1"),
                        dcc.Dropdown(
                            id="whatif-sku",
                            placeholder="Select a SKU to analyse…",
                            clearable=False,
                            searchable=True,
                        ),
                    ], md=5),
                    dbc.Col(
                        html.Small(
                            "Adjust the controls below to see how the demand forecast "
                            "and ordering costs change for the selected SKU.",
                            className="text-muted align-self-center",
                            style={"fontSize": "13px"},
                        ),
                        md=7,
                    ),
                ]),
                style={"padding": "14px 20px"},
            ),
        ], className="mb-4 shadow-sm"),

        # ── Q4 — Demand What-If ──────────────────────────────────────────────
        dbc.Card([
            dbc.CardHeader(
                html.Span([
                    html.Strong("Demand What-If Simulator"),
                    html.Span(
                        " ⓘ",
                        id="whatif-q4-info",
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
                            html.P("How it works:", className="mb-1 fw-bold",
                                   style={"fontSize": "12px"}),
                            html.Ul([
                                html.Li(
                                    "Adjust price, discount depth, or SNAP status below",
                                    style={"fontSize": "12px"},
                                ),
                                html.Li(
                                    "The model re-runs instantly with your hypothetical values",
                                    style={"fontSize": "12px"},
                                ),
                                html.Li(
                                    "The sweep chart shows how the forecast responds across "
                                    "the full range of the feature you last adjusted",
                                    style={"fontSize": "12px"},
                                ),
                                html.Li(
                                    "All three controls can be combined — the impact cards "
                                    "always show the total combined effect",
                                    style={"fontSize": "12px"},
                                ),
                            ], className="mb-0 ps-3"),
                        ],
                        target="whatif-q4-info",
                        placement="right",
                        style={"maxWidth": "370px", "textAlign": "left"},
                    ),
                ])
            ),
            dbc.CardBody([

                # Impact summary cards
                html.Div(id="whatif-impact-cards", className="mb-4"),

                # Controls (left) + Sweep chart (right)
                dbc.Row([
                    # ── controls ─────────────────────────────────────────────
                    dbc.Col([

                        # Price input
                        html.Div([
                            html.Div([
                                dbc.Label(
                                    "Price ($)",
                                    id="whatif-price-tip",
                                    className="small fw-bold mb-0",
                                    style={
                                        "textDecoration": "underline",
                                        "textDecorationStyle": "dashed",
                                        "textDecorationColor": "rgba(108,117,125,0.7)",
                                        "cursor": "help",
                                    },
                                ),
                                dbc.Tooltip(
                                    "Average selling price for this SKU. "
                                    "A lower price typically increases demand; "
                                    "a higher price may suppress it.",
                                    target="whatif-price-tip",
                                    placement="right",
                                ),
                            ], className="mb-1"),
                            dbc.InputGroup([
                                dbc.InputGroupText("$", style={"fontSize": "13px"}),
                                dbc.Input(
                                    id="whatif-price",
                                    type="number",
                                    min=0.01,
                                    step=0.01,
                                    placeholder="Current price",
                                    style={"fontSize": "13px"},
                                ),
                            ], size="sm"),
                        ], className="mb-4"),

                        # Discount depth slider
                        html.Div([
                            html.Div([
                                dbc.Label(
                                    "Discount Depth",
                                    id="whatif-discount-tip",
                                    className="small fw-bold mb-0",
                                    style={
                                        "textDecoration": "underline",
                                        "textDecorationStyle": "dashed",
                                        "textDecorationColor": "rgba(108,117,125,0.7)",
                                        "cursor": "help",
                                    },
                                ),
                                dbc.Tooltip(
                                    "Fraction of the price given as a discount. "
                                    "0 = no discount, 0.30 = 30% off. "
                                    "Discounts typically lift short-term demand.",
                                    target="whatif-discount-tip",
                                    placement="right",
                                ),
                            ], className="mb-2"),
                            dcc.Slider(
                                id="whatif-discount",
                                min=0.0,
                                max=0.5,
                                step=0.05,
                                value=0.0,
                                marks={
                                    0.0: "0%",
                                    0.1: "10%",
                                    0.2: "20%",
                                    0.3: "30%",
                                    0.4: "40%",
                                    0.5: "50%",
                                },
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ], className="mb-4"),

                        # SNAP toggle
                        html.Div([
                            html.Div([
                                dbc.Label(
                                    "SNAP Eligibility",
                                    id="whatif-snap-tip",
                                    className="small fw-bold mb-0",
                                    style={
                                        "textDecoration": "underline",
                                        "textDecorationStyle": "dashed",
                                        "textDecorationColor": "rgba(108,117,125,0.7)",
                                        "cursor": "help",
                                    },
                                ),
                                dbc.Tooltip(
                                    "SNAP (Supplemental Nutrition Assistance Program) "
                                    "benefit day eligibility. When active, certain food "
                                    "items see a demand uplift as benefit recipients "
                                    "make purchases.",
                                    target="whatif-snap-tip",
                                    placement="right",
                                ),
                            ], className="mb-1"),
                            dbc.Switch(
                                id="whatif-snap",
                                value=False,
                                label="",
                                style={"fontSize": "13px"},
                            ),
                        ], className="mb-4"),

                        html.Div([
                            dbc.Button(
                                "Run",
                                id="whatif-run-q4",
                                size="sm",
                                className="me-2",
                                style={"backgroundColor": "#7B61FF",
                                       "borderColor": "#7B61FF", "color": "#fff"},
                            ),
                            dbc.Button(
                                "Reset to SKU defaults",
                                id="whatif-reset-controls",
                                color="secondary",
                                outline=True,
                                size="sm",
                            ),
                        ], className="d-flex"),

                    ], md=4),

                    # ── sweep charts (tabbed) ─────────────────────────────────
                    dbc.Col([
                        dbc.Tabs([
                            dbc.Tab(
                                dcc.Loading(type="circle", children=dcc.Graph(
                                    id="whatif-sweep-price",
                                    config={"displayModeBar": False},
                                    style={"height": "310px"},
                                )),
                                label="Price Sweep",
                                tab_id="tab-price",
                            ),
                            dbc.Tab(
                                dcc.Loading(type="circle", children=dcc.Graph(
                                    id="whatif-sweep-discount",
                                    config={"displayModeBar": False},
                                    style={"height": "310px"},
                                )),
                                label="Discount Sweep",
                                tab_id="tab-discount",
                            ),
                            dbc.Tab(
                                dcc.Loading(type="circle", children=dcc.Graph(
                                    id="whatif-sweep-snap",
                                    config={"displayModeBar": False},
                                    style={"height": "310px"},
                                )),
                                label="SNAP Effect",
                                tab_id="tab-snap",
                            ),
                        ], id="whatif-sweep-tabs", active_tab="tab-price"),
                    ], md=8),
                ], className="mb-3"),

                html.Div(id="whatif-q4-nlg"),
            ]),
        ], className="mb-4 shadow-sm"),

        # ── Q10 — Cost Impact Simulator ──────────────────────────────────────
        dbc.Card([
            dbc.CardHeader(
                html.Span([
                    html.Strong("Cost Impact Simulator"),
                    html.Span(
                        " ⓘ",
                        id="whatif-q10-info",
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
                            html.P("What it calculates:", className="mb-1 fw-bold",
                                   style={"fontSize": "12px"}),
                            html.Ul([
                                html.Li(
                                    "Uses the SKU's forecast range (q10 / q50 / q90) to "
                                    "simulate 5,000 demand scenarios via Monte Carlo",
                                    style={"fontSize": "12px"},
                                ),
                                html.Li(
                                    "For each scenario, computes stockout cost (lost margin) "
                                    "and overstock cost (holding excess stock)",
                                    style={"fontSize": "12px"},
                                ),
                                html.Li(
                                    "Finds the order quantity that minimises expected total "
                                    "cost using the newsvendor critical ratio formula",
                                    style={"fontSize": "12px"},
                                ),
                                html.Li(
                                    "The cost curve shows how total expected cost changes "
                                    "across all possible order quantities — the green line "
                                    "marks the optimal point",
                                    style={"fontSize": "12px"},
                                ),
                            ], className="mb-0 ps-3"),
                        ],
                        target="whatif-q10-info",
                        placement="right",
                        style={"maxWidth": "390px", "textAlign": "left"},
                    ),
                ])
            ),
            dbc.CardBody([

                # Business-parameter inputs
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            dbc.Label(
                                "Unit Margin ($/unit)",
                                id="whatif-margin-tip",
                                className="small fw-bold mb-0",
                                style={
                                    "textDecoration": "underline",
                                    "textDecorationStyle": "dashed",
                                    "textDecorationColor": "rgba(108,117,125,0.7)",
                                    "cursor": "help",
                                },
                            ),
                            dbc.Tooltip(
                                "Profit margin lost per unit if a stockout occurs — "
                                "the opportunity cost of each missed sale. "
                                "A higher margin means the model recommends ordering more.",
                                target="whatif-margin-tip",
                                placement="right",
                            ),
                        ], className="mb-1"),
                        dbc.InputGroup([
                            dbc.InputGroupText("$", style={"fontSize": "13px"}),
                            dbc.Input(
                                id="whatif-unit-margin",
                                type="number",
                                value=3.50,
                                min=0.01,
                                step="any",
                                style={"fontSize": "13px"},
                            ),
                        ], size="sm"),
                    ], md=4),

                    dbc.Col([
                        html.Div([
                            dbc.Label(
                                "Holding Cost ($/unit)",
                                id="whatif-holding-tip",
                                className="small fw-bold mb-0",
                                style={
                                    "textDecoration": "underline",
                                    "textDecorationStyle": "dashed",
                                    "textDecorationColor": "rgba(108,117,125,0.7)",
                                    "cursor": "help",
                                },
                            ),
                            dbc.Tooltip(
                                "Cost to hold one unsold unit for the forecast period "
                                "(7 days). Covers storage, spoilage, and capital tied up "
                                "in stock. A higher holding cost nudges the optimal order "
                                "quantity downward.",
                                target="whatif-holding-tip",
                                placement="right",
                            ),
                        ], className="mb-1"),
                        dbc.InputGroup([
                            dbc.InputGroupText("$", style={"fontSize": "13px"}),
                            dbc.Input(
                                id="whatif-holding-cost",
                                type="number",
                                value=0.80,
                                min=0.01,
                                step="any",
                                style={"fontSize": "13px"},
                            ),
                        ], size="sm"),
                    ], md=4),

                    dbc.Col(
                        html.Div([
                            dbc.Button(
                                "Run",
                                id="whatif-run-costs",
                                size="sm",
                                className="me-2",
                                style={"backgroundColor": "#7B61FF",
                                       "borderColor": "#7B61FF", "color": "#fff"},
                            ),
                            dbc.Button(
                                "Reset defaults",
                                id="whatif-reset-costs",
                                color="secondary",
                                outline=True,
                                size="sm",
                            ),
                        ], className="d-flex"),
                        md=4,
                        className="d-flex align-items-end pb-1",
                    ),
                ], className="mb-4 g-3"),

                # Cost stat cards
                html.Div(id="whatif-cost-cards", className="mb-4"),

                # Cost curve chart
                dcc.Loading(type="circle", children=dcc.Graph(
                    id="whatif-cost-chart",
                    config={"displayModeBar": False},
                    style={"height": "380px"},
                )),

                html.Div(id="whatif-q10-nlg", className="mt-3"),
            ]),
        ], className="shadow-sm"),
    ])
