"""
Overview page — SKU overview and replenishment cards.
"""

from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc

_URGENCY_COLOUR = {"CRITICAL": "danger", "HIGH": "warning", "LOW": "success"}

_COL_TOOLTIPS = {
    "sku_id":              "Unique product code",
    "forecast_q50":        "Best single estimate of demand for the forecast period — think of it as the 'most likely' number",
    "forecast_q10":        "Low-end demand estimate — there's only a 10% chance actual demand is below this",
    "forecast_q90":        "High-end demand estimate — there's only a 10% chance actual demand is above this",
    "safety_stock":        "Extra buffer units kept to protect against unexpected demand spikes",
    "days_of_stock":       "How many days current stock will last at the expected daily demand rate",
    "reorder_qty":         "Number of units the system recommends ordering to replenish stock",
    "trigger_reorder":     "Whether the system recommends placing an order right now",
    "urgency":             "How urgent replenishment is — based on how many days of stock remain vs lead time",
    "confidence_band_pct": "How wide the forecast range is as a % of the expected demand — lower means more confident",
}


def layout() -> html.Div:
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H3([
                    "Replenishment Overview",
                    html.Span("ⓘ", id="overview-page-info", style={
                        "fontSize": "14px", "color": "rgba(108,117,125,0.6)",
                        "cursor": "help", "marginLeft": "8px", "verticalAlign": "middle",
                    }),
                ], className="mb-0"),
                dbc.Tooltip(
                    "See which products need restocking. This page shows stock levels, "
                    "urgency, and order recommendations for all your products at a glance.",
                    target="overview-page-info", placement="right",
                ),
            ], width="auto"),
        ], align="end", className="mb-3"),

        html.Div(id="dash-summary-stats", className="mb-4"),

        dbc.Row([
            dbc.Col([
                dbc.Label("Filter by urgency", html_for="urgency-filter"),
                dcc.Dropdown(
                    id="urgency-filter",
                    options=[
                        {"label": "All",      "value": "ALL"},
                        {"label": "CRITICAL", "value": "CRITICAL"},
                        {"label": "HIGH",     "value": "HIGH"},
                        {"label": "LOW",      "value": "LOW"},
                    ],
                    value="ALL",
                    clearable=False,
                    style={"width": "200px"},
                ),
            ], width="auto"),
            dbc.Col([
                dbc.Label("Filter by category", html_for="category-filter"),
                dcc.Dropdown(
                    id="category-filter",
                    options=[
                        {"label": "All categories", "value": "ALL"},
                        {"label": "FOODS_1",         "value": "FOODS_1"},
                        {"label": "FOODS_2",         "value": "FOODS_2"},
                        {"label": "FOODS_3",         "value": "FOODS_3"},
                    ],
                    value="ALL",
                    clearable=False,
                    style={"width": "200px"},
                ),
            ], width="auto"),
        ], className="mb-3 g-3"),

        dbc.Card([
            dbc.CardHeader(html.Strong("SKU Overview")),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("SKU", className="small fw-bold mb-1"),
                        dcc.Dropdown(
                            id="tbl-sku-filter",
                            placeholder="Search or select SKUs…",
                            multi=True,
                            searchable=True,
                            optionHeight=30,
                            style={"fontSize": "13px"},
                        ),
                    ], md=5),
                    dbc.Col([
                        dbc.Label("Urgency", className="small fw-bold mb-1"),
                        dcc.Dropdown(
                            id="tbl-urgency-filter",
                            options=[
                                {"label": "CRITICAL", "value": "CRITICAL"},
                                {"label": "HIGH",     "value": "HIGH"},
                                {"label": "LOW",      "value": "LOW"},
                            ],
                            placeholder="All",
                            multi=True,
                            searchable=False,
                            style={"fontSize": "13px"},
                        ),
                    ], md=3),
                    dbc.Col([
                        dbc.Label("Order Now?", className="small fw-bold mb-1"),
                        dcc.Dropdown(
                            id="tbl-trigger-filter",
                            options=[
                                {"label": "Yes", "value": "True"},
                                {"label": "No",  "value": "False"},
                            ],
                            placeholder="All",
                            multi=True,
                            searchable=False,
                            style={"fontSize": "13px"},
                        ),
                    ], md=2),
                    dbc.Col(
                        dbc.Button(
                            "Clear", id="tbl-clear-filters",
                            color="secondary", outline=True, size="sm",
                        ),
                        md=2, className="d-flex align-items-end pb-1",
                    ),
                ], className="mb-3 g-2"),
                html.Div(id="sku-overview-table"),
            ]),
        ], className="mb-4 shadow-sm"),

        dbc.Card([
            dbc.CardHeader(html.Strong("Replenishment Cards")),
            dbc.CardBody(html.Div(id="replenishment-cards-grid")),
        ], className="shadow-sm"),
    ])


def build_sku_overview_table(df_records: list[dict], dark: bool = False) -> dash_table.DataTable:
    columns = [
        {"name": "SKU",           "id": "sku_id"},
        {"name": "Forecast (q50)","id": "forecast_q50",       "type": "numeric"},
        {"name": "Low (q10)",     "id": "forecast_q10",       "type": "numeric"},
        {"name": "High (q90)",    "id": "forecast_q90",       "type": "numeric"},
        {"name": "Safety Stock",  "id": "safety_stock",       "type": "numeric"},
        {"name": "Days of Stock", "id": "days_of_stock",      "type": "numeric", "format": {"specifier": ".1f"}},
        {"name": "Order Qty",     "id": "reorder_qty",        "type": "numeric"},
        {"name": "Order Now?",    "id": "trigger_reorder"},
        {"name": "Urgency",       "id": "urgency"},
        {"name": "Confidence %",  "id": "confidence_band_pct","type": "numeric", "format": {"specifier": ".1f"}},
    ]

    if dark:
        s_header = {"backgroundColor": "#0A0C1A", "color": "#FFFFFF", "fontWeight": "bold",
                    "fontSize": "13px", "textDecoration": "underline",
                    "textDecorationStyle": "dashed", "textDecorationColor": "rgba(95,1,251,0.6)",
                    "cursor": "help", "border": "1px solid rgba(95,1,251,0.3)"}
        s_data   = {"backgroundColor": "#0F1225", "color": "#FFFFFF",
                    "border": "1px solid rgba(95,1,251,0.2)"}
        s_cell   = {"fontSize": "13px", "padding": "6px 10px", "textAlign": "left",
                    "backgroundColor": "#0F1225", "color": "#FFFFFF"}
        s_cond   = [
            {"if": {"filter_query": '{urgency} = "CRITICAL"'},
             "backgroundColor": "rgba(255,45,85,0.15)", "color": "#FF2D55"},
            {"if": {"filter_query": '{urgency} = "HIGH"'},
             "backgroundColor": "rgba(255,184,0,0.15)",  "color": "#FFB800"},
            {"if": {"filter_query": '{trigger_reorder} = "True"', "column_id": "trigger_reorder"},
             "color": "#FF2D55", "fontWeight": "bold"},
        ]
    else:
        s_header = {"backgroundColor": "#343a40", "color": "white", "fontWeight": "bold",
                    "fontSize": "13px", "textDecoration": "underline",
                    "textDecorationStyle": "dashed", "textDecorationColor": "rgba(255,255,255,0.5)",
                    "cursor": "help"}
        s_data   = {}
        s_cell   = {"fontSize": "13px", "padding": "6px 10px", "textAlign": "left"}
        s_cond   = [
            {"if": {"filter_query": '{urgency} = "CRITICAL"'}, "backgroundColor": "#f8d7da", "color": "#842029"},
            {"if": {"filter_query": '{urgency} = "HIGH"'},      "backgroundColor": "#fff3cd", "color": "#664d03"},
            {"if": {"filter_query": '{trigger_reorder} = "True"', "column_id": "trigger_reorder"},
             "color": "#dc3545", "fontWeight": "bold"},
        ]

    return dash_table.DataTable(
        id="sku-datatable",
        columns=columns,
        data=df_records,
        page_size=20,
        sort_action="native",
        tooltip_header={col["id"]: _COL_TOOLTIPS.get(col["id"], "") for col in columns},
        tooltip_delay=0,
        tooltip_duration=None,
        style_table={"overflowX": "auto"},
        style_header=s_header,
        style_data=s_data,
        style_cell=s_cell,
        style_data_conditional=s_cond,
    )


def build_replenishment_cards(records: list[dict]) -> html.Div:
    if not records:
        return html.P("No SKUs match the current filter.", className="text-muted")

    cards = []
    for r in records[:24]:
        urgency = r.get("urgency", "LOW")
        colour  = _URGENCY_COLOUR.get(urgency, "secondary")

        if urgency == "CRITICAL":
            action_el = html.Span("⚠ Reorder Now", className="text-danger fw-bold",
                                  style={"fontSize": "12px"})
        elif urgency == "HIGH":
            action_el = html.Span("👁 Watch", className="text-warning fw-bold",
                                  style={"fontSize": "12px"})
        else:
            action_el = html.Span("")

        card = dbc.Col(
            dbc.Card([
                dbc.CardHeader(
                    dbc.Row([
                        dbc.Col(html.Strong(r["sku_id"], style={"fontSize": "13px"})),
                        dbc.Col(dbc.Badge(urgency, color=colour, className="float-end"), width="auto"),
                    ]),
                    style={"padding": "8px 12px"},
                ),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Small("Forecast", className="text-muted d-block"),
                            html.Strong(f"{r['forecast_q50']:.1f} units"),
                        ], width=6),
                        dbc.Col([
                            html.Small("Order Qty", className="text-muted d-block"),
                            html.Strong(f"{r['reorder_qty']:.1f} units"),
                        ], width=6),
                    ], className="mb-2"),
                    dbc.Row([
                        dbc.Col([
                            html.Small("Days of Stock", className="text-muted d-block"),
                            html.Span(f"{r['days_of_stock']:.1f}"),
                        ], width=6),
                        dbc.Col([
                            html.Small("Safety Stock", className="text-muted d-block"),
                            html.Span(f"{r['safety_stock']:.1f}"),
                        ], width=6),
                    ], className="mb-2"),
                    html.Hr(style={"margin": "8px 0"}),
                    action_el,
                ], style={"padding": "10px 12px"}),
            ], className="h-100 shadow-sm"),
            xs=12, sm=6, md=4, lg=3, className="mb-3",
        )
        cards.append(card)

    return dbc.Row(cards)
