"""
Report Generator page — select SKUs, choose template and format, download PDF/DOCX.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc


_SECTIONS = [
    ("replenishment", "Replenishment Summary"),
    ("forecast",      "Demand Forecast"),
    ("shap",          "Local SHAP Analysis"),
    ("temporal",      "Temporal Pattern"),
    ("whatif",        "What-If Sensitivity"),
    ("cost",          "Cost Impact"),
    ("reliability",   "Model Reliability"),
]

_BTN_STYLE = {"backgroundColor": "#7B61FF", "borderColor": "#7B61FF", "color": "#fff"}


def layout() -> html.Div:
    return html.Div([

        # ── page header ──────────────────────────────────────────────────────
        dbc.Row([
            dbc.Col([
                html.H3([
                    "Report Generator",
                    html.Span("ⓘ", id="reports-page-info", style={
                        "fontSize": "14px", "color": "rgba(108,117,125,0.6)",
                        "cursor": "help", "marginLeft": "8px", "verticalAlign": "middle",
                    }),
                ], className="mb-0"),
                dbc.Tooltip(
                    "Download a ready-to-share report for any product. Choose a template and "
                    "get a Word document with forecasts, order recommendations, and explanations.",
                    target="reports-page-info", placement="right",
                ),
            ], width="auto"),
        ], align="end", className="mb-4"),

        dbc.Row([

            # ── left panel — controls ────────────────────────────────────────
            dbc.Col([

                # SKU selector
                dbc.Card([
                    dbc.CardHeader(html.Strong("Select SKUs")),
                    dbc.CardBody([
                        dcc.Dropdown(
                            id="report-sku-select",
                            placeholder="Select one or more SKUs…",
                            multi=True,
                            searchable=True,
                            clearable=True,
                        ),
                        html.Small(
                            "You can select multiple SKUs. Each gets its own section in the report.",
                            className="text-muted mt-2 d-block",
                            style={"fontSize": "12px"},
                        ),
                    ]),
                ], className="mb-3 shadow-sm"),

                # Template selector
                dbc.Card([
                    dbc.CardHeader(html.Strong("Report Template")),
                    dbc.CardBody([
                        dbc.RadioItems(
                            id="report-template",
                            options=[
                                {
                                    "label": html.Span([
                                        html.Strong("Replenishment Brief"),
                                        html.Small(
                                            " — 1-2 pages · replenishment + forecast + cost",
                                            className="text-muted ms-1",
                                        ),
                                    ]),
                                    "value": "brief",
                                },
                                {
                                    "label": html.Span([
                                        html.Strong("Full XAI Report"),
                                        html.Small(
                                            " — 4-6 pages · all sections with charts",
                                            className="text-muted ms-1",
                                        ),
                                    ]),
                                    "value": "full",
                                },
                                {
                                    "label": html.Span([
                                        html.Strong("Executive Summary"),
                                        html.Small(
                                            " — 1 page · bullet points only, no charts",
                                            className="text-muted ms-1",
                                        ),
                                    ]),
                                    "value": "exec",
                                },
                            ],
                            value="full",
                            inputStyle={"accentColor": "#7B61FF"},
                        ),
                    ]),
                ], className="mb-3 shadow-sm"),

                # Section toggles (Full XAI only)
                dbc.Collapse(
                    dbc.Card([
                        dbc.CardHeader(html.Strong("Sections to include")),
                        dbc.CardBody(
                            dbc.Checklist(
                                id="report-sections",
                                options=[{"label": lbl, "value": val} for val, lbl in _SECTIONS],
                                value=[val for val, _ in _SECTIONS],
                                inputStyle={"accentColor": "#7B61FF"},
                            ),
                        ),
                    ], className="shadow-sm"),
                    id="report-sections-collapse",
                    is_open=True,
                ),

            ], md=5, className="mb-3"),

            # ── right panel — format + params + generate ─────────────────────
            dbc.Col([

                # Output format — DOCX only (hidden store)
                dcc.Store(id="report-format", data="docx"),

                # Business parameters
                dbc.Card([
                    dbc.CardHeader(html.Strong("Business Parameters")),
                    dbc.CardBody(
                        dbc.Row([
                            dbc.Col([
                                dbc.Label(
                                    "Gross Margin %",
                                    id="report-margin-tip",
                                    className="small fw-bold mb-0",
                                    style={
                                        "textDecoration": "underline",
                                        "textDecorationStyle": "dashed",
                                        "textDecorationColor": "rgba(108,117,125,0.7)",
                                        "cursor": "help",
                                    },
                                ),
                                dbc.Tooltip(
                                    "Profit margin as % of sell price lost per unit on a stockout. "
                                    "Converted to $ per SKU using its actual sell price.",
                                    target="report-margin-tip", placement="top",
                                ),
                                dbc.InputGroup([
                                    dbc.Input(
                                        id="report-unit-margin",
                                        type="number", value=25,
                                        min=1, max=80, step=1,
                                        style={"fontSize": "13px"},
                                    ),
                                    dbc.InputGroupText("%", style={"fontSize": "13px"}),
                                ], size="sm", className="mt-1"),
                            ], md=6),
                            dbc.Col([
                                dbc.Label(
                                    "Holding Cost %",
                                    id="report-holding-tip",
                                    className="small fw-bold mb-0",
                                    style={
                                        "textDecoration": "underline",
                                        "textDecorationStyle": "dashed",
                                        "textDecorationColor": "rgba(108,117,125,0.7)",
                                        "cursor": "help",
                                    },
                                ),
                                dbc.Tooltip(
                                    "Cost to hold one unsold unit as % of sell price (storage, spoilage). "
                                    "Converted to $ per SKU using its actual sell price.",
                                    target="report-holding-tip", placement="top",
                                ),
                                dbc.InputGroup([
                                    dbc.Input(
                                        id="report-holding-cost",
                                        type="number", value=8,
                                        min=1, max=50, step=1,
                                        style={"fontSize": "13px"},
                                    ),
                                    dbc.InputGroupText("%", style={"fontSize": "13px"}),
                                ], size="sm", className="mt-1"),
                            ], md=6),
                        ], className="g-2"),
                    ),
                ], className="mb-3 shadow-sm"),

                # Generate button + status
                dbc.Card([
                    dbc.CardBody([
                        dbc.Button(
                            "Generate Report",
                            id="report-generate",
                            size="md",
                            className="w-100 mb-3",
                            style=_BTN_STYLE,
                        ),
                        dcc.Loading(
                            type="circle",
                            children=html.Div(id="report-status", className="text-muted",
                                              style={"fontSize": "13px", "textAlign": "center"}),
                        ),
                    ]),
                ], className="shadow-sm"),

                # Hidden download component
                dcc.Download(id="report-download"),

            ], md=7, className="mb-3"),
        ]),
    ])
