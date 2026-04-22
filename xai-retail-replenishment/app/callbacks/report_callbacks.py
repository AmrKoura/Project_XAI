"""
Callbacks for the Report Generator page.
"""

from __future__ import annotations

import datetime

from dash import Input, Output, State, ctx, html, dcc
import dash_bootstrap_components as dbc

import app.data_store as ds
from reporting.report_builder import build_report


def register_report_callbacks(app) -> None:

    # ── subtitle ──────────────────────────────────────────────────────────────
    @app.callback(
        Output("report-subtitle", "children"),
        Input("model-store", "data"),
    )
    def report_subtitle(model_key: str) -> str:
        cfg = ds.MODEL_CONFIGS.get(model_key, {})
        return cfg.get("label", "") + " · LGBM · M5 Walmart"

    # ── populate SKU dropdown ─────────────────────────────────────────────────
    @app.callback(
        Output("report-sku-select", "options"),
        Input("model-store", "data"),
    )
    def populate_sku_options(_model_key: str):
        return [{"label": s, "value": s} for s in ds.SKU_LIST]

    # ── show/hide sections checklist (Full XAI only) ──────────────────────────
    @app.callback(
        Output("report-sections-collapse", "is_open"),
        Input("report-template", "value"),
    )
    def toggle_sections(template: str) -> bool:
        return template == "full"

    # ── generate report → download ────────────────────────────────────────────
    @app.callback(
        Output("report-download", "data"),
        Output("report-status",   "children"),
        Input("report-generate",  "n_clicks"),
        State("report-sku-select",   "value"),
        State("report-template",     "value"),
        State("report-format",       "data"),
        State("report-sections",     "value"),
        State("report-unit-margin",  "value"),
        State("report-holding-cost", "value"),
        State("model-store",         "data"),
        prevent_initial_call=True,
    )
    def generate_report(
        _n, sku_ids, template, fmt, sections,
        unit_margin, holding_cost, model_key,
    ):
        # ── validation ────────────────────────────────────────────────────────
        if not sku_ids:
            return None, dbc.Alert(
                "Please select at least one SKU.", color="warning", className="mb-0 py-2",
            )

        try:
            unit_margin  = float(unit_margin)
            holding_cost = float(holding_cost)
        except (TypeError, ValueError):
            return None, dbc.Alert(
                "Invalid business parameters — please enter positive numbers.",
                color="danger", className="mb-0 py-2",
            )

        if unit_margin <= 0 or holding_cost <= 0:
            return None, dbc.Alert(
                "Unit margin and holding cost must be greater than zero.",
                color="danger", className="mb-0 py-2",
            )

        if template == "full" and not sections:
            return None, dbc.Alert(
                "Please select at least one section for the Full XAI Report.",
                color="warning", className="mb-0 py-2",
            )

        # ── generate ──────────────────────────────────────────────────────────
        try:
            raw = build_report(
                sku_ids      = sku_ids,
                template     = template or "full",
                fmt          = "docx",
                sections     = sections,
                unit_margin  = unit_margin,
                holding_cost = holding_cost,
                model_key    = model_key or "7d",
            )
        except Exception as exc:
            return None, dbc.Alert(
                f"Report generation failed: {exc}",
                color="danger", className="mb-0 py-2",
            )

        # ── filename ──────────────────────────────────────────────────────────
        today    = datetime.date.today().strftime("%Y%m%d")
        sku_slug = sku_ids[0] if len(sku_ids) == 1 else f"{len(sku_ids)}_skus"
        filename = f"xai_report_{sku_slug}_{today}.docx"

        status = dbc.Alert(
            [
                html.Strong(f"Report ready: "), filename,
                f" — {len(sku_ids)} SKU(s) — {template} template",
            ],
            color="success", className="mb-0 py-2",
        )

        return dcc.send_bytes(raw, filename), status
