"""
Callbacks for the Overview (Dashboard) page.
"""

from __future__ import annotations

from dash import Input, Output, html
import dash_bootstrap_components as dbc

import app.data_store as ds
from app.pages.dashboard import build_sku_overview_table, build_replenishment_cards


def register_dashboard_callbacks(app) -> None:

    # ── subtitle (updates on model switch) ────────────────────────────────────
    @app.callback(
        Output("overview-subtitle", "children"),
        Input("model-store", "data"),
    )
    def update_subtitle(model_key: str) -> str:
        cfg = ds.MODEL_CONFIGS.get(model_key, {})
        return cfg.get("label", "") + " · LGBM · M5 Walmart"

    # ── summary stat cards ────────────────────────────────────────────────────
    @app.callback(
        Output("dash-summary-stats", "children"),
        Input("url",              "pathname"),
        Input("model-store",      "data"),
        Input("theme-store",      "data"),
        Input("urgency-filter",   "value"),
        Input("category-filter",  "value"),
    )
    def update_summary_stats(pathname: str, _model_key: str, theme: str,
                             urgency: str, category: str):
        if pathname not in ("/", "/overview"):
            return []
        df = ds.cards_df.copy()
        if urgency  != "ALL": df = df[df["urgency"] == urgency]
        if category != "ALL": df = df[df["sku_id"].str.startswith(category)]
        total    = len(df)
        critical = int((df["urgency"] == "CRITICAL").sum())
        high     = int((df["urgency"] == "HIGH").sum())
        reorder  = int(df["trigger_reorder"].sum())
        total_colour = "light" if theme == "dark" else "dark"

        def _stat(value, label, colour):
            return dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.H3(str(value), className=f"text-{colour} mb-0"),
                        html.Small(label, className="text-muted"),
                    ], style={"padding": "14px 18px"}),
                ], className="shadow-sm text-center"),
                xs=6, md=3,
            )

        return dbc.Row([
            _stat(total,    "Total SKUs",       total_colour),
            _stat(critical, "Critical urgency", "danger"),
            _stat(high,     "High urgency",     "warning"),
            _stat(reorder,  "Order now",        "success"),
        ], className="g-3")

    # ── column filter options + reset ─────────────────────────────────────────
    @app.callback(
        Output("tbl-sku-filter",     "options"),
        Output("tbl-sku-filter",     "value"),
        Output("tbl-urgency-filter", "value"),
        Output("tbl-trigger-filter", "value"),
        Input("urgency-filter",      "value"),
        Input("category-filter",     "value"),
        Input("model-store",         "data"),
        Input("tbl-clear-filters",   "n_clicks"),
    )
    def reset_col_filters(urgency: str, category: str, _model_key: str, _clear):
        df = ds.cards_df.copy()
        if urgency  != "ALL": df = df[df["urgency"] == urgency]
        if category != "ALL": df = df[df["sku_id"].str.startswith(category)]
        opts = [{"label": s, "value": s} for s in sorted(df["sku_id"])]
        return opts, [], [], []

    # ── SKU overview table ────────────────────────────────────────────────────
    @app.callback(
        Output("sku-overview-table", "children"),
        Input("urgency-filter",    "value"),
        Input("category-filter",   "value"),
        Input("model-store",       "data"),
        Input("theme-store",       "data"),
        Input("tbl-sku-filter",    "value"),
        Input("tbl-urgency-filter","value"),
        Input("tbl-trigger-filter","value"),
    )
    def update_overview_table(urgency: str, category: str, _model_key: str, theme: str,
                              tbl_skus: list, tbl_urgency: list, tbl_trigger: list):
        df = ds.cards_df.copy()
        df["trigger_reorder"] = df["trigger_reorder"].astype(str)

        # page-level filters
        if urgency  != "ALL": df = df[df["urgency"] == urgency]
        if category != "ALL": df = df[df["sku_id"].str.startswith(category)]

        # column-level filters
        if tbl_skus:    df = df[df["sku_id"].isin(tbl_skus)]
        if tbl_urgency: df = df[df["urgency"].isin(tbl_urgency)]
        if tbl_trigger: df = df[df["trigger_reorder"].isin(tbl_trigger)]

        urgency_order = {"CRITICAL": 0, "HIGH": 1, "LOW": 2}
        df["_sort"] = df["urgency"].map(urgency_order)
        df = df.sort_values(["_sort", "reorder_qty"], ascending=[True, False]).drop(columns=["_sort"])
        return build_sku_overview_table(df.to_dict("records"), dark=(theme == "dark"))

    # ── replenishment cards grid ──────────────────────────────────────────────
    @app.callback(
        Output("replenishment-cards-grid", "children"),
        Input("urgency-filter",  "value"),
        Input("category-filter", "value"),
        Input("model-store",     "data"),
    )
    def update_replenishment_cards(urgency: str, category: str, _model_key: str):
        df = ds.cards_df.copy()

        if urgency == "ALL":
            df = df[df["urgency"].isin(["CRITICAL", "HIGH"])]
        else:
            df = df[df["urgency"] == urgency]

        if category != "ALL":
            df = df[df["sku_id"].str.startswith(category)]

        urgency_order = {"CRITICAL": 0, "HIGH": 1, "LOW": 2}
        df["_sort"] = df["urgency"].map(urgency_order)
        df = df.sort_values(["_sort", "reorder_qty"], ascending=[True, False])

        return build_replenishment_cards(df.to_dict("records"))
