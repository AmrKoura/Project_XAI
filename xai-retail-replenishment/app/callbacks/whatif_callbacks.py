"""
Callbacks for the What-If Simulator page (Q4 + Q10).
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from dash import Input, Output, State, ctx, html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

import app.data_store as ds
from xai.counterfactual import batch_counterfactuals
from xai.cost_impact import simulate_cost_distribution, optimal_order_quantity


def register_whatif_callbacks(app) -> None:

    # ── subtitle ───────────────────────────────────────────────────────────────
    @app.callback(
        Output("whatif-subtitle", "children"),
        Input("model-store", "data"),
    )
    def whatif_subtitle(model_key: str) -> str:
        cfg = ds.MODEL_CONFIGS.get(model_key, {})
        return cfg.get("label", "") + " · LGBM · M5 Walmart"

    # ── populate SKU dropdown ──────────────────────────────────────────────────
    @app.callback(
        Output("whatif-sku", "options"),
        Output("whatif-sku", "value"),
        Input("model-store", "data"),
    )
    def init_whatif_sku(_model_key):
        opts    = [{"label": s, "value": s} for s in ds.SKU_LIST]
        default = ds.SKU_LIST[0] if ds.SKU_LIST else None
        return opts, default

    # ── initialise / reset Q4 controls when SKU changes ───────────────────────
    @app.callback(
        Output("whatif-price",    "value"),
        Output("whatif-discount", "value"),
        Output("whatif-snap",     "value"),
        Input("whatif-sku",             "value"),
        Input("whatif-reset-controls",  "n_clicks"),
        Input("model-store",            "data"),
    )
    def init_whatif_controls(sku_id, _reset, _model_key):
        if not sku_id:
            return None, 0.0, 0
        try:
            X_row    = ds.get_sku_X_row(sku_id)
            price    = round(float(X_row["aggregated_sell_price"].iloc[0]), 4)
            discount = round(float(X_row["discount_depth"].iloc[0]),        2)
            snap     = bool(int(X_row["snap_relevant"].iloc[0]))
            return price, discount, snap
        except Exception:
            return None, 0.0, False

    # ── Q4 — impact cards + sweep charts + NLG ────────────────────────────────
    @app.callback(
        Output("whatif-impact-cards",   "children"),
        Output("whatif-sweep-price",    "figure"),
        Output("whatif-sweep-discount", "figure"),
        Output("whatif-sweep-snap",     "figure"),
        Output("whatif-q4-nlg",         "children"),
        Input("whatif-run-q4",          "n_clicks"),
        Input("whatif-reset-controls",  "n_clicks"),
        Input("whatif-sku",             "value"),
        Input("theme-store",            "data"),
        Input("model-store",            "data"),
        State("whatif-price",           "value"),
        State("whatif-discount",        "value"),
        State("whatif-snap",            "value"),
    )
    def update_whatif_q4(_run_q4, _reset, sku_id, theme, _model_key, price, discount, snap):
        dark      = (theme == "dark")
        empty_fig = _empty_figure(dark)

        if not sku_id or not ds.is_loaded():
            return [], empty_fig, empty_fig, empty_fig, []

        try:
            X_row = ds.get_sku_X_row(sku_id)
        except KeyError:
            return [], empty_fig, empty_fig, empty_fig, []

        orig_price    = float(X_row["aggregated_sell_price"].iloc[0])
        orig_discount = float(X_row["discount_depth"].iloc[0])
        orig_snap     = int(X_row["snap_relevant"].iloc[0])

        triggered = ctx.triggered_id or "whatif-sku"

        if triggered in ("whatif-sku", "model-store", "whatif-reset-controls"):
            cur_price    = orig_price
            cur_discount = orig_discount
            cur_snap     = orig_snap
        else:
            cur_price    = float(price)    if price    is not None else orig_price
            cur_discount = float(discount) if discount is not None else orig_discount
            cur_snap     = int(snap)       if snap     is not None else orig_snap

        # Combined modified row
        X_mod = X_row.copy()
        X_mod["aggregated_sell_price"] = cur_price
        X_mod["discount_depth"]        = cur_discount
        X_mod["snap_relevant"]         = cur_snap

        original_pred = float(ds.model.predict(X_row)[0])
        new_pred      = float(ds.model.predict(X_mod)[0])
        delta         = new_pred - original_pred

        # Compute all three sweeps
        price_batch    = batch_counterfactuals(
            ds.model, X_row, "aggregated_sell_price",
            np.linspace(orig_price * 0.5, orig_price * 1.5, 30),
        )
        discount_batch = batch_counterfactuals(
            ds.model, X_row, "discount_depth",
            np.linspace(0.0, 0.50, 30),
        )
        snap_batch     = batch_counterfactuals(
            ds.model, X_row, "snap_relevant", [0, 1],
        )

        price_fig    = _price_figure(price_batch,    cur_price,    original_pred, sku_id, dark)
        discount_fig = _discount_figure(discount_batch, cur_discount, original_pred, sku_id, dark)
        snap_fig     = _snap_figure(snap_batch,      cur_snap,     original_pred, sku_id, dark)

        impact_cards = _impact_cards(original_pred, new_pred, delta)
        nlg_div      = _q4_nlg(
            sku_id,
            orig_price, cur_price,
            orig_discount, cur_discount,
            orig_snap, cur_snap,
            original_pred, new_pred, delta,
        )

        return impact_cards, price_fig, discount_fig, snap_fig, nlg_div

    # ── Q10 — reset cost inputs to defaults ────────────────────────────────────
    @app.callback(
        Output("whatif-unit-margin",  "value"),
        Output("whatif-holding-cost", "value"),
        Input("whatif-reset-costs",   "n_clicks"),
        prevent_initial_call=True,
    )
    def reset_cost_inputs(_n):
        return ds.UNIT_MARGIN, ds.HOLDING_COST

    # ── Q10 — cost cards + cost curve + NLG ───────────────────────────────────
    @app.callback(
        Output("whatif-cost-cards", "children"),
        Output("whatif-cost-chart", "figure"),
        Output("whatif-q10-nlg",    "children"),
        Input("whatif-run-costs",    "n_clicks"),
        Input("whatif-reset-costs",  "n_clicks"),
        Input("whatif-sku",          "value"),
        Input("theme-store",         "data"),
        Input("model-store",         "data"),
        State("whatif-unit-margin",  "value"),
        State("whatif-holding-cost", "value"),
    )
    def update_whatif_q10(_run, _reset, sku_id, theme, _model_key, unit_margin, holding_cost):
        dark      = (theme == "dark")
        empty_fig = _empty_figure(dark)

        if not sku_id or not ds.is_loaded():
            return [], empty_fig, []

        if ctx.triggered_id == "whatif-reset-costs":
            unit_margin  = ds.UNIT_MARGIN
            holding_cost = ds.HOLDING_COST
        else:
            try:
                unit_margin  = float(unit_margin)
                holding_cost = float(holding_cost)
            except (TypeError, ValueError):
                return [], empty_fig, []

            if unit_margin <= 0 or holding_cost <= 0:
                return [], empty_fig, []

        iv = ds.forecasts.get(sku_id)
        if iv is None:
            return [], empty_fig, []

        q10, q50, q90 = iv["q10"], iv["q50"], iv["q90"]

        opt = optimal_order_quantity(q10, q50, q90, unit_margin, holding_cost)

        sim = simulate_cost_distribution(
            q10, q50, q90, opt["optimal_qty"],
            unit_margin, holding_cost,
            n_simulations=5_000, seed=42,
        )
        exp_so   = float(sim["stockout_cost"].mean())
        exp_os   = float(sim["overstock_cost"].mean())
        exp_tot  = exp_so + exp_os
        dominant = "STOCKOUT" if exp_so > exp_os else "OVERSTOCK"

        # Cost curve: sweep order quantities
        order_range = np.linspace(max(q10 * 0.5, 0.1), q90 * 1.5, 35)
        rows = []
        for oq in order_range:
            s = simulate_cost_distribution(
                q10, q50, q90, oq, unit_margin, holding_cost,
                n_simulations=2_000, seed=42,
            )
            rows.append({
                "order_qty":      float(oq),
                "mean_stockout":  float(s["stockout_cost"].mean()),
                "mean_overstock": float(s["overstock_cost"].mean()),
                "mean_total":     float(s["total_cost"].mean()),
            })
        curve = pd.DataFrame(rows)

        fig       = _cost_curve_figure(curve, opt, q50, sku_id, dark)
        cost_cds  = _cost_stat_cards(opt, exp_so, exp_os, exp_tot, dominant)
        nlg_div   = _q10_nlg(sku_id, q50, opt, exp_so, exp_os, unit_margin, holding_cost, dominant)

        return cost_cds, fig, nlg_div


# ── figure helpers ────────────────────────────────────────────────────────────

def _empty_figure(dark: bool) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig


def _theme(dark: bool) -> dict:
    return {
        "bg":     "rgba(0,0,0,0)",
        "grid":   "rgba(255,255,255,0.08)" if dark else "rgba(0,0,0,0.05)",
        "text":   "#e0e0e0" if dark else "#333333",
        "accent": "#7B61FF" if dark else "#5c6cf5",
        "line":   "#e0e0e0" if dark else "#111827",
    }


def _base_layout(c: dict, title: str) -> dict:
    return dict(
        paper_bgcolor=c["bg"],
        plot_bgcolor=c["bg"],
        font=dict(color=c["text"], size=12),
        margin=dict(l=40, r=20, t=50, b=40),
        showlegend=False,
        title=dict(text=title, font=dict(size=14), x=0.5, xanchor="center"),
    )


def _price_figure(
    batch_df: pd.DataFrame, cur_price: float, baseline: float,
    sku_id: str, dark: bool,
) -> go.Figure:
    c   = _theme(dark)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=batch_df["feature_value"], y=batch_df["prediction"],
        mode="lines", line=dict(color=c["accent"], width=2.5),
    ))
    fig.add_vline(
        x=cur_price, line_dash="dash", line_color="gray",
        annotation_text=f"Current ${cur_price:.2f}",
        annotation_position="top right",
        annotation_font_color=c["text"],
    )
    fig.add_hline(y=baseline, line_dash="dot", line_color="rgba(128,128,128,0.45)")
    fig.update_layout(
        **_base_layout(c, f"Price Sensitivity — {sku_id}"),
        xaxis=dict(title="Price ($)",           gridcolor=c["grid"], zerolinecolor=c["grid"]),
        yaxis=dict(title="7-day Forecast (units)", gridcolor=c["grid"], zerolinecolor=c["grid"]),
    )
    return fig


def _discount_figure(
    batch_df: pd.DataFrame, cur_discount: float, baseline: float,
    sku_id: str, dark: bool,
) -> go.Figure:
    c   = _theme(dark)
    fig = go.Figure()
    xs  = batch_df["feature_value"] * 100
    fig.add_trace(go.Scatter(
        x=xs, y=batch_df["prediction"],
        mode="lines", line=dict(color=c["accent"], width=2.5),
    ))
    fig.add_vline(
        x=cur_discount * 100, line_dash="dash", line_color="gray",
        annotation_text=f"Current {cur_discount * 100:.0f}%",
        annotation_position="top right",
        annotation_font_color=c["text"],
    )
    fig.add_hline(y=baseline, line_dash="dot", line_color="rgba(128,128,128,0.45)")
    fig.update_layout(
        **_base_layout(c, f"Discount Sensitivity — {sku_id}"),
        xaxis=dict(title="Discount Depth (%)",      gridcolor=c["grid"], zerolinecolor=c["grid"]),
        yaxis=dict(title="7-day Forecast (units)",  gridcolor=c["grid"], zerolinecolor=c["grid"]),
    )
    return fig


def _snap_figure(
    batch_df: pd.DataFrame, cur_snap: int, baseline: float,
    sku_id: str, dark: bool,
) -> go.Figure:
    c          = _theme(dark)
    bar_colors = [
        c["accent"] if int(v) == int(cur_snap) else "rgba(108,117,125,0.35)"
        for v in [0, 1]
    ]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["SNAP Off (0)", "SNAP On (1)"],
        y=batch_df["prediction"].tolist(),
        marker_color=bar_colors,
        text=[f"{v:.2f}" for v in batch_df["prediction"]],
        textposition="outside",
        textfont=dict(color=c["text"]),
    ))
    fig.add_hline(
        y=baseline, line_dash="dot", line_color="gray",
        annotation_text=f"Baseline {baseline:.1f}",
        annotation_position="top right",
        annotation_font_color=c["text"],
    )
    fig.update_layout(
        **_base_layout(c, f"SNAP Effect — {sku_id}"),
        xaxis=dict(title="SNAP Status",             gridcolor=c["grid"], zerolinecolor=c["grid"]),
        yaxis=dict(title="7-day Forecast (units)",  gridcolor=c["grid"], zerolinecolor=c["grid"]),
    )
    return fig


def _cost_curve_figure(
    curve: pd.DataFrame,
    opt: dict,
    q50: float,
    sku_id: str,
    dark: bool,
) -> go.Figure:
    c   = _theme(dark)
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=curve["order_qty"], y=curve["mean_stockout"],
        name="Stockout cost", mode="lines",
        line=dict(color="#ef4444", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=curve["order_qty"], y=curve["mean_overstock"],
        name="Overstock cost", mode="lines",
        line=dict(color="#3b82f6", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=curve["order_qty"], y=curve["mean_total"],
        name="Total cost", mode="lines",
        line=dict(color=c["line"], width=2.5),
    ))
    fig.add_vline(
        x=opt["optimal_qty"], line_dash="dash", line_color="#22c55e",
        annotation_text=f"Optimal: {opt['optimal_qty']:.1f}",
        annotation_position="top right",
        annotation_font_color="#22c55e",
    )
    fig.add_vline(
        x=q50, line_dash="dot", line_color="gray",
        annotation_text=f"q50: {q50:.1f}",
        annotation_position="top left",
        annotation_font_color=c["text"],
    )
    fig.update_layout(
        paper_bgcolor=c["bg"],
        plot_bgcolor=c["bg"],
        font=dict(color=c["text"], size=12),
        margin=dict(l=40, r=30, t=50, b=80),
        xaxis=dict(title="Order Quantity (units)", gridcolor=c["grid"], zerolinecolor=c["grid"]),
        yaxis=dict(title="Expected Cost ($)",      gridcolor=c["grid"], zerolinecolor=c["grid"]),
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0),
        title=dict(text=f"Cost vs Order Quantity — {sku_id}", font=dict(size=14), x=0.5, xanchor="center"),
    )
    return fig


# ── card helpers ──────────────────────────────────────────────────────────────

def _stat_card(label: str, value: str, sub, color: str | None = None) -> dbc.Card:
    val_cls = f"mb-0{' text-' + color if color else ''}"
    return dbc.Card(
        dbc.CardBody([
            html.Small(label, className="text-muted d-block"),
            html.H4(value, className=val_cls),
            html.Small(sub if isinstance(sub, list) else sub, className="text-muted"),
        ], style={"padding": "12px 16px"}),
        className="shadow-sm text-center h-100",
    )


def _impact_cards(original_pred: float, new_pred: float, delta: float) -> dbc.Row:
    pct         = abs(delta) / original_pred * 100 if original_pred != 0 else 0.0
    sign        = "+" if delta >= 0 else ""
    delta_color = "success" if delta >= 0 else "danger"
    orig_ceil   = math.ceil(original_pred)
    new_ceil    = math.ceil(new_pred)
    delta_ceil  = new_ceil - orig_ceil
    sign_ceil   = "+" if delta_ceil >= 0 else ""
    return dbc.Row([
        dbc.Col(_stat_card("Current Forecast", f"{orig_ceil}",
                           "units (baseline)"), md=4),
        dbc.Col(_stat_card("New Forecast",     f"{new_ceil}",
                           "units (what-if)"), md=4),
        dbc.Col(_stat_card("Combined Impact",  f"{sign_ceil}{delta_ceil}",
                           f"{sign}{pct:.1f}% vs baseline", delta_color), md=4),
    ], className="g-3")


_TIP_STYLE = {
    "textDecoration": "underline",
    "textDecorationStyle": "dashed",
    "textDecorationColor": "rgba(108,117,125,0.7)",
    "cursor": "help",
}


def _cost_stat_cards(
    opt: dict,
    exp_so: float,
    exp_os: float,
    exp_tot: float,
    dominant: str,
) -> dbc.Row:
    risk_color = "danger" if dominant == "STOCKOUT" else "warning"

    sl_sub = [
        f"units · ",
        html.Span("SL", id="cost-sl-tip", style=_TIP_STYLE),
        f" {opt['implied_service_level']:.0f}%",
        dbc.Tooltip(
            "Service Level — the probability that demand will be fully met without a stockout. "
            "We target 80%: a balanced threshold that covers most demand scenarios without "
            "tying up too much capital in excess stock.",
            target="cost-sl-tip", placement="top",
        ),
    ]

    cr_pct = f"{opt['critical_ratio']:.1%}"
    cr_sub = [
        "Critical ratio: ",
        html.Span(cr_pct, id="cost-cr-tip", style=_TIP_STYLE),
        dbc.Tooltip(
            f"The critical ratio ({cr_pct}) is calculated as: margin ÷ (margin + holding cost). "
            "It marks the tipping point between the two risks — above it, stockout is more "
            "costly than overstock, so it pays to order more. Below it, holding excess stock "
            "costs more, so you should order less.",
            target="cost-cr-tip", placement="top",
        ),
    ]

    return dbc.Row([
        dbc.Col(_stat_card(
            "Optimal Order Qty",
            f"{math.ceil(opt['optimal_qty'])}",
            sl_sub,
        ), md=4),
        dbc.Col(_stat_card(
            "Expected Total Cost",
            f"${exp_tot:.2f}",
            f"SO ${exp_so:.2f}  |  OS ${exp_os:.2f}",
        ), md=4),
        dbc.Col(_stat_card(
            "Dominant Risk",
            dominant,
            cr_sub,
            risk_color,
        ), md=4),
    ], className="g-3")


# ── NLG helpers ───────────────────────────────────────────────────────────────

def _section_label(text: str) -> html.P:
    return html.P(text, className="fw-bold mb-1 mt-2", style={"fontSize": "13px"})


def _bullet_list(items: list) -> html.Ul:
    return html.Ul(
        [html.Li(i, style={"fontSize": "13px"}) for i in items],
        className="mb-0 ps-3",
    )


def _q4_nlg(
    sku_id: str,
    orig_price: float,    cur_price: float,
    orig_discount: float, cur_discount: float,
    orig_snap: int,       cur_snap: int,
    original_pred: float,
    new_pred: float,
    delta: float,
) -> html.Div:
    sign      = "+" if delta >= 0 else ""
    pct       = abs(delta) / original_pred * 100 if original_pred != 0 else 0.0
    direction = "increase" if delta >= 0 else "decrease"

    orig_ceil  = math.ceil(original_pred)
    new_ceil   = math.ceil(new_pred)
    delta_ceil = new_ceil - orig_ceil
    sign_ceil  = "+" if delta_ceil >= 0 else ""

    changed = []
    if abs(cur_price - orig_price) > 0.001:
        changed.append(f"Price: ${orig_price:.2f} → ${cur_price:.2f}")
    if abs(cur_discount - orig_discount) > 0.001:
        changed.append(
            f"Discount: {orig_discount * 100:.0f}% → {cur_discount * 100:.0f}%"
        )
    if cur_snap != orig_snap:
        changed.append(
            f"SNAP: {'Off' if orig_snap == 0 else 'On'} → {'Off' if cur_snap == 0 else 'On'}"
        )

    bullets = [
        f"Changes applied: {', '.join(changed) if changed else 'none (showing baseline)'}",
        f"Combined forecast impact: {sign_ceil}{delta_ceil} units ({sign}{pct:.1f}% {direction}) vs baseline",
        "Use the tabs above to explore how each feature independently drives the forecast.",
    ]

    return html.Div([
        html.Hr(),
        _section_label(f"What-If Impact — {sku_id}"),
        _bullet_list(bullets),
    ])


def _q10_nlg(
    sku_id: str,
    q50: float,
    opt: dict,
    exp_so: float,
    exp_os: float,
    unit_margin: float,
    holding_cost: float,
    dominant: str,
) -> html.Div:
    exp_tot      = exp_so + exp_os
    opt_ceil     = math.ceil(opt["optimal_qty"])
    q50_ceil     = math.ceil(q50)
    order_vs_q50 = opt_ceil - q50_ceil
    vs_txt = (
        f"{abs(order_vs_q50)} units above q50"
        if order_vs_q50 >= 0
        else f"{abs(order_vs_q50)} units below q50"
    )

    if dominant == "STOCKOUT":
        rec = (
            f"The stockout cost (${unit_margin:.2f}/unit) outweighs holding cost "
            f"(${holding_cost:.2f}/unit) — consider ordering above the point forecast."
        )
    else:
        rec = (
            f"The holding cost (${holding_cost:.2f}/unit) outweighs stockout cost "
            f"(${unit_margin:.2f}/unit) — stay close to or below the point forecast."
        )

    bullets = [
        f"Point forecast (q50): {q50_ceil} units",
        f"Optimal order quantity: {opt_ceil} units ({vs_txt})",
        f"Expected stockout cost at optimal: ${exp_so:.2f}",
        f"Expected overstock cost at optimal: ${exp_os:.2f}",
        f"Expected total cost: ${exp_tot:.2f}",
        rec,
    ]

    return html.Div([
        html.Hr(),
        _section_label(f"Cost-Impact Summary — {sku_id}"),
        _bullet_list(bullets),
    ])
