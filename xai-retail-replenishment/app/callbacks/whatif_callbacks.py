"""
Callbacks for the What-If Simulator page (Q4 + Q10).
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from dash import Input, Output, State, ctx, html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

import app.data_store as ds
from xai.cost_impact import simulate_cost_distribution, optimal_order_quantity


def register_whatif_callbacks(app) -> None:

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
        Output("whatif-price",        "value"),
        Output("whatif-snap",         "value"),
        Output("whatif-custom-order", "value"),
        Input("whatif-sku",            "value"),
        Input("whatif-reset-controls", "n_clicks"),
        Input("model-store",           "data"),
    )
    def init_whatif_controls(sku_id, _reset, _model_key):
        if not sku_id:
            return None, False, None
        try:
            X_row = ds.get_sku_X_row(sku_id)
            price = round(float(X_row["aggregated_sell_price"].iloc[0]), 2)
            snap  = bool(int(X_row["snap_ca"].iloc[0]))
            return price, snap, None
        except Exception:
            return None, False, None

    # ── Lock / clear Scenario A ───────────────────────────────────────────────
    @app.callback(
        Output("whatif-scenario-a-store", "data"),
        Input("whatif-lock-a",            "n_clicks"),
        Input("whatif-clear-a",           "n_clicks"),
        Input("whatif-sku",               "value"),
        Input("whatif-reset-controls",    "n_clicks"),
        State("whatif-current-store",     "data"),
        prevent_initial_call=True,
    )
    def lock_scenario_a(_lock, _clear, _sku, _reset, current):
        if ctx.triggered_id in ("whatif-clear-a", "whatif-sku", "whatif-reset-controls"):
            return None
        if ctx.triggered_id == "whatif-lock-a" and current:
            return current
        return None

    # ── Scenario comparison ────────────────────────────────────────────────────
    @app.callback(
        Output("whatif-scenario-comparison", "children"),
        Input("whatif-scenario-a-store",     "data"),
        Input("whatif-current-store",        "data"),
        Input("theme-store",                 "data"),
    )
    def render_comparison(scenario_a, current, theme):
        if not scenario_a or not current:
            return []
        dark = theme == "dark"
        return _scenario_comparison(scenario_a, current, dark)

    # ── Combined scenario + cost impact ───────────────────────────────────────
    @app.callback(
        Output("whatif-causal-warning",   "children"),
        Output("whatif-impact-cards",     "children"),
        Output("whatif-compare-chart",    "figure"),
        Output("whatif-cost-cards",       "children"),
        Output("whatif-order-comparison", "children"),
        Output("whatif-cost-chart",       "figure"),
        Output("whatif-risk-chart",       "figure"),
        Output("whatif-q4-nlg",           "children"),
        Output("whatif-q10-nlg",          "children"),
        Output("whatif-current-store",    "data"),
        Input("whatif-run-q4",         "n_clicks"),
        Input("whatif-reset-controls", "n_clicks"),
        Input("whatif-sku",            "value"),
        Input("theme-store",           "data"),
        Input("model-store",           "data"),
        State("whatif-price",          "value"),
        State("whatif-snap",           "value"),
        State("whatif-unit-margin",    "value"),
        State("whatif-holding-cost",   "value"),
        State("whatif-custom-order",   "value"),
    )
    def update_whatif(
        _run, _reset, sku_id, theme, _model_key,
        price, snap, margin_pct, holding_pct, custom_order,
    ):
        dark      = (theme == "dark")
        empty_fig = _empty_figure(dark)
        empty     = ([], [], empty_fig, [], [], empty_fig, empty_fig, [], [], None)

        if not sku_id or not ds.is_loaded():
            return empty

        try:
            X_row = ds.get_sku_X_row(sku_id)
        except KeyError:
            return empty


        orig_price = float(X_row["aggregated_sell_price"].iloc[0])
        orig_snap  = int(X_row["snap_ca"].iloc[0])

        triggered = ctx.triggered_id or "whatif-sku"
        if triggered in ("whatif-sku", "model-store", "whatif-reset-controls"):
            cur_price    = orig_price
            cur_snap     = orig_snap
            margin_pct   = 25.0
            holding_pct  = 8.0
            custom_order = None
        else:
            cur_price   = float(price)       if price       is not None else orig_price
            cur_snap    = int(snap)          if snap        is not None else orig_snap
            margin_pct  = float(margin_pct  or 25)
            holding_pct = float(holding_pct or 8)

        # ── Forecast simulation ───────────────────────────────────────────────
        original_pred = float(ds.forecasts.get(sku_id, {}).get("q50", 0.0))

        X_mod = X_row.copy()
        X_mod["aggregated_sell_price"] = cur_price
        X_mod["snap_ca"]               = cur_snap
        new_pred_raw = max(0.0, float(ds.model.predict(X_mod)[0]))

        # Ceil for consistency with all other forecast displays
        new_pred = float(math.ceil(new_pred_raw))
        delta    = new_pred - original_pred

        # ── Build scenario q10/q90 from ceiled q50 ───────────────────────────
        std_v  = float(ds.sku_std.get(sku_id, ds.sku_std.mean())) if ds.sku_std is not None else 1.0
        iv_margin = 1.282 * std_v
        q50 = new_pred
        q10 = max(0.0, q50 - iv_margin)
        q90 = q50 + iv_margin

        # ── Cost impact ───────────────────────────────────────────────────────
        unit_margin  = orig_price * margin_pct  / 100.0
        holding_cost = orig_price * holding_pct / 100.0

        opt = optimal_order_quantity(q10, q50, q90, unit_margin, holding_cost)
        sim = simulate_cost_distribution(
            q10, q50, q90, opt["optimal_qty"],
            unit_margin, holding_cost,
            n_simulations=5_000, seed=42,
        )
        exp_so   = float(sim["stockout_cost"].mean())
        exp_os   = float(sim["overstock_cost"].mean())
        dominant = "STOCKOUT" if exp_so > exp_os else "OVERSTOCK"
        p90_cost = float(np.percentile(sim["total_cost"], 90))
        p95_cost = float(np.percentile(sim["total_cost"], 95))

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

        # ── Custom order qty comparison ───────────────────────────────────────
        custom_qty = float(custom_order) if custom_order is not None else None
        order_comparison = _order_comparison(
            custom_qty, opt, q10, q50, q90,
            unit_margin, holding_cost,
        )

        # ── Counterintuitive detection ────────────────────────────────────────
        caution = _causal_warning(orig_price, cur_price, orig_snap, cur_snap,
                                  original_pred, new_pred)

        # Describe changes for chart hovers
        changes = []
        if abs(cur_price - orig_price) > 0.001:
            changes.append(f"price {'↑' if cur_price > orig_price else '↓'} "
                           f"${orig_price:.2f}→${cur_price:.2f}")
        if cur_snap != orig_snap:
            changes.append(f"SNAP {'on→off' if orig_snap and not cur_snap else 'off→on'}")
        changes_txt = ", ".join(changes) if changes else "no changes"

        current = {
            "sku_id":    sku_id,
            "price":     cur_price,
            "snap":      cur_snap,
            "margin_pct":   margin_pct,
            "holding_pct":  holding_pct,
            "forecast":     int(new_pred),
            "orig_forecast": int(original_pred),
            "optimal_order": math.ceil(opt["optimal_qty"]),
            "exp_cost":   round(exp_so + exp_os, 2),
            "exp_so":     round(exp_so, 2),
            "exp_os":     round(exp_os, 2),
            "p90":        round(p90_cost, 2),
            "p95":        round(p95_cost, 2),
            "dominant":   dominant,
        }

        return (
            caution,
            _impact_cards(original_pred, new_pred, delta),
            _compare_figure(original_pred, new_pred, sku_id, changes_txt, dark),
            _cost_stat_cards(opt, exp_so, exp_os, exp_so + exp_os, dominant),
            order_comparison,
            _cost_curve_figure(curve, opt, q50, custom_qty, sku_id, dark),
            _risk_distribution_figure(sim, sku_id, dark),
            _q4_nlg(sku_id, orig_price, cur_price, orig_snap, cur_snap,
                    original_pred, new_pred, delta),
            _q10_nlg(sku_id, q50, orig_price, margin_pct, holding_pct,
                     opt, exp_so, exp_os, unit_margin, holding_cost, dominant,
                     p90_cost, p95_cost),
            current,
        )


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


def _scenario_comparison(a: dict, b: dict, dark: bool) -> list:
    """Render a side-by-side comparison table + bar chart for two scenarios."""
    c = _theme(dark)

    def _snap_label(v): return "On" if v else "Off"
    def _delta(a_val, b_val, fmt=".1f"):
        d = b_val - a_val
        return f"{'+' if d >= 0 else ''}{d:{fmt}}"

    # ── comparison table ─────────────────────────────────────────────────────
    rows = [
        ("Price ($)",          f"${a['price']:.2f}",          f"${b['price']:.2f}"),
        ("SNAP CA",            _snap_label(a["snap"]),         _snap_label(b["snap"])),
        ("Margin %",           f"{a['margin_pct']:.0f}%",      f"{b['margin_pct']:.0f}%"),
        ("Holding %",          f"{a['holding_pct']:.0f}%",     f"{b['holding_pct']:.0f}%"),
        ("Scenario Forecast",  f"{a['forecast']} units",       f"{b['forecast']} units"),
        ("Optimal Order",      f"{a['optimal_order']} units",  f"{b['optimal_order']} units"),
        ("Expected Cost",      f"${a['exp_cost']:.2f}",        f"${b['exp_cost']:.2f}"),
        ("Risk P90",           f"${a['p90']:.2f}",             f"${b['p90']:.2f}"),
        ("Risk P95",           f"${a['p95']:.2f}",             f"${b['p95']:.2f}"),
        ("Dominant Risk",      a["dominant"],                   b["dominant"]),
    ]

    th_style = {"backgroundColor": "#343a40" if not dark else "#0A0C1A",
                "color": "#fff", "fontWeight": "bold",
                "fontSize": "13px", "padding": "8px 12px"}
    td_style = {"fontSize": "13px", "padding": "7px 12px"}
    td_b_style = {**td_style, "color": "#6d28d9" if not dark else "#5F01FB",
                  "fontWeight": "600"}

    table_rows = []
    for label, val_a, val_b in rows:
        table_rows.append(html.Tr([
            html.Td(label, style={**td_style, "color": "#6c757d", "fontWeight": "500"}),
            html.Td(val_a, style=td_style),
            html.Td(val_b, style=td_b_style),
        ]))

    table = html.Table([
        html.Thead(html.Tr([
            html.Th("", style=th_style),
            html.Th("Scenario A (locked)", style=th_style),
            html.Th("Scenario B (current)", style=th_style),
        ])),
        html.Tbody(table_rows),
    ], className="table table-sm table-bordered mb-0")

    # ── bar chart ─────────────────────────────────────────────────────────────
    fig = go.Figure()
    color_a  = "#6d28d9" if not dark else "#5F01FB"
    color_b  = "#f97316"

    for label, key, suffix in [
        ("Forecast (units)", "forecast", " units"),
        ("Optimal Order (units)", "optimal_order", " units"),
        ("Expected Cost ($)", "exp_cost", ""),
        ("P95 Risk ($)", "p95", ""),
    ]:
        fig.add_trace(go.Bar(
            name=label,
            x=["Scenario A", "Scenario B"],
            y=[a[key], b[key]],
            marker_color=[color_a, color_b],
            text=[f"{a[key]}{suffix}", f"{b[key]}{suffix}"],
            textposition="outside",
            textfont=dict(size=11, color=c["text"]),
            hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y}}<extra></extra>",
            visible=False,
        ))

    fig.data[0].visible = True  # default: show Forecast

    fig.update_layout(
        paper_bgcolor=c["bg"], plot_bgcolor=c["bg"],
        font=dict(color=c["text"], size=12),
        margin=dict(l=40, r=20, t=50, b=50),
        showlegend=False,
        yaxis=dict(gridcolor=c["grid"]),
        xaxis=dict(gridcolor=c["grid"]),
        updatemenus=[dict(
            type="buttons", direction="right",
            x=0.0, y=1.18, xanchor="left",
            buttons=[
                dict(label=lbl,
                     method="update",
                     args=[{"visible": [i == j for j in range(4)]}])
                for i, lbl in enumerate(["Forecast", "Optimal Order",
                                         "Expected Cost", "P95 Risk"])
            ],
            bgcolor="#f8f9fa" if not dark else "#1a1a2e",
            font=dict(color=c["text"], size=11),
        )],
    )

    return [
        html.H6("Scenario Comparison", className="fw-bold mb-3"),
        dbc.Row([
            dbc.Col(table, md=6),
            dbc.Col(
                dcc.Graph(figure=fig, config={"displayModeBar": False},
                          style={"height": "280px"}),
                md=6,
            ),
        ]),
    ]


def _causal_warning(
    orig_price: float, cur_price: float,
    orig_snap: int,   cur_snap: int,
    original_pred: float, new_pred: float,
) -> list:
    """Return a warning alert if the forecast moves counterintuitively."""
    price_changed = abs(cur_price - orig_price) > 0.001
    if not price_changed:
        return []

    price_up    = cur_price > orig_price
    forecast_up = new_pred  > original_pred

    if price_up != forecast_up:   # intuitive — price up, demand down (or vice versa)
        return []

    direction = "increased" if price_up else "decreased"
    return [dbc.Alert([
        html.Strong("⚠ Counterintuitive result. "),
        html.Span(
            f"Price {direction} but the forecast also {direction}. "
            "This reflects a historical correlation the model learned from training data — "
            "not a causal effect. In the M5 dataset, price changes often coincided with "
            "high-demand periods (e.g. holidays), so the model associates higher prices "
            "with higher sales. Treat price-driven what-if results with caution.",
        ),
    ], color="warning", className="mb-3", style={"fontSize": "13px"})]


def _compare_figure(
    original_pred: float, new_pred: float,
    sku_id: str, changes_txt: str, dark: bool,
) -> go.Figure:
    c         = _theme(dark)
    orig_ceil = math.ceil(original_pred)
    new_ceil  = math.ceil(new_pred)
    delta     = new_ceil - orig_ceil
    new_color = "#22c55e" if delta >= 0 else "#ef4444"

    hover = [
        f"<b>Current Forecast</b><br>{orig_ceil} units<br>Source: future predictions CSV",
        f"<b>Scenario Forecast</b><br>{new_ceil} units<br>"
        f"Changes: {changes_txt}<br>"
        f"Δ {'+' if delta >= 0 else ''}{delta} units vs current",
    ]

    fig = go.Figure(go.Bar(
        x=["Current Forecast", "Scenario Forecast"],
        y=[orig_ceil, new_ceil],
        marker_color=[c["accent"], new_color],
        text=[f"{orig_ceil} units", f"{new_ceil} units"],
        textposition="outside",
        textfont=dict(color=c["text"], size=13),
        customdata=hover,
        hovertemplate="%{customdata}<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout(c, f"Forecast Comparison — {sku_id}"),
        xaxis=dict(gridcolor=c["grid"], zerolinecolor=c["grid"]),
        yaxis=dict(title="Units", gridcolor=c["grid"], zerolinecolor=c["grid"]),
    )
    return fig


def _order_comparison(
    custom_qty: float | None,
    opt: dict,
    q10: float, q50: float, q90: float,
    unit_margin: float, holding_cost: float,
) -> list:
    """Show comparison between custom order qty and optimal."""
    if custom_qty is None:
        return []

    opt_qty  = opt["optimal_qty"]
    diff     = custom_qty - opt_qty

    sim = simulate_cost_distribution(
        q10, q50, q90, custom_qty,
        unit_margin, holding_cost,
        n_simulations=3_000, seed=42,
    )
    sim_opt = simulate_cost_distribution(
        q10, q50, q90, opt_qty,
        unit_margin, holding_cost,
        n_simulations=3_000, seed=42,
    )

    custom_cost = float(sim["total_cost"].mean())
    opt_cost    = float(sim_opt["total_cost"].mean())
    extra_cost  = custom_cost - opt_cost

    sign   = "+" if diff >= 0 else ""
    colour = "warning" if abs(extra_cost) > 0.01 else "success"
    icon   = "⚠" if abs(extra_cost) > 0.01 else "✓"

    return [dbc.Alert([
        html.Strong(f"{icon} Your order: {int(custom_qty)} units  "),
        html.Span(f"(optimal is {math.ceil(opt_qty)} units, {sign}{diff:+.0f})"),
        html.Br(),
        html.Span(
            f"Expected cost at your quantity: ${custom_cost:.2f}  "
            f"vs  optimal: ${opt_cost:.2f}  →  "
            f"{'extra cost' if extra_cost > 0 else 'saving'}: "
            f"${abs(extra_cost):.2f}",
            style={"fontSize": "13px"},
        ),
    ], color=colour, className="py-2", style={"fontSize": "13px"})]


def _cost_curve_figure(
    curve: pd.DataFrame,
    opt: dict,
    q50: float,
    custom_qty: float | None,
    sku_id: str,
    dark: bool,
) -> go.Figure:
    c   = _theme(dark)
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=curve["order_qty"], y=curve["mean_stockout"],
        name="Stockout cost", mode="lines",
        line=dict(color="#ef4444", width=2),
        hovertemplate=(
            "<b>Order %{x:.0f} units</b><br>"
            "Stockout cost: $%{y:.2f}<br>"
            "<i>Cost of running out of stock (lost margin)</i><extra></extra>"
        ),
    ))
    fig.add_trace(go.Scatter(
        x=curve["order_qty"], y=curve["mean_overstock"],
        name="Overstock cost", mode="lines",
        line=dict(color="#3b82f6", width=2),
        hovertemplate=(
            "<b>Order %{x:.0f} units</b><br>"
            "Overstock cost: $%{y:.2f}<br>"
            "<i>Cost of holding unsold units</i><extra></extra>"
        ),
    ))
    fig.add_trace(go.Scatter(
        x=curve["order_qty"], y=curve["mean_total"],
        name="Total cost", mode="lines",
        line=dict(color=c["line"], width=2.5),
        hovertemplate=(
            "<b>Order %{x:.0f} units</b><br>"
            "Total cost: $%{y:.2f}<br>"
            "<i>Stockout + overstock combined</i><extra></extra>"
        ),
    ))
    fig.add_vline(
        x=opt["optimal_qty"], line_dash="dash", line_color="#22c55e",
        annotation_text=f"Optimal: {math.ceil(opt['optimal_qty'])}u",
        annotation_position="top right",
        annotation_font_color="#22c55e",
    )
    fig.add_vline(
        x=q50, line_dash="dot", line_color="gray",
        annotation_text=f"Forecast: {int(q50)}u",
        annotation_position="top left",
        annotation_font_color=c["text"],
    )
    if custom_qty is not None:
        fig.add_vline(
            x=custom_qty, line_dash="dashdot", line_color="#f97316",
            annotation_text=f"Your order: {int(custom_qty)}u",
            annotation_position="bottom right",
            annotation_font_color="#f97316",
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


def _risk_distribution_figure(
    sim: pd.DataFrame, sku_id: str, dark: bool,
) -> go.Figure:
    c      = _theme(dark)
    costs  = sim["total_cost"].values
    n      = len(costs)
    mean_c = float(costs.mean())
    p90    = float(np.percentile(costs, 90))
    p95    = float(np.percentile(costs, 95))

    fig = go.Figure(go.Histogram(
        x=costs,
        nbinsx=40,
        histnorm="percent",
        marker_color=c["accent"],
        opacity=0.75,
        hovertemplate=(
            "Cost range: $%{x:.2f}<br>"
            "%{y:.1f}% of simulated weeks fell here<extra></extra>"
        ),
        name="",
    ))

    # Reference lines as Scatter traces so they support hovertemplate
    refs = [
        (
            mean_c, "#22c55e", "Mean",
            f"<b>Average cost: ${mean_c:.2f}</b><br>"
            f"Across {n:,} simulated demand scenarios,<br>"
            f"the expected total cost is ${mean_c:.2f} per period.<br>"
            f"This is the number shown in the cost cards above.",
        ),
        (
            p90, "#f97316", "P90",
            f"<b>90th percentile: ${p90:.2f}</b><br>"
            f"In 1 out of every 10 simulated weeks,<br>"
            f"the total cost exceeds ${p90:.2f}.<br>"
            f"This is your moderate downside risk.",
        ),
        (
            p95, "#ef4444", "P95",
            f"<b>95th percentile: ${p95:.2f}</b><br>"
            f"Only 5% of simulated weeks cost more than ${p95:.2f}.<br>"
            f"This is your worst-case realistic scenario.<br>"
            f"If you can absorb this, the optimal order is safe.",
        ),
    ]

    for val, color, name, tooltip in refs:
        fig.add_trace(go.Scatter(
            x=[val, val], y=[0, 100],
            mode="lines",
            line=dict(color=color, dash="dash", width=2),
            name=name,
            hovertemplate=tooltip + "<extra></extra>",
        ))

    fig.update_layout(
        paper_bgcolor=c["bg"], plot_bgcolor=c["bg"],
        font=dict(color=c["text"], size=12),
        margin=dict(l=40, r=20, t=50, b=50),
        title=dict(
            text=f"Cost Risk Distribution — {n:,} Simulated Demand Scenarios (optimal order qty)",
            font=dict(size=12), x=0.5, xanchor="center",
        ),
        xaxis=dict(
            title="Total cost for the period ($)",
            gridcolor=c["grid"], zerolinecolor=c["grid"],
        ),
        yaxis=dict(
            title="% of simulated weeks",
            gridcolor=c["grid"],
            range=[0, None],
        ),
        legend=dict(
            orientation="h", y=-0.25, x=0,
            font=dict(size=11),
        ),
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
    orig_price: float, cur_price: float,
    orig_snap: int,    cur_snap: int,
    original_pred: float,
    new_pred: float,
    delta: float,
) -> html.Div:
    pct        = abs(delta) / original_pred * 100 if original_pred != 0 else 0.0
    direction  = "increase" if delta >= 0 else "decrease"
    orig_ceil  = math.ceil(original_pred)
    new_ceil   = math.ceil(new_pred)
    delta_ceil = new_ceil - orig_ceil
    sign_ceil  = "+" if delta_ceil >= 0 else ""

    changed = []
    if abs(cur_price - orig_price) > 0.001:
        changed.append(f"Price: ${orig_price:.2f} → ${cur_price:.2f}")
    if cur_snap != orig_snap:
        changed.append(f"SNAP: {'Off → On' if cur_snap else 'On → Off'}")

    bullets = [
        f"Scenario: {', '.join(changed) if changed else 'no changes vs forecast period defaults'}",
        f"New forecast: {new_ceil} units (was {orig_ceil}) — {sign_ceil}{delta_ceil} units "
        f"({pct:.1f}% {direction})",
    ]
    if not changed:
        bullets.append("Adjust price or SNAP above and hit Run to simulate a scenario.")

    return html.Div([
        html.Hr(),
        _section_label(f"Simulation Result — {sku_id}"),
        _bullet_list(bullets),
    ])


def _q10_nlg(
    sku_id: str,
    q50: float,
    sell_price: float,
    margin_pct: float,
    holding_pct: float,
    opt: dict,
    exp_so: float,
    exp_os: float,
    unit_margin: float,
    holding_cost: float,
    dominant: str,
    p90_cost: float,
    p95_cost: float,
) -> html.Div:
    exp_tot      = exp_so + exp_os
    opt_ceil     = math.ceil(opt["optimal_qty"])
    q50_ceil     = math.ceil(q50)
    order_vs_q50 = opt_ceil - q50_ceil
    vs_txt       = (f"{abs(order_vs_q50)} units above scenario forecast"
                    if order_vs_q50 >= 0
                    else f"{abs(order_vs_q50)} units below scenario forecast")

    if dominant == "STOCKOUT":
        rec = (f"Stockout risk dominates — margin lost per missed unit (${unit_margin:.2f}) "
               f"exceeds holding cost (${holding_cost:.2f}). Order above the forecast.")
    else:
        rec = (f"Overstock risk dominates — holding cost per unit (${holding_cost:.2f}) "
               f"exceeds stockout margin (${unit_margin:.2f}). Stay at or below the forecast.")

    p90_mult = p90_cost / exp_tot if exp_tot > 0 else 1.0
    p95_mult = p95_cost / exp_tot if exp_tot > 0 else 1.0

    bullets = [
        f"Scenario forecast: {q50_ceil} units at ${sell_price:.2f}/unit",
        f"Margin: {margin_pct:.0f}% = ${unit_margin:.2f}/unit  |  Holding: {holding_pct:.0f}% = ${holding_cost:.2f}/unit",
        f"Optimal order: {opt_ceil} units ({vs_txt})",
        f"Expected cost at optimal: ${exp_tot:.2f}  (stockout ${exp_so:.2f}  +  overstock ${exp_os:.2f})",
        rec,
        f"Risk tail: 1 in 10 weeks you could pay ${p90_cost:.2f} ({p90_mult:.1f}× the average) — "
        f"and in the worst 5% of weeks up to ${p95_cost:.2f} ({p95_mult:.1f}× the average).",
    ]

    return html.Div([
        html.Hr(),
        _section_label(f"Cost-Impact Summary — {sku_id}"),
        _bullet_list(bullets),
    ])
