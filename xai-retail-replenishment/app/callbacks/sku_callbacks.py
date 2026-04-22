"""
Callbacks for the SKU Explorer page.

Handles SKU selection, forecast chart, SHAP waterfall, and NLG brief.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, html
import dash_bootstrap_components as dbc

import app.data_store as ds
from xai.local_shap import get_top_contributors
from xai.uncertainty import compute_prediction_interval, confidence_label
from xai.temporal_shap import (
    compute_temporal_shap, classify_demand_pattern, get_top_temporal_drivers,
)


# ── helpers ───────────────────────────────────────────────────────────────────

_URGENCY_COLOUR = {"CRITICAL": "danger", "HIGH": "warning", "LOW": "success"}


def _summary_cards(iv: dict, card: dict, std_v: float, dark: bool = False) -> html.Div:
    q50   = iv.get("q50", 0)
    q10   = iv.get("q10", 0)
    q90   = iv.get("q90", 0)
    pi    = compute_prediction_interval(q50, std_v)
    conf  = confidence_label(pi["width"], q50)
    conf_colour = {"high": "success", "moderate": "warning", "low": "danger"}.get(conf, "secondary")

    urgency = card.get("urgency", "LOW")
    days    = card.get("days_of_stock", 0)

    total_colour = "light" if dark else "dark"

    def _stat(value, label, colour, label_id=None):
        label_el = (
            html.Small(label, id=label_id, className="text-muted",
                       style={"cursor": "help", "borderBottom": "1px dashed #6c757d"})
            if label_id
            else html.Small(label, className="text-muted")
        )
        return dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.H3(str(value), className=f"text-{colour} mb-0"),
                    label_el,
                ], style={"padding": "14px 18px"}),
            ], className="shadow-sm text-center"),
            xs=6, md=True,
        )

    return html.Div([
        dbc.Row([
            _stat(f"{q50:.0f} units",         "Forecast (q50)",    total_colour),
            _stat(f"[{q10:.0f} – {q90:.0f}]", "80% interval",      conf_colour,  label_id="interval-card-label"),
            _stat(conf.upper(),                "Confidence",        conf_colour,  label_id="confidence-card-label"),
            _stat(f"{days:.1f} days",          "Days of stock",     total_colour),
            _stat(urgency,                     "Urgency",           _URGENCY_COLOUR.get(urgency, "secondary")),
        ], className="g-3"),

        dbc.Tooltip(
            [
                html.P(
                    "There is an 80% chance that actual demand will fall inside this range.",
                    className="mb-2", style={"fontSize": "12px"},
                ),
                html.P("How to read it:", className="mb-1 fw-bold", style={"fontSize": "12px"}),
                html.Ul([
                    html.Li("The two numbers are the lower and upper bound of expected demand", style={"fontSize": "12px"}),
                    html.Li("A narrow range = the model is confident (low uncertainty)", style={"fontSize": "12px"}),
                    html.Li("A wide range = higher variability, plan with more caution", style={"fontSize": "12px"}),
                    html.Li("The interval is derived from this SKU's historical forecast errors", style={"fontSize": "12px"}),
                ], className="mb-0 ps-3"),
            ],
            target="interval-card-label",
            placement="bottom",
            style={"maxWidth": "360px", "textAlign": "left"},
        ),

        dbc.Tooltip(
            [
                html.P(
                    "Confidence measures how wide the forecast uncertainty range is "
                    "relative to the predicted demand — it is NOT a percentage score.",
                    className="mb-2", style={"fontSize": "12px"},
                ),
                html.P("Where it comes from:", className="mb-1 fw-bold", style={"fontSize": "12px"}),
                html.P(
                    "CV = interval width ÷ forecast (q50). "
                    "A SKU forecasted at 10 units with a [2–40] interval has CV = 38/10 = 3.8 → LOW.",
                    className="mb-2", style={"fontSize": "12px"},
                ),
                html.P("Thresholds:", className="mb-1 fw-bold", style={"fontSize": "12px"}),
                html.Ul([
                    html.Li("HIGH — interval width < 40% of the forecast", style={"fontSize": "12px"}),
                    html.Li("MODERATE — width between 40% and 100% of the forecast", style={"fontSize": "12px"}),
                    html.Li("LOW — width exceeds the forecast value itself (> 100%)", style={"fontSize": "12px"}),
                ], className="mb-0 ps-3"),
            ],
            target="confidence-card-label",
            placement="bottom",
            style={"maxWidth": "380px", "textAlign": "left"},
        ),
    ])


_NEON_BG      = "#0F1225"
_NEON_FONT    = "#FFFFFF"
_NEON_GRID    = "rgba(95,1,251,0.2)"
_NEON_PURPLE  = "#5F01FB"
_NEON_RED     = "#FF2D55"
_NEON_MUTED   = "rgba(255,255,255,0.50)"

_NEON_LAYOUT = dict(
    paper_bgcolor    = _NEON_BG,
    plot_bgcolor     = _NEON_BG,
    font             = {"color": _NEON_FONT},
    xaxis_gridcolor  = _NEON_GRID,
    xaxis_linecolor  = _NEON_GRID,
    xaxis_zerolinecolor = _NEON_GRID,
    yaxis_gridcolor  = _NEON_GRID,
    yaxis_linecolor  = _NEON_GRID,
    yaxis_zerolinecolor = _NEON_GRID,
)


def _forecast_figure(sku_id: str, std_v: float, dark: bool = False) -> go.Figure:
    try:
        sku_df, sku_preds = ds.get_sku_test_rows(sku_id)
    except KeyError:
        return go.Figure()

    if sku_df.empty:
        return go.Figure()

    dates = (
        pd.to_datetime(sku_df["date"]).tolist()
        if "date" in sku_df.columns
        else list(range(len(sku_df)))
    )

    actuals = sku_df[ds.TARGET].values
    preds   = sku_preds

    margin = 1.28 * std_v
    upper  = np.maximum(0, preds + margin)
    lower  = np.maximum(0, preds - margin)

    fig = go.Figure()

    pred_color = _NEON_PURPLE if dark else "#636efa"
    act_color  = _NEON_RED    if dark else "#ef553b"
    band_color = "rgba(95,1,251,0.12)" if dark else "rgba(99,110,250,0.15)"

    fig.add_trace(go.Scatter(
        x=list(dates) + list(reversed(dates)),
        y=list(upper) + list(reversed(lower)),
        fill="toself", fillcolor=band_color,
        line={"width": 0}, name="80% interval",
        showlegend=True, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=preds, mode="lines+markers", name="Predicted",
        line={"color": pred_color, "width": 2},
        marker={"size": 5, "color": pred_color},
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=actuals, mode="lines+markers", name="Actual",
        line={"color": act_color, "width": 2, "dash": "dot"},
        marker={"size": 5, "color": act_color},
    ))

    layout_extra = _NEON_LAYOUT if dark else {"template": "plotly_white"}
    fig.update_layout(
        **layout_extra,
        margin={"t": 20, "b": 40, "l": 60, "r": 20},
        legend={"orientation": "h", "y": -0.2},
        xaxis_title="Date",
        yaxis_title="Sales (units)",
        hovermode="x unified",
    )
    return fig


def _shap_figure(sku_id: str, dark: bool = False) -> go.Figure:
    try:
        shap_exp = ds.get_local_shap(sku_id)
    except Exception:
        return go.Figure()

    top    = get_top_contributors(shap_exp, n=10)
    top    = top.sort_values("shap_value", ascending=True)
    labels = [f.split("__", 1)[-1] if "__" in f else f for f in top["feature"]]

    if dark:
        colors     = [_NEON_RED if v < 0 else _NEON_PURPLE for v in top["shap_value"]]
        text_color = _NEON_FONT
        layout_extra = _NEON_LAYOUT
    else:
        colors     = ["#ef4444" if v < 0 else "#3b82f6" for v in top["shap_value"]]
        text_color = "#333333"
        layout_extra = {"template": "plotly_white"}

    fig = go.Figure(go.Bar(
        x=top["shap_value"],
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in top["shap_value"]],
        textposition="outside",
        textfont={"color": text_color},
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:+.4f}<extra></extra>",
    ))

    fig.update_layout(
        **layout_extra,
        margin={"t": 20, "b": 20, "l": 20, "r": 60},
        xaxis_title="SHAP value (impact on forecast)",
        yaxis_automargin=True,
        showlegend=False,
    )
    return fig


def _feat_val(shap_exp, *names) -> float | None:
    """Extract a feature's raw value from a SHAP explanation by name."""
    fn = [str(f) for f in (getattr(shap_exp, "feature_names", None) or [])]
    if shap_exp.data is None or not fn:
        return None
    data = np.asarray(shap_exp.data)
    if data.ndim > 1:
        data = data[0]
    for name in names:
        if name in fn:
            try:
                return float(data[fn.index(name)])
            except Exception:
                pass
    return None


def _section_label(text: str) -> html.P:
    return html.P(
        text,
        className="mb-1 mt-0 fw-bold text-uppercase",
        style={"fontSize": "11px", "letterSpacing": "0.5px", "color": "#6c757d"},
    )


def _bullet_list(items: list) -> html.Ul:
    return html.Ul(items, style={"fontSize": "13px", "paddingLeft": "18px", "marginBottom": "0"})


def _nlg_content(sku_id: str, iv: dict, std_v: float) -> html.Div:
    q50 = iv.get("q50", 0.0)
    q10 = iv.get("q10", 0.0)
    q90 = iv.get("q90", 0.0)

    # ── Confidence ────────────────────────────────────────────────────────────
    pi   = compute_prediction_interval(q50, std_v)
    conf = confidence_label(pi["width"], q50)
    conf_colour = {"high": "success", "moderate": "warning", "low": "danger"}.get(conf, "secondary")
    conf_desc = {
        "high":     "The forecast is reliable — demand is unlikely to deviate much.",
        "moderate": "Some variability is expected — keep a reasonable safety buffer.",
        "low":      "Actual demand could differ significantly — plan conservatively.",
    }.get(conf, "")

    # ── SHAP-derived insights ─────────────────────────────────────────────────
    typical = trend = forecast_pct = None
    has_promo = False
    try:
        shap_exp = ds.get_local_shap(sku_id)
        base_val = float(np.asarray(shap_exp.base_values).reshape(-1)[0])
        typical  = _feat_val(shap_exp, "num__item_mean_train", "item_mean_train") or base_val

        roll7 = _feat_val(shap_exp, "num__sales_roll_mean_7", "sales_roll_mean_7")
        lag7  = _feat_val(shap_exp, "num__sales_lag_7",        "sales_lag_7")
        if roll7 is not None and lag7 is not None:
            if abs(roll7 - lag7) <= 0.75:
                trend = "stable"
            elif roll7 > lag7:
                trend = "upward"
            else:
                trend = "softening"

        if typical and typical > 0:
            forecast_pct = (q50 - typical) / typical * 100

        discount  = _feat_val(shap_exp, "num__discount_depth", "discount_depth")
        has_promo = discount is not None and discount > 0.05
    except Exception:
        pass

    # ── Forecast summary bullets ──────────────────────────────────────────────
    q1_items = [html.Li(f"Forecast: {int(round(q50))} units over the next {ds.HORIZON} days")]
    if typical is not None:
        q1_items.append(html.Li(f"Typical demand for this SKU: ~{typical:.1f} units/period"))
    if forecast_pct is not None:
        direction = "above" if forecast_pct >= 0 else "below"
        note = "in line with normal" if abs(forecast_pct) < 5 else ("higher than usual" if forecast_pct > 0 else "lower than usual")
        q1_items.append(html.Li(f"Demand is {abs(forecast_pct):.1f}% {direction} average — {note}"))
    if trend:
        q1_items.append(html.Li(f"Recent sales trend: {trend}"))
    q1_items.append(
        html.Li("Active promotion detected — may be boosting demand" if has_promo
                else "No active promotions detected")
    )

    return html.Div([
        _section_label("Forecast Summary"),
        _bullet_list(q1_items),

        html.Hr(style={"margin": "12px 0"}),

        _section_label("Model Confidence"),
        _bullet_list([
            html.Li(f"Prediction range: {int(round(q10))} – {int(round(q90))} units"),
            html.Li([
                "Confidence: ",
                dbc.Badge(conf.upper(), color=conf_colour, className="ms-1 me-1"),
            ]),
            html.Li(conf_desc),
        ]),
    ])


_PATTERN_COLOUR = {
    "spike":    "danger",
    "growing":  "success",
    "declining":"warning",
    "stable":   "primary",
    "seasonal": "info",
}


def _temporal_heatmap(sku_id: str, dark: bool = False) -> go.Figure:
    try:
        t_df = _get_temporal_df(sku_id)
    except Exception:
        return go.Figure()

    skip      = {"date", "prediction", "total_shap"}
    feat_cols = [c for c in t_df.columns if c not in skip]
    if not feat_cols:
        return go.Figure()

    top_feats = (
        t_df[feat_cols].var()
        .sort_values(ascending=False)
        .head(10)
        .index.tolist()
    )

    heat_z  = t_df[top_feats].T.values
    heat_x  = t_df["date"].astype(str).tolist()
    abs_max = float(np.abs(heat_z).max()) or 1.0

    if dark:
        colorscale   = [[0.0, _NEON_RED], [0.5, "#0F1225"], [1.0, _NEON_PURPLE]]
        cb_font      = {"color": _NEON_FONT}
        layout_extra = _NEON_LAYOUT
    else:
        colorscale   = "RdBu"
        cb_font      = {"color": "#333333"}
        layout_extra = {"template": "plotly_white"}

    fig = go.Figure(go.Heatmap(
        z=heat_z,
        x=heat_x,
        y=top_feats,
        colorscale=colorscale,
        zmid=0,
        zmin=-abs_max,
        zmax=abs_max,
        colorbar={"title": {"text": "SHAP", "font": cb_font},
                  "thickness": 12, "tickfont": cb_font},
        hovertemplate="<b>%{y}</b><br>Date: %{x}<br>SHAP: %{z:.4f}<extra></extra>",
    ))

    fig.update_layout(
        **layout_extra,
        margin={"t": 20, "b": 60, "l": 20, "r": 20},
        xaxis_tickangle=-35,
        xaxis_automargin=True,
        yaxis_automargin=True,
    )
    return fig


def _temporal_summary(sku_id: str) -> html.Div:
    try:
        t_df    = _get_temporal_df(sku_id)
        pattern = classify_demand_pattern(t_df, sku_id)
        drivers = get_top_temporal_drivers(t_df, n=5)
    except Exception as exc:
        return html.P(f"Temporal analysis unavailable: {exc}", className="text-muted")

    colour = _PATTERN_COLOUR.get(pattern["pattern"], "secondary")

    _dashed = {"cursor": "help", "borderBottom": "1px dashed #6c757d"}

    def _row(label, value, label_id=None):
        label_el = (
            html.Small(label, id=label_id, className="text-muted", style=_dashed)
            if label_id
            else html.Small(label, className="text-muted")
        )
        return html.Tr([html.Td(label_el), html.Td(html.Small(str(value)))])

    drivers_rows = [
        html.Tr([
            html.Td(html.Small(str(r["feature"]), style={"fontSize": "11px"})),
            html.Td(html.Small(f"{r['temporal_variance']:.4f}", style={"fontSize": "11px"})),
        ])
        for _, r in drivers.iterrows()
    ]

    return html.Div([
        html.Div([
            dbc.Badge(
                pattern["pattern"].upper(),
                color=colour,
                className="me-2",
                style={"fontSize": "13px"},
            ),
            html.Small(
                f"{pattern['confidence']:.0%} confidence",
                id="temporal-pattern-confidence",
                className="text-muted",
                style={"cursor": "help", "borderBottom": "1px dashed #6c757d"},
            ),
            dbc.Tooltip(
                [
                    html.P(
                        "This percentage shows how clearly the system spotted this pattern "
                        "— it is not a forecast accuracy score.",
                        className="mb-2", style={"fontSize": "12px"},
                    ),
                    html.P(
                        "Think of it like this: 90% means the pattern is very obvious in the data "
                        "(e.g. a very sharp spike or a very steady trend). "
                        "50% means the data is mixed and the pattern is harder to pin down.",
                        className="mb-0", style={"fontSize": "12px"},
                    ),
                ],
                target="temporal-pattern-confidence",
                placement="bottom",
                style={"maxWidth": "340px", "textAlign": "left"},
            ),
        ], className="mb-3"),

        html.Table([
            html.Tbody([
                _row("Mean forecast",  f"{pattern['mean_pred']:.1f} units",           "tp-mean"),
                _row("Std dev",        f"{pattern['std_pred']:.1f} units",            "tp-std"),
                _row("Trend slope",    f"{pattern['trend_slope']:+.2f} units/period", "tp-slope"),
                _row("CV",             f"{pattern['cv']:.3f}",                        "tp-cv"),
            ])
        ], className="table table-sm mb-3"),

        dbc.Tooltip(
            "The average number of units this SKU is expected to sell per forecast period. "
            "Higher = more demand.",
            target="tp-mean", placement="left", style={"maxWidth": "260px", "fontSize": "12px"},
        ),
        dbc.Tooltip(
            "How much sales typically vary around the mean. "
            "A low value means demand is steady and predictable. "
            "A high value means sales jump around a lot — harder to plan for.",
            target="tp-std", placement="left", style={"maxWidth": "280px", "fontSize": "12px"},
        ),
        dbc.Tooltip(
            "Whether demand is going up or down over time. "
            "A positive number (e.g. +0.07) means sales are slowly rising each period. "
            "A negative number means they are falling. Close to zero means no clear trend.",
            target="tp-slope", placement="left", style={"maxWidth": "300px", "fontSize": "12px"},
        ),
        dbc.Tooltip(
            [
                html.P("CV (Coefficient of Variation) = Std dev ÷ Mean forecast.",
                       className="mb-2", style={"fontSize": "12px"}),
                html.P("It shows how volatile demand is relative to its size.",
                       className="mb-2", style={"fontSize": "12px"}),
                html.Ul([
                    html.Li("Below 0.15 → very stable, easy to plan", style={"fontSize": "12px"}),
                    html.Li("0.15 – 0.40 → some variability, manageable", style={"fontSize": "12px"}),
                    html.Li("Above 0.40 → high volatility, keep extra safety stock", style={"fontSize": "12px"}),
                ], className="mb-0 ps-3"),
            ],
            target="tp-cv", placement="left", style={"maxWidth": "310px", "textAlign": "left"},
        ),

        html.P(html.Strong("Top temporal drivers"), className="mb-1", style={"fontSize": "12px"}),
        html.Table([
            html.Thead(html.Tr([
                html.Th(html.Small("Feature"),  style={"fontSize": "11px"}),
                html.Th(
                    html.Small(
                        "Variance",
                        id="tp-variance-header",
                        style={"cursor": "help", "borderBottom": "1px dashed #6c757d"},
                    ),
                    style={"fontSize": "11px"},
                ),
            ])),
            html.Tbody(drivers_rows),
        ], className="table table-sm"),

        dbc.Tooltip(
            "How much this feature's influence on the forecast changes from one time period to the next. "
            "A high variance means this feature sometimes pushes demand up a lot and other times barely matters — "
            "making it a key source of unpredictability for this SKU.",
            target="tp-variance-header",
            placement="left",
            style={"maxWidth": "310px", "fontSize": "12px"},
        ),
    ])


def _temporal_nlg(sku_id: str) -> html.Div:
    try:
        t_df    = _get_temporal_df(sku_id)
        pattern = classify_demand_pattern(t_df, sku_id)
        drivers = get_top_temporal_drivers(t_df, n=3)
    except Exception as exc:
        return html.P(f"Temporal analysis unavailable: {exc}", className="text-muted small")

    _descriptions = {
        "spike":    "Sales show an irregular spike — one or more periods had unusually high or low demand.",
        "growing":  "Demand is growing — sales are consistently rising period over period.",
        "declining":"Demand is declining — sales are gradually falling over time.",
        "stable":   "Demand is stable — sales remain consistent with little variation.",
        "seasonal": "Demand follows a cyclical pattern — likely tied to recurring events or seasons.",
    }
    desc       = _descriptions.get(pattern["pattern"], "")
    colour     = _PATTERN_COLOUR.get(pattern["pattern"], "secondary")
    top_driver = drivers.iloc[0]["feature"] if not drivers.empty else "unknown"

    return html.Div([
        html.Hr(style={"margin": "12px 0"}),
        _section_label("Demand Pattern"),
        _bullet_list([
            html.Li([
                "Pattern: ",
                dbc.Badge(pattern["pattern"].upper(), color=colour, className="ms-1 me-1"),
                html.Span(f"({pattern['confidence']:.0%} confident)", className="text-muted",
                          style={"fontSize": "12px"}),
            ]),
            html.Li(desc),
            html.Li(f"Most time-sensitive feature: {top_driver}"),
        ]),
    ])


# ── temporal SHAP cache (cleared on model switch) ────────────────────────────
_temporal_cache: dict[str, pd.DataFrame] = {}


def _get_temporal_df(sku_id: str) -> pd.DataFrame:
    if sku_id not in _temporal_cache:
        dates    = pd.to_datetime(ds.test_df["date"])
        item_ids = ds.test_df["item_id"]
        _temporal_cache[sku_id] = compute_temporal_shap(
            ds.model, ds.X_test, dates, item_ids, sku_id
        )
    return _temporal_cache[sku_id]


# ── callback registration ─────────────────────────────────────────────────────

def register_sku_callbacks(app) -> None:

    @app.callback(
        Output("sku-selector", "options"),
        Output("sku-selector", "value"),
        Input("model-store", "data"),
    )
    def update_sku_options(_model_key: str):
        _temporal_cache.clear()
        opts    = [{"label": s, "value": s} for s in ds.SKU_LIST]
        default = ds.SKU_LIST[0] if ds.SKU_LIST else None
        return opts, default

    @app.callback(
        Output("sku-summary-cards",    "children"),
        Output("sku-forecast-chart",   "figure"),
        Output("sku-shap-chart",       "figure"),
        Output("sku-nlg-brief",        "children"),
        Output("sku-temporal-heatmap", "figure"),
        Output("sku-temporal-summary", "children"),
        Output("sku-temporal-nlg",     "children"),
        Input("sku-selector",  "value"),
        Input("model-store",   "data"),
        Input("theme-store",   "data"),
    )
    def update_sku_page(sku_id: str, _model_key: str, theme: str):
        if not sku_id:
            return [], {}, {}, "", {}, "", ""

        dark = theme == "dark"

        global_std = float(ds.sku_std.mean()) if ds.sku_std is not None else 1.0
        std_v      = float(ds.sku_std.get(sku_id, global_std)) if ds.sku_std is not None else global_std

        iv       = ds.forecasts.get(sku_id, {"q10": 0, "q50": 0, "q90": 0, "width": 0})
        card_row = ds.cards_df[ds.cards_df["sku_id"] == sku_id]
        card     = card_row.iloc[0].to_dict() if not card_row.empty else {}

        summary      = _summary_cards(iv, card, std_v, dark=dark)
        forecast_fig = _forecast_figure(sku_id, std_v, dark=dark)
        shap_fig     = _shap_figure(sku_id, dark=dark)
        nlg          = _nlg_content(sku_id, iv, std_v)
        t_heat       = _temporal_heatmap(sku_id, dark=dark)
        t_summary    = _temporal_summary(sku_id)
        t_nlg        = _temporal_nlg(sku_id)

        return summary, forecast_fig, shap_fig, nlg, t_heat, t_summary, t_nlg
