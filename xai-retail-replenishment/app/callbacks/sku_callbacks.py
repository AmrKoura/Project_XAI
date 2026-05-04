"""
Callbacks for the SKU Explorer page.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, html, dash_table
import dash_bootstrap_components as dbc

import app.data_store as ds
from xai.local_shap import get_top_contributors
from xai.uncertainty import compute_prediction_interval, confidence_label
from xai.temporal_shap import (
    compute_temporal_shap, classify_demand_pattern, get_top_temporal_drivers,
)
from xai.comparative_shap import compare_models_for_sku

# ── colour constants ──────────────────────────────────────────────────────────

_NEON_BG     = "#0F1225"
_NEON_FONT   = "#FFFFFF"
_NEON_GRID   = "rgba(95,1,251,0.2)"
_NEON_PURPLE = "#5F01FB"
_NEON_RED    = "#FF2D55"

_NEON_LAYOUT = dict(
    paper_bgcolor       = _NEON_BG,
    plot_bgcolor        = _NEON_BG,
    font                = {"color": _NEON_FONT},
    xaxis_gridcolor     = _NEON_GRID,
    xaxis_linecolor     = _NEON_GRID,
    xaxis_zerolinecolor = _NEON_GRID,
    yaxis_gridcolor     = _NEON_GRID,
    yaxis_linecolor     = _NEON_GRID,
    yaxis_zerolinecolor = _NEON_GRID,
)

_URGENCY_COLOUR = {"CRITICAL": "danger", "HIGH": "warning", "LOW": "success"}
_PATTERN_COLOUR = {"spike": "danger", "growing": "success",
                   "declining": "warning", "stable": "primary", "seasonal": "info"}

# ── helpers ───────────────────────────────────────────────────────────────────

def _date_badge() -> html.Div:
    """Compact date-context line shown under page titles."""
    return html.Div([
        dbc.Badge(
            f"Forecasting: {ds.forecast_range_str()}",
            color="primary", className="me-2",
            style={"fontSize": "12px", "fontWeight": "normal"},
        ),
        html.Small(
            f"Based on data through {ds.DATA_LAST_DATE}",
            className="text-muted",
            style={"fontSize": "12px"},
        ),
    ])


def _section_label(text: str) -> html.P:
    return html.P(text, className="mb-1 mt-0 fw-bold text-uppercase",
                  style={"fontSize": "11px", "letterSpacing": "0.5px", "color": "#6c757d"})

def _bullet_list(items: list) -> html.Ul:
    return html.Ul(items, style={"fontSize": "13px", "paddingLeft": "18px", "marginBottom": "0"})

def _clean(name: str) -> str:
    return ds.feature_label(name)

# ── summary cards ─────────────────────────────────────────────────────────────

def _summary_cards(iv: dict, card: dict, std_v: float, dark: bool = False) -> html.Div:
    q50  = iv.get("q50", 0)
    q10  = iv.get("q10", 0)
    q90  = iv.get("q90", 0)
    pi   = compute_prediction_interval(q50, std_v)
    conf = confidence_label(pi["width"], q50)
    conf_colour  = {"high": "success", "moderate": "warning", "low": "danger"}.get(conf, "secondary")
    urgency      = card.get("urgency", "LOW")
    days         = card.get("days_of_stock", 0)
    total_colour = "light" if dark else "dark"

    def _stat(value, label, colour, label_id=None, highlight=False):
        if highlight:
            bg = _NEON_PURPLE if dark else "#0d6efd"
            label_el = (
                html.Small(label, id=label_id,
                           style={"color": "rgba(255,255,255,0.75)", "cursor": "help",
                                  "borderBottom": "1px dashed rgba(255,255,255,0.5)"})
                if label_id else
                html.Small(label, style={"color": "rgba(255,255,255,0.75)"})
            )
            div_style = {
                "padding": "14px 18px",
                "textAlign": "center",
                "borderRadius": "0.375rem",
                "boxShadow": "0 0.125rem 0.25rem rgba(0,0,0,0.15)",
            }
            if dark:
                div_style["backgroundColor"] = _NEON_PURPLE
            return dbc.Col(
                html.Div([
                    html.H3(str(value), className="mb-0", style={"color": "#ffffff"}),
                    label_el,
                ], className="" if dark else "bg-primary", style=div_style),
                xs=6, md=True,
            )

        label_el = (
            html.Small(label, id=label_id, className="text-muted",
                       style={"cursor": "help", "borderBottom": "1px dashed #6c757d"})
            if label_id else html.Small(label, className="text-muted")
        )
        return dbc.Col(
            dbc.Card([dbc.CardBody([
                html.H3(str(value), className=f"text-{colour} mb-0"),
                label_el,
            ], style={"padding": "14px 18px"})], className="shadow-sm text-center"),
            xs=6, md=True,
        )

    return html.Div([
        dbc.Row([
            _stat(f"{int(round(q50))} units", "Forecast", total_colour, highlight=True),
            _stat(f"[{int(round(q10))} – {int(round(q90))}]",       "80% interval",  conf_colour, label_id="interval-card-label"),
            _stat(conf.upper(),                                       "Confidence",    conf_colour, label_id="confidence-card-label"),
            _stat(f"{days:.1f} days",                                 "Days of stock", total_colour),
            _stat(urgency,                                            "Urgency",       _URGENCY_COLOUR.get(urgency, "secondary")),
        ], className="g-3"),
        dbc.Tooltip(
            "80% chance actual demand falls inside this range. Narrow = confident, Wide = more uncertainty.",
            target="interval-card-label", placement="bottom",
        ),
        dbc.Tooltip(
            "HIGH: interval width < 40% of forecast. MODERATE: 40–100%. LOW: > 100% (very uncertain).",
            target="confidence-card-label", placement="bottom",
        ),
    ])

# ── forecast figure ───────────────────────────────────────────────────────────

def _forecast_figure(sku_id: str, std_v: float, dark: bool = False) -> go.Figure:
    try:
        sku_df, sku_preds = ds.get_sku_test_rows(sku_id)
    except KeyError:
        return go.Figure()
    if sku_df.empty:
        return go.Figure()

    dates   = pd.to_datetime(sku_df["date"]).tolist() if "date" in sku_df.columns else list(range(len(sku_df)))
    actuals = sku_df[ds.TARGET].values
    preds   = sku_preds
    margin  = 1.28 * std_v
    upper   = np.maximum(0, preds + margin)
    lower   = np.maximum(0, preds - margin)

    # Classify each actual point: inside band, above, or below
    inside  = (actuals >= lower) & (actuals <= upper)
    above   = actuals > upper
    below   = actuals < lower

    if dark:
        pred_color  = _NEON_PURPLE
        band_color  = "rgba(95,1,251,0.12)"
        col_inside  = "#00FF87"
        col_outside = "#FFB800"
    else:
        pred_color  = "#636efa"
        band_color  = "rgba(99,110,250,0.15)"
        col_inside  = "#22c55e"
        col_outside = "#f97316"

    marker_colors = [col_inside if i else col_outside for i in inside]
    layout_extra  = _NEON_LAYOUT if dark else {"template": "plotly_white"}

    # Rich hover for predicted line
    pred_hover = [
        f"<b>Week of {d.strftime('%b %d, %Y')}</b><br>"
        f"Forecast: {p:.1f} units<br>"
        f"80% range: {lo:.1f} – {hi:.1f} units"
        for d, p, lo, hi in zip(dates, preds, lower, upper)
    ]

    # Rich hover for actual line
    act_hover = []
    for i, (d, a, lo, hi) in enumerate(zip(dates, actuals, lower, upper)):
        err       = float(a) - float(preds[i])
        label     = "within forecast range ✓" if lo <= a <= hi else ("above forecast range ⚠" if a > hi else "below forecast range ⚠")
        direction = "over-predicted" if err < 0 else ("under-predicted" if err > 0 else "exact")
        err_line  = f"Error: {abs(err):.1f} units (model {direction})" if err != 0 else "Error: exact"
        act_hover.append(
            f"<b>Week of {d.strftime('%b %d, %Y')}</b><br>"
            f"Actual: {a:.1f} units — {label}<br>"
            f"{err_line}"
        )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(dates) + list(reversed(dates)),
        y=list(upper) + list(reversed(lower)),
        fill="toself", fillcolor=band_color,
        line={"width": 0}, name="80% interval", showlegend=True, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=preds, mode="lines+markers", name="Predicted",
        line={"color": pred_color, "width": 2},
        marker={"size": 5, "color": pred_color},
        customdata=pred_hover,
        hovertemplate="%{customdata}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=actuals, mode="lines+markers", name="Actual",
        line={"color": marker_colors[0] if len(set(marker_colors)) == 1 else "#94a3b8", "width": 2, "dash": "dot"},
        marker={"size": 7, "color": marker_colors},
        customdata=act_hover,
        hovertemplate="%{customdata}<extra></extra>",
    ))
    fig.update_layout(
        **layout_extra,
        margin={"t": 20, "b": 40, "l": 60, "r": 20},
        legend={"orientation": "h", "y": -0.2},
        xaxis_title="Date", yaxis_title="Sales (units)",
        hovermode="x unified",
    )
    return fig

# ── SHAP waterfall ────────────────────────────────────────────────────────────

def _shap_figure(sku_id: str, dark: bool = False) -> go.Figure:
    try:
        shap_exp = ds.get_local_shap(sku_id)
    except Exception:
        return go.Figure()

    # Preserve rank before sorting for display
    top = get_top_contributors(shap_exp, n=10)
    top["rank"] = range(1, len(top) + 1)
    top = top.sort_values("shap_value", ascending=True).reset_index(drop=True)

    labels       = [_clean(f) for f in top["feature"]]
    colors       = [(_NEON_RED if dark else "#ef4444") if v < 0 else (_NEON_PURPLE if dark else "#3b82f6") for v in top["shap_value"]]
    layout_extra = _NEON_LAYOUT if dark else {"template": "plotly_white"}

    def _fmt_val(feat: str, val: float) -> str:
        f = feat.lower()
        if any(f.startswith(p) for p in ("event_", "snap", "is_month")):
            return "Active" if val > 0.5 else "Inactive"
        if "price" in f:
            return f"${val:.2f}"
        return f"{val:.1f}"

    hover = []
    for _, row in top.iterrows():
        rank    = int(row["rank"])
        label   = _clean(str(row["feature"]))
        sv      = float(row["shap_value"])
        fv      = row["feature_value"]
        dir_txt = "Increased" if sv >= 0 else "Decreased"
        fmt_val = _fmt_val(str(row["feature"]), float(fv)) if not pd.isna(fv) else "N/A"
        hover.append(
            f"<b>#{rank} driver — {label}</b><br>"
            f"{dir_txt} forecast by {abs(sv):.1f} units<br>"
            f"Feature value: {fmt_val}"
        )

    fig = go.Figure(go.Bar(
        x=top["shap_value"], y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v:+.2f}" for v in top["shap_value"]],
        textposition="outside", textfont={"color": _NEON_FONT if dark else "#333333"},
        customdata=hover,
        hovertemplate="%{customdata}<extra></extra>",
    ))
    fig.update_layout(
        **layout_extra,
        margin={"t": 20, "b": 20, "l": 20, "r": 60},
        xaxis_title="Impact on forecast (units)",
        yaxis_automargin=True, showlegend=False,
    )
    return fig

# ── NLG brief ─────────────────────────────────────────────────────────────────

def _nlg_content(sku_id: str, iv: dict, std_v: float) -> html.Div:
    q50 = iv.get("q50", 0.0)
    q10 = iv.get("q10", 0.0)
    q90 = iv.get("q90", 0.0)

    pi   = compute_prediction_interval(q50, std_v)
    conf = confidence_label(pi["width"], q50)
    conf_colour = {"high": "success", "moderate": "warning", "low": "danger"}.get(conf, "secondary")
    conf_desc   = {
        "high":     "The forecast is reliable — demand is unlikely to deviate much.",
        "moderate": "Some variability is expected — keep a reasonable safety buffer.",
        "low":      "Actual demand could differ significantly — plan conservatively.",
    }.get(conf, "")

    top_drivers = []
    try:
        shap_exp    = ds.get_local_shap(sku_id)
        top_df      = get_top_contributors(shap_exp, n=3)
        top_drivers = [_clean(f) for f in top_df["feature"].tolist()]
    except Exception:
        pass

    q1_items = [
        html.Li(f"Forecast: {int(round(q50))} units for {ds.forecast_range_str()}"),
        html.Li(f"Range: {int(round(q10))} – {int(round(q90))} units (80% probability)"),
        html.Li(f"Based on historical data through {ds.DATA_LAST_DATE}"),
    ]
    if top_drivers:
        q1_items.append(html.Li(f"Top drivers: {', '.join(top_drivers)}"))

    return html.Div([
        _section_label("Forecast Summary"),
        _bullet_list(q1_items),
        html.Hr(style={"margin": "12px 0"}),
        _section_label("Model Confidence"),
        _bullet_list([
            html.Li(["Confidence: ", dbc.Badge(conf.upper(), color=conf_colour, className="ms-1 me-1")]),
            html.Li(conf_desc),
        ]),
    ])

# ── temporal heatmap ──────────────────────────────────────────────────────────

_temporal_cache: dict[str, pd.DataFrame] = {}


def _get_temporal_df(sku_id: str) -> pd.DataFrame:
    if sku_id not in _temporal_cache:
        dates    = pd.to_datetime(ds.test_df["date"])
        item_ids = ds.test_df["item_id"]
        _temporal_cache[sku_id] = compute_temporal_shap(
            ds.model, ds.X_test, dates, item_ids, sku_id
        )
    return _temporal_cache[sku_id]


def _temporal_line_chart(sku_id: str, dark: bool = False) -> go.Figure:
    layout_extra = _NEON_LAYOUT if dark else {"template": "plotly_white"}

    try:
        t_df = _get_temporal_df(sku_id)
    except Exception:
        return go.Figure()

    skip      = {"date", "prediction", "total_shap"}
    feat_cols = [c for c in t_df.columns if c not in skip]
    if not feat_cols:
        return go.Figure()

    top_feats    = t_df[feat_cols].var().sort_values(ascending=False).head(5).index.tolist()
    dates        = pd.to_datetime(t_df["date"]).tolist()
    palette_dark = ["#5F01FB", "#FF2D55", "#00FF87", "#FFB800", "#00C8FF"]
    palette_lite = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6"]
    palette      = palette_dark if dark else palette_lite

    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(120,120,120,0.35)", line_width=1)

    for i, feat in enumerate(top_feats):
        shap_vals = t_df[feat].values.astype(float)
        label     = _clean(feat)
        color     = palette[i % len(palette)]
        hover     = [
            f"<b>{label}</b><br>"
            f"{'increased' if v >= 0 else 'decreased'} forecast by {abs(v):.1f} units"
            for v in shap_vals
        ]
        fig.add_trace(go.Scatter(
            x=dates, y=shap_vals,
            mode="lines+markers", name=label,
            line={"color": color, "width": 2},
            marker={"size": 5, "color": color},
            customdata=hover,
            hovertemplate="%{customdata}<extra></extra>",
        ))

    fig.update_layout(
        **layout_extra,
        margin={"t": 20, "b": 70, "l": 60, "r": 20},
        xaxis_title="Date",
        yaxis_title="Impact on forecast (units)",
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.3, "x": 0},
        xaxis_tickangle=-35, xaxis_automargin=True,
    )
    return fig


def _temporal_summary(sku_id: str) -> html.Div:
    try:
        t_df    = _get_temporal_df(sku_id)
        pattern = classify_demand_pattern(t_df, sku_id)
        drivers = get_top_temporal_drivers(t_df, n=5)
    except Exception as exc:
        if "TreeExplainer" in str(exc) or "not yet supported" in str(exc):
            return html.Div([
                html.P("Pattern classification not available for Neural Networks.",
                       className="text-muted small mb-1"),
                html.P("Switch to LightGBM, XGBoost, or Random Forest to enable.",
                       className="text-muted small"),
            ])
        return html.P(f"Temporal analysis unavailable: {exc}", className="text-muted small")

    colour = _PATTERN_COLOUR.get(pattern["pattern"], "secondary")
    drivers_rows = [
        html.Tr([
            html.Td(html.Small(_clean(str(r["feature"])), style={"fontSize": "11px"})),
            html.Td(html.Small(f"{r['temporal_variance']:.3f}", style={"fontSize": "11px"})),
        ])
        for _, r in drivers.iterrows()
    ]

    return html.Div([
        html.Div([
            dbc.Badge(pattern["pattern"].upper(), color=colour, className="me-2",
                      style={"fontSize": "13px"}),
            html.Small(
                f"{pattern['confidence']:.0%} confidence",
                id="temporal-pattern-confidence",
                className="text-muted",
                style={"cursor": "help", "borderBottom": "1px dashed #6c757d"},
            ),
            dbc.Tooltip(
                "How strongly the sales data fits the detected pattern — not prediction accuracy. "
                "Based on spike intensity, trend slope strength, or demand stability (CV).",
                target="temporal-pattern-confidence",
                placement="bottom",
            ),
        ], className="mb-3"),
        html.Table([html.Tbody([
            html.Tr([
                html.Td(html.Small("Mean forecast", id="tp-mean", className="text-muted",
                                   style={"cursor": "help", "borderBottom": "1px dashed #6c757d"})),
                html.Td(html.Small(f"{pattern['mean_pred']:.1f} units")),
            ]),
            html.Tr([
                html.Td(html.Small("Std dev", id="tp-std", className="text-muted",
                                   style={"cursor": "help", "borderBottom": "1px dashed #6c757d"})),
                html.Td(html.Small(f"{pattern['std_pred']:.1f} units")),
            ]),
            html.Tr([
                html.Td(html.Small("Trend slope", id="tp-slope", className="text-muted",
                                   style={"cursor": "help", "borderBottom": "1px dashed #6c757d"})),
                html.Td(html.Small(f"{pattern['trend_slope']:+.2f} units/period")),
            ]),
        ])], className="table table-sm mb-3"),
        dbc.Tooltip("Average weekly forecast across the test period for this SKU.",
                    target="tp-mean", placement="left"),
        dbc.Tooltip("How much the weekly forecast varies around the mean — higher = more volatile demand.",
                    target="tp-std", placement="left"),
        dbc.Tooltip("Change in forecast per week (positive = growing, negative = declining, near zero = flat).",
                    target="tp-slope", placement="left"),
        html.P(html.Strong("Most time-sensitive features"), className="mb-1", style={"fontSize": "12px"}),
        html.P(html.Small("Higher score = feature's influence changes more across weeks",
                           className="text-muted"), className="mb-1"),
        html.Table([
            html.Thead(html.Tr([
                html.Th(html.Small("Feature"), style={"fontSize": "11px"}),
                html.Th(html.Small("Score"), style={"fontSize": "11px"}),
            ])),
            html.Tbody(drivers_rows),
        ], className="table table-sm"),
    ])


def _temporal_nlg(sku_id: str) -> html.Div:
    try:
        t_df    = _get_temporal_df(sku_id)
        pattern = classify_demand_pattern(t_df, sku_id)
        drivers = get_top_temporal_drivers(t_df, n=3)
    except Exception as exc:
        if "TreeExplainer" in str(exc) or "not yet supported" in str(exc):
            return html.Div([
                html.Hr(style={"margin": "12px 0"}),
                _section_label("Demand Pattern"),
                html.P("SHAP-based pattern analysis is only available for tree models (LightGBM, XGBoost, Random Forest).",
                       className="text-muted", style={"fontSize": "13px"}),
            ])
        return html.Div()

    _descriptions = {
        "spike":    "Sales show an irregular spike — one or more periods had unusually high or low demand.",
        "growing":  "Demand is growing — sales are consistently rising period over period.",
        "declining":"Demand is declining — sales are gradually falling over time.",
        "stable":   "Demand is stable — sales remain consistent with little variation.",
        "seasonal": "Demand follows a cyclical pattern — likely tied to recurring events or seasons.",
    }
    colour     = _PATTERN_COLOUR.get(pattern["pattern"], "secondary")
    top_driver = _clean(drivers.iloc[0]["feature"]) if not drivers.empty else "unknown"

    return html.Div([
        html.Hr(style={"margin": "12px 0"}),
        _section_label("Demand Pattern"),
        _bullet_list([
            html.Li(["Pattern: ",
                     dbc.Badge(pattern["pattern"].upper(), color=colour, className="ms-1 me-1"),
                     html.Span(f"({pattern['confidence']:.0%} confident)",
                               className="text-muted", style={"fontSize": "12px"})]),
            html.Li(_descriptions.get(pattern["pattern"], "")),
            html.Li(f"Most time-sensitive feature: {top_driver}"),
        ]),
    ])

# ── censored demand & SHAP distortion (V4: zero-sales proxy) ─────────────────

_stockout_cache: dict[str, pd.DataFrame] = {}


def _get_stockout_df(sku_id: str) -> pd.DataFrame:
    """Compute per-week stockout flags for a SKU using zero actual sales as proxy."""
    if sku_id in _stockout_cache:
        return _stockout_cache[sku_id]
    sku_df, preds = ds.get_sku_test_rows(sku_id)
    df = sku_df[["date", ds.TARGET]].copy()
    df["date"]        = pd.to_datetime(df["date"])
    df["prediction"]  = preds
    df["is_stockout"] = (df[ds.TARGET] == 0).astype(int)
    df["demand_gap"]  = np.where(df["is_stockout"] == 1,
                                  np.maximum(0, df["prediction"]),
                                  0.0)
    _stockout_cache[sku_id] = df.sort_values("date").reset_index(drop=True)
    return _stockout_cache[sku_id]


def _censored_demand_figure(sku_id: str, dark: bool = False) -> go.Figure:
    try:
        df = _get_stockout_df(sku_id)
    except Exception:
        return go.Figure()

    act_color    = _NEON_FONT   if dark else "#1e293b"
    pred_color   = _NEON_RED    if dark else "#ef4444"
    shade_color  = "rgba(255,45,85,0.18)" if dark else "rgba(239,68,68,0.12)"
    layout_extra = _NEON_LAYOUT if dark else {"template": "plotly_white"}

    fig = go.Figure()

    # Shade stockout weeks
    for _, row in df[df["is_stockout"] == 1].iterrows():
        x0 = row["date"]
        x1 = x0 + pd.Timedelta(days=ds.HORIZON)
        fig.add_vrect(x0=x0, x1=x1, fillcolor=shade_color, layer="below", line_width=0)

    act_hover  = []
    pred_hover = []
    act_marker_colors = []

    for _, row in df.iterrows():
        actual    = float(row[ds.TARGET])
        pred      = float(row["prediction"])
        gap       = float(row["demand_gap"])
        stockout  = bool(row["is_stockout"])
        date_str  = pd.Timestamp(row["date"]).strftime("%b %d, %Y")

        if stockout:
            act_hover.append(
                f"<b>Week of {date_str}</b><br>"
                f"Actual: 0 units — suspected stockout ⚠<br>"
                f"Estimated lost demand: {gap:.0f} units"
            )
            act_marker_colors.append(_NEON_RED if dark else "#ef4444")
        else:
            act_hover.append(
                f"<b>Week of {date_str}</b><br>"
                f"Actual: {actual:.0f} units — normal week ✓"
            )
            act_marker_colors.append(act_color)

        pred_hover.append(
            f"<b>Week of {date_str}</b><br>"
            f"Model forecast: {pred:.1f} units"
            + (f"<br>Gap vs actual: {gap:.0f} units (likely lost demand)" if stockout else "")
        )

    fig.add_trace(go.Scatter(
        x=df["date"], y=df[ds.TARGET],
        name="Actual sales", mode="lines+markers",
        line={"color": act_color, "width": 2},
        marker={"size": 6, "color": act_marker_colors},
        customdata=act_hover,
        hovertemplate="%{customdata}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["prediction"],
        name="Model prediction", mode="lines",
        line={"color": pred_color, "width": 2, "dash": "dash"},
        customdata=pred_hover,
        hovertemplate="%{customdata}<extra></extra>",
    ))

    n_stockout   = int(df["is_stockout"].sum())
    total_lost   = float(df["demand_gap"].sum())
    title_suffix = f"  ({n_stockout} zero-sales weeks | est. {total_lost:.0f} lost units)"

    fig.update_layout(
        **layout_extra,
        title={"text": f"Actual vs Predicted{title_suffix}", "font": {"size": 12}},
        margin={"t": 40, "b": 40, "l": 50, "r": 20},
        xaxis_title="Date", yaxis_title="Units",
        legend={"orientation": "h", "y": -0.3},
        hovermode="x unified",
    )
    return fig


def _shap_distortion_figure(sku_id: str, dark: bool = False) -> go.Figure:
    """Compare avg SHAP values during zero-sales vs normal weeks for top features."""
    try:
        df   = _get_stockout_df(sku_id).copy()
        t_df = _get_temporal_df(sku_id).copy()
    except Exception:
        return go.Figure()

    skip      = {"date", "prediction", "total_shap"}
    feat_cols = [c for c in t_df.columns if c not in skip]
    if not feat_cols:
        return go.Figure()

    # Merge on date to align rows correctly regardless of sort order
    df["date"]   = pd.to_datetime(df["date"])
    t_df["date"] = pd.to_datetime(t_df["date"])
    merged = t_df.merge(df[["date", "is_stockout"]], on="date", how="inner")
    if merged.empty:
        return go.Figure()
    t_df = merged

    normal_mask   = t_df["is_stockout"] == 0
    stockout_mask = t_df["is_stockout"] == 1

    if not stockout_mask.any():
        layout_extra = _NEON_LAYOUT if dark else {"template": "plotly_white"}
        fig = go.Figure()
        fig.update_layout(**layout_extra,
                          title={"text": "No zero-sales weeks detected", "font": {"size": 12}},
                          margin={"t": 40, "b": 20, "l": 20, "r": 20})
        return fig
    if not normal_mask.any():
        return go.Figure()

    # Top 8 features by absolute mean SHAP overall
    top_feats = (
        t_df[feat_cols].abs().mean()
        .sort_values(ascending=False)
        .head(8).index.tolist()
    )

    normal_means   = t_df.loc[normal_mask, top_feats].mean()
    stockout_means = t_df.loc[stockout_mask, top_feats].mean()

    labels       = [_clean(f) for f in top_feats]
    layout_extra = _NEON_LAYOUT if dark else {"template": "plotly_white"}
    color_n      = _NEON_PURPLE if dark else "#3b82f6"
    color_s      = _NEON_RED    if dark else "#ef4444"

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=normal_means.values, orientation="h",
        name="Normal weeks", marker_color=color_n, opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        y=labels, x=stockout_means.values, orientation="h",
        name="Zero-sales weeks", marker_color=color_s, opacity=0.85,
    ))
    fig.update_layout(
        **layout_extra,
        barmode="group",
        title={"text": "Avg SHAP: Normal vs Zero-Sales Weeks", "font": {"size": 12}},
        margin={"t": 40, "b": 20, "l": 20, "r": 20},
        xaxis_title="Mean SHAP value",
        yaxis_automargin=True,
        legend={"orientation": "h", "y": -0.15},
    )
    return fig


def _stockout_sku_nlg(sku_id: str) -> html.Div:
    try:
        df = _get_stockout_df(sku_id)
    except Exception:
        return html.Div()

    n_stockout = int(df["is_stockout"].sum())
    total      = len(df)
    total_lost = float(df["demand_gap"].sum())

    if n_stockout == 0:
        return html.Div([
            html.Hr(style={"margin": "12px 0"}),
            _section_label("Stockout Analysis"),
            html.P("No zero-sales weeks detected in the test period — demand signal appears clean.",
                   className="text-muted", style={"fontSize": "13px"}),
        ])

    rate     = n_stockout / total * 100
    severity = "HIGH" if rate > 30 else "MODERATE" if rate > 10 else "LOW"
    sev_col  = {"HIGH": "danger", "MODERATE": "warning", "LOW": "success"}.get(severity, "secondary")

    return html.Div([
        html.Hr(style={"margin": "12px 0"}),
        _section_label("Stockout Analysis"),
        _bullet_list([
            html.Li([
                f"Zero-sales weeks: {n_stockout} / {total} ({rate:.0f}%) — severity: ",
                dbc.Badge(severity, color=sev_col, className="ms-1"),
            ]),
            html.Li(f"Estimated lost demand: {total_lost:.0f} units across stockout weeks"),
            html.Li("Red-shaded weeks had zero recorded sales — model predictions show what true demand likely was"),
            html.Li("The SHAP chart shows how the model behaves differently when sales are missing"),
        ]),
    ])


# ── comparative SHAP (model vs model) ─────────────────────────────────────────

_model_comp_cache: dict[tuple, pd.DataFrame] = {}
_model_b_cache:    dict[str, object]          = {}


def _get_model_comp_df(sku_id: str, model_type_b: str) -> pd.DataFrame:
    key = (sku_id, ds.current_model_type, model_type_b)
    if key not in _model_comp_cache:
        if model_type_b not in _model_b_cache:
            _model_b_cache[model_type_b] = ds.load_second_model(model_type_b)
        model_b = _model_b_cache[model_type_b]
        _model_comp_cache[key] = compare_models_for_sku(
            ds.model, model_b, ds.X_test, ds.test_df["item_id"], sku_id
        )
    return _model_comp_cache[key]


def _comp_diff_figure(sku_id: str, label_a: str, label_b: str, model_type_b: str, dark: bool = False) -> go.Figure:
    try:
        diff_df = _get_model_comp_df(sku_id, model_type_b)
    except Exception:
        return go.Figure()

    top          = diff_df.head(12).copy().sort_values("diff", ascending=True)
    top["feat_clean"] = top["feature"].apply(_clean)
    colors       = [(_NEON_RED if dark else "#ef4444") if v < 0 else (_NEON_PURPLE if dark else "#3b82f6") for v in top["diff"]]
    layout_extra = _NEON_LAYOUT if dark else {"template": "plotly_white"}

    hover = [
        f"<b>{_clean(r['feature'])}</b><br>"
        f"{'More influential in ' + label_a if r['diff'] > 0 else 'More influential in ' + label_b}<br>"
        f"Difference: {r['diff']:+.2f} units"
        for _, r in top.iterrows()
    ]

    fig = go.Figure(go.Bar(
        x=top["diff"], y=top["feat_clean"], orientation="h",
        marker_color=colors,
        text=[f"{v:+.2f}" for v in top["diff"]],
        textposition="outside", textfont={"color": _NEON_FONT if dark else "#333333"},
        customdata=hover,
        hovertemplate="%{customdata}<extra></extra>",
    ))
    fig.update_layout(
        **layout_extra,
        title={"text": f"SHAP Difference ({label_a} − {label_b})", "font": {"size": 13}},
        margin={"t": 40, "b": 20, "l": 20, "r": 70},
        xaxis_title="SHAP diff (A − B)", yaxis_automargin=True, showlegend=False,
    )
    return fig


def _comp_side_figure(sku_id: str, label_a: str, label_b: str, model_type_b: str, dark: bool = False) -> go.Figure:
    try:
        diff_df = _get_model_comp_df(sku_id, model_type_b)
    except Exception:
        return go.Figure()

    top          = diff_df.head(10).copy()
    top["feat_clean"] = top["feature"].apply(_clean)
    layout_extra = _NEON_LAYOUT if dark else {"template": "plotly_white"}
    color_a      = _NEON_PURPLE if dark else "#3b82f6"
    color_b      = _NEON_RED    if dark else "#f97316"

    def _dir(v): return "increased" if v >= 0 else "decreased"

    hover_a, hover_b = [], []
    for _, row in top.iterrows():
        feat  = row["feat_clean"]
        sa, sb, diff = float(row["shap_a"]), float(row["shap_b"]), float(row["diff"])
        agree = "models agree on direction" if (sa >= 0) == (sb >= 0) else "models disagree on direction"
        hover_a.append(
            f"<b>{feat}</b><br>"
            f"{label_a}: {_dir(sa)} forecast by {abs(sa):.1f} units<br>"
            f"{label_b}: {_dir(sb)} forecast by {abs(sb):.1f} units<br>"
            f"Gap: {abs(diff):.1f} units — {agree}"
        )
        hover_b.append(
            f"<b>{feat}</b><br>"
            f"{label_b}: {_dir(sb)} forecast by {abs(sb):.1f} units<br>"
            f"{label_a}: {_dir(sa)} forecast by {abs(sa):.1f} units<br>"
            f"Gap: {abs(diff):.1f} units — {agree}"
        )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=top["feat_clean"], x=top["shap_a"], orientation="h",
        name=label_a, marker_color=color_a, opacity=0.85,
        customdata=hover_a,
        hovertemplate="%{customdata}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=top["feat_clean"], x=top["shap_b"], orientation="h",
        name=label_b, marker_color=color_b, opacity=0.85,
        customdata=hover_b,
        hovertemplate="%{customdata}<extra></extra>",
    ))
    fig.update_layout(
        **layout_extra,
        barmode="group",
        title={"text": f"SHAP Values — {label_a} vs {label_b}", "font": {"size": 13}},
        margin={"t": 40, "b": 20, "l": 20, "r": 20},
        xaxis_title="Impact on forecast (units)", yaxis_automargin=True,
        legend={"orientation": "h", "y": -0.15},
    )
    return fig


def _comp_nlg(sku_id: str, label_a: str, label_b: str, model_type_b: str) -> html.Div:
    try:
        diff_df = _get_model_comp_df(sku_id, model_type_b)
    except Exception as exc:
        return html.P(f"Comparison unavailable: {exc}", className="text-muted small")

    pred_a    = diff_df.attrs.get("pred_a", 0)
    pred_b    = diff_df.attrs.get("pred_b", 0)
    delta     = pred_a - pred_b
    direction = "higher" if delta >= 0 else "lower"
    top1      = diff_df.iloc[0]
    top1_feat = _clean(top1["feature"])
    top1_diff = float(top1["diff"])
    weights   = f"weights it more heavily" if top1_diff > 0 else "weights it less heavily"

    return html.Div([
        html.Hr(style={"margin": "12px 0"}),
        _section_label("Forecast Difference"),
        html.P([
            html.Strong(label_a), f": {pred_a:.1f} units vs ",
            html.Strong(label_b), f": {pred_b:.1f} units — ",
            html.Strong(f"{abs(delta):.1f} units {direction}"),
            f" for {label_a}.",
        ], style={"fontSize": "13px"}, className="mb-2"),
        _section_label("Biggest Reasoning Difference"),
        html.P(
            f"'{top1_feat}' is where the models disagree most ({abs(top1_diff):.2f} units). "
            f"{label_a} {weights} compared to {label_b}.",
            style={"fontSize": "13px"}, className="mb-0",
        ),
    ])

# ── stockout risk ─────────────────────────────────────────────────────────────

def _stockout_risk_table(dark: bool = False) -> html.Div:
    df = ds.cards_df
    if df is None or df.empty:
        return html.P("No data available.", className="text-muted")

    # Only show SKUs where a reorder is actually triggered
    df = df[df["trigger_reorder"] == True]

    if df.empty:
        return html.P("No SKUs currently require a reorder.", className="text-muted")

    display = df[["sku_id", "stock_on_hand", "days_of_stock",
                  "reorder_point", "reorder_qty", "urgency"]].copy()

    urgency_order = {"CRITICAL": 0, "HIGH": 1, "LOW": 2}
    display["_sort"] = display["urgency"].map(urgency_order)
    display = display.sort_values(["_sort", "days_of_stock"], ascending=[True, True]).drop(columns="_sort").head(30)

    columns = [
        {"name": "SKU",           "id": "sku_id"},
        {"name": "Stock on Hand", "id": "stock_on_hand",  "type": "numeric"},
        {"name": "Days of Stock", "id": "days_of_stock",  "type": "numeric", "format": {"specifier": ".1f"}},
        {"name": "Reorder Point", "id": "reorder_point",  "type": "numeric", "format": {"specifier": ".1f"}},
        {"name": "Order Qty",     "id": "reorder_qty",    "type": "numeric"},
        {"name": "Risk Level",    "id": "urgency"},
    ]

    if dark:
        s_header = {"backgroundColor": "#0A0C1A", "color": "#FFFFFF", "fontWeight": "bold",
                    "fontSize": "13px", "border": "1px solid rgba(95,1,251,0.3)"}
        s_data   = {"backgroundColor": "#0F1225", "color": "#FFFFFF",
                    "border": "1px solid rgba(95,1,251,0.2)"}
        s_cell   = {"fontSize": "13px", "padding": "6px 10px", "textAlign": "left",
                    "backgroundColor": "#0F1225", "color": "#FFFFFF"}
        s_cond   = [
            {"if": {"filter_query": '{urgency} = "CRITICAL"'},
             "backgroundColor": "rgba(255,45,85,0.12)", "color": "#FF2D55"},
            {"if": {"filter_query": '{urgency} = "HIGH"'},
             "backgroundColor": "rgba(255,184,0,0.12)", "color": "#FFB800"},
            {"if": {"filter_query": '{urgency} = "LOW"'},
             "backgroundColor": "rgba(0,255,135,0.08)", "color": "#00FF87"},
        ]
    else:
        s_header = {"backgroundColor": "#343a40", "color": "white",
                    "fontWeight": "bold", "fontSize": "13px"}
        s_data   = {}
        s_cell   = {"fontSize": "13px", "padding": "6px 10px", "textAlign": "left"}
        s_cond   = [
            {"if": {"filter_query": '{urgency} = "CRITICAL"'},
             "backgroundColor": "#f8d7da", "color": "#842029"},
            {"if": {"filter_query": '{urgency} = "HIGH"'},
             "backgroundColor": "#fff3cd", "color": "#664d03"},
            {"if": {"filter_query": '{urgency} = "LOW"'},
             "backgroundColor": "#d1e7dd", "color": "#0a3622"},
        ]

    return html.Div([
        html.P("SKUs where a reorder is triggered (Order Now = True), sorted by urgency then days of stock. "
               "CRITICAL = stock runs out before next delivery. HIGH = at reorder point but stock covers lead time.",
               className="text-muted small mb-3"),
        dash_table.DataTable(
            columns=columns,
            data=display.to_dict("records"),
            page_size=15,
            sort_action="native",
            style_table={"overflowX": "auto"},
            style_header=s_header,
            style_data=s_data,
            style_cell=s_cell,
            style_data_conditional=s_cond,
        ),
    ])

# ── callback registration ─────────────────────────────────────────────────────

def register_sku_callbacks(app) -> None:

    @app.callback(
        Output("sku-date-context", "children"),
        Input("model-store",      "data"),
        Input("model-type-store", "data"),
    )
    def update_sku_date_context(_mk, _mt):
        return _date_badge()

    @app.callback(
        Output("sku-selector", "options"),
        Output("sku-selector", "value"),
        Input("model-store",      "data"),
        Input("model-type-store", "data"),
    )
    def update_sku_options(_mk, _mt):
        _temporal_cache.clear()
        _model_comp_cache.clear()
        _model_b_cache.clear()
        _stockout_cache.clear()
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
        Input("sku-selector",     "value"),
        Input("model-store",      "data"),
        Input("model-type-store", "data"),
        Input("theme-store",      "data"),
    )
    def update_sku_page(sku_id: str, _mk, _mt, theme: str):
        if not sku_id:
            return [], {}, {}, "", {}, "", ""
        dark       = theme == "dark"
        global_std = float(ds.sku_std.mean()) if ds.sku_std is not None else 1.0
        std_v      = float(ds.sku_std.get(sku_id, global_std)) if ds.sku_std is not None else global_std
        iv         = ds.forecasts.get(sku_id, {"q10": 0, "q50": 0, "q90": 0, "width": 0})
        card_row   = ds.cards_df[ds.cards_df["sku_id"] == sku_id]
        card       = card_row.iloc[0].to_dict() if not card_row.empty else {}

        return (
            _summary_cards(iv, card, std_v, dark=dark),
            _forecast_figure(sku_id, std_v, dark=dark),
            _shap_figure(sku_id, dark=dark),
            _nlg_content(sku_id, iv, std_v),
            _temporal_line_chart(sku_id, dark=dark),
            _temporal_summary(sku_id),
            _temporal_nlg(sku_id),
        )

    # ── Censored demand + SHAP distortion ────────────────────────────────────
    @app.callback(
        Output("sku-censored-chart", "figure"),
        Output("sku-stockout-nlg",   "children"),
        Input("sku-selector",     "value"),
        Input("model-store",      "data"),
        Input("model-type-store", "data"),
        Input("theme-store",      "data"),
    )
    def update_censored_demand(sku_id: str, _mk, _mt, theme: str):
        if not sku_id:
            return {}, []
        dark = theme == "dark"
        return (
            _censored_demand_figure(sku_id, dark=dark),
            _stockout_sku_nlg(sku_id),
        )

    # ── Comparative SHAP — Model A label ────────────────────────────────────
    @app.callback(
        Output("comp-model-a-display", "children"),
        Input("model-type-store", "data"),
    )
    def update_comp_model_a_display(model_type: str):
        label = ds.MODEL_TYPES.get(model_type or "lightgbm", model_type)
        return dbc.Badge(label, color="primary", style={"fontSize": "13px"})

    # ── Comparative SHAP — Model B dropdown ──────────────────────────────────
    @app.callback(
        Output("comp-model-b", "options"),
        Output("comp-model-b", "value"),
        Input("model-type-store", "data"),
        Input("url",              "pathname"),
    )
    def update_comp_model_b(model_type: str, pathname: str):
        if pathname != "/sku-explorer":
            return [], None
        _model_comp_cache.clear()
        _model_b_cache.clear()
        opts    = [{"label": lbl, "value": key}
                   for key, lbl in ds.MODEL_TYPES.items()
                   if key != model_type and key != "neural_network"]
        default = opts[0]["value"] if opts else None
        return opts, default

    # ── Comparative SHAP — charts ─────────────────────────────────────────────
    @app.callback(
        Output("comp-diff-chart", "figure"),
        Output("comp-side-chart", "figure"),
        Output("comp-nlg",        "children"),
        Input("sku-selector",     "value"),
        Input("comp-model-b",     "value"),
        Input("model-type-store", "data"),
        Input("theme-store",      "data"),
        Input("url",              "pathname"),
    )
    def update_comp_charts(sku_id: str, model_type_b: str, model_type_a: str,
                           theme: str, pathname: str):
        if pathname != "/sku-explorer" or not sku_id or not model_type_b:
            return {}, {}, []
        dark    = theme == "dark"
        label_a = ds.MODEL_TYPES.get(model_type_a, model_type_a)
        label_b = ds.MODEL_TYPES.get(model_type_b, model_type_b)
        return (
            _comp_diff_figure(sku_id, label_a, label_b, model_type_b, dark=dark),
            _comp_side_figure(sku_id, label_a, label_b, model_type_b, dark=dark),
            _comp_nlg(sku_id, label_a, label_b, model_type_b),
        )

