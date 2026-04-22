"""
Callbacks for the Explanations page (Q2, Q5, Q7, Q8).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, html, dash_table
import dash_bootstrap_components as dbc

import app.data_store as ds
from xai.comparative_shap import find_similar_skus, compare_skus_aggregate
from xai.stockout_analysis import analyze_zero_lag_shap_impact, estimate_censored_demand


# ── neon palette (shared with sku_callbacks) ──────────────────────────────────

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


# ── helpers ───────────────────────────────────────────────────────────────────

def _clean(name: str) -> str:
    return name.replace("num__", "").replace("cat__", "")


def _section_label(text: str) -> html.P:
    return html.P(
        text,
        className="mb-1 mt-0 fw-bold text-uppercase",
        style={"fontSize": "11px", "letterSpacing": "0.5px", "color": "#6c757d"},
    )


def _bullet_list(items: list) -> html.Ul:
    return html.Ul(items, style={"fontSize": "13px", "paddingLeft": "18px", "marginBottom": "0"})


# ── Q2 — Global SHAP chart ────────────────────────────────────────────────────

def _global_shap_figure(dark: bool = False) -> go.Figure:
    df = ds.global_shap_df
    if df is None or df.empty:
        return go.Figure()

    top = df.head(15).copy()
    top["feature_clean"] = top["feature"].apply(_clean)
    top = top.sort_values("mean_abs_shap", ascending=True)

    bar_color    = _NEON_PURPLE  if dark else "#3b82f6"
    text_color   = _NEON_FONT    if dark else "#333333"
    layout_extra = _NEON_LAYOUT  if dark else {"template": "plotly_white"}

    fig = go.Figure(go.Bar(
        x=top["mean_abs_shap"],
        y=top["feature_clean"],
        orientation="h",
        marker_color=bar_color,
        text=[f"{v:.3f}" for v in top["mean_abs_shap"]],
        textposition="outside",
        textfont={"color": text_color},
        hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>",
    ))

    fig.update_layout(
        **layout_extra,
        margin={"t": 20, "b": 20, "l": 20, "r": 70},
        xaxis_title="Mean |SHAP value|",
        yaxis_automargin=True,
        showlegend=False,
    )
    return fig


# ── Q2 — Global SHAP NLG card ─────────────────────────────────────────────────

def _global_shap_nlg() -> html.Div:
    df = ds.global_shap_df
    if df is None or df.empty:
        return html.P("No data available.", className="text-muted")

    top5       = df.head(5)
    top_names  = [_clean(f) for f in top5["feature"]]
    top_vals   = top5["mean_abs_shap"].tolist()
    total_shap = float(df["mean_abs_shap"].sum())
    top5_share = float(top5["mean_abs_shap"].sum()) / total_shap * 100 if total_shap > 0 else 0.0

    lag_feats   = [f for f in top_names if "lag" in f or "roll" in f]
    price_feats = [f for f in top_names if "price" in f or "discount" in f]
    event_feats = [f for f in top_names if "event" in f]

    if len(lag_feats) >= 2:
        driver_txt = "recent sales history"
    elif len(price_feats) >= 2:
        driver_txt = "pricing and promotion signals"
    elif len(event_feats) >= 2:
        driver_txt = "event and holiday indicators"
    else:
        driver_txt = "a mix of lag, price, and contextual signals"

    bottom5_names = [_clean(f) for f in df.tail(5)["feature"]]

    feature_items = [
        html.Li([
            html.Strong(f"{i}. {name}"),
            html.Span(
                f"  ({val:.3f})",
                className="text-muted",
                style={"fontSize": "12px"},
            ),
        ])
        for i, (name, val) in enumerate(zip(top_names, top_vals), 1)
    ]

    return html.Div([
        _section_label("Key Driver"),
        html.P(
            f"Demand forecasts are primarily driven by {driver_txt}.",
            style={"fontSize": "13px"},
            className="mb-2",
        ),

        html.Hr(style={"margin": "10px 0"}),

        _section_label("Top 5 Features"),
        _bullet_list(feature_items),

        html.Hr(style={"margin": "10px 0"}),

        _section_label("SHAP Coverage"),
        html.P(
            [
                f"Top 5 features account for ",
                html.Strong(f"{top5_share:.1f}%"),
                f" of total model influence across all {len(df)} features.",
            ],
            style={"fontSize": "13px"},
            className="mb-2",
        ),

        html.Hr(style={"margin": "10px 0"}),

        _section_label("Lowest Impact"),
        html.P(
            ", ".join(bottom5_names),
            className="text-muted mb-0",
            style={"fontSize": "12px"},
        ),
    ])


# ── Q8 — Audit summary cards ──────────────────────────────────────────────────

def _audit_summary_cards(dark: bool = False) -> html.Div:
    df = ds.feature_audit_df
    if df is None or df.empty:
        return html.Div()

    total   = len(df)
    flagged = int((df["flag"] != "ok").sum())
    ok      = total - flagged
    total_colour = "light" if dark else "dark"

    def _stat(value, label, colour):
        return dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.H3(str(value), className=f"text-{colour} mb-0"),
                    html.Small(label, className="text-muted"),
                ], style={"padding": "14px 18px"}),
            ], className="shadow-sm text-center"),
            xs=6, md=4,
        )

    return dbc.Row([
        _stat(total,   "Total features", total_colour),
        _stat(flagged, "Flagged",        "danger"),
        _stat(ok,      "Healthy (ok)",   "success"),
    ], className="g-3")


# ── Q8 — Audit DataTable ──────────────────────────────────────────────────────

_AUDIT_COL_TOOLTIPS = {
    "feature":     "The name of the input feature used by the model",
    "std":         "Standard deviation of this feature across the test set — how much the values vary. Zero means the feature is constant and carries no information",
    "missing_pct": "Percentage of rows where this feature's value is missing (NaN). Values above 10% are flagged",
    "flag":        "Data quality verdict. ok = healthy. zero_variance = constant column. mostly_zero = over 80% of values are 0. high_corr = nearly identical to another feature (>0.95 correlation). high_missing = too many missing values",
}


def _audit_table(dark: bool = False, df: pd.DataFrame = None) -> dash_table.DataTable:
    if df is None:
        df = ds.feature_audit_df
    if df is None or df.empty:
        return html.P("No audit data available.", className="text-muted")

    display = df.copy()
    display["feature"] = display["feature"].apply(_clean)

    columns = [
        {"name": "Feature",   "id": "feature"},
        {"name": "Std Dev",   "id": "std",         "type": "numeric",
         "format": {"specifier": ".4f"}},
        {"name": "Missing %", "id": "missing_pct", "type": "numeric",
         "format": {"specifier": ".1f"}},
        {"name": "Status",    "id": "flag"},
    ]

    if dark:
        s_header = {
            "backgroundColor": "#0A0C1A", "color": "#FFFFFF",
            "fontWeight": "bold", "fontSize": "13px",
            "border": "1px solid rgba(95,1,251,0.3)",
            "textDecoration": "underline", "textDecorationStyle": "dashed",
            "textDecorationColor": "rgba(95,1,251,0.6)", "cursor": "help",
        }
        s_data = {
            "backgroundColor": "#0F1225", "color": "#FFFFFF",
            "border": "1px solid rgba(95,1,251,0.2)",
        }
        s_cell = {
            "fontSize": "13px", "padding": "6px 10px", "textAlign": "left",
            "backgroundColor": "#0F1225", "color": "#FFFFFF",
        }
        s_cond = [
            {"if": {"filter_query": '{flag} = "ok"'},
             "backgroundColor": "rgba(0,255,135,0.08)", "color": "#00FF87"},
            {"if": {"filter_query": '{flag} contains "zero_variance"'},
             "backgroundColor": "rgba(255,45,85,0.12)", "color": "#FF2D55"},
            {"if": {"filter_query": '{flag} contains "high_missing"'},
             "backgroundColor": "rgba(255,45,85,0.12)", "color": "#FF2D55"},
            {"if": {"filter_query": '{flag} contains "high_corr"'},
             "backgroundColor": "rgba(255,184,0,0.12)", "color": "#FFB800"},
            {"if": {"filter_query": '{flag} contains "mostly_zero"'},
             "backgroundColor": "rgba(255,184,0,0.12)", "color": "#FFB800"},
        ]
    else:
        s_header = {
            "backgroundColor": "#343a40", "color": "white",
            "fontWeight": "bold", "fontSize": "13px",
            "textDecoration": "underline", "textDecorationStyle": "dashed",
            "textDecorationColor": "rgba(255,255,255,0.5)", "cursor": "help",
        }
        s_data = {}
        s_cell = {"fontSize": "13px", "padding": "6px 10px", "textAlign": "left"}
        s_cond = [
            {"if": {"filter_query": '{flag} = "ok"'},
             "backgroundColor": "#d1e7dd", "color": "#0a3622"},
            {"if": {"filter_query": '{flag} contains "zero_variance"'},
             "backgroundColor": "#f8d7da", "color": "#842029"},
            {"if": {"filter_query": '{flag} contains "high_missing"'},
             "backgroundColor": "#f8d7da", "color": "#842029"},
            {"if": {"filter_query": '{flag} contains "high_corr"'},
             "backgroundColor": "#fff3cd", "color": "#664d03"},
            {"if": {"filter_query": '{flag} contains "mostly_zero"'},
             "backgroundColor": "#fff3cd", "color": "#664d03"},
        ]

    return dash_table.DataTable(
        id="audit-datatable",
        columns=columns,
        data=display[["feature", "std", "missing_pct", "flag"]].to_dict("records"),
        page_size=20,
        sort_action="native",
        tooltip_header={col["id"]: _AUDIT_COL_TOOLTIPS.get(col["id"], "") for col in columns},
        tooltip_delay=0,
        tooltip_duration=None,
        style_table={"overflowX": "auto"},
        style_header=s_header,
        style_data=s_data,
        style_cell=s_cell,
        style_data_conditional=s_cond,
    )


# ── Q5 — Comparative SHAP ────────────────────────────────────────────────────

_comparative_cache: dict[tuple, pd.DataFrame] = {}


def _get_comp_df(sku_a: str, sku_b: str) -> pd.DataFrame:
    key = (sku_a, sku_b)
    if key not in _comparative_cache:
        _comparative_cache[key] = compare_skus_aggregate(
            ds.model, ds.X_test, ds.test_df["item_id"], sku_a, sku_b
        )
    return _comparative_cache[key]


def _get_similar_skus(sku_a: str, n: int = 5) -> list[str]:
    mask = ds.test_df["item_id"] == sku_a
    if not mask.any():
        return []
    pos_idx = int(np.where(mask.values)[0][0])
    similar_idxs = find_similar_skus(
        ds.X_test, pos_idx, n=n, item_ids=ds.test_df["item_id"]
    )
    seen, result = set(), []
    for i in similar_idxs:
        sku = ds.test_df.loc[i, "item_id"]
        if sku not in seen:
            seen.add(sku)
            result.append(sku)
    return result


def _comp_diff_figure(sku_a: str, sku_b: str, dark: bool = False) -> go.Figure:
    try:
        diff_df = _get_comp_df(sku_a, sku_b)
    except Exception:
        return go.Figure()

    top = diff_df.head(12).copy().sort_values("diff", ascending=True)
    colors = [_NEON_RED if v < 0 else _NEON_PURPLE for v in top["diff"]] if dark \
             else ["#ef4444" if v < 0 else "#3b82f6" for v in top["diff"]]
    text_color   = _NEON_FONT  if dark else "#333333"
    layout_extra = _NEON_LAYOUT if dark else {"template": "plotly_white"}

    fig = go.Figure(go.Bar(
        x=top["diff"],
        y=top["feature"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.2f}" for v in top["diff"]],
        textposition="outside",
        textfont={"color": text_color},
        hovertemplate="<b>%{y}</b><br>Diff (A−B): %{x:+.3f}<extra></extra>",
    ))
    fig.update_layout(
        **layout_extra,
        title={"text": f"SHAP Difference  ({sku_a} − {sku_b})", "font": {"size": 13}},
        margin={"t": 40, "b": 20, "l": 20, "r": 70},
        xaxis_title="SHAP diff (A − B)",
        yaxis_automargin=True,
        showlegend=False,
    )
    return fig


def _comp_side_figure(sku_a: str, sku_b: str, dark: bool = False) -> go.Figure:
    try:
        diff_df = _get_comp_df(sku_a, sku_b)
    except Exception:
        return go.Figure()

    top = diff_df.head(10)
    layout_extra = _NEON_LAYOUT if dark else {"template": "plotly_white"}
    color_a = _NEON_PURPLE if dark else "#3b82f6"
    color_b = _NEON_RED    if dark else "#f97316"

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=top["feature"], x=top["shap_a"], orientation="h",
        name=sku_a, marker_color=color_a, opacity=0.85,
        hovertemplate="<b>%{y}</b><br>SHAP A: %{x:+.3f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=top["feature"], x=top["shap_b"], orientation="h",
        name=sku_b, marker_color=color_b, opacity=0.85,
        hovertemplate="<b>%{y}</b><br>SHAP B: %{x:+.3f}<extra></extra>",
    ))
    fig.update_layout(
        **layout_extra,
        barmode="group",
        title={"text": "SHAP Values Side-by-Side", "font": {"size": 13}},
        margin={"t": 40, "b": 20, "l": 20, "r": 20},
        xaxis_title="SHAP value",
        yaxis_automargin=True,
        legend={"orientation": "h", "y": -0.15},
    )
    return fig


def _comp_nlg(sku_a: str, sku_b: str) -> html.Div:
    try:
        diff_df = _get_comp_df(sku_a, sku_b)
    except Exception as exc:
        return html.P(f"Comparison unavailable: {exc}", className="text-muted small")

    pred_a = diff_df.attrs.get("pred_a", 0)
    pred_b = diff_df.attrs.get("pred_b", 0)
    delta  = pred_a - pred_b
    direction = "higher" if delta >= 0 else "lower"
    top5 = diff_df.head(5)
    top1_feat = _clean(top5.iloc[0]["feature"])
    top1_diff = float(top5.iloc[0]["diff"])

    driver_dir = "pushes forecast UP for SKU A" if top1_diff > 0 else "pushes forecast DOWN for SKU A"

    feature_items = [
        html.Li([
            html.Strong(_clean(row["feature"])),
            html.Span(
                f" — A: {row['shap_a']:+.2f} · B: {row['shap_b']:+.2f} · diff: {row['diff']:+.2f}",
                className="text-muted", style={"fontSize": "12px"},
            ),
        ])
        for _, row in top5.iterrows()
    ]

    return html.Div([
        html.Hr(style={"margin": "12px 0"}),
        _section_label("Forecast Difference"),
        html.P([
            html.Strong(sku_a), f": {pred_a:.1f} units vs ",
            html.Strong(sku_b), f": {pred_b:.1f} units — ",
            html.Strong(f"{abs(delta):.1f} units {direction}"), f" for {sku_a}.",
        ], style={"fontSize": "13px"}, className="mb-2"),

        _section_label("Top 5 Drivers of Difference"),
        _bullet_list(feature_items),

        html.Hr(style={"margin": "10px 0"}),
        _section_label("Primary Driver"),
        html.P(
            f"'{top1_feat}' is the biggest differentiator ({abs(top1_diff):.2f} SHAP units) "
            f"— it {driver_dir}.",
            style={"fontSize": "13px"}, className="mb-0",
        ),
    ])


# ── Q7 — Stockout Risk ────────────────────────────────────────────────────────

_stockout_shap_cache:     dict[str, pd.DataFrame] = {}
_stockout_censored_cache: dict[str, pd.DataFrame] = {}


def _get_stockout_impact(sku_id: str) -> pd.DataFrame:
    if sku_id not in _stockout_shap_cache:
        _stockout_shap_cache[sku_id] = analyze_zero_lag_shap_impact(
            ds.model, ds.X_test, ds.test_df["item_id"], sku_id, lag_col=ds.LAG_COL
        )
    return _stockout_shap_cache[sku_id]


def _get_censored_df(sku_id: str) -> pd.DataFrame:
    if sku_id not in _stockout_censored_cache:
        _stockout_censored_cache[sku_id] = estimate_censored_demand(
            ds.full_df, sku_id,
            sku_col="item_id", sales_col=ds.TARGET,
            date_col="date",   lag_col=ds.LAG_COL,
        )
    return _stockout_censored_cache[sku_id]


_STOCKOUT_COL_TOOLTIPS = {
    "item_id":           "The unique product identifier (SKU)",
    "total_periods":     "Total number of time periods observed for this SKU across train and test data",
    "stockout_periods":  "Number of periods where sales_lag_7 ≤ 0.5 — used as a proxy for suspected stockout (no inventory data available)",
    "stockout_rate_pct": "Percentage of periods flagged as suspected stockouts. HIGH > 50%, MODERATE 20–50%, LOW < 20%",
    "risk_level":        "Overall stockout risk classification based on stockout rate. HIGH means the model's demand signal is likely heavily distorted by past stockouts",
}


def _stockout_global_table(dark: bool = False, df: pd.DataFrame = None) -> html.Div:
    if df is None:
        df = ds.stockout_risk_df
    if df is None or df.empty:
        return html.P("No stockout data available.", className="text-muted")

    display = df.head(20).copy()
    display["stockout_rate_pct"] = (display["stockout_rate"] * 100).round(1)
    display["risk_level"] = display["stockout_rate"].apply(
        lambda r: "HIGH" if r > 0.5 else "MODERATE" if r > 0.2 else "LOW"
    )

    columns = [
        {"name": "SKU",              "id": "item_id"},
        {"name": "Total Periods",    "id": "total_periods",    "type": "numeric"},
        {"name": "Stockout Periods", "id": "stockout_periods", "type": "numeric"},
        {"name": "Stockout Rate %",  "id": "stockout_rate_pct","type": "numeric",
         "format": {"specifier": ".1f"}},
        {"name": "Risk Level",       "id": "risk_level"},
    ]

    if dark:
        s_header = {
            "backgroundColor": "#0A0C1A", "color": "#FFFFFF",
            "fontWeight": "bold", "fontSize": "13px",
            "border": "1px solid rgba(95,1,251,0.3)",
            "textDecoration": "underline", "textDecorationStyle": "dashed",
            "textDecorationColor": "rgba(95,1,251,0.6)", "cursor": "help",
        }
        s_data = {"backgroundColor": "#0F1225", "color": "#FFFFFF",
                  "border": "1px solid rgba(95,1,251,0.2)"}
        s_cell = {"fontSize": "13px", "padding": "6px 10px", "textAlign": "left",
                  "backgroundColor": "#0F1225", "color": "#FFFFFF"}
        s_cond = [
            {"if": {"filter_query": '{risk_level} = "HIGH"'},
             "backgroundColor": "rgba(255,45,85,0.12)", "color": "#FF2D55"},
            {"if": {"filter_query": '{risk_level} = "MODERATE"'},
             "backgroundColor": "rgba(255,184,0,0.12)", "color": "#FFB800"},
            {"if": {"filter_query": '{risk_level} = "LOW"'},
             "backgroundColor": "rgba(0,255,135,0.08)", "color": "#00FF87"},
        ]
    else:
        s_header = {
            "backgroundColor": "#343a40", "color": "white",
            "fontWeight": "bold", "fontSize": "13px",
            "textDecoration": "underline", "textDecorationStyle": "dashed",
            "textDecorationColor": "rgba(255,255,255,0.5)", "cursor": "help",
        }
        s_data = {}
        s_cell = {"fontSize": "13px", "padding": "6px 10px", "textAlign": "left"}
        s_cond = [
            {"if": {"filter_query": '{risk_level} = "HIGH"'},
             "backgroundColor": "#f8d7da", "color": "#842029"},
            {"if": {"filter_query": '{risk_level} = "MODERATE"'},
             "backgroundColor": "#fff3cd", "color": "#664d03"},
            {"if": {"filter_query": '{risk_level} = "LOW"'},
             "backgroundColor": "#d1e7dd", "color": "#0a3622"},
        ]

    return dash_table.DataTable(
        columns=columns,
        data=display[["item_id", "total_periods", "stockout_periods",
                       "stockout_rate_pct", "risk_level"]].to_dict("records"),
        page_size=10,
        sort_action="native",
        tooltip_header={col["id"]: _STOCKOUT_COL_TOOLTIPS.get(col["id"], "") for col in columns},
        tooltip_delay=0,
        tooltip_duration=None,
        style_table={"overflowX": "auto"},
        style_header=s_header,
        style_data=s_data,
        style_cell=s_cell,
        style_data_conditional=s_cond,
    )


def _stockout_pred_figure(sku_id: str, dark: bool = False) -> go.Figure:
    try:
        impact_df = _get_stockout_impact(sku_id)
    except Exception:
        return go.Figure()

    colors = [_NEON_RED if z else _NEON_PURPLE for z in impact_df["is_zero_lag"]] if dark \
             else ["#ef4444" if z else "#3b82f6" for z in impact_df["is_zero_lag"]]
    layout_extra = _NEON_LAYOUT if dark else {"template": "plotly_white"}

    fig = go.Figure(go.Bar(
        x=list(range(len(impact_df))),
        y=impact_df["prediction"],
        marker_color=colors,
        hovertemplate="Period %{x}<br>Forecast: %{y:.2f}<extra></extra>",
    ))
    fig.update_layout(
        **layout_extra,
        title={"text": "Forecast per Period (red = suspected stockout)", "font": {"size": 13}},
        margin={"t": 40, "b": 30, "l": 50, "r": 20},
        xaxis_title="Period index (test set)",
        yaxis_title="Forecast (units)",
        showlegend=False,
    )
    return fig


def _stockout_shap_figure(sku_id: str, dark: bool = False) -> go.Figure:
    try:
        impact_df = _get_stockout_impact(sku_id)
    except Exception:
        return go.Figure()

    colors = [_NEON_RED if z else _NEON_PURPLE for z in impact_df["is_zero_lag"]] if dark \
             else ["#ef4444" if z else "#3b82f6" for z in impact_df["is_zero_lag"]]
    layout_extra = _NEON_LAYOUT if dark else {"template": "plotly_white"}

    fig = go.Figure(go.Bar(
        x=list(range(len(impact_df))),
        y=impact_df["shap_lag7"],
        marker_color=colors,
        hovertemplate="Period %{x}<br>SHAP lag: %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="gray", line_width=1)
    fig.update_layout(
        **layout_extra,
        title={"text": "SHAP of sales_lag_7 (red = suspected stockout)", "font": {"size": 13}},
        margin={"t": 40, "b": 30, "l": 50, "r": 20},
        xaxis_title="Period index",
        yaxis_title="SHAP value",
        showlegend=False,
    )
    return fig


def _stockout_censored_figure(sku_id: str, dark: bool = False) -> go.Figure:
    try:
        cdf = _get_censored_df(sku_id)
    except Exception:
        return go.Figure()

    layout_extra = _NEON_LAYOUT if dark else {"template": "plotly_white"}
    act_color   = _NEON_FONT  if dark else "#1e293b"
    est_color   = _NEON_RED   if dark else "#ef4444"
    shade_color = "rgba(255,45,85,0.18)" if dark else "rgba(239,68,68,0.12)"

    fig = go.Figure()

    # Shade each contiguous suspected-stockout block
    if "is_potential_stockout" in cdf.columns:
        s      = cdf["is_potential_stockout"].astype(bool)
        run_id = (s != s.shift()).cumsum()
        for _, grp in cdf[s].groupby(run_id[s]):
            x0 = grp["date"].iloc[0]
            x1 = grp["date"].iloc[-1]
            if x0 == x1:
                x1 = x1 + pd.Timedelta(days=7)
            fig.add_vrect(
                x0=x0, x1=x1,
                fillcolor=shade_color,
                layer="below",
                line_width=0,
            )

    fig.add_trace(go.Scatter(
        x=cdf["date"], y=cdf[ds.TARGET],
        name="Actual sales", mode="lines",
        line={"color": act_color, "width": 2},
    ))
    fig.add_trace(go.Scatter(
        x=cdf["date"], y=cdf["estimated_demand"],
        name="Estimated (if no stockout)", mode="lines",
        line={"color": est_color, "width": 2, "dash": "dash"},
    ))
    fig.update_layout(
        **layout_extra,
        title={"text": "Actual vs Estimated Demand (red shading = suspected stockout periods)",
               "font": {"size": 13}},
        margin={"t": 40, "b": 40, "l": 50, "r": 20},
        xaxis_title="Date",
        yaxis_title="Units",
        legend={"orientation": "h", "y": -0.2},
        hovermode="x unified",
    )
    return fig


def _stockout_nlg(sku_id: str) -> html.Div:
    try:
        impact_df = _get_stockout_impact(sku_id)
        cdf       = _get_censored_df(sku_id)
    except Exception as exc:
        return html.P(f"Analysis unavailable: {exc}", className="text-muted small")

    n_stockout   = int(impact_df["is_zero_lag"].sum())
    total        = len(impact_df)
    total_lost   = float(cdf["demand_gap"].sum())
    zero_mask    = impact_df["is_zero_lag"] == 1
    avg_dist     = float(impact_df.loc[zero_mask, "shap_distortion"].mean()) if zero_mask.any() else 0.0

    severity     = "HIGH" if abs(avg_dist) > 5 else "MODERATE" if abs(avg_dist) > 2 else "LOW"
    sev_colour   = {"HIGH": "danger", "MODERATE": "warning", "LOW": "success"}.get(severity, "secondary")
    stockout_pct = n_stockout / total * 100 if total > 0 else 0.0

    return html.Div([
        html.Hr(style={"margin": "12px 0"}),
        _section_label("Stockout Risk Summary"),
        _bullet_list([
            html.Li([
                f"Suspected stockout periods: ",
                html.Strong(f"{n_stockout} / {total}"),
                f" ({stockout_pct:.0f}% of test history)",
            ]),
            html.Li([
                "SHAP distortion severity: ",
                dbc.Badge(severity, color=sev_colour, className="ms-1"),
                html.Span(f" (avg distortion: {avg_dist:.2f} units/period)",
                          className="text-muted ms-1", style={"fontSize": "12px"}),
            ]),
            html.Li(f"Estimated lost demand: {total_lost:.1f} units across stockout periods"),
        ]),
        html.Hr(style={"margin": "10px 0"}),
        _section_label("Recommendation"),
        html.P(
            "Treat forecasts for this SKU with caution — the model may have learned from "
            "artificially low sales. Consider enriching with actual inventory data to distinguish "
            "true zero demand from stockouts.",
            style={"fontSize": "13px"}, className="mb-0",
        ),
    ])


# ── Q9 — Model Reliability & Cold-Start ──────────────────────────────────────

def _reliability_stat_cards(dark: bool = False) -> html.Div:
    total_colour = "light" if dark else "dark"
    n_cold       = int(ds.cold_start_df["is_cold_start"].sum()) if ds.cold_start_df is not None else 0
    n_total      = len(ds.cold_start_df) if ds.cold_start_df is not None else 0
    n_estab      = n_total - n_cold
    coverage     = ds.interval_coverage or 0.0
    cov_colour   = "success" if coverage >= 75 else "warning" if coverage >= 65 else "danger"
    high_n       = ds.confidence_dist.get("high", 0)

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

    dash_style = {
        "textDecoration": "underline",
        "textDecorationStyle": "dashed",
        "textDecorationColor": "rgba(108,117,125,0.7)",
        "cursor": "help",
    }

    coverage_col = dbc.Col(
        [
            dbc.Card([
                dbc.CardBody([
                    html.H3(f"{coverage:.1f}%", className=f"text-{cov_colour} mb-0"),
                    html.Small(
                        "Interval Coverage (target 80%)",
                        id="coverage-label-tip",
                        className="text-muted",
                        style=dash_style,
                    ),
                ], style={"padding": "14px 18px"}),
            ], className="shadow-sm text-center"),
            dbc.Tooltip(
                [
                    html.P(
                        "What % of actual sales fell inside the model's predicted 80% range "
                        "(between q10 and q90). A well-calibrated model should hit ~80%.",
                        className="mb-2",
                        style={"fontSize": "12px"},
                    ),
                    html.Ul([
                        html.Li("≈ 80% → model uncertainty is well-calibrated", style={"fontSize": "12px"}),
                        html.Li("Much > 80% → intervals are too wide (over-cautious)", style={"fontSize": "12px"}),
                        html.Li("Much < 80% → model is under-estimating uncertainty", style={"fontSize": "12px"}),
                    ], className="mb-0 ps-3"),
                ],
                target="coverage-label-tip",
                placement="bottom",
                style={"maxWidth": "340px", "textAlign": "left"},
            ),
        ],
        xs=6, md=3,
    )

    return dbc.Row([
        coverage_col,
        _stat(n_cold,  "Cold-Start SKUs",            "danger" if n_cold > 0 else "success"),
        _stat(n_estab, "Established SKUs",            total_colour),
        _stat(high_n,  "High-Confidence Predictions", "success"),
    ], className="g-3")


def _confidence_dist_chart(dark: bool = False) -> go.Figure:
    dist         = ds.confidence_dist
    if not dist:
        return go.Figure()

    layout_extra = _NEON_LAYOUT if dark else {"template": "plotly_white"}
    labels       = ["high", "moderate", "low"]
    counts       = [dist.get(l, 0) for l in labels]
    total        = sum(counts) or 1

    if dark:
        colors = ["#00FF87", "#FFB800", _NEON_RED]
    else:
        colors = ["#198754", "#ffc107", "#dc3545"]

    text_color = _NEON_FONT if dark else "#333333"

    fig = go.Figure(go.Bar(
        x=[l.upper() for l in labels],
        y=counts,
        marker_color=colors,
        text=[f"{c:,}<br>({c/total*100:.1f}%)" for c in counts],
        textposition="outside",
        textfont={"color": text_color, "size": 12},
        hovertemplate="<b>%{x}</b><br>Count: %{y:,}<extra></extra>",
    ))
    fig.update_layout(
        **layout_extra,
        title={"text": "Confidence Distribution (all test predictions)", "font": {"size": 13}},
        margin={"t": 40, "b": 20, "l": 40, "r": 20},
        yaxis_title="# Predictions",
        showlegend=False,
    )
    return fig


def _subgroup_eval_table(dark: bool = False) -> html.Div:
    df = ds.subgroup_eval_df
    if df is None or df.empty:
        return html.P("No subgroup data available.", className="text-muted")

    columns = [
        {"name": "Category", "id": "group"},
        {"name": "N",        "id": "n",     "type": "numeric"},
        {"name": "MAE",      "id": "MAE",   "type": "numeric", "format": {"specifier": ".3f"}},
        {"name": "RMSE",     "id": "RMSE",  "type": "numeric", "format": {"specifier": ".3f"}},
        {"name": "SMAPE %",  "id": "SMAPE", "type": "numeric", "format": {"specifier": ".2f"}},
        {"name": "BIAS",     "id": "BIAS",  "type": "numeric", "format": {"specifier": ".3f"}},
    ]

    _SUBGROUP_TOOLTIPS = {
        "group": "Product category group (first two segments of the SKU ID)",
        "n":     "Number of test-set predictions in this category",
        "MAE":   "Mean Absolute Error — average absolute difference between forecast and actual sales",
        "RMSE":  "Root Mean Squared Error — penalises large errors more than MAE",
        "SMAPE": "Symmetric Mean Absolute Percentage Error — scale-independent accuracy metric (lower is better)",
        "BIAS":  "Average signed error (forecast − actual). Positive = over-forecasting, negative = under-forecasting",
    }

    if dark:
        s_header = {
            "backgroundColor": "#0A0C1A", "color": "#FFFFFF",
            "fontWeight": "bold", "fontSize": "13px",
            "border": "1px solid rgba(95,1,251,0.3)",
            "textDecoration": "underline", "textDecorationStyle": "dashed",
            "textDecorationColor": "rgba(95,1,251,0.6)", "cursor": "help",
        }
        s_data = {"backgroundColor": "#0F1225", "color": "#FFFFFF",
                  "border": "1px solid rgba(95,1,251,0.2)"}
        s_cell = {"fontSize": "13px", "padding": "14px 10px", "textAlign": "left",
                  "backgroundColor": "#0F1225", "color": "#FFFFFF"}
    else:
        s_header = {
            "backgroundColor": "#343a40", "color": "white",
            "fontWeight": "bold", "fontSize": "13px",
            "textDecoration": "underline", "textDecorationStyle": "dashed",
            "textDecorationColor": "rgba(255,255,255,0.5)", "cursor": "help",
        }
        s_data = {}
        s_cell = {"fontSize": "13px", "padding": "14px 10px", "textAlign": "left"}

    return dash_table.DataTable(
        columns=columns,
        data=df.to_dict("records"),
        sort_action="native",
        tooltip_header={col["id"]: _SUBGROUP_TOOLTIPS.get(col["id"], "") for col in columns},
        tooltip_delay=0,
        tooltip_duration=None,
        style_table={"overflowX": "auto", "minHeight": "300px"},
        style_header=s_header,
        style_data=s_data,
        style_cell=s_cell,
    )


# ── Q9 — NLG summary under metrics table ─────────────────────────────────────

def _reliability_nlg() -> html.Div:
    df       = ds.subgroup_eval_df
    dist     = ds.confidence_dist or {}
    coverage = ds.interval_coverage or 0.0

    if df is None or df.empty:
        return html.Div()

    best  = df.loc[df["SMAPE"].idxmin()]
    worst = df.loc[df["SMAPE"].idxmax()]

    avg_bias    = float(df["BIAS"].mean())
    bias_dir    = "over-forecasting" if avg_bias > 0 else "under-forecasting"
    total_preds = sum(dist.values()) or 1
    high_pct    = dist.get("high", 0) / total_preds * 100

    if abs(coverage - 80) < 5:
        cov_comment = "on target"
    elif coverage > 80:
        cov_comment = "above target — prediction intervals may be too wide"
    else:
        cov_comment = "below target — model may be under-estimating uncertainty"

    return html.Div([
        html.Hr(style={"margin": "12px 0"}),
        _section_label("Performance Summary"),
        _bullet_list([
            html.Li([
                "Best-performing category: ",
                html.Strong(best["group"]),
                f" — SMAPE {best['SMAPE']:.1f}%",
            ]),
            html.Li([
                "Weakest category: ",
                html.Strong(worst["group"]),
                f" — SMAPE {worst['SMAPE']:.1f}%",
            ]),
            html.Li([
                "Systematic bias: model leans toward ",
                html.Strong(bias_dir),
                f" across categories (avg {avg_bias:+.2f} units/period).",
            ]),
            html.Li([
                f"Interval coverage is {coverage:.1f}% — ",
                html.Strong(cov_comment), ".",
            ]),
            html.Li(f"{high_pct:.0f}% of test predictions are classified as high-confidence."),
        ]),
    ])


# ── callback registration ─────────────────────────────────────────────────────

def register_explanations_callbacks(app) -> None:

    @app.callback(
        Output("explanations-subtitle", "children"),
        Input("model-store", "data"),
    )
    def update_subtitle(model_key: str) -> str:
        cfg = ds.MODEL_CONFIGS.get(model_key, {})
        return cfg.get("label", "") + " · LGBM · M5 Walmart"

    # ── Q8 — populate audit feature options + reset ───────────────────────────
    @app.callback(
        Output("audit-feature-filter", "options"),
        Output("audit-feature-filter", "value"),
        Output("audit-flag-filter",    "value"),
        Input("model-store",           "data"),
        Input("url",                   "pathname"),
        Input("audit-clear-filters",   "n_clicks"),
    )
    def reset_audit_filters(_model_key: str, pathname: str, _clear):
        df = ds.feature_audit_df
        if df is None or df.empty:
            return [], [], []
        opts = [{"label": _clean(f), "value": f} for f in sorted(df["feature"])]
        return opts, [], []

    @app.callback(
        Output("global-shap-chart",   "figure"),
        Output("global-shap-nlg",     "children"),
        Output("audit-summary-cards", "children"),
        Output("feature-audit-table", "children"),
        Input("url",                  "pathname"),
        Input("model-store",          "data"),
        Input("theme-store",          "data"),
        Input("audit-feature-filter", "value"),
        Input("audit-flag-filter",    "value"),
    )
    def update_explanations(pathname: str, _model_key: str, theme: str,
                            audit_features: list, audit_flags: list):
        if pathname != "/explanations":
            return {}, [], [], []
        dark = theme == "dark"

        # apply audit column filters
        df = ds.feature_audit_df
        if df is not None and not df.empty:
            if audit_features: df = df[df["feature"].isin(audit_features)]
            if audit_flags:    df = df[df["flag"].isin(audit_flags)]

        return (
            _global_shap_figure(dark=dark),
            _global_shap_nlg(),
            _audit_summary_cards(dark=dark),
            _audit_table(dark=dark, df=df),
        )

    # ── Q5 — populate SKU A options ───────────────────────────────────────────
    @app.callback(
        Output("comp-sku-a", "options"),
        Output("comp-sku-a", "value"),
        Input("model-store", "data"),
        Input("url",         "pathname"),
    )
    def update_comp_sku_a(model_key: str, pathname: str):
        if pathname != "/explanations":
            return [], None
        _comparative_cache.clear()
        opts    = [{"label": s, "value": s} for s in ds.SKU_LIST]
        default = ds.SKU_LIST[0] if ds.SKU_LIST else None
        return opts, default

    # ── Q5 — auto-suggest SKU B based on similarity ───────────────────────────
    @app.callback(
        Output("comp-sku-b", "options"),
        Output("comp-sku-b", "value"),
        Input("comp-sku-a",  "value"),
        Input("model-store", "data"),
    )
    def update_comp_sku_b(sku_a: str, _model_key: str):
        if not sku_a:
            return [], None
        similar = _get_similar_skus(sku_a, n=10)
        opts    = [{"label": s, "value": s} for s in similar]
        default = similar[0] if similar else None
        return opts, default

    # ── Q5 — render comparison charts + NLG ──────────────────────────────────
    @app.callback(
        Output("comp-diff-chart", "figure"),
        Output("comp-side-chart", "figure"),
        Output("comp-nlg",        "children"),
        Input("comp-sku-a",   "value"),
        Input("comp-sku-b",   "value"),
        Input("theme-store",  "data"),
        Input("url",          "pathname"),
    )
    def update_comp_charts(sku_a: str, sku_b: str, theme: str, pathname: str):
        if pathname != "/explanations" or not sku_a or not sku_b or sku_a == sku_b:
            return {}, {}, []
        dark = theme == "dark"
        return (
            _comp_diff_figure(sku_a, sku_b, dark=dark),
            _comp_side_figure(sku_a, sku_b, dark=dark),
            _comp_nlg(sku_a, sku_b),
        )

    # ── Q7 — stockout column filter options + reset ───────────────────────────
    @app.callback(
        Output("stockout-sku-col-filter",  "options"),
        Output("stockout-sku-col-filter",  "value"),
        Output("stockout-risk-col-filter", "value"),
        Input("model-store",               "data"),
        Input("url",                       "pathname"),
        Input("stockout-col-clear-filters","n_clicks"),
    )
    def reset_stockout_col_filters(_model_key: str, _pathname: str, _clear):
        df = ds.stockout_risk_df
        if df is None or df.empty:
            return [], [], []
        opts = [{"label": r, "value": r} for r in sorted(df["item_id"])]
        return opts, [], []

    # ── Q7 — global risk table + SKU selector ────────────────────────────────
    @app.callback(
        Output("stockout-global-table",   "children"),
        Output("stockout-sku-selector",   "options"),
        Output("stockout-sku-selector",   "value"),
        Input("url",                      "pathname"),
        Input("model-store",              "data"),
        Input("theme-store",              "data"),
        Input("stockout-sku-col-filter",  "value"),
        Input("stockout-risk-col-filter", "value"),
    )
    def update_stockout_global(pathname: str, _model_key: str, theme: str,
                               col_skus: list, col_risks: list):
        if pathname != "/explanations":
            return [], [], None
        _stockout_shap_cache.clear()
        _stockout_censored_cache.clear()
        dark = theme == "dark"

        df = ds.stockout_risk_df
        if df is None or df.empty:
            return _stockout_global_table(dark=dark, df=df), [], None

        # apply column filters to the display table
        filtered = df.copy()
        filtered["risk_level"] = filtered["stockout_rate"].apply(
            lambda r: "HIGH" if r > 0.5 else "MODERATE" if r > 0.2 else "LOW"
        )
        if col_skus:  filtered = filtered[filtered["item_id"].isin(col_skus)]
        if col_risks: filtered = filtered[filtered["risk_level"].isin(col_risks)]

        table   = _stockout_global_table(dark=dark, df=filtered)
        at_risk = df[df["stockout_rate"] > 0.2].sort_values("stockout_rate", ascending=False)
        opts    = [{"label": f"{r['item_id']} ({r['stockout_rate']*100:.0f}%)", "value": r["item_id"]}
                   for _, r in at_risk.iterrows()]
        default = opts[0]["value"] if opts else None
        return table, opts, default

    # ── Q7 — per-SKU detail charts + NLG ─────────────────────────────────────
    @app.callback(
        Output("stockout-pred-chart",     "figure"),
        Output("stockout-shap-chart",     "figure"),
        Output("stockout-censored-chart", "figure"),
        Output("stockout-nlg",            "children"),
        Input("stockout-sku-selector", "value"),
        Input("theme-store",           "data"),
        Input("url",                   "pathname"),
    )
    def update_stockout_detail(sku_id: str, theme: str, pathname: str):
        if pathname != "/explanations" or not sku_id:
            return {}, {}, {}, []
        dark = theme == "dark"
        return (
            _stockout_pred_figure(sku_id, dark=dark),
            _stockout_shap_figure(sku_id, dark=dark),
            _stockout_censored_figure(sku_id, dark=dark),
            _stockout_nlg(sku_id),
        )

    # ── Q9 — Model Reliability & Cold-Start ──────────────────────────────────
    @app.callback(
        Output("reliability-stat-cards", "children"),
        Output("confidence-dist-chart",  "figure"),
        Output("subgroup-eval-table",    "children"),
        Output("reliability-nlg",        "children"),
        Input("url",         "pathname"),
        Input("model-store", "data"),
        Input("theme-store", "data"),
    )
    def update_reliability(pathname: str, _model_key: str, theme: str):
        if pathname != "/explanations":
            return [], {}, [], []
        dark = theme == "dark"
        return (
            _reliability_stat_cards(dark=dark),
            _confidence_dist_chart(dark=dark),
            _subgroup_eval_table(dark=dark),
            _reliability_nlg(),
        )
