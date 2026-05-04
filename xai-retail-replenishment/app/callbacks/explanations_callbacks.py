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
    return ds.feature_label(name)


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
        fig = go.Figure()
        if ds.current_model_type == "neural_network":
            fig.add_annotation(
                text="Global SHAP is not available for Neural Networks.<br>Switch to LightGBM, XGBoost, or Random Forest.",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=14, color="#6c757d"),
                align="center",
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(visible=False), yaxis=dict(visible=False),
                margin=dict(t=20, b=20, l=20, r=20),
            )
        return fig

    top = df.head(15).copy()
    top["rank"] = range(1, len(top) + 1)
    total_shap   = float(df["mean_abs_shap"].sum()) or 1.0
    top["share"] = top["mean_abs_shap"] / total_shap * 100
    top["feature_clean"] = top["feature"].apply(_clean)
    top = top.sort_values("mean_abs_shap", ascending=True)

    bar_color    = _NEON_PURPLE  if dark else "#3b82f6"
    text_color   = _NEON_FONT    if dark else "#333333"
    layout_extra = _NEON_LAYOUT  if dark else {"template": "plotly_white"}

    hover = [
        f"<b>#{int(r['rank'])} — {r['feature_clean']}</b><br>"
        f"On average, shifts the forecast by ±{r['mean_abs_shap']:.1f} units<br>"
        f"Accounts for {r['share']:.1f}% of total model influence"
        for _, r in top.iterrows()
    ]

    fig = go.Figure(go.Bar(
        x=top["mean_abs_shap"],
        y=top["feature_clean"],
        orientation="h",
        marker_color=bar_color,
        text=[f"{v:.2f}" for v in top["mean_abs_shap"]],
        textposition="outside",
        textfont={"color": text_color},
        customdata=hover,
        hovertemplate="%{customdata}<extra></extra>",
    ))

    fig.update_layout(
        **layout_extra,
        margin={"t": 20, "b": 20, "l": 20, "r": 70},
        xaxis_title="Average impact on forecast (units)",
        yaxis_automargin=True,
        showlegend=False,
    )
    return fig


# ── Combined Feature Intelligence NLG ────────────────────────────────────────

def _combined_nlg() -> html.Div:
    shap_df  = ds.global_shap_df
    audit_df = ds.feature_audit_df

    if shap_df is None or shap_df.empty:
        if ds.current_model_type == "neural_network":
            return html.P("SHAP analysis not available for Neural Networks.",
                          className="text-muted", style={"fontSize": "13px"})
        return html.P("Visit this page to compute SHAP (first visit may take ~30s).",
                      className="text-muted", style={"fontSize": "13px"})

    total_shap = float(shap_df["mean_abs_shap"].sum()) or 1.0
    top5       = shap_df.head(5)
    top_names  = [_clean(f) for f in top5["feature"]]
    top5_share = float(top5["mean_abs_shap"].sum()) / total_shap * 100

    # ── Cross-reference with audit ────────────────────────────────────────────
    risky_features = []
    if audit_df is not None and not audit_df.empty:
        shap_clean = shap_df.copy()
        shap_clean["feature_raw"] = shap_clean["feature"].str.replace(
            r"^(num__|cat__)", "", regex=True
        )
        p67 = shap_df["mean_abs_shap"].quantile(0.67)

        for _, row in audit_df.iterrows():
            if row["flag"] == "ok":
                continue
            raw = row["feature"]
            match = shap_clean[shap_clean["feature_raw"] == raw]
            if not match.empty and float(match.iloc[0]["mean_abs_shap"]) >= p67:
                risky_features.append((_clean(raw), row["flag"]))

        total_features = len(audit_df)
        flagged = int((audit_df["flag"] != "ok").sum())
    else:
        total_features, flagged = 0, 0

    # ── Build NLG ────────────────────────────────────────────────────────────
    blocks = []

    # Key drivers
    blocks += [
        _section_label("What drives demand?"),
        _bullet_list([
            html.Li([
                html.Strong(f"{i}. {name}"),
                html.Span(f" — {v:.1f}% of model influence",
                           className="text-muted", style={"fontSize": "12px"}),
            ])
            for i, (name, v) in enumerate(
                zip(top_names,
                    [float(r["mean_abs_shap"]) / total_shap * 100
                     for _, r in top5.iterrows()]),
                1,
            )
        ]),
        html.P(
            [f"Together the top 5 features account for ",
             html.Strong(f"{top5_share:.0f}%"), " of total model influence."],
            style={"fontSize": "13px"}, className="mt-2 mb-0",
        ),
    ]

    # Data health
    if total_features > 0:
        health_colour = "success" if flagged == 0 else ("warning" if flagged <= 3 else "danger")
        blocks += [
            html.Hr(style={"margin": "10px 0"}),
            _section_label("Data health"),
            html.P([
                html.Strong(f"{total_features - flagged} / {total_features}"),
                " features pass the quality audit. ",
                dbc.Badge(f"{flagged} flagged", color=health_colour, className="ms-1"),
            ], style={"fontSize": "13px"}, className="mb-0"),
        ]

    # Action items
    blocks.append(html.Hr(style={"margin": "10px 0"}))
    if risky_features:
        items = [
            html.Li([
                html.Strong(feat),
                html.Span(f" — {flag.replace('_', ' ')} but high model reliance",
                           className="text-muted", style={"fontSize": "12px"}),
            ])
            for feat, flag in risky_features
        ]
        blocks += [
            _section_label("⚠ Action items"),
            html.P("These features have data quality issues the model relies on heavily:",
                   style={"fontSize": "13px"}, className="mb-1"),
            _bullet_list(items),
        ]
    else:
        blocks += [
            _section_label("✓ No critical issues"),
            html.P(
                "All high-importance features have clean data. "
                "The model's core reasoning is on solid ground.",
                style={"fontSize": "13px"}, className="mb-0",
            ),
        ]

    return html.Div(blocks)


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
    "feature":    "The name of the input feature used by the model",
    "flag":       "Data quality verdict: ok = healthy | zero_variance = constant (no info) | mostly_zero = >80% zeros | high_corr = near-duplicate of another feature | high_missing = >10% missing",
    "reliance":   "How much the model relies on this feature based on its average SHAP contribution. High = top third, Medium = middle, Low = bottom third. Only available for tree-based models.",
    "risk":       "Combined verdict: Safe = no quality issue OR model ignores it. Risky = quality issue AND model relies on it heavily — this is the most important flag to act on.",
    "missing_pct":"Percentage of rows where this feature's value is missing (NaN)",
    "std":        "Standard deviation across the test set — how much the feature varies. Zero = constant column (no information for the model)",
}


def _audit_table(dark: bool = False, df: pd.DataFrame = None) -> dash_table.DataTable:
    if df is None:
        df = ds.feature_audit_df
    if df is None or df.empty:
        return html.P("No audit data available.", className="text-muted")

    display = df.copy()

    # ── Cross-reference with Global SHAP ─────────────────────────────────────
    shap_df = ds.global_shap_df
    if shap_df is not None and not shap_df.empty:
        shap_clean = shap_df.copy()
        # Strip sklearn prefixes so names match the audit feature names
        shap_clean["feature"] = shap_clean["feature"].str.replace(
            r"^(num__|cat__)", "", regex=True
        )
        shap_clean = shap_clean[["feature", "mean_abs_shap"]].drop_duplicates("feature")

        # Relative thresholds: top/middle/bottom third
        p33 = shap_clean["mean_abs_shap"].quantile(0.33)
        p67 = shap_clean["mean_abs_shap"].quantile(0.67)

        def _reliance(v):
            if v >= p67: return "High"
            if v >= p33: return "Medium"
            return "Low"

        shap_clean["reliance"] = shap_clean["mean_abs_shap"].apply(_reliance)
        display = display.merge(shap_clean[["feature", "reliance"]], on="feature", how="left")
        display["reliance"] = display["reliance"].fillna("N/A")
    else:
        display["reliance"] = "N/A"

    # ── Risk column: quality flag × model reliance ────────────────────────────
    def _risk(row):
        if row["flag"] == "ok":
            return "✓ Safe"
        if row["reliance"] == "High":
            return "⚠ Risky"
        if row["reliance"] == "Medium":
            return "~ Moderate"
        return "✓ Safe"   # flagged but Low reliance → model ignores it

    display["risk"] = display.apply(_risk, axis=1)
    display["feature"] = display["feature"].apply(_clean)

    columns = [
        {"name": "Feature",        "id": "feature"},
        {"name": "Status",         "id": "flag"},
        {"name": "Model Reliance", "id": "reliance"},
        {"name": "Risk",           "id": "risk"},
        {"name": "Missing %",      "id": "missing_pct", "type": "numeric",
         "format": {"specifier": ".1f"}},
        {"name": "Std Dev",        "id": "std",         "type": "numeric",
         "format": {"specifier": ".3f"}},
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
            {"if": {"filter_query": '{risk} = "⚠ Risky"',  "column_id": "risk"},
             "backgroundColor": "rgba(255,45,85,0.15)",  "color": "#FF2D55", "fontWeight": "bold"},
            {"if": {"filter_query": '{risk} = "~ Moderate"', "column_id": "risk"},
             "backgroundColor": "rgba(255,184,0,0.12)", "color": "#FFB800", "fontWeight": "bold"},
            {"if": {"filter_query": '{risk} = "✓ Safe"',  "column_id": "risk"},
             "backgroundColor": "rgba(0,255,135,0.08)",  "color": "#00FF87"},
            {"if": {"filter_query": '{reliance} = "High"',   "column_id": "reliance"},
             "color": "#FF2D55"},
            {"if": {"filter_query": '{reliance} = "Medium"', "column_id": "reliance"},
             "color": "#FFB800"},
            {"if": {"filter_query": '{reliance} = "Low"',    "column_id": "reliance"},
             "color": "#00FF87"},
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
            {"if": {"filter_query": '{risk} = "⚠ Risky"',  "column_id": "risk"},
             "backgroundColor": "#f8d7da", "color": "#842029", "fontWeight": "bold"},
            {"if": {"filter_query": '{risk} = "~ Moderate"', "column_id": "risk"},
             "backgroundColor": "#fff3cd", "color": "#664d03", "fontWeight": "bold"},
            {"if": {"filter_query": '{risk} = "✓ Safe"',  "column_id": "risk"},
             "backgroundColor": "#d1e7dd", "color": "#0a3622"},
            {"if": {"filter_query": '{reliance} = "High"',   "column_id": "reliance"},
             "color": "#dc3545", "fontWeight": "600"},
            {"if": {"filter_query": '{reliance} = "Medium"', "column_id": "reliance"},
             "color": "#856404", "fontWeight": "600"},
            {"if": {"filter_query": '{reliance} = "Low"',    "column_id": "reliance"},
             "color": "#0a3622", "fontWeight": "600"},
        ]

    return dash_table.DataTable(
        id="audit-datatable",
        columns=columns,
        data=display[["feature", "flag", "reliance", "risk", "missing_pct", "std"]].to_dict("records"),
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
    coverage   = ds.interval_coverage or 0.0
    cov_colour = "success" if coverage >= 75 else "warning" if coverage >= 65 else "danger"

    # Compute overall weighted MAE and BIAS from subgroup table
    df = ds.subgroup_eval_df
    if df is not None and not df.empty and "n" in df.columns:
        total_n    = df["n"].sum() or 1
        overall_mae  = float((df["MAE"]  * df["n"]).sum() / total_n)
        overall_bias = float((df["BIAS"] * df["n"]).sum() / total_n)
    else:
        overall_mae, overall_bias = None, None

    _dash = {
        "textDecoration": "underline", "textDecorationStyle": "dashed",
        "textDecorationColor": "rgba(108,117,125,0.7)", "cursor": "help",
    }

    # ── Card 1: Interval Coverage ────────────────────────────────────────────
    cov_col = dbc.Col([
        dbc.Card([dbc.CardBody([
            html.H3(f"{coverage:.1f}%", className=f"text-{cov_colour} mb-0"),
            html.Small("Interval Coverage", id="coverage-label-tip",
                       className="text-muted", style=_dash),
        ], style={"padding": "14px 18px"})], className="shadow-sm text-center"),
        dbc.Tooltip([
            html.P("% of actual sales that fell inside the model's 80% prediction band. "
                   "A well-calibrated model should be close to 80%.",
                   className="mb-2", style={"fontSize": "12px"}),
            html.Ul([
                html.Li("≈ 80% → intervals are well-calibrated", style={"fontSize": "12px"}),
                html.Li("Much < 80% → intervals too narrow — model is overconfident", style={"fontSize": "12px"}),
                html.Li("Much > 80% → intervals too wide — model is over-cautious", style={"fontSize": "12px"}),
            ], className="mb-0 ps-3"),
        ], target="coverage-label-tip", placement="bottom",
           style={"maxWidth": "340px", "textAlign": "left"}),
    ], xs=6, md=4)

    # ── Card 2: Overall MAE ──────────────────────────────────────────────────
    mae_val  = f"{overall_mae:.1f} units" if overall_mae is not None else "N/A"
    mae_col  = dbc.Col([
        dbc.Card([dbc.CardBody([
            html.H3(mae_val, className="text-dark mb-0" if not dark else "text-light mb-0"),
            html.Small("Avg. Error per Week (MAE)", id="mae-label-tip",
                       className="text-muted", style=_dash),
        ], style={"padding": "14px 18px"})], className="shadow-sm text-center"),
        dbc.Tooltip(
            "On average, the model's forecast is off by this many units per week. "
            "Lower is better. Compare against typical sales volume to judge significance.",
            target="mae-label-tip", placement="bottom",
        ),
    ], xs=6, md=4)

    # ── Card 3: Systematic Bias ──────────────────────────────────────────────
    if overall_bias is not None:
        if abs(overall_bias) < 0.5:
            bias_label, bias_colour = "BALANCED", "success"
        elif overall_bias > 0:
            bias_label, bias_colour = "OVER-FORECASTING", "warning"
        else:
            bias_label, bias_colour = "UNDER-FORECASTING", "warning"
        bias_val = html.Div([
            dbc.Badge(bias_label, color=bias_colour, className="mb-1",
                      style={"fontSize": "13px"}),
            html.Br(),
            html.Small(f"avg {overall_bias:+.2f} units/week",
                       className="text-muted", style={"fontSize": "11px"}),
        ])
    else:
        bias_val, bias_colour = html.Span("N/A"), "secondary"

    bias_col = dbc.Col([
        dbc.Card([dbc.CardBody([
            bias_val,
            html.Small("Systematic Bias", id="bias-label-tip",
                       className="text-muted d-block", style=_dash),
        ], style={"padding": "14px 18px"})], className="shadow-sm text-center"),
        dbc.Tooltip(
            "Whether the model consistently over- or under-forecasts on average. "
            "Over-forecasting → excess stock. Under-forecasting → stockouts.",
            target="bias-label-tip", placement="bottom",
        ),
    ], xs=6, md=4)

    return dbc.Row([cov_col, mae_col, bias_col], className="g-3")


def _category_accuracy_chart(dark: bool = False) -> go.Figure:
    df = ds.subgroup_eval_df
    if df is None or df.empty:
        return go.Figure()

    layout_extra = _NEON_LAYOUT if dark else {"template": "plotly_white"}
    df_sorted    = df.sort_values("SMAPE", ascending=True).reset_index(drop=True)

    def _colour(smape):
        if smape < 50:
            return "#00FF87" if dark else "#22c55e"
        if smape < 70:
            return "#FFB800" if dark else "#f59e0b"
        return _NEON_RED if dark else "#ef4444"

    def _verdict(smape):
        return "Good" if smape < 50 else ("OK" if smape < 70 else "Poor")

    colors = [_colour(s) for s in df_sorted["SMAPE"]]
    hover  = [
        f"<b>{row['group']}</b><br>"
        f"Accuracy: {_verdict(row['SMAPE'])} (SMAPE {row['SMAPE']:.1f}%)<br>"
        f"Avg error: {row['MAE']:.1f} units | Bias: {row['BIAS']:+.2f} units/week"
        for _, row in df_sorted.iterrows()
    ]
    text_color = _NEON_FONT if dark else "#333333"

    fig = go.Figure(go.Bar(
        x=df_sorted["SMAPE"],
        y=df_sorted["group"],
        orientation="h",
        marker_color=colors,
        text=[f"{s:.1f}%" for s in df_sorted["SMAPE"]],
        textposition="outside",
        textfont={"color": text_color, "size": 11},
        customdata=hover,
        hovertemplate="%{customdata}<extra></extra>",
    ))
    fig.update_layout(
        **layout_extra,
        title={"text": "Forecast Accuracy by Category (SMAPE %)", "font": {"size": 13}},
        margin={"t": 40, "b": 20, "l": 20, "r": 60},
        xaxis_title="SMAPE % (lower is better)",
        yaxis_automargin=True,
        showlegend=False,
    )
    return fig


def _subgroup_eval_table(dark: bool = False) -> html.Div:
    df = ds.subgroup_eval_df
    if df is None or df.empty:
        return html.P("No subgroup data available.", className="text-muted")

    display = df.copy()
    display["reliability"] = display["SMAPE"].apply(
        lambda s: "Good" if s < 50 else ("OK" if s < 70 else "Poor")
    )

    columns = [
        {"name": "Category",    "id": "group"},
        {"name": "Reliability", "id": "reliability"},
        {"name": "MAE",         "id": "MAE",   "type": "numeric", "format": {"specifier": ".2f"}},
        {"name": "SMAPE %",     "id": "SMAPE", "type": "numeric", "format": {"specifier": ".1f"}},
        {"name": "BIAS",        "id": "BIAS",  "type": "numeric", "format": {"specifier": ".2f"}},
        {"name": "N",           "id": "n",     "type": "numeric"},
    ]

    _SUBGROUP_TOOLTIPS = {
        "group":       "Product category (first two segments of the SKU ID)",
        "reliability": "Overall verdict based on SMAPE: Good < 50%, OK 50–70%, Poor > 70%",
        "MAE":         "Mean Absolute Error — average absolute difference between forecast and actual (in units)",
        "SMAPE":       "Symmetric Mean Absolute Percentage Error — scale-independent accuracy (lower is better)",
        "BIAS":        "Avg signed error (forecast − actual). Positive = over-forecasting, negative = under-forecasting",
        "n":           "Number of test-set predictions in this category",
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

    rel_cond = [
        {"if": {"filter_query": '{reliability} = "Good"', "column_id": "reliability"},
         "backgroundColor": ("rgba(0,255,135,0.12)" if dark else "#d1e7dd"),
         "color": ("#00FF87" if dark else "#0a3622"), "fontWeight": "bold"},
        {"if": {"filter_query": '{reliability} = "OK"', "column_id": "reliability"},
         "backgroundColor": ("rgba(255,184,0,0.12)" if dark else "#fff3cd"),
         "color": ("#FFB800" if dark else "#664d03"), "fontWeight": "bold"},
        {"if": {"filter_query": '{reliability} = "Poor"', "column_id": "reliability"},
         "backgroundColor": ("rgba(255,45,85,0.12)" if dark else "#f8d7da"),
         "color": ("#FF2D55" if dark else "#842029"), "fontWeight": "bold"},
    ]

    return dash_table.DataTable(
        columns=columns,
        data=display[["group", "reliability", "MAE", "SMAPE", "BIAS", "n"]].to_dict("records"),
        sort_action="native",
        tooltip_header={col["id"]: _SUBGROUP_TOOLTIPS.get(col["id"], "") for col in columns},
        tooltip_delay=0,
        tooltip_duration=None,
        style_table={"overflowX": "auto"},
        style_header=s_header,
        style_data=s_data,
        style_cell=s_cell,
        style_data_conditional=rel_cond,
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


# ── lazy SHAP / audit computation ────────────────────────────────────────────

def _ensure_global_shap() -> None:
    """Compute and cache global SHAP for the current model type if not already available."""
    if ds.global_shap_df is not None:
        return
    if ds.current_model_type == "neural_network":
        print("[explanations] Global SHAP skipped — not supported for Neural Network.")
        return
    try:
        from xai.global_shap import compute_global_shap, rank_feature_importance
        from pathlib import Path
        print(f"[explanations] Computing global SHAP for {ds.MODEL_TYPES[ds.current_model_type]} (50 samples)…")
        shap_vals         = compute_global_shap(ds.model, ds.X_test, max_samples=50)
        ds.global_shap_df = rank_feature_importance(shap_vals)
        _path = ds._REPORTS / f"global_shap_{ds.MODEL_CONFIGS[ds.current_model_key]['suffix']}_{ds.current_model_type}.csv"
        Path(_path).parent.mkdir(parents=True, exist_ok=True)
        ds.global_shap_df.to_csv(_path, index=False)
        print(f"[explanations] Global SHAP saved → {_path.name}")
    except Exception as exc:
        print(f"[explanations] SHAP computation failed: {exc}")


def _ensure_feature_audit() -> None:
    """Compute and cache feature audit if not already available."""
    if ds.feature_audit_df is not None:
        return
    try:
        from xai.global_shap import feature_quality_audit
        from pathlib import Path
        print("[explanations] Computing feature audit (first visit)…")
        ds.feature_audit_df = feature_quality_audit(ds.X_test)
        _path = ds._REPORTS / ds.MODEL_CONFIGS[ds.current_model_key]["audit_file"]
        Path(_path).parent.mkdir(parents=True, exist_ok=True)
        ds.feature_audit_df.to_csv(_path, index=False)
        print("[explanations] Feature audit saved.")
    except Exception as exc:
        print(f"[explanations] Audit computation failed: {exc}")


# ── callback registration ─────────────────────────────────────────────────────

def register_explanations_callbacks(app) -> None:

    # ── audit feature dropdown options + reset ────────────────────────────────
    @app.callback(
        Output("audit-feature-filter", "options"),
        Output("audit-feature-filter", "value"),
        Output("audit-flag-filter",    "value"),
        Input("model-store",           "data"),
        Input("model-type-store",      "data"),
        Input("url",                   "pathname"),
        Input("audit-clear-filters",   "n_clicks"),
    )
    def reset_audit_filters(_mk, _mt, pathname: str, _clear):
        df = ds.feature_audit_df
        if df is None or df.empty:
            return [], [], []
        opts = [{"label": _clean(f), "value": f} for f in sorted(df["feature"])]
        return opts, [], []

    # ── Global SHAP + Feature Audit ───────────────────────────────────────────
    @app.callback(
        Output("global-shap-chart",   "figure"),
        Output("global-shap-nlg",     "children"),
        Output("audit-summary-cards", "children"),
        Output("feature-audit-table", "children"),
        Input("url",                  "pathname"),
        Input("model-store",          "data"),
        Input("model-type-store",     "data"),
        Input("theme-store",          "data"),
        Input("audit-feature-filter", "value"),
        Input("audit-flag-filter",    "value"),
    )
    def update_explanations(pathname: str, _mk, _mt, theme: str,
                            audit_features: list, audit_flags: list):
        if pathname != "/explanations":
            return {}, [], [], []
        dark = theme == "dark"

        _ensure_global_shap()
        _ensure_feature_audit()

        df = ds.feature_audit_df
        if df is not None and not df.empty:
            if audit_features: df = df[df["feature"].isin(audit_features)]
            if audit_flags:    df = df[df["flag"].isin(audit_flags)]

        return (
            _global_shap_figure(dark=dark),
            _combined_nlg(),
            _audit_summary_cards(dark=dark),
            _audit_table(dark=dark, df=df),
        )

    # ── Model Reliability & Cold-Start ────────────────────────────────────────
    @app.callback(
        Output("reliability-stat-cards", "children"),
        Output("confidence-dist-chart",  "figure"),
        Output("subgroup-eval-table",    "children"),
        Output("reliability-nlg",        "children"),
        Input("url",              "pathname"),
        Input("model-store",      "data"),
        Input("model-type-store", "data"),
        Input("theme-store",      "data"),
    )
    def update_reliability(pathname: str, _mk, _mt, theme: str):
        if pathname != "/explanations":
            return [], {}, [], []
        dark = theme == "dark"
        return (
            _reliability_stat_cards(dark=dark),
            _category_accuracy_chart(dark=dark),
            _subgroup_eval_table(dark=dark),
            _reliability_nlg(),
        )
