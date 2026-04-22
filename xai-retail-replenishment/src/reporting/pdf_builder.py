"""
PDF report builder using ReportLab Platypus.

Templates:
  "brief"  — Replenishment Brief   (1-2 pp, key metrics + forecast chart)
  "full"   — Full XAI Report        (4-6 pp, all sections + charts)
  "exec"   — Executive Summary      (1 p,  bullets only, no charts)
"""

from __future__ import annotations

import datetime
import math
from io import BytesIO

import numpy as np
import pandas as pd

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether,
)

import app.data_store as ds
from xai.uncertainty import compute_prediction_interval, confidence_label
from xai.cost_impact import simulate_cost_distribution, optimal_order_quantity
from xai.counterfactual import batch_counterfactuals
from xai.local_shap import get_top_contributors
from xai.temporal_shap import classify_demand_pattern

from .chart_exporter import fig_to_bytesio

# ── Colours ───────────────────────────────────────────────────────────────────

PURPLE    = HexColor("#7B61FF")
DARK_TEXT = HexColor("#1a1a2e")
LIGHT_BG  = HexColor("#f8f9fa")
BORDER    = HexColor("#dee2e6")
MUTED     = HexColor("#6c757d")
SUCCESS   = HexColor("#198754")
WARNING_C = HexColor("#e6a800")
DANGER    = HexColor("#dc3545")
WHITE     = white

URGENCY_COLOR = {"CRITICAL": DANGER, "HIGH": WARNING_C, "LOW": SUCCESS}

# ── Page geometry ─────────────────────────────────────────────────────────────

PAGE_MARGIN = 1.8 * cm
A4_W, _     = A4
CONTENT_W   = A4_W - 2 * PAGE_MARGIN

# ── Paragraph styles ──────────────────────────────────────────────────────────

def _make_styles() -> dict:
    return {
        "title": ParagraphStyle(
            "rpt_title", fontSize=22, textColor=PURPLE, spaceAfter=4,
            fontName="Helvetica-Bold", alignment=TA_CENTER,
        ),
        "subtitle": ParagraphStyle(
            "rpt_sub", fontSize=10, textColor=MUTED, spaceAfter=16,
            fontName="Helvetica", alignment=TA_CENTER,
        ),
        "h2": ParagraphStyle(
            "rpt_h2", fontSize=13, textColor=WHITE, spaceAfter=0,
            fontName="Helvetica-Bold", spaceBefore=14,
            backColor=PURPLE, leftIndent=6, rightIndent=6,
        ),
        "h3": ParagraphStyle(
            "rpt_h3", fontSize=10, textColor=DARK_TEXT, spaceAfter=4,
            fontName="Helvetica-Bold", spaceBefore=8,
        ),
        "body": ParagraphStyle(
            "rpt_body", fontSize=9, textColor=DARK_TEXT, spaceAfter=3,
            fontName="Helvetica", leading=14,
        ),
        "muted": ParagraphStyle(
            "rpt_muted", fontSize=8, textColor=MUTED, spaceAfter=3,
            fontName="Helvetica",
        ),
        "bullet": ParagraphStyle(
            "rpt_bullet", fontSize=9, textColor=DARK_TEXT, spaceAfter=2,
            fontName="Helvetica", leftIndent=14, bulletIndent=4, leading=13,
        ),
        "sku_banner": ParagraphStyle(
            "rpt_banner", fontSize=15, textColor=WHITE, spaceAfter=6,
            fontName="Helvetica-Bold", alignment=TA_LEFT,
            backColor=PURPLE, leftIndent=8,
        ),
        "metric_label": ParagraphStyle(
            "rpt_ml", fontSize=8, textColor=MUTED, spaceAfter=1,
            fontName="Helvetica",
        ),
        "metric_value": ParagraphStyle(
            "rpt_mv", fontSize=14, textColor=DARK_TEXT, spaceAfter=2,
            fontName="Helvetica-Bold",
        ),
    }


S = _make_styles()


# ── Table factory ─────────────────────────────────────────────────────────────

def _table(data: list, col_widths=None, header_bg=PURPLE, zebra=True) -> Table:
    rows = len(data)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR",  (0, 0), (-1, 0), WHITE),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, 0), 9),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("FONTNAME",   (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",   (0, 1), (-1, -1), 9),
        ("GRID",       (0, 0), (-1, -1), 0.4, BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if zebra:
        for r in range(1, rows):
            bg = LIGHT_BG if r % 2 == 0 else WHITE
            style.append(("BACKGROUND", (0, r), (-1, r), bg))
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle(style))
    return t


def _metric_card_row(pairs: list[tuple[str, str]]) -> Table:
    """Render (label, value) pairs as a horizontal metrics bar."""
    n     = len(pairs)
    w_each = CONTENT_W / n
    label_row = [Paragraph(lbl, S["metric_label"]) for lbl, _ in pairs]
    value_row = [Paragraph(val, S["metric_value"]) for _, val in pairs]
    data = [label_row, value_row]
    style = [
        ("ALIGN",   (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",  (0, 0), (-1, -1), "MIDDLE"),
        ("BOX",     (0, 0), (-1, -1), 0.5, BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, BORDER),
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]
    t = Table(data, colWidths=[w_each] * n)
    t.setStyle(TableStyle(style))
    return t


def _chart(fig, w_cm=16.5, h_cm=8.0) -> Image | Spacer:
    buf = fig_to_bytesio(fig, width=int(w_cm * 37.8), height=int(h_cm * 37.8))
    if buf.getbuffer().nbytes < 100:
        return Spacer(1, h_cm * cm)
    return Image(buf, width=w_cm * cm, height=h_cm * cm)


# ── Data collector ────────────────────────────────────────────────────────────

def _sku_data(sku_id: str, unit_margin: float, holding_cost: float) -> dict:
    iv       = ds.forecasts.get(sku_id, {"q10": 0, "q50": 0, "q90": 0})
    card_df  = ds.cards_df[ds.cards_df["sku_id"] == sku_id]
    card     = card_df.iloc[0].to_dict() if not card_df.empty else {}
    std_v    = float(ds.sku_std.get(sku_id, float(ds.sku_std.mean())))
    pi       = compute_prediction_interval(iv["q50"], std_v)
    conf     = confidence_label(pi["width"], iv["q50"])

    # Cold-start flag  (cold_start_df uses "sku_id" column, not "item_id")
    is_cold = False
    if ds.cold_start_df is not None and not ds.cold_start_df.empty:
        cs_row  = ds.cold_start_df[ds.cold_start_df["sku_id"] == sku_id]
        is_cold = not cs_row.empty and bool(cs_row.iloc[0].get("is_cold_start", False))

    # Stockout risk
    so_rate = 0.0
    if ds.stockout_risk_df is not None and not ds.stockout_risk_df.empty:
        so_row  = ds.stockout_risk_df[ds.stockout_risk_df["item_id"] == sku_id]
        so_rate = float(so_row.iloc[0]["stockout_rate"]) if not so_row.empty else 0.0

    # SHAP top drivers
    top_shap: list[tuple[str, float]] = []
    try:
        shap_exp = ds.get_local_shap(sku_id)
        top_df   = get_top_contributors(shap_exp, n=5)
        for _, row in top_df.iterrows():
            feat = row["feature"].split("__", 1)[-1] if "__" in row["feature"] else row["feature"]
            top_shap.append((feat, float(row["shap_value"])))
    except Exception:
        pass

    # Temporal pattern
    pattern = "unknown"
    try:
        from app.callbacks.sku_callbacks import _get_temporal_df
        t_df    = _get_temporal_df(sku_id)
        pattern = classify_demand_pattern(t_df, sku_id)["pattern"]
    except Exception:
        pass

    # Cost impact
    q10, q50, q90 = iv["q10"], iv["q50"], iv["q90"]
    opt = optimal_order_quantity(q10, q50, q90, unit_margin, holding_cost)
    sim = simulate_cost_distribution(
        q10, q50, q90, opt["optimal_qty"],
        unit_margin, holding_cost, n_simulations=2_000, seed=42,
    )

    return dict(
        sku_id=sku_id, iv=iv, card=card, std_v=std_v, pi=pi, conf=conf,
        is_cold=is_cold, so_rate=so_rate, top_shap=top_shap, pattern=pattern,
        opt=opt,
        exp_so=float(sim["stockout_cost"].mean()),
        exp_os=float(sim["overstock_cost"].mean()),
        exp_tot=float(sim["total_cost"].mean()),
        unit_margin=unit_margin, holding_cost=holding_cost,
    )


# ── Section builders (return list of Flowables) ───────────────────────────────

def _cover(sku_ids: list[str], model_key: str, template_label: str, story: list) -> None:
    cfg   = ds.MODEL_CONFIGS.get(model_key, ds.MODEL_CONFIGS["7d"])
    today = datetime.date.today().strftime("%d %B %Y")
    story += [
        Spacer(1, 1.2 * cm),
        Paragraph("XAI Retail Replenishment Report", S["title"]),
        Paragraph(
            f"{cfg['label']} &nbsp;·&nbsp; LightGBM &nbsp;·&nbsp; M5 Walmart",
            S["subtitle"],
        ),
        HRFlowable(width=CONTENT_W, color=PURPLE, thickness=2, spaceAfter=6),
        Paragraph(
            f"Template: <b>{template_label}</b> &nbsp;|&nbsp; Generated: {today} &nbsp;|&nbsp; "
            f"SKUs: {len(sku_ids)}",
            S["muted"],
        ),
        Spacer(1, 0.4 * cm),
    ]


def _summary_table(sku_ids: list[str], unit_margin: float, holding_cost: float, story: list) -> None:
    if len(sku_ids) < 2:
        return
    hdr  = ["SKU", "Urgency", "Forecast (q50)", "Opt. Order", "Confidence", "Stockout Risk"]
    rows = [hdr]
    for sid in sku_ids:
        d = _sku_data(sid, unit_margin, holding_cost)
        rows.append([
            sid,
            d["card"].get("urgency", "—"),
            f"{math.ceil(d['iv']['q50'])} units",
            f"{math.ceil(d['opt']['optimal_qty'])} units",
            d["conf"].upper(),
            f"{d['so_rate'] * 100:.0f}%",
        ])
    w = CONTENT_W / 6
    story += [
        Paragraph("Portfolio Summary", S["h2"]),
        Spacer(1, 0.15 * cm),
        _table(rows, col_widths=[w * 1.4, w * 0.9, w, w, w * 0.9, w * 0.8]),
        Spacer(1, 0.4 * cm),
    ]


def _sec_replenishment(d: dict, story: list) -> None:
    card    = d["card"]
    urgency = card.get("urgency", "—")
    u_color = URGENCY_COLOR.get(urgency, MUTED)
    story += [
        Paragraph("Replenishment Summary", S["h2"]),
        Spacer(1, 0.15 * cm),
        _metric_card_row([
            ("Urgency",          urgency),
            ("Days of Stock",    f"{card.get('days_of_stock', 0):.1f}"),
            ("Stock on Hand",    f"{card.get('stock_on_hand', 0):.0f} u"),
            ("Safety Stock",     f"{card.get('safety_stock', 0):.0f} u"),
            ("Recommended Order", f"{math.ceil(card.get('reorder_qty', 0))} u"),
        ]),
        Spacer(1, 0.2 * cm),
    ]
    trigger = card.get("trigger_reorder", False)
    note    = "⚠ Reorder recommended NOW." if trigger else "Stock levels are adequate for the forecast period."
    story.append(Paragraph(f"<b>Action:</b> {note}", S["body"]))
    story.append(Spacer(1, 0.2 * cm))


def _sec_forecast(d: dict, include_chart: bool, story: list) -> None:
    iv   = d["iv"]
    conf = d["conf"]
    conf_desc = {"high": "reliable", "moderate": "moderate variability", "low": "high variability"}.get(conf, "")
    story += [
        Paragraph("Demand Forecast", S["h2"]),
        Spacer(1, 0.15 * cm),
        _metric_card_row([
            ("q10 (pessimistic)", f"{math.ceil(iv['q10'])} u"),
            ("q50 (median)",      f"{math.ceil(iv['q50'])} u"),
            ("q90 (optimistic)",  f"{math.ceil(iv['q90'])} u"),
            ("Confidence",        conf.upper()),
            ("Interval Coverage", f"{ds.interval_coverage:.1f}%"),
        ]),
        Spacer(1, 0.2 * cm),
        Paragraph(
            f"The model forecasts <b>{math.ceil(iv['q50'])} units</b> over the next "
            f"<b>{ds.HORIZON} days</b>. Confidence is <b>{conf.upper()}</b> — {conf_desc}. "
            f"{'Cold-start SKU: limited history may reduce accuracy.' if d['is_cold'] else ''}",
            S["body"],
        ),
    ]
    if include_chart:
        try:
            from app.callbacks.sku_callbacks import _forecast_figure
            fig = _forecast_figure(d["sku_id"], d["std_v"], dark=False)
            story.append(_chart(fig, h_cm=7.0))
        except Exception:
            pass
    story.append(Spacer(1, 0.2 * cm))


def _sec_shap(d: dict, story: list) -> None:
    story += [Paragraph("Local SHAP Analysis", S["h2"]), Spacer(1, 0.15 * cm)]
    if d["top_shap"]:
        hdr  = [["Feature", "SHAP Value", "Direction"]]
        rows = hdr + [
            [feat, f"{val:+.4f}", "▲ increases forecast" if val > 0 else "▼ decreases forecast"]
            for feat, val in d["top_shap"]
        ]
        story.append(_table(rows, col_widths=[7 * cm, 4 * cm, 7.5 * cm]))
    else:
        story.append(Paragraph("SHAP data not available for this SKU.", S["muted"]))

    try:
        from app.callbacks.sku_callbacks import _shap_figure
        fig = _shap_figure(d["sku_id"], dark=False)
        story.append(_chart(fig, h_cm=6.5))
    except Exception:
        pass
    story.append(Spacer(1, 0.2 * cm))


def _sec_temporal(d: dict, story: list) -> None:
    story += [
        Paragraph("Temporal Demand Pattern", S["h2"]),
        Spacer(1, 0.15 * cm),
        Paragraph(
            f"Classified demand pattern: <b>{d['pattern'].upper()}</b>. "
            "The heatmap below shows how each feature's SHAP contribution evolves over time.",
            S["body"],
        ),
    ]
    try:
        from app.callbacks.sku_callbacks import _temporal_heatmap
        fig = _temporal_heatmap(d["sku_id"], dark=False)
        story.append(_chart(fig, h_cm=7.0))
    except Exception:
        pass
    story.append(Spacer(1, 0.2 * cm))


def _sec_whatif(d: dict, story: list) -> None:
    story += [Paragraph("What-If Sensitivity", S["h2"]), Spacer(1, 0.15 * cm)]
    try:
        from app.callbacks.whatif_callbacks import _price_figure
        X_row       = ds.get_sku_X_row(d["sku_id"])
        orig_price  = float(X_row["aggregated_sell_price"].iloc[0])
        orig_pred   = float(ds.model.predict(X_row)[0])
        price_batch = batch_counterfactuals(
            ds.model, X_row, "aggregated_sell_price",
            np.linspace(orig_price * 0.5, orig_price * 1.5, 30),
        )
        fig = _price_figure(price_batch, orig_price, orig_pred, d["sku_id"], dark=False)
        story.append(Paragraph(
            f"Price sensitivity sweep: current price ${orig_price:.2f}. "
            "The chart shows how the 7-day forecast responds across the price range.",
            S["body"],
        ))
        story.append(_chart(fig, h_cm=7.0))
    except Exception:
        story.append(Paragraph("What-If data not available.", S["muted"]))
    story.append(Spacer(1, 0.2 * cm))


def _sec_cost(d: dict, story: list) -> None:
    opt = d["opt"]
    story += [
        Paragraph("Cost Impact Analysis", S["h2"]),
        Spacer(1, 0.15 * cm),
        _metric_card_row([
            ("Optimal Order Qty",   f"{math.ceil(opt['optimal_qty'])} u"),
            ("Exp. Stockout Cost",  f"${d['exp_so']:.2f}"),
            ("Exp. Overstock Cost", f"${d['exp_os']:.2f}"),
            ("Expected Total Cost", f"${d['exp_tot']:.2f}"),
            ("Critical Ratio",      f"{opt['critical_ratio']:.1%}"),
        ]),
        Spacer(1, 0.2 * cm),
        Paragraph(
            f"Unit margin: ${d['unit_margin']:.2f} &nbsp;|&nbsp; "
            f"Holding cost: ${d['holding_cost']:.2f}/unit &nbsp;|&nbsp; "
            f"Implied service level: {opt['implied_service_level']:.0f}%",
            S["muted"],
        ),
    ]
    try:
        from app.callbacks.whatif_callbacks import _cost_curve_figure
        iv  = d["iv"]
        q10, q50, q90 = iv["q10"], iv["q50"], iv["q90"]
        order_range   = np.linspace(max(q10 * 0.5, 0.1), q90 * 1.5, 25)
        rows = []
        for oq in order_range:
            s = simulate_cost_distribution(
                q10, q50, q90, oq,
                d["unit_margin"], d["holding_cost"],
                n_simulations=1_000, seed=42,
            )
            rows.append({
                "order_qty":      float(oq),
                "mean_stockout":  float(s["stockout_cost"].mean()),
                "mean_overstock": float(s["overstock_cost"].mean()),
                "mean_total":     float(s["total_cost"].mean()),
            })
        curve = pd.DataFrame(rows)
        fig   = _cost_curve_figure(curve, opt, q50, d["sku_id"], dark=False)
        story.append(_chart(fig, h_cm=7.0))
    except Exception:
        pass
    story.append(Spacer(1, 0.2 * cm))


def _sec_reliability(d: dict, story: list) -> None:
    story += [
        Paragraph("Model Reliability", S["h2"]),
        Spacer(1, 0.15 * cm),
        _metric_card_row([
            ("Confidence Level",   d["conf"].upper()),
            ("Interval Coverage",  f"{ds.interval_coverage:.1f}%"),
            ("Stockout Risk",      f"{d['so_rate'] * 100:.0f}%"),
            ("Cold Start",         "Yes" if d["is_cold"] else "No"),
        ]),
        Spacer(1, 0.2 * cm),
        Paragraph(
            f"The 80% prediction interval covers <b>{ds.interval_coverage:.1f}%</b> of "
            f"actual test-set demand. A higher coverage means the uncertainty bounds "
            f"are well-calibrated.",
            S["body"],
        ),
    ]
    if d["is_cold"]:
        story.append(Paragraph(
            "⚠ Cold-start SKU: fewer than 13 weeks of history. "
            "Forecasts may be less accurate until more data is collected.",
            S["body"],
        ))
    story.append(Spacer(1, 0.2 * cm))


# ── Per-SKU section assemblers ────────────────────────────────────────────────

def _build_full(story: list, sku_id: str, sections: list,
                unit_margin: float, holding_cost: float) -> None:
    d = _sku_data(sku_id, unit_margin, holding_cost)
    story.append(Paragraph(f"  {sku_id}", S["sku_banner"]))
    story.append(Spacer(1, 0.2 * cm))
    if "replenishment" in sections:
        _sec_replenishment(d, story)
    if "forecast" in sections:
        _sec_forecast(d, include_chart=True, story=story)
    if "shap" in sections:
        _sec_shap(d, story)
    if "temporal" in sections:
        _sec_temporal(d, story)
    if "whatif" in sections:
        _sec_whatif(d, story)
    if "cost" in sections:
        _sec_cost(d, story)
    if "reliability" in sections:
        _sec_reliability(d, story)


def _build_brief(story: list, sku_id: str, unit_margin: float, holding_cost: float) -> None:
    d = _sku_data(sku_id, unit_margin, holding_cost)
    story.append(Paragraph(f"  {sku_id}", S["sku_banner"]))
    story.append(Spacer(1, 0.2 * cm))
    _sec_replenishment(d, story)
    _sec_forecast(d, include_chart=True, story=story)
    _sec_cost(d, story)


def _build_exec(story: list, sku_id: str, unit_margin: float, holding_cost: float) -> None:
    d    = _sku_data(sku_id, unit_margin, holding_cost)
    card = d["card"]
    story.append(Paragraph(f"  {sku_id}", S["sku_banner"]))
    story.append(Spacer(1, 0.2 * cm))
    bullets = [
        f"Urgency: <b>{card.get('urgency', '—')}</b> — "
        f"{'Reorder immediately' if card.get('trigger_reorder') else 'No immediate action needed'}",
        f"Forecast: <b>{math.ceil(d['iv']['q50'])} units</b> over {ds.HORIZON} days "
        f"(range {math.ceil(d['iv']['q10'])}–{math.ceil(d['iv']['q90'])})",
        f"Confidence: <b>{d['conf'].upper()}</b> — "
        f"interval coverage {ds.interval_coverage:.0f}%",
        f"Optimal order quantity: <b>{math.ceil(d['opt']['optimal_qty'])} units</b> "
        f"(newsvendor, SL {d['opt']['implied_service_level']:.0f}%)",
        f"Expected total cost at optimal: <b>${d['exp_tot']:.2f}</b> "
        f"(SO ${d['exp_so']:.2f} | OS ${d['exp_os']:.2f})",
    ]
    if d["top_shap"]:
        top3 = ", ".join(f"{f} ({v:+.3f})" for f, v in d["top_shap"][:3])
        bullets.append(f"Top SHAP drivers: {top3}")
    if d["is_cold"]:
        bullets.append("⚠ Cold-start SKU — limited history, treat forecast with caution")
    for b in bullets:
        story.append(Paragraph(f"• {b}", S["bullet"]))
    story.append(Spacer(1, 0.3 * cm))


# ── Public entry point ────────────────────────────────────────────────────────

def build_pdf(
    sku_ids:      list[str],
    template:     str,
    sections:     list[str],
    unit_margin:  float,
    holding_cost: float,
    model_key:    str,
) -> bytes:
    """Assemble and return a PDF as bytes."""
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=PAGE_MARGIN, leftMargin=PAGE_MARGIN,
        topMargin=PAGE_MARGIN,   bottomMargin=PAGE_MARGIN,
        title="XAI Retail Replenishment Report",
    )

    tpl_labels = {"brief": "Replenishment Brief", "full": "Full XAI Report", "exec": "Executive Summary"}
    story: list = []

    _cover(sku_ids, model_key, tpl_labels.get(template, template), story)
    _summary_table(sku_ids, unit_margin, holding_cost, story)

    for i, sku_id in enumerate(sku_ids):
        if i > 0:
            story.append(PageBreak())
        if template == "exec":
            _build_exec(story, sku_id, unit_margin, holding_cost)
        elif template == "brief":
            _build_brief(story, sku_id, unit_margin, holding_cost)
        else:
            _build_full(story, sku_id, sections, unit_margin, holding_cost)

    doc.build(story)
    return buf.getvalue()
