"""
Public API for report generation.

Usage:
    from reporting.report_builder import build_report
    pdf_bytes = build_report(["FOODS_1_001"], template="full", fmt="pdf", ...)
"""

from __future__ import annotations

from .pdf_builder  import build_pdf
from .docx_builder import build_docx

_DEFAULT_SECTIONS = [
    "replenishment", "forecast", "shap", "temporal", "whatif", "cost", "reliability",
]


def build_report(
    sku_ids:      list[str],
    template:     str   = "full",
    fmt:          str   = "pdf",
    sections:     list[str] | None = None,
    unit_margin:  float = 3.50,
    holding_cost: float = 0.80,
    model_key:    str   = "7d",
) -> bytes:
    """
    Generate a report and return raw bytes.

    Parameters
    ----------
    sku_ids      : List of SKU IDs to include.
    template     : "brief" | "full" | "exec"
    fmt          : "pdf" | "docx"
    sections     : Active sections for "full" template (default: all seven).
    unit_margin  : $/unit margin (for cost impact calculation).
    holding_cost : $/unit holding cost.
    model_key    : Active model ("7d" | "14d" | "28d").

    Returns
    -------
    bytes — PDF or DOCX file content ready for download.
    """
    if not sku_ids:
        raise ValueError("At least one SKU must be selected.")

    active_sections = sections if sections is not None else _DEFAULT_SECTIONS

    if fmt == "docx":
        return build_docx(sku_ids, template, active_sections, unit_margin, holding_cost, model_key)
    return build_pdf(sku_ids, template, active_sections, unit_margin, holding_cost, model_key)
