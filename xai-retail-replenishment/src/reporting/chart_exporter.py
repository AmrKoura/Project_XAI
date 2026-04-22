"""
Convert Plotly figures to PNG bytes for embedding in PDF/DOCX reports.
"""

from __future__ import annotations

from io import BytesIO
import plotly.graph_objects as go


def fig_to_png_bytes(
    fig: go.Figure,
    width: int = 1400,
    height: int = 660,
) -> bytes:
    """Return PNG bytes for a Plotly figure (requires kaleido)."""
    try:
        # Force light background so charts look clean in the report
        fig = fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            font_color="#333333",
        )
        return fig.to_image(format="png", width=width, height=height, scale=1)
    except Exception as exc:
        print(f"[chart_exporter] Warning: could not render chart — {exc}")
        return b""


def fig_to_bytesio(fig: go.Figure, width: int = 1400, height: int = 660) -> BytesIO:
    return BytesIO(fig_to_png_bytes(fig, width, height))
