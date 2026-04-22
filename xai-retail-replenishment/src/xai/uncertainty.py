"""
Q3 + Q9 — Forecast uncertainty and cold-start analysis.

Answers:
  Q3: "How confident is the model in this forecast?"
  Q9: "Is the model less reliable for this SKU due to limited history?"

Since the model is a point estimator (MSE loss), prediction intervals are
derived from per-SKU training residuals:
  q50 = point prediction
  q10 = q50 - 1.28 * residual_std  (lower 80% bound)
  q90 = q50 + 1.28 * residual_std  (upper 80% bound)

Cold-start detection uses the engineered feature `weeks_since_first_seen`
which is already in the dataset, avoiding redundant date recomputation.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error


# ── interval helpers ──────────────────────────────────────────────────────────

def compute_sku_residual_std(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    item_ids: pd.Series,
) -> pd.Series:
    """Compute per-SKU residual standard deviation from training predictions.

    Parameters
    ----------
    y_true, y_pred : array-like
        Actual and predicted values on the training set.
    item_ids : pd.Series
        SKU identifier aligned with y_true / y_pred.

    Returns
    -------
    pd.Series
        Index = sku_id, values = residual std. SKUs with only one observation
        fall back to the global residual std.
    """
    residuals  = np.asarray(y_pred) - np.asarray(y_true)
    global_std = float(np.std(residuals))

    df = pd.DataFrame({"item_id": item_ids.values, "residual": residuals})
    sku_std = df.groupby("item_id")["residual"].std().fillna(global_std)

    # Replace zero std (single-obs SKUs) with global std.
    sku_std = sku_std.replace(0.0, global_std)
    return sku_std


def compute_prediction_interval(
    q50: float,
    residual_std: float,
    z: float = 1.28,
) -> dict[str, float]:
    """Return the prediction interval width and bounds.

    Intervals are derived from the per-SKU residual std on the training set.
    z=1.28 gives an ~80% interval (10th–90th percentile under normality).

    Parameters
    ----------
    q50 : float
        Point prediction (median).
    residual_std : float
        Per-SKU residual standard deviation from training.
    z : float
        Z-score multiplier. Default 1.28 → 80% interval.

    Returns
    -------
    dict[str, float]
        ``{'q10': lower, 'q50': q50, 'q90': upper, 'width': upper - lower}``
    """
    margin = z * residual_std
    q10    = max(0.0, q50 - margin)
    q90    = q50 + margin
    return {
        "q10":   round(q10,         2),
        "q50":   round(q50,         2),
        "q90":   round(q90,         2),
        "width": round(q90 - q10,   2),
    }


def confidence_label(interval_width: float, median: float) -> str:
    """Map forecast uncertainty to a human-readable confidence label.

    Uses the coefficient of variation of the interval width relative to
    the median prediction so the label scales with the forecast magnitude.

    Parameters
    ----------
    interval_width : float
    median : float

    Returns
    -------
    str
        One of ``'high'``, ``'moderate'``, ``'low'``.
    """
    if median <= 0:
        return "low"
    cv = interval_width / median
    if cv < 0.4:
        return "high"
    if cv < 1.0:
        return "moderate"
    return "low"


# ── cold-start detection ──────────────────────────────────────────────────────

def detect_cold_start_skus(
    df: pd.DataFrame,
    sku_col: str = "item_id",
    weeks_col: str = "weeks_since_first_seen",
    min_history_weeks: int = 13,
    date_col: str | None = "date",
) -> pd.DataFrame:
    """Identify SKUs with insufficient historical data.

    Uses the engineered feature ``weeks_since_first_seen`` when available
    (already in the dataset), falling back to date-based computation.

    Parameters
    ----------
    df : pd.DataFrame
    sku_col : str
    weeks_col : str
        Name of the weeks-since-first-seen feature. Used when present.
    min_history_weeks : int
        SKUs with fewer weeks of history are flagged. Default 13 (~3 months).
    date_col : str | None
        Fallback: date column used when ``weeks_col`` is not in df.

    Returns
    -------
    pd.DataFrame
        Columns: ``['sku_id', 'max_weeks_seen', 'is_cold_start']``,
        one row per SKU.
    """
    if weeks_col in df.columns:
        # Use engineered feature directly — most recent (max) value per SKU.
        sku_weeks = (
            df.groupby(sku_col)[weeks_col]
            .max()
            .reset_index()
            .rename(columns={sku_col: "sku_id", weeks_col: "max_weeks_seen"})
        )
    elif date_col is not None and date_col in df.columns:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        sku_weeks = (
            df.assign(_date=dates)
            .groupby(sku_col)["_date"]
            .agg(lambda x: (x.max() - x.min()).days // 7)
            .reset_index()
            .rename(columns={sku_col: "sku_id", "_date": "max_weeks_seen"})
        )
    else:
        raise ValueError("Either weeks_col or date_col must be present in df.")

    sku_weeks["is_cold_start"] = sku_weeks["max_weeks_seen"] < min_history_weeks
    return sku_weeks.sort_values("max_weeks_seen").reset_index(drop=True)


# ── subgroup evaluation ───────────────────────────────────────────────────────

def subgroup_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: pd.Series,
) -> pd.DataFrame:
    """Evaluate forecast accuracy per subgroup (e.g. cold-start vs established).

    Parameters
    ----------
    y_true, y_pred : np.ndarray
    groups : pd.Series
        Group label for each observation (e.g. 'cold_start' / 'established').

    Returns
    -------
    pd.DataFrame
        Columns: ``['group', 'n', 'MAE', 'RMSE', 'SMAPE', 'BIAS']``.
    """
    y_true  = np.asarray(y_true)
    y_pred  = np.asarray(y_pred)
    labels  = groups.values

    rows = []
    for grp in np.unique(labels):
        mask   = labels == grp
        yt, yp = y_true[mask], y_pred[mask]
        n      = int(mask.sum())
        mae    = float(mean_absolute_error(yt, yp))
        rmse   = float(np.sqrt(np.mean((yp - yt) ** 2)))
        denom  = np.abs(yt) + np.abs(yp) + 1e-8
        smape  = float(100.0 * np.mean(2.0 * np.abs(yp - yt) / denom))
        bias   = float(np.mean(yp - yt))
        rows.append({"group": grp, "n": n, "MAE": round(mae, 4),
                     "RMSE": round(rmse, 4), "SMAPE": round(smape, 2), "BIAS": round(bias, 4)})

    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)


# ── NLG ──────────────────────────────────────────────────────────────────────

def generate_confidence_text(
    sku_id: str,
    q50: float,
    residual_std: float,
    is_cold_start: bool,
    forecast_window: int = 7,
) -> str:
    """Generate a natural-language confidence statement for a SKU.

    Internally computes the prediction interval and confidence label so
    the caller only needs the raw inputs.

    Parameters
    ----------
    sku_id : str
    q50 : float
        Point prediction.
    residual_std : float
        Per-SKU residual std from training.
    is_cold_start : bool
    forecast_window : int

    Returns
    -------
    str
    """
    interval  = compute_prediction_interval(q50, residual_std)
    label     = confidence_label(interval["width"], q50)
    q10       = interval["q10"]
    q90       = interval["q90"]
    width     = interval["width"]

    label_descriptions = {
        "high":     "The model is confident in this forecast — the uncertainty range is narrow.",
        "moderate": "The model has moderate confidence — some demand variability is expected.",
        "low":      "The model has low confidence — the forecast range is wide and should be treated with caution.",
    }

    lines = [
        f"=== Forecast Confidence: {sku_id} ===",
        f"Point forecast (next {forecast_window} days): {q50:.1f} units",
        f"80% prediction interval: [{q10:.1f}, {q90:.1f}]  (width: {width:.1f} units)",
        f"Confidence level: {label.upper()}",
        label_descriptions.get(label, ""),
    ]

    if is_cold_start:
        lines.append("")
        lines.append(
            "WARNING (Q9 — Cold Start): This SKU has limited sales history. "
            "The model has seen fewer than 13 weeks of data for this item, "
            "which reduces forecast reliability. Widen safety stock buffer accordingly."
        )
    else:
        lines.append("")
        lines.append("This SKU has sufficient history — cold-start risk is low.")

    return "\n".join(lines)
