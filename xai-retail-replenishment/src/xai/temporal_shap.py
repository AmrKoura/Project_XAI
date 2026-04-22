"""
Q6 — Temporal SHAP analysis.

Answers: "Is this SKU experiencing a demand spike or stable growth?"

Groups SHAP values by time-related features to reveal whether recent
demand changes are driven by seasonal patterns, trends, or anomalies.
"""

import pandas as pd
import numpy as np
import shap
from scipy import sparse


def _compute_shap_matrix(model: object, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Return (shap_matrix, feature_names) for all rows in X."""
    if hasattr(model, "named_steps") and "prep" in model.named_steps and "model" in model.named_steps:
        preprocessor = model.named_steps["prep"]
        estimator    = model.named_steps["model"]

        X_t = preprocessor.transform(X)
        if sparse.issparse(X_t):
            X_t = X_t.toarray()

        try:
            feature_names = list(preprocessor.get_feature_names_out())
        except Exception:
            feature_names = [f"feature_{i}" for i in range(X_t.shape[1])]

        explainer  = shap.TreeExplainer(estimator)
        shap_raw   = explainer(X_t)
        values     = np.asarray(shap_raw.values)
        return values, feature_names

    raise TypeError("Model must be a sklearn Pipeline with 'prep' and 'model' steps.")


def compute_temporal_shap(
    model: object,
    X: pd.DataFrame,
    dates: pd.Series,
    item_ids: pd.Series,
    sku_id: str,
) -> pd.DataFrame:
    """Compute SHAP values for one SKU across all its dates, ordered chronologically.

    Parameters
    ----------
    model : object
        Trained sklearn Pipeline.
    X : pd.DataFrame
        Full feature matrix.
    dates : pd.Series
        Date column aligned with X's index.
    item_ids : pd.Series
        SKU identifier column aligned with X's index.
    sku_id : str
        The SKU to analyse.

    Returns
    -------
    pd.DataFrame
        One row per date for the SKU. Columns: ``date``, ``prediction``,
        one column per feature containing that feature's SHAP value,
        plus ``total_shap`` (sum of all SHAP values = prediction − base_value).
    """
    mask    = item_ids == sku_id
    X_sku   = X.loc[mask].copy()
    d_sku   = dates.loc[mask].copy()

    if X_sku.empty:
        raise ValueError(f"SKU '{sku_id}' not found.")

    order   = d_sku.argsort()
    X_sku   = X_sku.iloc[order]
    d_sku   = d_sku.iloc[order]

    values, feature_names = _compute_shap_matrix(model, X_sku)
    preds   = model.predict(X_sku)

    clean_names = [str(f).replace("num__", "").replace("cat__", "") for f in feature_names]

    df = pd.DataFrame(values, columns=clean_names)
    df.insert(0, "date",       d_sku.values)
    df.insert(1, "prediction", preds)
    df["total_shap"] = values.sum(axis=1)
    return df.reset_index(drop=True)


def classify_demand_pattern(
    temporal_shap_df: pd.DataFrame,
    sku_id: str,
) -> dict:
    """Classify demand pattern as spike, stable, seasonal, or declining.

    Uses the prediction column from ``compute_temporal_shap`` to infer the
    overall demand trajectory across time periods.

    Parameters
    ----------
    temporal_shap_df : pd.DataFrame
        Output of ``compute_temporal_shap``.
    sku_id : str

    Returns
    -------
    dict
        Keys: ``pattern`` (str label), ``confidence`` (float 0–1),
        ``mean_pred`` (float), ``std_pred`` (float), ``trend_slope`` (float).
    """
    preds = temporal_shap_df["prediction"].values.astype(float)
    n     = len(preds)

    mean_pred  = float(preds.mean())
    std_pred   = float(preds.std())
    cv         = std_pred / mean_pred if mean_pred > 0 else 0.0

    # Linear trend slope (units per period).
    x           = np.arange(n)
    slope       = float(np.polyfit(x, preds, 1)[0]) if n >= 3 else 0.0
    slope_pct   = slope / mean_pred if mean_pred > 0 else 0.0

    # Spike: one period is > 2 std above the mean.
    z_scores    = (preds - mean_pred) / (std_pred + 1e-8)
    has_spike   = bool(np.any(np.abs(z_scores) > 2.0))

    # Pattern classification.
    if has_spike and cv > 0.3:
        pattern    = "spike"
        confidence = min(1.0, float(np.max(np.abs(z_scores))) / 3.0)
    elif slope_pct > 0.03:
        pattern    = "growing"
        confidence = min(1.0, abs(slope_pct) * 10)
    elif slope_pct < -0.03:
        pattern    = "declining"
        confidence = min(1.0, abs(slope_pct) * 10)
    elif cv < 0.15:
        pattern    = "stable"
        confidence = 1.0 - cv
    else:
        pattern    = "seasonal"
        confidence = 0.5

    return {
        "sku_id":      sku_id,
        "pattern":     pattern,
        "confidence":  round(confidence, 3),
        "mean_pred":   round(mean_pred,  2),
        "std_pred":    round(std_pred,   2),
        "trend_slope": round(slope,      4),
        "cv":          round(cv,         4),
    }


def get_top_temporal_drivers(
    temporal_shap_df: pd.DataFrame,
    n: int = 5,
) -> pd.DataFrame:
    """Identify which features drive the most SHAP variance over time.

    A feature with high temporal SHAP variance is one whose importance
    fluctuates across dates — a signal of time-varying demand drivers.

    Parameters
    ----------
    temporal_shap_df : pd.DataFrame
        Output of ``compute_temporal_shap``.
    n : int

    Returns
    -------
    pd.DataFrame
        Columns: ``['feature', 'mean_shap', 'std_shap', 'temporal_variance']``,
        sorted by ``temporal_variance`` descending.
    """
    skip = {"date", "prediction", "total_shap"}
    feat_cols = [c for c in temporal_shap_df.columns if c not in skip]

    rows = []
    for col in feat_cols:
        vals = temporal_shap_df[col].values.astype(float)
        rows.append({
            "feature":          col,
            "mean_shap":        float(vals.mean()),
            "std_shap":         float(vals.std()),
            "temporal_variance": float(vals.var()),
        })

    df = pd.DataFrame(rows).sort_values("temporal_variance", ascending=False).head(n).reset_index(drop=True)
    return df


def generate_temporal_explanation_text(
    sku_id: str,
    pattern_info: dict,
    top_drivers: pd.DataFrame,
) -> str:
    """Generate a natural-language explanation of the temporal demand pattern.

    Parameters
    ----------
    sku_id : str
    pattern_info : dict
        Output of ``classify_demand_pattern``.
    top_drivers : pd.DataFrame
        Output of ``get_top_temporal_drivers``.

    Returns
    -------
    str
    """
    pattern    = pattern_info["pattern"]
    confidence = pattern_info["confidence"]
    mean_pred  = pattern_info["mean_pred"]
    std_pred   = pattern_info["std_pred"]
    slope      = pattern_info["trend_slope"]

    pattern_descriptions = {
        "spike":    "an irregular demand spike — one or more periods show unusually high or low sales far above the norm",
        "growing":  "a growing demand trend — sales are consistently increasing period over period",
        "declining":"a declining demand trend — sales are gradually falling over time",
        "stable":   "stable demand — sales remain consistent with little variation across periods",
        "seasonal": "a seasonal or cyclical pattern — demand fluctuates regularly, likely tied to recurring events or cycles",
    }
    desc = pattern_descriptions.get(pattern, "an unclassified pattern")

    lines = [
        f"=== Temporal Demand Pattern: {sku_id} ===",
        f"Pattern detected: {pattern.upper()} (confidence: {confidence:.0%})",
        f"Description: This SKU shows {desc}.",
        f"Mean 7-day forecast: {mean_pred:.1f} units  |  Std dev: {std_pred:.1f} units",
    ]

    if pattern in ("growing", "declining"):
        lines.append(f"Trend slope: {slope:+.2f} units per period.")

    lines.append("")
    lines.append("Top features driving temporal SHAP variance (most time-sensitive drivers):")
    for _, row in top_drivers.iterrows():
        feat = str(row["feature"])
        mean_s = float(row["mean_shap"])
        std_s  = float(row["std_shap"])
        lines.append(
            f"  {feat:35s}  mean SHAP={mean_s:+.3f}  std={std_s:.3f}"
        )

    top1 = top_drivers.iloc[0]["feature"] if not top_drivers.empty else "unknown"
    lines.append("")
    lines.append(
        f"The most time-sensitive driver is '{top1}' — its SHAP contribution "
        f"varies the most across dates, making it the primary source of demand volatility for {sku_id}."
    )

    return "\n".join(lines)
