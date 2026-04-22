"""
Q7 — Stockout distortion analysis.

Answers: "Could past stockouts have distorted the demand signal for this SKU?"

Since inventory (stock_on_hand) data is unavailable, stockout periods are
identified via a sales-based proxy: rows where the 7-day lag sales are at or
near zero. The analysis examines how the model's SHAP values for sales_lag_7
behave during these suspected stockout periods vs normal periods, and
estimates what demand likely was had stock been available.
"""

import pandas as pd
import numpy as np
import shap
from scipy import sparse


# ── helpers ──────────────────────────────────────────────────────────────────

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

        explainer = shap.TreeExplainer(estimator)
        shap_raw  = explainer(X_t)
        return np.asarray(shap_raw.values), feature_names

    raise TypeError("Model must be a sklearn Pipeline with 'prep' and 'model' steps.")


# ── public API ────────────────────────────────────────────────────────────────

def flag_potential_stockouts(
    df: pd.DataFrame,
    lag_col: str = "sales_lag_7",
    zero_threshold: float = 0.5,
) -> pd.DataFrame:
    """Add a binary stockout indicator using a sales-lag proxy.

    A period is flagged as a potential stockout when the 7-day lagged sales
    are at or near zero — the same signal a stockout would leave in the data,
    since the model cannot distinguish true zero demand from censored demand.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``lag_col``.
    lag_col : str
        Lag feature to use as the stockout proxy. Default ``'sales_lag_7'``.
    zero_threshold : float
        Values <= this are treated as zero-sales (potential stockout).

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with a new ``'is_potential_stockout'`` column (0/1).
    """
    if lag_col not in df.columns:
        raise ValueError(f"Column '{lag_col}' not found in df.")

    out = df.copy()
    out["is_potential_stockout"] = (out[lag_col] <= zero_threshold).astype(int)
    return out


def analyze_zero_lag_shap_impact(
    model: object,
    X: pd.DataFrame,
    item_ids: pd.Series,
    sku_id: str,
    lag_col: str = "sales_lag_7",
    zero_threshold: float = 0.5,
) -> pd.DataFrame:
    """Compare SHAP values of sales_lag_7 during zero-lag vs normal periods.

    When sales_lag_7 = 0, the model interprets it as "no recent demand."
    If this was actually a stockout, the model is being told the wrong thing —
    it underestimates demand. This function quantifies that distortion.

    Parameters
    ----------
    model : object
    X : pd.DataFrame
    item_ids : pd.Series
    sku_id : str
    lag_col : str
    zero_threshold : float

    Returns
    -------
    pd.DataFrame
        One row per date for the SKU. Columns: ``lag_value``, ``is_zero_lag``,
        ``prediction``, ``shap_lag7`` (SHAP of the lag feature),
        ``shap_distortion`` (difference vs non-zero-lag mean SHAP).
    """
    mask  = item_ids == sku_id
    X_sku = X.loc[mask].copy()

    if X_sku.empty:
        raise ValueError(f"SKU '{sku_id}' not found.")
    if lag_col not in X_sku.columns:
        raise ValueError(f"'{lag_col}' not in feature columns.")

    values, feature_names = _compute_shap_matrix(model, X_sku)
    clean_names = [str(f).replace("num__", "").replace("cat__", "") for f in feature_names]

    # Find the lag column index in the transformed feature space.
    lag_col_clean = lag_col.replace("num__", "")
    lag_idx = next(
        (i for i, n in enumerate(clean_names) if n == lag_col_clean),
        None,
    )
    if lag_idx is None:
        raise ValueError(f"Could not find '{lag_col}' in SHAP feature names.")

    shap_lag   = values[:, lag_idx]
    lag_values = X_sku[lag_col].values
    preds      = model.predict(X_sku)
    is_zero    = (lag_values <= zero_threshold).astype(int)

    # Distortion = how much lower the SHAP is vs non-zero periods.
    normal_mean_shap = float(shap_lag[is_zero == 0].mean()) if (is_zero == 0).any() else 0.0
    shap_distortion  = shap_lag - normal_mean_shap

    return pd.DataFrame({
        "lag_value":        lag_values,
        "is_zero_lag":      is_zero,
        "prediction":       preds,
        "shap_lag7":        shap_lag,
        "shap_distortion":  shap_distortion,
    }).reset_index(drop=True)


def estimate_censored_demand(
    df: pd.DataFrame,
    sku_id: str,
    sku_col: str = "item_id",
    sales_col: str = "aggregated_sales_7",
    date_col: str = "date",
    lag_col: str = "sales_lag_7",
    zero_threshold: float = 0.5,
) -> pd.DataFrame:
    """Estimate what demand would have been during suspected stockout periods.

    Uses linear interpolation from surrounding non-zero periods to infer
    the likely true demand that was masked by the stockout.

    Parameters
    ----------
    df : pd.DataFrame
    sku_id : str
    sku_col, sales_col, date_col, lag_col : str
    zero_threshold : float

    Returns
    -------
    pd.DataFrame
        SKU rows sorted by date with ``'estimated_demand'`` column.
        Non-stockout rows keep their actual sales value;
        stockout rows get the interpolated estimate.
    """
    sku_df = df[df[sku_col] == sku_id].copy().sort_values(date_col).reset_index(drop=True)

    if sku_df.empty:
        raise ValueError(f"SKU '{sku_id}' not found.")

    is_stockout = sku_df[lag_col] <= zero_threshold
    sku_df["is_potential_stockout"] = is_stockout.astype(int)

    # Replace zero-lag rows with NaN then interpolate.
    estimated = sku_df[sales_col].copy().astype(float)
    estimated[is_stockout] = np.nan
    estimated = estimated.interpolate(method="linear", limit_direction="both")

    sku_df["estimated_demand"] = estimated
    sku_df["demand_gap"] = (sku_df["estimated_demand"] - sku_df[sales_col]).clip(lower=0)
    return sku_df


def generate_stockout_text(
    sku_id: str,
    n_stockout_periods: int,
    avg_shap_impact: float,
    estimated_lost_demand: float | None = None,
    total_periods: int | None = None,
) -> str:
    """Generate a natural-language stockout distortion warning.

    Parameters
    ----------
    sku_id : str
    n_stockout_periods : int
        Number of periods flagged as potential stockouts.
    avg_shap_impact : float
        Mean SHAP distortion during stockout periods (negative = underestimation).
    estimated_lost_demand : float | None
        Total estimated demand gap across stockout periods.
    total_periods : int | None
        Total number of periods for this SKU (used to compute stockout rate).

    Returns
    -------
    str
    """
    lines = [f"=== Stockout Distortion Analysis: {sku_id} ==="]

    if total_periods is not None and total_periods > 0:
        rate = n_stockout_periods / total_periods * 100
        lines.append(f"Suspected stockout periods: {n_stockout_periods} / {total_periods} ({rate:.1f}% of history)")
    else:
        lines.append(f"Suspected stockout periods: {n_stockout_periods}")

    if n_stockout_periods == 0:
        lines.append("No stockout distortion detected. Demand signal appears clean.")
        return "\n".join(lines)

    severity = "HIGH" if abs(avg_shap_impact) > 5 else "MODERATE" if abs(avg_shap_impact) > 2 else "LOW"
    lines.append(f"Average SHAP distortion during stockout periods: {avg_shap_impact:.3f} units  [{severity} severity]")

    direction = "underestimating" if avg_shap_impact < 0 else "overestimating"
    lines.append(
        f"The model is likely {direction} demand for {sku_id} because the sales lag "
        f"feature carries zero values from periods where stock may have been unavailable."
    )

    if estimated_lost_demand is not None and estimated_lost_demand > 0:
        lines.append(f"Estimated total censored demand: {estimated_lost_demand:.1f} units across stockout periods.")
        lines.append(
            "This demand was never observed — the model learned from artificially low sales "
            "and may continue to under-forecast this SKU."
        )

    lines.append("")
    lines.append(
        "Recommendation: treat forecasts for this SKU with caution. "
        "Consider enriching with actual inventory data to distinguish true zero demand from stockouts."
    )

    return "\n".join(lines)
