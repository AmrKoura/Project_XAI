"""
Q1 — Local SHAP explanations.

Answers: "Why is the system recommending a reorder for this SKU right now?"

Computes per-instance SHAP values to produce waterfall charts and
natural-language explanations for individual SKU forecasts.
"""

import pandas as pd
import numpy as np
import shap
from scipy import sparse
import math


def _extract_single_values(shap_values: shap.Explanation) -> np.ndarray:
    """Return a 1D SHAP vector for a single instance explanation."""
    raw_values = shap_values.values
    if sparse.issparse(raw_values):
        dense = raw_values.toarray()
        if dense.ndim == 2:
            return np.asarray(dense[0]).reshape(-1)
        return np.asarray(dense).reshape(-1)

    values = np.asarray(raw_values)
    if values.ndim == 0:
        raw = values.item()
        if sparse.issparse(raw):
            dense = raw.toarray()
            if dense.ndim == 2:
                return np.asarray(dense[0]).reshape(-1)
            return np.asarray(dense).reshape(-1)
        arr = np.asarray(raw)
        if arr.ndim == 0:
            return np.array([arr.item()])
        return arr.reshape(-1)
    if values.ndim == 1:
        return values
    if values.ndim == 2:
        return values[0]
    # Fallback for uncommon multi-output shapes.
    return np.asarray(values[0]).reshape(-1)


def _extract_single_data(shap_values: shap.Explanation) -> np.ndarray:
    """Return a 1D feature-value vector for a single instance explanation."""
    if shap_values.data is None:
        return np.array([])

    data = np.asarray(shap_values.data)
    if data.ndim == 1:
        return data
    return data[0]


def compute_local_shap(
    model: object,
    X: pd.DataFrame,
    idx: int,
) -> shap.Explanation:
    """Compute SHAP values for a single prediction.

    Parameters
    ----------
    model : object
        Trained LightGBM / XGBoost model.
    X : pd.DataFrame
        Feature matrix.
    idx : int
        Row index of the SKU instance to explain.

    Returns
    -------
    shap.Explanation
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    if len(X) == 0:
        raise ValueError("X is empty; cannot compute local SHAP.")

    if idx < 0 or idx >= len(X):
        raise IndexError(f"idx must be between 0 and {len(X) - 1}.")

    x_row = X.iloc[[idx]]

    # Prefer tree explainer on transformed features when using a sklearn pipeline.
    if hasattr(model, "named_steps") and "prep" in model.named_steps and "model" in model.named_steps:
        preprocessor = model.named_steps["prep"]
        estimator = model.named_steps["model"]

        x_row_t = preprocessor.transform(x_row)

        explainer = shap.TreeExplainer(estimator)
        shap_raw = explainer(x_row_t)

        try:
            feature_names = list(preprocessor.get_feature_names_out())
        except Exception:
            n_features = x_row_t.shape[1]
            feature_names = [f"feature_{i}" for i in range(n_features)]

        if sparse.issparse(x_row_t):
            data_row = x_row_t.toarray()[0]
        else:
            data_row = np.asarray(x_row_t)[0]

        values_row = _extract_single_values(shap_raw)
        base_value = float(np.asarray(shap_raw.base_values).reshape(-1)[0])

        return shap.Explanation(
            values=values_row,
            base_values=base_value,
            data=data_row,
            feature_names=feature_names,
        )

    # Generic fallback explainer for non-pipeline models.
    background = X.sample(min(200, len(X)), random_state=42)
    explainer = shap.Explainer(model.predict, background)
    shap_raw = explainer(x_row)
    if hasattr(shap_raw, "__getitem__"):
        return shap_raw[0]
    return shap_raw


def get_top_contributors(
    shap_values: shap.Explanation,
    n: int = 5,
) -> pd.DataFrame:
    """Extract the top-N features contributing to the prediction.

    Parameters
    ----------
    shap_values : shap.Explanation
    n : int

    Returns
    -------
    pd.DataFrame
        Columns: ``['feature', 'shap_value', 'feature_value']``.
    """
    if n <= 0:
        raise ValueError("n must be > 0.")

    values = _extract_single_values(shap_values)
    data = _extract_single_data(shap_values)

    if getattr(shap_values, "feature_names", None) is not None:
        feature_names = [str(name) for name in shap_values.feature_names]
    else:
        feature_names = [f"feature_{i}" for i in range(len(values))]

    if len(data) == 0:
        data = np.repeat(np.nan, len(values))

    out = pd.DataFrame(
        {
            "feature": feature_names,
            "shap_value": values,
            "feature_value": data,
        }
    )

    out["abs_shap"] = out["shap_value"].abs()
    out = out.sort_values("abs_shap", ascending=False).head(n).drop(columns=["abs_shap"]).reset_index(drop=True)
    return out


def generate_local_explanation_text(
    shap_values: shap.Explanation,
    forecast_value: float,
    sku_id: str,
    forecast_window: int = 7,
) -> str:
    """Generate a plain-English explanation for a single SKU forecast.

    Parameters
    ----------
    shap_values : shap.Explanation
    forecast_value : float
    sku_id : str

    Returns
    -------
    str
        Human-readable explanation string.
    """
    values = _extract_single_values(shap_values)
    data = _extract_single_data(shap_values)
    feature_names = [str(name) for name in getattr(shap_values, "feature_names", [])]

    def _value_for(*candidates: str) -> float | None:
        if len(feature_names) == 0 or len(data) == 0:
            return None
        for candidate in candidates:
            if candidate in feature_names:
                idx = feature_names.index(candidate)
                try:
                    return float(data[idx])
                except Exception:
                    return None
        return None

    # Prefer explicit historical average feature when present.
    typical = _value_for("num__item_mean_train", "item_mean_train")
    if typical is None:
        base_val = float(np.asarray(shap_values.base_values).reshape(-1)[0])
        typical = base_val

    forecast_pct = None
    if typical is not None and typical != 0:
        forecast_pct = ((forecast_value - typical) / typical) * 100.0

    roll7  = _value_for("num__sales_roll_mean_7",  "sales_roll_mean_7")
    lag7   = _value_for("num__sales_lag_7",         "sales_lag_7")
    trend_txt = "insufficient data"
    if roll7 is not None and lag7 is not None:
        if abs(roll7 - lag7) <= 0.75:
            trend_txt = f"stable (7-day rolling avg {roll7:.1f}, lag-7 {lag7:.1f})"
        elif roll7 > lag7:
            trend_txt = f"upward (7-day rolling avg {roll7:.1f}, lag-7 {lag7:.1f})"
        else:
            trend_txt = f"softening (7-day rolling avg {roll7:.1f}, lag-7 {lag7:.1f})"

    discount_depth = _value_for("num__discount_depth", "discount_depth")
    promo_active = bool(discount_depth is not None and discount_depth > 0.05)

    active_events = []
    if len(feature_names) > 0 and len(data) > 0:
        for name, val in zip(feature_names, data):
            if "event_" in str(name) and isinstance(val, (int, float, np.number)) and float(val) > 0.5:
                active_events.append(str(name).replace("num__", ""))

    if promo_active:
        promo_txt = f"Promotion signals detected (discount depth {discount_depth:.2f})."
    elif len(active_events) > 0:
        promo_txt = (
            "No active promotions detected. "
            + f"Recent event signals in the 28-day lookback: {', '.join(active_events[:3])}."
        )
    else:
        promo_txt = "No active promotions or events detected."

    rounded_units = int(math.ceil(forecast_value))
    lines = [
        f"SKU: {sku_id}",
        f"Forecast (next {forecast_window} days): {forecast_value:.1f} units ({rounded_units})",
        f"This SKU's typical demand: ~{typical:.1f} units/period",
    ]

    if forecast_pct is not None:
        pct = round(float(forecast_pct), 1)
        if abs(pct) < 0.05:
            pct = 0.0
        direction = "above" if pct >= 0 else "below"
        label = "stable" if abs(pct) < 5 else ("elevated" if pct > 0 else "softer")
        lines.append(
            f"Forecast is {abs(pct):.1f}% {direction} historical average -> {label} demand expected."
        )
    else:
        lines.append("Forecast vs historical average could not be computed.")

    lines.append(f"Recent trend: {trend_txt}")
    lines.append(promo_txt)
    return "\n".join(lines)
