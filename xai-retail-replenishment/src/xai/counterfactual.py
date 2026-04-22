"""
Q4 — Counterfactual analysis.

Answers: "What would happen to the forecast if a promotion were added or removed?"

Modifies individual features (promotion status, price, lead time) and
re-runs the model to show the causal-like effect on the forecast.
Partial Dependence Plots (PDP) are used as supporting evidence.
"""

import pandas as pd
import numpy as np


def generate_counterfactual(
    model: object,
    X_row: pd.DataFrame,
    feature: str,
    new_value: float | int,
) -> dict:
    """Generate a what-if counterfactual for a single feature change.

    Parameters
    ----------
    model : object
        Trained sklearn Pipeline.
    X_row : pd.DataFrame
        Single-row DataFrame of feature values for the instance to explain.
    feature : str
        Feature column to modify (must be in X_row.columns).
    new_value : float | int
        New value to assign to the feature.

    Returns
    -------
    dict
        ``{'original_pred', 'counterfactual_pred', 'delta',
           'original_feature_val', 'new_feature_val'}``
    """
    if not isinstance(X_row, pd.DataFrame):
        raise TypeError("X_row must be a single-row pd.DataFrame, not a Series.")
    if len(X_row) != 1:
        raise ValueError(f"X_row must have exactly 1 row, got {len(X_row)}.")
    if feature not in X_row.columns:
        raise ValueError(f"Feature '{feature}' not found in X_row columns.")

    original_feature_val = float(X_row[feature].iloc[0])
    original_pred        = float(model.predict(X_row)[0])

    X_cf = X_row.copy()
    X_cf[feature] = new_value
    cf_pred = float(model.predict(X_cf)[0])

    return {
        "original_feature_val":      original_feature_val,
        "new_feature_val":           float(new_value),
        "original_pred":             round(original_pred, 4),
        "counterfactual_pred":       round(cf_pred,       4),
        "delta":                     round(cf_pred - original_pred, 4),
    }


def batch_counterfactuals(
    model: object,
    X_row: pd.DataFrame,
    feature: str,
    values: list[float | int],
) -> pd.DataFrame:
    """Sweep a feature across multiple values and record predictions.

    Parameters
    ----------
    model : object
    X_row : pd.DataFrame
        Single-row DataFrame for the instance.
    feature : str
    values : list[float | int]
        Grid of values to sweep the feature over.

    Returns
    -------
    pd.DataFrame
        Columns: ``['feature_value', 'prediction', 'delta']``.
    """
    if not isinstance(X_row, pd.DataFrame):
        raise TypeError("X_row must be a single-row pd.DataFrame.")
    if len(X_row) != 1:
        raise ValueError(f"X_row must have exactly 1 row, got {len(X_row)}.")
    if feature not in X_row.columns:
        raise ValueError(f"Feature '{feature}' not found in X_row columns.")

    original_pred = float(model.predict(X_row)[0])
    rows = []

    for v in values:
        X_cf = X_row.copy()
        X_cf[feature] = v
        pred = float(model.predict(X_cf)[0])
        rows.append({
            "feature_value": float(v),
            "prediction":    round(pred, 4),
            "delta":         round(pred - original_pred, 4),
        })

    return pd.DataFrame(rows)


def partial_dependence(
    model: object,
    X: pd.DataFrame,
    feature: str,
    grid_resolution: int = 50,
) -> pd.DataFrame:
    """Compute Partial Dependence Plot data for a feature.

    For each grid point, replaces the feature value for ALL instances with
    that grid value and averages the predictions. This marginalises over
    the joint distribution of other features, showing the isolated effect
    of the chosen feature on the forecast.

    Parameters
    ----------
    model : object
    X : pd.DataFrame
    feature : str
    grid_resolution : int
        Number of evenly-spaced grid points across the feature's range.

    Returns
    -------
    pd.DataFrame
        Columns: ``['feature_value', 'avg_prediction', 'std_prediction']``.
    """
    if feature not in X.columns:
        raise ValueError(f"Feature '{feature}' not found in X columns.")

    feat_min  = float(X[feature].min())
    feat_max  = float(X[feature].max())
    grid      = np.linspace(feat_min, feat_max, grid_resolution)

    rows = []
    for v in grid:
        X_mod = X.copy()
        X_mod[feature] = v
        preds = model.predict(X_mod)
        rows.append({
            "feature_value":  round(float(v),           4),
            "avg_prediction": round(float(preds.mean()), 4),
            "std_prediction": round(float(preds.std()),  4),
        })

    return pd.DataFrame(rows)


def generate_counterfactual_text(
    sku_id: str,
    feature: str,
    original_feature_val: float,
    new_feature_val: float,
    original_pred: float,
    new_pred: float,
    delta: float,
) -> str:
    """Generate a natural-language what-if statement.

    Parameters
    ----------
    sku_id : str
    feature : str
    original_feature_val : float
        The feature's value before the change.
    new_feature_val : float
        The feature's value after the change.
    original_pred : float
        Forecast before the change.
    new_pred : float
        Forecast after the change.
    delta : float
        new_pred - original_pred.

    Returns
    -------
    str
    """
    direction  = "increase" if delta > 0 else "decrease"
    pct_change = (abs(delta) / original_pred * 100) if original_pred != 0 else 0.0

    feat_clean = feature.replace("num__", "").replace("_", " ")

    lines = [
        f"=== What-If Analysis: {sku_id} ===",
        f"Feature changed: {feat_clean}",
        f"  Before: {original_feature_val:.4g}  →  After: {new_feature_val:.4g}",
        f"",
        f"Forecast before change: {original_pred:.1f} units",
        f"Forecast after  change: {new_pred:.1f} units",
        f"Impact: {'+' if delta >= 0 else ''}{delta:.1f} units "
        f"({pct_change:.1f}% {direction})",
    ]

    # Contextual interpretation based on feature name.
    feat_lower = feature.lower()
    if "price" in feat_lower or "sell_price" in feat_lower:
        if delta < 0:
            lines.append("Interpretation: A higher price is associated with lower predicted demand.")
        else:
            lines.append("Interpretation: A lower price is associated with higher predicted demand.")
    elif "discount" in feat_lower:
        if delta > 0:
            lines.append("Interpretation: Applying a discount is expected to boost demand.")
        else:
            lines.append("Interpretation: Removing the discount reduces the demand forecast.")
    elif "snap" in feat_lower:
        if delta > 0:
            lines.append("Interpretation: SNAP benefit eligibility is associated with higher demand.")
        else:
            lines.append("Interpretation: Removing SNAP eligibility reduces the forecast.")
    elif "lag" in feat_lower or "roll" in feat_lower:
        lines.append(
            "Interpretation: Changing historical sales signals directly shifts "
            "the model's expectation of future demand."
        )

    return "\n".join(lines)
