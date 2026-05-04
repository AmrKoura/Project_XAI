"""
Q5 — Comparative SHAP analysis.

Answers: "Why is the reorder quantity higher for this SKU than a similar one?"

Compares local SHAP values between two SKUs to highlight which features
drive the difference in their forecasts and reorder quantities.
"""

import pandas as pd
import numpy as np
import shap
from scipy import sparse

# Features that describe a SKU's identity (stable across dates).
SKU_IDENTITY_FEATURES = [
    "item_mean_train",
    "item_std_train",
    "item_cv_train",
    "weeks_since_first_seen",
]


def _transform(model, X_in):
    """Apply all pre-model pipeline steps to X_in and return transformed array + feature names."""
    estimator = model.named_steps["model"]
    X_t = X_in
    for name, step in model.named_steps.items():
        if name == "model":
            break
        X_t = step.transform(X_t)
    if sparse.issparse(X_t):
        X_t = X_t.toarray()
    feature_names = list(X_in.columns) if hasattr(X_in, "columns") else [f"f{i}" for i in range(X_t.shape[1])]
    return estimator, X_t, feature_names


def _get_shap_for_row(model: object, X: pd.DataFrame, idx: int):
    """Return (feature_names, shap_values_1d, prediction) for one row."""
    if not (hasattr(model, "named_steps") and "model" in model.named_steps):
        raise TypeError("Model must be a sklearn Pipeline with a 'model' step.")

    x_row = X.iloc[[idx]]
    estimator, x_t, feature_names = _transform(model, x_row)

    explainer = shap.TreeExplainer(estimator)
    shap_raw  = explainer(x_t)
    values    = np.asarray(shap_raw.values)
    if values.ndim == 2:
        values = values[0]

    pred = float(model.predict(x_row)[0])
    return feature_names, values, pred


def _get_shap_for_batch(model: object, X: pd.DataFrame, idxs: list[int]):
    """Return (feature_names, shap_matrix, predictions) for multiple rows."""
    if not (hasattr(model, "named_steps") and "model" in model.named_steps):
        raise TypeError("Model must be a sklearn Pipeline with a 'model' step.")

    X_sub = X.iloc[idxs]
    estimator, x_t, feature_names = _transform(model, X_sub)

    explainer = shap.TreeExplainer(estimator)
    shap_raw  = explainer(x_t)
    values    = np.asarray(shap_raw.values)

    preds = model.predict(X_sub)
    return feature_names, values, preds


def find_similar_skus(
    X: pd.DataFrame,
    target_idx: int,
    n: int = 5,
    feature_cols: list[str] | None = None,
    item_ids: pd.Series | None = None,
) -> list[int]:
    """Find the N most similar SKUs to a target based on SKU-identity features.

    Uses normalized Euclidean distance on stable SKU-level features only
    (mean demand, std, CV, maturity) rather than date-driven lag features,
    so similarity reflects the SKU's demand profile — not a specific day's values.

    Parameters
    ----------
    X : pd.DataFrame
    target_idx : int
    n : int
    feature_cols : list[str] | None
        Override the default SKU identity features if needed.
    item_ids : pd.Series | None
        SKU identifier series aligned with X's index. When provided, all rows
        belonging to the same SKU as target_idx are excluded from the pool so
        the result is always a different SKU.

    Returns
    -------
    list[int]
        Row indices of similar SKUs (from different SKUs than the target).
    """
    if target_idx < 0 or target_idx >= len(X):
        raise IndexError(f"target_idx={target_idx} out of range.")

    # Use SKU identity features that are available in X.
    if feature_cols is not None:
        cols = feature_cols
    else:
        cols = [c for c in SKU_IDENTITY_FEATURES if c in X.columns]
        if not cols:
            cols = X.select_dtypes(include=[np.number]).columns.tolist()

    data = X[cols].copy().fillna(0).astype(float)
    std = data.std().replace(0, 1)
    data_norm = (data - data.mean()) / std

    target_vec = data_norm.iloc[target_idx].values

    # Exclude ALL rows of the same SKU (not just the single target row).
    if item_ids is not None:
        target_sku = item_ids.iloc[target_idx]
        mask = item_ids != target_sku
        other = data_norm.loc[mask]
    else:
        other = data_norm.drop(index=data_norm.index[target_idx])

    dists = np.linalg.norm(other.values - target_vec, axis=1)
    closest = np.argsort(dists)[:n]

    original_indices = other.index.tolist()
    return [original_indices[i] for i in closest]


def compare_two_skus(
    model: object,
    X: pd.DataFrame,
    idx_a: int,
    idx_b: int,
) -> pd.DataFrame:
    """Compare SHAP values between two SKU instances (single-row comparison).

    Parameters
    ----------
    model : object
    X : pd.DataFrame
    idx_a, idx_b : int
        Row indices of the two SKUs to compare.

    Returns
    -------
    pd.DataFrame
        Columns: ``['feature', 'shap_a', 'shap_b', 'diff']``, sorted by |diff|.
    """
    if idx_a < 0 or idx_a >= len(X):
        raise IndexError(f"idx_a={idx_a} out of range.")
    if idx_b < 0 or idx_b >= len(X):
        raise IndexError(f"idx_b={idx_b} out of range.")

    names_a, vals_a, pred_a = _get_shap_for_row(model, X, idx_a)
    names_b, vals_b, pred_b = _get_shap_for_row(model, X, idx_b)

    df = pd.DataFrame({
        "feature": [str(f).replace("num__", "").replace("cat__", "") for f in names_a],
        "shap_a":  vals_a,
        "shap_b":  vals_b,
        "diff":    vals_a - vals_b,
    })
    df["abs_diff"] = df["diff"].abs()
    df = df.sort_values("abs_diff", ascending=False).drop(columns=["abs_diff"]).reset_index(drop=True)
    df.attrs["pred_a"] = pred_a
    df.attrs["pred_b"] = pred_b
    return df


def compare_skus_aggregate(
    model: object,
    X: pd.DataFrame,
    item_ids: pd.Series,
    sku_a: str,
    sku_b: str,
) -> pd.DataFrame:
    """Compare two SKUs by averaging SHAP values across all their rows.

    This is more stable than a single-row comparison because date-specific
    lag fluctuations cancel out, leaving the persistent demand-driver profile
    for each SKU.

    Parameters
    ----------
    model : object
    X : pd.DataFrame
    item_ids : pd.Series
        Series of SKU identifiers aligned with X's index.
    sku_a, sku_b : str
        SKU identifiers to compare.

    Returns
    -------
    pd.DataFrame
        Columns: ``['feature', 'shap_a', 'shap_b', 'diff']``, sorted by |diff|.
        ``df.attrs['pred_a']`` and ``df.attrs['pred_b']`` hold mean predictions.
    """
    idxs_a = X.index[item_ids == sku_a].tolist()
    idxs_b = X.index[item_ids == sku_b].tolist()

    if not idxs_a:
        raise ValueError(f"SKU '{sku_a}' not found in item_ids.")
    if not idxs_b:
        raise ValueError(f"SKU '{sku_b}' not found in item_ids.")

    # Use positional indices for iloc.
    pos_a = [X.index.get_loc(i) for i in idxs_a]
    pos_b = [X.index.get_loc(i) for i in idxs_b]

    names, vals_a_mat, preds_a = _get_shap_for_batch(model, X, pos_a)
    _,     vals_b_mat, preds_b = _get_shap_for_batch(model, X, pos_b)

    mean_shap_a = vals_a_mat.mean(axis=0)
    mean_shap_b = vals_b_mat.mean(axis=0)

    df = pd.DataFrame({
        "feature": [str(f).replace("num__", "").replace("cat__", "") for f in names],
        "shap_a":  mean_shap_a,
        "shap_b":  mean_shap_b,
        "diff":    mean_shap_a - mean_shap_b,
    })
    df["abs_diff"] = df["diff"].abs()
    df = df.sort_values("abs_diff", ascending=False).drop(columns=["abs_diff"]).reset_index(drop=True)
    df.attrs["pred_a"] = float(preds_a.mean())
    df.attrs["pred_b"] = float(preds_b.mean())
    df.attrs["n_rows_a"] = len(idxs_a)
    df.attrs["n_rows_b"] = len(idxs_b)
    return df


def compare_models_for_sku(
    model_a: object,
    model_b: object,
    X: pd.DataFrame,
    item_ids: pd.Series,
    sku_id: str,
) -> pd.DataFrame:
    """Compare SHAP values for one SKU across two different models.

    Averages SHAP values across all test rows for the SKU so date-specific
    noise cancels out, leaving each model's persistent reasoning pattern.

    Returns
    -------
    pd.DataFrame
        Columns: ``['feature', 'shap_a', 'shap_b', 'diff']``, sorted by |diff|.
        ``df.attrs['pred_a']`` and ``df.attrs['pred_b']`` hold mean predictions.
    """
    idxs = X.index[item_ids == sku_id].tolist()
    if not idxs:
        raise ValueError(f"SKU '{sku_id}' not found.")

    pos = [X.index.get_loc(i) for i in idxs]

    names, vals_a, preds_a = _get_shap_for_batch(model_a, X, pos)
    _,     vals_b, preds_b = _get_shap_for_batch(model_b, X, pos)

    mean_a = vals_a.mean(axis=0)
    mean_b = vals_b.mean(axis=0)

    df = pd.DataFrame({
        "feature": [str(f).replace("num__", "").replace("cat__", "") for f in names],
        "shap_a":  mean_a,
        "shap_b":  mean_b,
        "diff":    mean_a - mean_b,
    })
    df["abs_diff"] = df["diff"].abs()
    df = df.sort_values("abs_diff", ascending=False).drop(columns=["abs_diff"]).reset_index(drop=True)
    df.attrs["pred_a"] = float(preds_a.mean())
    df.attrs["pred_b"] = float(preds_b.mean())
    return df


def generate_comparative_text(
    sku_a: str,
    sku_b: str,
    diff_df: pd.DataFrame,
) -> str:
    """Generate a natural-language explanation of why SKU A differs from SKU B.

    Parameters
    ----------
    sku_a, sku_b : str
    diff_df : pd.DataFrame
        Output of ``compare_two_skus`` or ``compare_skus_aggregate``.

    Returns
    -------
    str
    """
    pred_a  = diff_df.attrs.get("pred_a")
    pred_b  = diff_df.attrs.get("pred_b")
    n_a     = diff_df.attrs.get("n_rows_a")
    n_b     = diff_df.attrs.get("n_rows_b")
    top     = diff_df.head(5)

    lines = [f"=== Comparative SHAP: {sku_a} vs {sku_b} ==="]

    if n_a is not None and n_b is not None:
        lines.append(f"(Aggregated over {n_a} rows for {sku_a}, {n_b} rows for {sku_b})")

    if pred_a is not None and pred_b is not None:
        delta = pred_a - pred_b
        direction = "higher" if delta > 0 else "lower"
        lines.append(
            f"Mean forecast — {sku_a}: {pred_a:.1f} units  |  {sku_b}: {pred_b:.1f} units  "
            f"({abs(delta):.1f} units {direction} for {sku_a})"
        )

    lines.append("")
    lines.append("Top 5 features driving the difference:")
    for _, row in top.iterrows():
        feat = str(row["feature"])
        diff = float(row["diff"])
        shap_a = float(row["shap_a"])
        shap_b = float(row["shap_b"])
        direction = "pushes forecast UP" if diff > 0 else "pushes forecast DOWN"
        lines.append(
            f"  {feat:35s}  SHAP_A={shap_a:+.3f}  SHAP_B={shap_b:+.3f}  diff={diff:+.3f}  → {direction} for {sku_a}"
        )

    top1 = top.iloc[0]
    top1_feat = str(top1["feature"])
    top1_diff = float(top1["diff"])
    driver_dir = "higher demand" if top1_diff > 0 else "lower demand"
    lines.append("")
    lines.append(
        f"Primary driver: '{top1_feat}' accounts for {abs(top1_diff):.3f} SHAP units more "
        f"for {sku_a}, indicating {driver_dir} relative to {sku_b}."
    )

    return "\n".join(lines)
