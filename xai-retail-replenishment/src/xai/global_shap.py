"""
Q2 + Q8 — Global SHAP explanations and data/feature audit.

Answers:
  Q2: "What are the most important features driving demand across all SKUs?"
  Q8: "Are there any data quality issues affecting the model's reliability?"

Computes global SHAP summary plots and performs feature-level data
quality auditing (missing rates, variance, correlation checks).
"""

import pandas as pd
import numpy as np
import shap
from scipy import sparse


def compute_global_shap(
    model: object,
    X: pd.DataFrame,
    max_samples: int = 1000,
) -> shap.Explanation:
    """Compute SHAP values across a sample of the full dataset.

    Parameters
    ----------
    model : object
        Trained LightGBM / XGBoost sklearn Pipeline.
    X : pd.DataFrame
        Feature matrix (raw, before preprocessing).
    max_samples : int
        Maximum rows to sample for SHAP computation.

    Returns
    -------
    shap.Explanation
        SHAP values with feature names attached.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    if len(X) == 0:
        raise ValueError("X is empty; cannot compute global SHAP.")

    sample = X.sample(min(max_samples, len(X)), random_state=42)

    if hasattr(model, "named_steps") and "prep" in model.named_steps and "model" in model.named_steps:
        preprocessor = model.named_steps["prep"]
        estimator = model.named_steps["model"]

        X_t = preprocessor.transform(sample)

        try:
            feature_names = list(preprocessor.get_feature_names_out())
        except Exception:
            n_features = X_t.shape[1]
            feature_names = [f"feature_{i}" for i in range(n_features)]

        if sparse.issparse(X_t):
            X_t = X_t.toarray()

        explainer = shap.TreeExplainer(estimator)
        shap_raw = explainer(X_t)

        base_values = np.asarray(shap_raw.base_values)

        return shap.Explanation(
            values=np.asarray(shap_raw.values),
            base_values=base_values,
            data=np.asarray(X_t),
            feature_names=feature_names,
        )

    # Fallback: generic kernel explainer (slow — avoid for large datasets).
    background = X.sample(min(100, len(X)), random_state=42)
    explainer = shap.Explainer(model.predict, background)
    return explainer(sample)


def rank_feature_importance(shap_values: shap.Explanation) -> pd.DataFrame:
    """Rank features by mean absolute SHAP value.

    Parameters
    ----------
    shap_values : shap.Explanation

    Returns
    -------
    pd.DataFrame
        Columns: ``['feature', 'mean_abs_shap']``, sorted descending.
    """
    values = np.asarray(shap_values.values)
    if values.ndim == 1:
        values = values.reshape(1, -1)

    mean_abs = np.abs(values).mean(axis=0)

    feature_names = getattr(shap_values, "feature_names", None)
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(mean_abs.shape[0])]

    df = pd.DataFrame({
        "feature": [str(f) for f in feature_names],
        "mean_abs_shap": mean_abs,
    })
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def feature_quality_audit(X: pd.DataFrame) -> pd.DataFrame:
    """Audit features for data quality issues.

    Checks missing rates, zero-variance columns, high correlations,
    and suspicious distributions that could undermine model reliability.

    Parameters
    ----------
    X : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Audit results per feature: ``['feature', 'missing_pct', 'std', 'flag']``.
    """
    rows = []
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = X[numeric_cols].corr().abs() if len(numeric_cols) > 1 else pd.DataFrame()

    high_corr_features: set[str] = set()
    if not corr_matrix.empty:
        for i, col_i in enumerate(numeric_cols):
            for j, col_j in enumerate(numeric_cols):
                if i < j and corr_matrix.loc[col_i, col_j] > 0.95:
                    high_corr_features.add(col_j)

    for col in X.columns:
        missing_pct = float(X[col].isna().mean() * 100)
        std = float(X[col].std()) if pd.api.types.is_numeric_dtype(X[col]) else float("nan")

        flags = []
        if missing_pct > 10:
            flags.append(f"high_missing({missing_pct:.1f}%)")
        if pd.api.types.is_numeric_dtype(X[col]) and std == 0.0:
            flags.append("zero_variance")
        if col in high_corr_features:
            flags.append("high_corr(>0.95)")
        if pd.api.types.is_numeric_dtype(X[col]):
            non_null = X[col].dropna()
            if len(non_null) > 0:
                zero_pct = float((non_null == 0).mean() * 100)
                if zero_pct > 80:
                    flags.append(f"mostly_zero({zero_pct:.1f}%)")

        rows.append({
            "feature": col,
            "missing_pct": round(missing_pct, 2),
            "std": round(std, 4) if not np.isnan(std) else np.nan,
            "flag": "; ".join(flags) if flags else "ok",
        })

    return pd.DataFrame(rows)


def generate_global_explanation_text(importance_df: pd.DataFrame) -> str:
    """Generate a natural-language summary of global feature importance.

    Parameters
    ----------
    importance_df : pd.DataFrame
        Output of ``rank_feature_importance`` — must have columns
        ``['feature', 'mean_abs_shap']``.

    Returns
    -------
    str
        Human-readable global explanation string.
    """
    if importance_df.empty:
        return "No feature importance data available."

    top5 = importance_df.head(5)
    top_names = [str(f).replace("num__", "").replace("cat__", "") for f in top5["feature"]]
    top_shap = top5["mean_abs_shap"].tolist()

    total_shap = float(importance_df["mean_abs_shap"].sum())
    top5_share = float(top5["mean_abs_shap"].sum()) / total_shap * 100 if total_shap > 0 else 0.0

    # Identify dominant driver category.
    lag_features = [f for f in top_names if "lag" in f or "roll" in f]
    price_features = [f for f in top_names if "price" in f or "discount" in f]
    event_features = [f for f in top_names if "event" in f]

    if len(lag_features) >= 2:
        driver_txt = "recent sales history (lag and rolling features)"
    elif len(price_features) >= 2:
        driver_txt = "pricing and promotion signals"
    elif len(event_features) >= 2:
        driver_txt = "event and holiday indicators"
    else:
        driver_txt = "a mix of lag, price, and contextual signals"

    lines = [
        "=== Global Feature Importance Summary ===",
        f"The model's demand forecasts are primarily driven by {driver_txt}.",
        "",
        "Top 5 features by mean absolute SHAP contribution:",
    ]
    for i, (name, val) in enumerate(zip(top_names, top_shap), 1):
        lines.append(f"  {i}. {name:35s}  mean |SHAP| = {val:.4f}")

    lines.append("")
    lines.append(
        f"These top 5 features collectively account for {top5_share:.1f}% "
        f"of total SHAP mass across all {len(importance_df)} features."
    )

    bottom = importance_df.tail(5)
    bottom_names = [str(f).replace("num__", "").replace("cat__", "") for f in bottom["feature"]]
    lines.append(
        f"Lowest-impact features: {', '.join(bottom_names)}."
    )

    return "\n".join(lines)
