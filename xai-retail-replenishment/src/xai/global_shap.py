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


def compute_global_shap(
    model: object,
    X: pd.DataFrame,
    max_samples: int = 1000,
) -> shap.Explanation:
    """Compute SHAP values across a sample of the full dataset.

    Parameters
    ----------
    model : object
    X : pd.DataFrame
    max_samples : int

    Returns
    -------
    shap.Explanation
    """
    ...


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
    ...


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
    ...


def generate_global_explanation_text(importance_df: pd.DataFrame) -> str:
    """Generate a natural-language summary of global feature importance.

    Parameters
    ----------
    importance_df : pd.DataFrame

    Returns
    -------
    str
    """
    ...
