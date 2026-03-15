"""
Q5 — Comparative SHAP analysis.

Answers: "Why is the reorder quantity higher for this SKU than a similar one?"

Compares local SHAP values between two SKUs to highlight which features
drive the difference in their forecasts and reorder quantities.
"""

import pandas as pd
import numpy as np
import shap


def compare_two_skus(
    model: object,
    X: pd.DataFrame,
    idx_a: int,
    idx_b: int,
) -> pd.DataFrame:
    """Compare SHAP values between two SKU instances.

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
    ...


def find_similar_skus(
    X: pd.DataFrame,
    target_idx: int,
    n: int = 5,
    feature_cols: list[str] | None = None,
) -> list[int]:
    """Find the N most similar SKUs to a target based on feature distance.

    Parameters
    ----------
    X : pd.DataFrame
    target_idx : int
    n : int
    feature_cols : list[str] | None

    Returns
    -------
    list[int]
        Row indices of similar SKUs.
    """
    ...


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

    Returns
    -------
    str
    """
    ...
