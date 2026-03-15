"""
Reusable Plotly chart builders.

Provides functions to create consistent, styled plots for the dashboard
and notebook analysis: time series, SHAP waterfalls, bar charts,
prediction intervals, PDP curves, and cost distributions.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def plot_time_series_with_bands(
    dates: pd.Series,
    actual: pd.Series,
    q10: pd.Series,
    q50: pd.Series,
    q90: pd.Series,
    title: str = "Forecast vs Actual",
) -> go.Figure:
    """Plot actual sales alongside quantile forecast bands.

    Parameters
    ----------
    dates : pd.Series
    actual : pd.Series
    q10, q50, q90 : pd.Series
    title : str

    Returns
    -------
    go.Figure
    """
    ...


def plot_shap_waterfall(
    shap_values: object,
    max_display: int = 10,
) -> go.Figure:
    """Create a SHAP waterfall chart for a single prediction.

    Parameters
    ----------
    shap_values : object
        SHAP Explanation for one instance.
    max_display : int

    Returns
    -------
    go.Figure
    """
    ...


def plot_global_shap_summary(
    importance_df: pd.DataFrame,
    top_n: int = 15,
) -> go.Figure:
    """Horizontal bar chart of global feature importance.

    Parameters
    ----------
    importance_df : pd.DataFrame
    top_n : int

    Returns
    -------
    go.Figure
    """
    ...


def plot_partial_dependence(
    pdp_df: pd.DataFrame,
    feature: str,
) -> go.Figure:
    """Plot a Partial Dependence curve for a single feature.

    Parameters
    ----------
    pdp_df : pd.DataFrame
    feature : str

    Returns
    -------
    go.Figure
    """
    ...


def plot_cost_distribution(
    sim_df: pd.DataFrame,
) -> go.Figure:
    """Histogram of simulated total cost from Monte Carlo runs.

    Parameters
    ----------
    sim_df : pd.DataFrame

    Returns
    -------
    go.Figure
    """
    ...
