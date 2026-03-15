"""
Train LightGBM and XGBoost gradient-boosted tree models.

LightGBM is the primary forecasting engine; XGBoost is trained as a
benchmark. Both support point and quantile regression.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict | None = None,
    quantiles: list[float] | None = None,
) -> dict[str, lgb.Booster]:
    """Train LightGBM models for point and quantile forecasts.

    Parameters
    ----------
    X_train, y_train : pd.DataFrame, pd.Series
        Training features and target.
    X_val, y_val : pd.DataFrame, pd.Series
        Validation features and target.
    params : dict | None
        LightGBM hyperparameters (uses config defaults if None).
    quantiles : list[float] | None
        Quantiles to train, e.g. ``[0.1, 0.5, 0.9]``.

    Returns
    -------
    dict[str, lgb.Booster]
        ``{'point': ..., 'q10': ..., 'q50': ..., 'q90': ...}``
    """
    ...


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict | None = None,
) -> xgb.Booster:
    """Train an XGBoost model as a benchmark forecaster.

    Parameters
    ----------
    X_train, y_train : pd.DataFrame, pd.Series
    X_val, y_val : pd.DataFrame, pd.Series
    params : dict | None

    Returns
    -------
    xgb.Booster
    """
    ...


def save_model(model: object, path: str) -> None:
    """Persist a trained model to disk.

    Parameters
    ----------
    model : object
        LightGBM Booster or XGBoost Booster.
    path : str
        Output file path.
    """
    ...


def load_model(path: str, model_type: str = "lightgbm") -> object:
    """Load a previously saved model.

    Parameters
    ----------
    path : str
    model_type : str
        ``'lightgbm'`` or ``'xgboost'``.

    Returns
    -------
    object
    """
    ...
